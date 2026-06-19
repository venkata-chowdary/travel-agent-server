from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import HTTPException, status
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from sqlalchemy.ext.asyncio import AsyncSession

from ai.agent import run_travel_agent
from ai.schemas import TravelAgentChatResponse, TransportSelection
from auth.models import User
from chat.service import load_chat_history, save_chat_turn
from trips.service import (
    create_draft_trip,
    create_trip,
    get_draft_trip_by_session,
    get_trip,
    get_trip_by_session,
    link_transport_options_to_trip,
    save_transport_options,
    update_trip_from_plan,
)


def _dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", by_alias=True)
    return value


def _ai_message_content(row: Any) -> str:
    """Build a richer AIMessage string so the LLM can reason about prior trip plans."""
    payload = row.payload or {}
    if payload.get("response_type") == "trip_plan":
        plan = payload.get("trip_plan") or {}
        dest = plan.get("destination", "")
        days = plan.get("days", "")
        summary = plan.get("summary", "")
        budget = (plan.get("budget") or {}).get("total", "")
        parts = [row.content]
        if dest and days:
            parts.append(f"Planned: {days}-day trip to {dest}.")
        if summary:
            parts.append(summary)
        if budget:
            parts.append(f"Total budget: {budget}.")
        return " ".join(p for p in parts if p)
    if payload.get("response_type") == "transport_choice":
        tc = payload.get("transport_choice") or {}
        origin = tc.get("origin", "")
        dest = tc.get("destination", "")
        route = f"{origin} → {dest}" if origin and dest else origin or dest or "unknown route"
        return f"{row.content} Transport options were presented for {route}. User has not yet selected."
    return row.content


def _trip_context_message(trip: Any) -> BaseMessage:
    return SystemMessage(
        content=(
            "The user is editing an existing saved trip. Revise this trip rather than treating "
            "the request as a brand-new plan.\n\n"
            f"Trip ID: {trip.id}\n"
            f"Destination: {trip.destination}\n"
            f"Origin: {trip.origin or 'unknown'}\n"
            f"Dates: {trip.start_date or 'unknown'} to {trip.end_date or 'unknown'}\n"
            f"Days: {trip.days}\n"
            f"Travelers: {trip.travelers}\n"
            f"Summary: {trip.summary}\n"
            f"Budget: {_dump(trip.budget)}\n"
            f"Itinerary: {_dump(trip.itinerary)}\n"
            f"Selected transport: {_dump(trip.transport_options)}"
        )
    )


class AgentService:
    def __init__(self, session: AsyncSession, user: User) -> None:
        self._session = session
        self._user = user

    async def chat(
        self,
        message: str,
        session_id: str,
        transport_selection: TransportSelection | None,
        target_trip_id: UUID | None,
    ) -> TravelAgentChatResponse:
        lc_history = await self._build_lc_history(session_id)

        if target_trip_id is not None:
            trip = await get_trip(self._session, self._user.id, target_trip_id)
            if trip is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="trip not found")
            await self._session.refresh(trip)
            lc_history.insert(0, _trip_context_message(trip))

        # Release the DB connection back to the pool before the long AI call so
        # it doesn't time out while the agents are running (~minutes).
        await self._session.close()

        response = await run_travel_agent(
            self._user.id, message, session_id,
            history=lc_history,
            transport_selection=transport_selection,
        )

        user_payload = {k: v for k, v in {
            "target_trip_id": str(target_trip_id) if target_trip_id else None,
            "transport_selection": _dump(transport_selection) if transport_selection else None,
        }.items() if v is not None}

        async with self._session.begin():
            await save_chat_turn(
                self._session, session_id, self._user.id,
                message, response.assistant_message,
                user_payload=user_payload,
                assistant_payload=response.model_dump(mode="json", by_alias=True),
                commit=False,
            )
            if response.response_type == "transport_choice" and response.transport_choice is not None:
                await self._save_transport(session_id, target_trip_id, response)
            if response.response_type == "trip_plan" and response.trip_plan is not None:
                await self._save_trip_plan(session_id, target_trip_id, response)

        return response

    # ── private helpers ────────────────────────────────────────────────────────

    async def _build_lc_history(self, session_id: str) -> list[BaseMessage]:
        rows = await load_chat_history(self._session, session_id, self._user.id)
        messages: list[BaseMessage] = []
        for r in rows:
            if r.role == "user":
                ts = (r.payload or {}).get("transport_selection")
                if ts:
                    opts = ts.get("selected_options") or []
                    lines = ["User selected transport:"]
                    for opt in opts:
                        lines.append(
                            f"  {opt.get('leg','').capitalize()}: {opt.get('provider','')} "
                            f"{opt.get('from','')} → {opt.get('to','')} "
                            f"depart {opt.get('depart','')} arrive {opt.get('arrive','')} "
                            f"₹{opt.get('price','')}"
                        )
                    messages.append(SystemMessage(content="\n".join(lines)))
                messages.append(HumanMessage(content=r.content))
            else:
                messages.append(AIMessage(content=_ai_message_content(r)))
        return messages

    async def _save_transport(
        self,
        session_id: str,
        target_trip_id: UUID | None,
        response: TravelAgentChatResponse,
    ) -> None:
        if target_trip_id is not None:
            trip_id = target_trip_id
        else:
            draft = await create_draft_trip(
                self._session, self._user.id, session_id, response.transport_choice, commit=False,
            )
            if draft is None:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to create draft trip",
                )
            trip_id = draft.id
        await save_transport_options(
            self._session, session_id, trip_id, response.transport_choice, commit=False,
        )

    async def _save_trip_plan(
        self,
        session_id: str,
        target_trip_id: UUID | None,
        response: TravelAgentChatResponse,
    ) -> None:
        transport_was_skipped = False
        if target_trip_id is not None:
            finalized = await update_trip_from_plan(
                self._session, self._user.id, target_trip_id, response.trip_plan, commit=False,
            )
            if finalized is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="trip not found")
        else:
            draft = await get_draft_trip_by_session(self._session, self._user.id, session_id)
            if draft is not None:
                finalized = await update_trip_from_plan(
                    self._session, self._user.id, draft.id, response.trip_plan, commit=False,
                )
                if finalized is None:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND, detail="draft trip not found",
                    )
            else:
                transport_was_skipped = True
                # Guard against duplicate trips if the planner retries in the same session.
                existing = await get_trip_by_session(self._session, self._user.id, session_id)
                if existing is not None:
                    finalized = await update_trip_from_plan(
                        self._session, self._user.id, existing.id, response.trip_plan, commit=False,
                    )
                    if finalized is None:
                        # Row disappeared between SELECT and UPDATE; create fresh.
                        finalized = await create_trip(
                            self._session, self._user.id, response.trip_plan,
                            session_id=session_id, commit=False,
                        )
                else:
                    finalized = await create_trip(
                        self._session, self._user.id, response.trip_plan,
                        session_id=session_id, commit=False,
                    )
        await link_transport_options_to_trip(
            self._session, finalized.id, response.trip_plan.transport_options,
            was_skipped=transport_was_skipped, commit=False,
        )
        response.trip_plan = response.trip_plan.model_copy(update={"id": str(finalized.id)})
