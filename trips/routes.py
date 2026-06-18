import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from auth.dependencies import get_current_user
from auth.models import User
from db import get_db_session
from pydantic import BaseModel
from trips.schemas import TripCreate, TripResponse, TripTransportOptionResponse
from trips.service import create_trip, delete_trip, get_trip, get_trip_transport_options, list_trips, update_trip_status

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/trips", tags=["Trips"])


@router.get("", response_model=list[TripResponse])
async def list_current_user_trips(
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> list[TripResponse]:
    return await list_trips(session, current_user.id, limit=limit, offset=offset)


@router.get("/{trip_id}", response_model=TripResponse)
async def get_current_user_trip(
    trip_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> TripResponse:
    trip = await get_trip(session, current_user.id, trip_id)
    if trip is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="trip not found")
    return trip


@router.post("", response_model=TripResponse, status_code=status.HTTP_201_CREATED)
async def create_current_user_trip(
    payload: TripCreate,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> TripResponse:
    logger.info("Creating trip for user %s: %s", current_user.id, payload.destination)
    return await create_trip(session, current_user.id, payload)


class TripStatusUpdate(BaseModel):
    status: str


@router.patch("/{trip_id}/status", response_model=TripResponse)
async def update_current_user_trip_status(
    trip_id: UUID,
    payload: TripStatusUpdate,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> TripResponse:
    trip = await update_trip_status(session, current_user.id, trip_id, payload.status)
    if trip is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="trip not found")
    logger.info("Trip %s status updated to %s by user %s", trip_id, payload.status, current_user.id)
    return trip


@router.get("/{trip_id}/transport", response_model=list[TripTransportOptionResponse])
async def get_trip_transport(
    trip_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> list[TripTransportOptionResponse]:
    trip = await get_trip(session, current_user.id, trip_id)
    if trip is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="trip not found")
    return await get_trip_transport_options(session, trip_id)


@router.delete("/{trip_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_current_user_trip(
    trip_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> Response:
    deleted = await delete_trip(session, current_user.id, trip_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="trip not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
