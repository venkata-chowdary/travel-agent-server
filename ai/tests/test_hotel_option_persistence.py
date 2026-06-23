from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

from ai.schemas.hotel import HotelChoiceResponse, HotelOption
from trips.models import TripHotelOption
from trips.service import (
    _attach_selected_hotel_options,
    link_hotel_options_to_trip,
    save_hotel_options,
)


class _ScalarRows:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class _Result:
    def __init__(self, *, rows=None, scalar_value=None):
        self._rows = rows or []
        self._scalar_value = scalar_value

    def scalars(self):
        return _ScalarRows(self._rows)

    def scalar(self):
        return self._scalar_value


class _Session:
    def __init__(self, first_result: _Result | None = None):
        self.first_result = first_result or _Result()
        self.statements = []
        self.committed = False

    async def execute(self, statement):
        self.statements.append(statement)
        if len(self.statements) == 1:
            return self.first_result
        return _Result()

    async def commit(self):
        self.committed = True


def _hotel_option(option_id: str, name: str, price: int) -> HotelOption:
    return HotelOption(
        id=option_id,
        name=name,
        provider="MockStay",
        area="Candolim",
        hotel_type="hotel",
        price_per_night=price,
        total_price=price * 2,
        rating=4.5,
        amenities=["wifi", "pool"],
        distance_from_center_km=2.1,
        available_rooms=3,
        refundable=True,
        breakfast_included=True,
    )


def _statement_values(statement) -> dict[str, object]:
    return {
        column.name: bind.value
        for column, bind in getattr(statement, "_values", {}).items()
    }


@pytest.mark.asyncio
async def test_save_hotel_options_builds_plain_insert_rows() -> None:
    trip_id = uuid4()
    choice = HotelChoiceResponse(
        destination="Goa",
        checkin="2026-07-10",
        checkout="2026-07-12",
        nights=2,
        travelers=2,
        options=[
            _hotel_option("hotel_1", "Casa Shoreline", 9000),
            _hotel_option("hotel_2", "Palm Court", 7000),
        ],
        recommended_id="hotel_1",
        summary="Pick a stay.",
    )
    session = _Session()

    await save_hotel_options(session, "session-1", trip_id, choice, commit=False)

    rows = session.statements[0]._multi_values[0]
    first = {column.name: value for column, value in rows[0].items()}
    second = {column.name: value for column, value in rows[1].items()}

    assert first["session_id"] == "session-1"
    assert first["trip_id"] == trip_id
    assert first["destination"] == "Goa"
    assert first["checkin"] == "2026-07-10"
    assert first["checkout"] == "2026-07-12"
    assert first["name"] == "Casa Shoreline"
    assert first["is_recommended"] is True
    assert second["is_recommended"] is False
    assert session.committed is False


@pytest.mark.asyncio
async def test_link_hotel_options_marks_selected_and_keeps_others_available() -> None:
    session = _Session(first_result=_Result(scalar_value=2))
    selected = _hotel_option("hotel_1", "Casa Shoreline", 9000)

    await link_hotel_options_to_trip(session, uuid4(), selected, commit=False)

    update_values = [_statement_values(statement) for statement in session.statements[1:]]

    assert {"status": "available"} in update_values
    assert {"status": "selected"} in update_values
    assert {"hotel_status": "selected"} in update_values
    assert session.committed is False


@pytest.mark.asyncio
async def test_link_hotel_options_marks_skipped_when_no_hotel_was_selected() -> None:
    session = _Session(first_result=_Result(scalar_value=0))

    await link_hotel_options_to_trip(session, uuid4(), None, was_skipped=True, commit=False)

    assert _statement_values(session.statements[1]) == {"hotel_status": "skipped"}
    assert session.committed is False


@pytest.mark.asyncio
async def test_selected_hotel_rows_replace_legacy_trip_json() -> None:
    trip_id = uuid4()
    trip = SimpleNamespace(id=trip_id, hotel_options=[{"id": "legacy"}])
    selected = TripHotelOption(
        id=uuid4(),
        session_id="session-1",
        trip_id=trip_id,
        option_id="hotel_1",
        destination="Goa",
        checkin="2026-07-10",
        checkout="2026-07-12",
        nights=2,
        travelers=2,
        name="Casa Shoreline",
        provider="MockStay",
        area="Candolim",
        hotel_type="hotel",
        price_per_night=9000,
        total_price=18000,
        rating=4.5,
        amenities=["wifi"],
        distance_from_center_km=2.1,
        available_rooms=3,
        refundable=True,
        breakfast_included=True,
        status="selected",
        is_recommended=True,
        created_at=datetime.now(timezone.utc),
    )
    session = _Session(first_result=_Result(rows=[selected]))

    await _attach_selected_hotel_options(session, [trip])

    assert trip.hotel_options == [selected.to_hotel_option()]
