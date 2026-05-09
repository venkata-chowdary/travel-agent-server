from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Query

from mock_apis import services
from mock_apis.schemas import ActivityInterest, BusType, FlightObjective, HotelType, MealType, TrainClass

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/mock", tags=["Mock Travel APIs"])


@router.get("/flights", summary="Search flights between two cities")
async def get_flights(
    source: Annotated[str, Query(description="IATA departure code: HYD, BLR, MAA, BOM, DEL, GOI, JAI, COK")],
    destination: Annotated[str, Query(description="IATA arrival code")],
    date: Annotated[str, Query(description="Travel date YYYY-MM-DD")],
    objective: Annotated[FlightObjective | None, Query(description="Sort by: cheapest | fastest | best_value")] = None,
    max_price: Annotated[float | None, Query(description="Max ticket price in INR", ge=0)] = None,
    airline: Annotated[str | None, Query(description="Filter by airline name")] = None,
):
    return services.search_flights(
        source=source,
        destination=destination,
        date=date,
        objective=objective,
        max_price=max_price,
        airline=airline,
    )


@router.get("/trains", summary="Search trains between two cities")
async def get_trains(
    source: Annotated[str, Query(description="Source station code")],
    destination: Annotated[str, Query(description="Destination station code")],
    date: Annotated[str, Query(description="Travel date YYYY-MM-DD")],
    class_type: Annotated[TrainClass | None, Query(description="Class: sleeper | 3ac | 2ac | chair_car")] = None,
    max_price: Annotated[float | None, Query(ge=0)] = None,
):
    return services.search_trains(
        source=source,
        destination=destination,
        date=date,
        class_type=class_type,
        max_price=max_price,
    )


@router.get("/buses", summary="Search buses between two cities")
async def get_buses(
    source: Annotated[str, Query(description="Source city/code")],
    destination: Annotated[str, Query(description="Destination city/code")],
    date: Annotated[str, Query(description="Travel date YYYY-MM-DD")],
    bus_type: Annotated[BusType | None, Query(description="Bus type: sleeper | semi_sleeper | ac | non_ac")] = None,
    max_price: Annotated[float | None, Query(ge=0)] = None,
):
    return services.search_buses(
        source=source,
        destination=destination,
        date=date,
        bus_type=bus_type,
        max_price=max_price,
    )


@router.get("/hotels", summary="Search hotels at a destination")
async def get_hotels(
    destination: Annotated[str, Query(description="City name: Goa, Bengaluru, Jaipur ...")],
    checkin: Annotated[str, Query(description="Check-in date YYYY-MM-DD")],
    checkout: Annotated[str, Query(description="Check-out date YYYY-MM-DD")],
    guests: Annotated[int, Query(ge=1, description="Number of guests")] = 1,
    max_price_per_night: Annotated[float | None, Query(ge=0)] = None,
    hotel_type: Annotated[HotelType | None, Query(description="budget | mid_range | luxury")] = None,
    min_rating: Annotated[float | None, Query(ge=0, le=5)] = None,
):
    return services.search_hotels(
        destination=destination,
        checkin=checkin,
        checkout=checkout,
        guests=guests,
        max_price_per_night=max_price_per_night,
        hotel_type=hotel_type,
        min_rating=min_rating,
    )


@router.get("/restaurants", summary="Search restaurants at a destination")
async def get_restaurants(
    destination: Annotated[str, Query(description="City name")],
    cuisine: Annotated[str | None, Query(description="Cuisine type: Indian, Seafood, Continental ...")] = None,
    max_price_per_person: Annotated[float | None, Query(ge=0)] = None,
    meal_type: Annotated[MealType | None, Query(description="breakfast | lunch | dinner | snacks")] = None,
    veg_only: Annotated[bool | None, Query(description="true = vegetarian friendly only")] = None,
    min_rating: Annotated[float | None, Query(ge=0, le=5)] = None,
):
    return services.search_restaurants(
        destination=destination,
        cuisine=cuisine,
        max_price_per_person=max_price_per_person,
        meal_type=meal_type,
        veg_only=veg_only,
        min_rating=min_rating,
    )


@router.get("/activities", summary="Search activities and experiences at a destination")
async def get_activities(
    destination: Annotated[str, Query(description="City name")],
    interest: Annotated[ActivityInterest | None, Query(description="adventure | beach | culture | nature | nightlife | food | wellness")] = None,
    max_price: Annotated[float | None, Query(ge=0)] = None,
    duration_hours: Annotated[float | None, Query(ge=0, description="Max activity duration in hours")] = None,
    min_rating: Annotated[float | None, Query(ge=0, le=5)] = None,
):
    return services.search_activities(
        destination=destination,
        interest=interest,
        max_price=max_price,
        duration_hours=duration_hours,
        min_rating=min_rating,
    )
