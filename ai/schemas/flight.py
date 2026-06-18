from __future__ import annotations

from pydantic import BaseModel, Field

try:
    # Pydantic v2
    from pydantic import ConfigDict  # type: ignore
except Exception:  # pragma: no cover
    ConfigDict = None  # type: ignore


class FlightOption(BaseModel):
    """
    Shared backend/frontend flight option shape.

    This matches `remix-of-eventspark/src/lib/types.ts::FlightOption` exactly.
    """

    if ConfigDict is not None:
        model_config = ConfigDict(populate_by_name=True)
    else:  # Pydantic v1 fallback
        class Config:
            allow_population_by_field_name = True

    id: str = Field(min_length=1)
    airline: str = Field(min_length=1)
    from_: str = Field(min_length=1, alias="from")
    to: str = Field(min_length=1)
    depart: str = Field(min_length=1)  # "HH:MM"
    arrive: str = Field(min_length=1)  # "HH:MM"
    duration: str = Field(min_length=1)  # "2h 15m"
    price: int = Field(ge=0)
    stops: int = Field(default=0, ge=0)
