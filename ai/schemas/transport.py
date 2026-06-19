from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


TransportMode = Literal["flight", "train", "bus"]
TransportLeg = Literal["outbound", "return"]


class TransportOption(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(min_length=1)
    mode: TransportMode
    leg: TransportLeg
    provider: str = Field(min_length=1)
    from_: str = Field(min_length=1, alias="from")
    to: str = Field(min_length=1)
    depart: str = Field(min_length=1)
    arrive: str = Field(min_length=1)
    duration: str = Field(min_length=1)
    price: int = Field(ge=0)
    available_seats: int = Field(default=0, ge=0)
    rating: float | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class TransportChoiceResponse(BaseModel):
    origin: str = Field(min_length=1)
    destination: str = Field(min_length=1)
    start_date: str = Field(min_length=1)
    end_date: str | None = None
    days: int = Field(ge=1, le=30)
    travelers: int = Field(default=1, ge=1, le=20)
    outbound_options: list[TransportOption] = Field(default_factory=list)
    return_options: list[TransportOption] = Field(default_factory=list)
    recommended_outbound_id: str | None = None
    recommended_return_id: str | None = None
    unavailable_modes: list[str] = Field(default_factory=list)
    summary: str = Field(min_length=1)
    supervisor_note: str = Field(default="")


class TransportSelection(BaseModel):
    origin: str = Field(min_length=1)
    destination: str = Field(min_length=1)
    start_date: str = Field(min_length=1)
    end_date: str | None = None
    days: int = Field(ge=1, le=30)
    travelers: int = Field(default=1, ge=1, le=20)
    selected_options: list[TransportOption] = Field(default_factory=list)
