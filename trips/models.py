import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint, func, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from db import Base


class Trip(Base):
    __tablename__ = "trips"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    session_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    destination: Mapped[str] = mapped_column(String(255), nullable=False)
    origin: Mapped[str | None] = mapped_column(String(255), nullable=True)
    start_date: Mapped[str | None] = mapped_column(String(64), nullable=True)
    end_date: Mapped[str | None] = mapped_column(String(64), nullable=True)
    days: Mapped[int] = mapped_column(Integer, nullable=False)
    travelers: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")
    status: Mapped[str] = mapped_column(String(32), index=True, nullable=False, default="planning", server_default="planning")
    cover_emoji: Mapped[str] = mapped_column(String(32), nullable=False, default="\u2708\ufe0f", server_default="plane")
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    budget: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb"))
    itinerary: Mapped[list] = mapped_column(JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb"))
    hotel_options: Mapped[list] = mapped_column(JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb"))
    hotel_status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="not_searched", server_default="not_searched"
    )
    flight_options: Mapped[list] = mapped_column(JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb"))
    transport_options: Mapped[list] = mapped_column(JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb"))
    transport_status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="not_searched", server_default="not_searched"
    )
    daily_forecast: Mapped[list] = mapped_column(JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb"))
    trip_risks: Mapped[list] = mapped_column(JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb"))
    verification_tips: Mapped[list] = mapped_column(JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        index=True,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
    )

    __table_args__ = (
        Index(
            "uq_trips_draft_user_session",
            "user_id",
            "session_id",
            unique=True,
            postgresql_where=text("status = 'draft'"),
        ),
    )


# Status values: available | selected | skipped | expired
class TripTransportOption(Base):
    __tablename__ = "trip_transport_options"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    trip_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("trips.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    option_id: Mapped[str] = mapped_column(String(255), nullable=False)
    mode: Mapped[str] = mapped_column(String(20), nullable=False)       # flight / train / bus
    leg: Mapped[str] = mapped_column(String(20), nullable=False)        # outbound / return
    provider: Mapped[str] = mapped_column(String(255), nullable=False)
    from_city: Mapped[str] = mapped_column(String(255), nullable=False)
    to_city: Mapped[str] = mapped_column(String(255), nullable=False)
    depart: Mapped[str] = mapped_column(String(64), nullable=False)
    arrive: Mapped[str] = mapped_column(String(64), nullable=False)
    duration: Mapped[str] = mapped_column(String(64), nullable=False)
    price: Mapped[int] = mapped_column(Integer, nullable=False)
    available_seats: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    rating: Mapped[float | None] = mapped_column(Float, nullable=True)
    details: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb"))
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="available", server_default="available")
    is_recommended: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
    )

    __table_args__ = (
        UniqueConstraint("session_id", "option_id", name="uq_trip_transport_session_option"),
    )


# Status values: available | selected | skipped | expired
class TripHotelOption(Base):
    __tablename__ = "trip_hotel_options"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    trip_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("trips.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    option_id: Mapped[str] = mapped_column(String(255), nullable=False)
    destination: Mapped[str] = mapped_column(String(255), nullable=False)
    checkin: Mapped[str] = mapped_column(String(64), nullable=False)
    checkout: Mapped[str] = mapped_column(String(64), nullable=False)
    nights: Mapped[int] = mapped_column(Integer, nullable=False)
    travelers: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    provider: Mapped[str] = mapped_column(String(255), nullable=False)
    area: Mapped[str | None] = mapped_column(String(255), nullable=True)
    hotel_type: Mapped[str] = mapped_column(String(64), nullable=False)
    price_per_night: Mapped[int] = mapped_column(Integer, nullable=False)
    total_price: Mapped[int] = mapped_column(Integer, nullable=False)
    rating: Mapped[float] = mapped_column(Float, nullable=False)
    amenities: Mapped[list] = mapped_column(JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb"))
    distance_from_center_km: Mapped[float | None] = mapped_column(Float, nullable=True)
    available_rooms: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    refundable: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")
    breakfast_included: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="available", server_default="available")
    is_recommended: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
    )

    __table_args__ = (
        UniqueConstraint(
            "session_id",
            "destination",
            "checkin",
            "checkout",
            "option_id",
            name="uq_trip_hotel_session_search_option",
        ),
    )

    def to_hotel_option(self) -> dict:
        return {
            "id": self.option_id,
            "name": self.name,
            "provider": self.provider,
            "area": self.area,
            "hotel_type": self.hotel_type,
            "price_per_night": self.price_per_night,
            "total_price": self.total_price,
            "rating": self.rating,
            "amenities": self.amenities,
            "distance_from_center_km": self.distance_from_center_km,
            "available_rooms": self.available_rooms,
            "refundable": self.refundable,
            "breakfast_included": self.breakfast_included,
        }
