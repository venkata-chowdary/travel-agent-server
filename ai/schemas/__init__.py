from .flight import FlightOption, FlightSearchResult
from .preferences import PastTrip, PreferenceContext, TravelPreferences, UserProfile
from .travel import (
    BudgetBreakdown,
    ItineraryDay,
    ItineraryItem,
    TravelAgentStructuredResponse,
    WeatherNotes,
)
from .weather import DailyForecast, TripRisk, WeatherForecastResponse

__all__ = [
    "BudgetBreakdown",
    "DailyForecast",
    "FlightOption",
    "FlightSearchResult",
    "ItineraryDay",
    "ItineraryItem",
    "TravelAgentStructuredResponse",
    "PastTrip",
    "PreferenceContext",
    "TravelPreferences",
    "UserProfile",
    "TripRisk",
    "WeatherForecastResponse",
    "WeatherNotes",
]
