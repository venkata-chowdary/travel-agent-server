from __future__ import annotations

from ai.schemas import PreferenceContext, TravelPreferences


def format_preferences_block(
    prefs: PreferenceContext | TravelPreferences | None,
) -> str:
    if prefs is None:
        return ""
    if isinstance(prefs, PreferenceContext):
        return _format_preference_context(prefs)
    return _format_travel_preferences(prefs)


def _format_preference_context(ctx: PreferenceContext) -> str:
    fields = {
        "Travel style": ctx.travel_style,
        "Budget style": ctx.budget_style,
        "Preferred transport": ", ".join(ctx.preferred_transport) if ctx.preferred_transport else None,
        "Food preference": ctx.food_preference,
        "Hotel preference": ctx.hotel_preference,
        "Avoid": ", ".join(ctx.avoid) if ctx.avoid else None,
        "Home city": ctx.home_city,
        "Preferred currency": ctx.currency,
        "Memory confidence": f"{ctx.memory_confidence:.0%}" if ctx.memory_confidence else None,
    }
    set_fields = {k: v for k, v in fields.items() if v is not None}
    if not set_fields:
        return ""
    lines = "\n".join(f"  - {k}: {v}" for k, v in set_fields.items())
    return f"\n\nTraveler profile (synthesized from preferences and past trips):\n{lines}"


def _format_travel_preferences(prefs: TravelPreferences) -> str:
    fields = {
        "Budget range": prefs.budget_range,
        "Travel style": ", ".join(prefs.travel_style) if prefs.travel_style else None,
        "Dietary restrictions": ", ".join(prefs.dietary_restrictions) if prefs.dietary_restrictions else None,
        "Cabin class": prefs.cabin_class,
        "Accommodation": prefs.accommodation_type,
        "Trip pace": prefs.pace,
        "Home city": prefs.home_city,
        "Preferred currency": prefs.currency,
    }
    set_fields = {k: v for k, v in fields.items() if v is not None}
    if not set_fields:
        return ""
    lines = "\n".join(f"  - {k}: {v}" for k, v in set_fields.items())
    return f"\n\nTraveler preferences (use these to personalize your response):\n{lines}"
