from __future__ import annotations


def filter_by_route(items: list[dict], source: str, destination: str) -> list[dict]:
    src, dst = source.upper().strip(), destination.upper().strip()
    return [
        i for i in items
        if i.get("source", "").upper() == src and i.get("destination", "").upper() == dst
    ]


def filter_by_destination(items: list[dict], destination: str) -> list[dict]:
    dst = destination.strip().lower()
    return [i for i in items if i.get("destination", "").lower() == dst]


def filter_by_max_price(items: list[dict], max_price: float, field: str = "price") -> list[dict]:
    return [i for i in items if i.get(field, float("inf")) <= max_price]


def filter_by_min_rating(items: list[dict], min_rating: float) -> list[dict]:
    return [i for i in items if i.get("rating", 0) >= min_rating]


def filter_available(items: list[dict]) -> list[dict]:
    excluded = {"sold_out", "cancelled"}
    return [i for i in items if i.get("status", "available").lower() not in excluded]


def filter_by_field(items: list[dict], field: str, value: str) -> list[dict]:
    val = value.strip().lower()
    return [i for i in items if str(i.get(field, "")).lower() == val]


def filter_by_bool(items: list[dict], field: str, value: bool) -> list[dict]:
    return [i for i in items if bool(i.get(field)) is value]
