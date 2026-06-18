from ai.agents.transport_agent import build_transport_choice, resolve_city_code


def test_resolve_city_code_maps_uppercase_goa_alias_to_goi():
    assert resolve_city_code("GOA") == "GOI"


def test_transport_choice_finds_dummy_options_for_hyderabad_to_uppercase_goa():
    result = build_transport_choice(
        origin="Hyderabad",
        destination="GOA",
        start_date="2026-06-20",
        days=2,
        travelers=1,
    )

    assert result.outbound_options
    assert result.return_options
    assert {option.mode for option in result.outbound_options} == {"flight", "train", "bus"}
    assert {option.mode for option in result.return_options} == {"flight", "train", "bus"}
    assert result.recommended_outbound_id is not None
    assert result.recommended_return_id is not None
