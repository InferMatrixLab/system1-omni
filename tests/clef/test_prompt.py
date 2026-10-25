import json
from pathlib import Path

from models.clef.prompt import SUFFIX, prompt_pieces, render, token_lengths

FIXTURE = Path(__file__).parent / "fixtures" / "text_contract.json"


def test_render_keeps_strings_and_sorts_object_keys():
    assert render("Checkout") == "Checkout"
    assert render({"b": 1, "a": 2}) == '{"a":2,"b":1}'
    assert render(None) == "null"


def test_fixture_pieces_match_the_pinned_encoder_trace():
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert fixture["provenance"]["loader_sha256"]
    assert fixture["provenance"]["tokenizer_sha256"]
    for case in fixture["cases"]:
        pieces = prompt_pieces(case["request"])
        assert pieces == case["pieces"]
        assert pieces[-1] == SUFFIX
        assert "<think>\n\n</think>" in pieces[-1]
        assert sum(case["piece_token_lengths"]) == len(case["input_ids"])
        assert case["fixed_length"] + case["state_tokens"] == len(case["input_ids"])
        assert token_lengths(lambda text: len(text.encode("utf-8")), case["request"])[1] > 0


def test_choice_options_are_sorted_in_the_prompt_only():
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    pieces = prompt_pieces(fixture["cases"][0]["request"])
    billing = pieces.index('{"description":"Payments or invoices","option_id":"billing"}')
    technical = pieces.index('{"description":"Bugs or outages","option_id":"technical"}')
    assert billing < technical
    omitted = prompt_pieces(fixture["cases"][1]["request"])
    assert "urgent" in omitted
    assert omitted[omitted.index("\nFIELD 1\nID: urgent\nTYPE: noul\nINSTRUCTION: ") + 1] == "urgent"


def test_extra_noul_keys_do_not_become_options():
    pieces = prompt_pieces(
        {
            "state": "note",
            "questions": {
                "urgent": {
                    "type": "noul",
                    "criteria": {"true": "Yes.", "maybe": "Ignored by the encoder."},
                }
            },
        }
    )
    rendered = "\n".join(pieces)
    assert "maybe" not in rendered
    assert '"option_id":"true"' in rendered
    assert '"option_id":"false"' in rendered
