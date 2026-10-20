#!/usr/bin/env python3
"""Record the native compiler-equivalence fixtures from the pinned reference.

Runs the reference worker's own wire parsing, preparation and pinned
``valen.data.compiler.Compiler`` -- everything the native Rust processing layer
must reproduce -- without loading the backbone, so it needs no GPU:

    PYTHONPATH=src .venv/bin/python recipe/valen/generate_native_fixtures.py \
        --valen-source weights/Valen --checkpoint weights/Valen-Preview-0923 \
        --base weights/Qwen3.5-2B --out tests/valen/data

The output ``native_compiler.json`` pins, per recorded request: the rendered
chat-template state (before and after image-placeholder expansion), the
question records, the base token ids, every branch suffix, the candidate-end
and decision-end positions, the token accounting, and a rejection corpus with
the reference worker's exact error bodies and status codes. The Rust suite
(tests/valen/native*.rs) asserts equality against it.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import io
import json
import subprocess
import sys
from pathlib import Path

REFERENCE_REVISION = "750bfcfbb48a5275534a9c912257ebe83ca57a97"
BASE_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
CHECKPOINT_SHA256 = (
    "836622efe78fe757e2627aa6050c223d461c424ea430a1c110c15e6f42f5a012"
)

IMAGE_PAD = 248_056
NOUL_TRUE = "True / 是：满足问题中的条件。"
NOUL_FALSE = "False / 否：不满足问题中的条件。"

REPO = Path(__file__).resolve().parents[2]


def png_bytes(width, height, colour):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (width, height), colour).save(buf, format="PNG")
    return buf.getvalue()


def jpeg_bytes(width, height, colour):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (width, height), colour).save(buf, format="JPEG")
    return buf.getvalue()


def apng_bytes():
    from PIL import Image

    frames = [Image.new("RGB", (32, 32), (i * 40 % 255, 0, 0)) for i in range(2)]
    buf = io.BytesIO()
    frames[0].save(
        buf, format="PNG", save_all=True, append_images=frames[1:], duration=100, loop=0
    )
    return buf.getvalue()


def truncated_png_bytes():
    data = png_bytes(64, 64, (1, 2, 3))
    return data[: len(data) // 2]


def data_url(data, mime):
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")


def body_text(value) -> str:
    """The exact request bytes the corpus runs through the worker path."""

    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


# --- the corpus -------------------------------------------------------------
#
# Valid cases: run parse -> prepare -> compile and record the full layout.
# The optional "image" note regenerates the same image on the Rust side; only
# the dimensions matter to token layout, so encoder differences are harmless.

CHOICE_MIN = {
    "type": "choice",
    "instructions": "Choose the only candidate.",
    "criteria": {"ok": "OK"},
}


def valid_cases():
    text_state = "The card was charged twice for one order."
    return [
        {
            "name": "text_choice",
            "body": {
                "model": "valen-preview-0923",
                "state": text_state,
                "questions": {
                    "refund": {
                        "type": "choice",
                        "instructions": "Decide the refund action.",
                        "criteria": {
                            "refund": "Refund the duplicate charge",
                            "wait": "Wait for review",
                        },
                    }
                },
            },
        },
        {
            "name": "text_trim_edge",
            "body": {
                "model": "valen-preview-0923",
                "state": "\u00a0\u3000 The order 42 was flagged, twice. \u001c\u2028 ",
                "questions": {
                    "unicode": {
                        "type": "choice",
                        "instructions": " 分类这个订单。 ",
                        "criteria": {"正常": "订单正常", "异常": "订单异常"},
                    },
                    "structured": {
                        "type": "noul",
                        "instructions": "Is the JSON state an object?",
                        "criteria": {"true": "It parses as an object"},
                    },
                },
            },
        },
        {
            "name": "text_state_object",
            "body": {
                "model": "valen-preview-0923",
                "state": {"orders": [1, 2], "note": None, "ratio": 2.5},
                "questions": {"pick": CHOICE_MIN},
            },
        },
        {
            "name": "choice_null_label_uses_key",
            "body": {
                "model": "valen-preview-0923",
                "state": "A short state.",
                "questions": {
                    "q": {
                        "type": "choice",
                        "instructions": "Pick.",
                        "criteria": {"fallback": None, "described": "Described."},
                    }
                },
            },
        },
        {
            "name": "image_choice",
            "body": json.loads(
                (REPO / "recipe/valen/example-request.json").read_text(encoding="utf-8")
            ),
            "image": {"kind": "embedded"},
        },
        {
            "name": "image_choice_generated",
            "body": {
                "model": "valen-preview-0923",
                "state": {"image": None},
                "questions": {"what": {
                    "type": "choice",
                    "instructions": "Describe the picture.",
                    "criteria": {"blue": "It is blue.", "dark": "It is dark."},
                }},
            },
            "image": {
                "kind": "generated",
                "format": "png",
                "size": [1000, 700],
                "colour": [40, 90, 160],
            },
        },
        {
            "name": "image_choice_jpeg",
            "body": {
                "model": "valen-preview-0923",
                "state": {"image": None},
                "questions": {"what": CHOICE_MIN},
            },
            "image": {
                "kind": "generated",
                "format": "jpeg",
                "size": [640, 480],
                "colour": [200, 30, 30],
            },
        },
        {
            "name": "image_boundary_pixels",
            "body": {
                "model": "valen-preview-0923",
                "state": {"image": None},
                "questions": {"what": CHOICE_MIN},
            },
            "image": {
                "kind": "generated",
                "format": "png",
                "size": [2048, 512],
                "colour": [0, 0, 0],
            },
        },
        {
            "name": "image_upscale_small",
            "body": {
                "model": "valen-preview-0923",
                "state": {"image": None},
                "questions": {"what": CHOICE_MIN},
            },
            "image": {
                "kind": "generated",
                "format": "png",
                "size": [100, 40],
                "colour": [250, 250, 250],
            },
        },
        {
            "name": "text_mixed_types",
            "body": {
                "model": "valen-preview-0923",
                "state": "Shared state text.",
                "questions": {
                    "pick": {
                        "type": "choice",
                        "instructions": "Pick one.",
                        "criteria": {
                            "a": "Option A",
                            "b": "Option B",
                            "c": "Option C",
                        },
                    },
                    "yn": {"type": "noul", "instructions": "Is it?"},
                    "yn2": {
                        "type": "noul",
                        "instructions": "Really?",
                        "criteria": {"true": "Certainly so"},
                    },
                    "grade": {
                        "type": "score",
                        "instructions": "Grade it.",
                        "criteria": ["low", "mid", "high"],
                    },
                },
            },
        },
        {
            "name": "image_mixed_types",
            "body": {
                "model": "valen-preview-0923",
                "state": {"image": None},
                "questions": {
                    "what": {
                        "type": "choice",
                        "instructions": "What is shown?",
                        "criteria": {"dot": "A dot.", "square": "A square."},
                    },
                    "grade": {
                        "type": "score",
                        "instructions": "Rate the clarity.",
                        "criteria": ["murky", "ok", "crisp"],
                    },
                },
            },
            "image": {
                "kind": "generated",
                "format": "png",
                "size": [32, 32],
                "colour": [90, 40, 160],
            },
        },
    ]


def rejection_cases():
    model = "valen-preview-0923"
    ok_state = "A short state."
    choice_ok = {
        "type": "choice",
        "instructions": "Pick.",
        "criteria": {"a": "Option A"},
    }
    cases = [
        {"name": "body_not_json", "body": "{not json"},
        {"name": "body_trailing", "body": '{"model": "%s"} 1' % model},
        {
            "name": "body_duplicate_key",
            "body": '{"model": "%s", "model": "%s", "state": "s", "questions": {}}' % (model, model),
        },
        {"name": "body_nan", "body": '{"model": NaN}'},
        {"name": "body_infinity", "body": '{"model": Infinity}'},
        {"name": "body_negative_infinity", "body": '{"a": -Infinity}'},
        {"name": "body_number_out_of_range", "body": '{"a": 1e999}'},
        {"name": "body_not_object", "body": "[1, 2]"},
        {"name": "body_empty", "body": ""},
        {"name": "body_lone_surrogate", "body_b64": "e30gfSJ9"},
        {"name": "body_not_utf8", "body_b64": "AP5+"},
        {"name": "body_oversize", "synthesize": "body_oversize"},
        {
            "name": "wrong_model",
            "body": {"model": "valen-preview-0924", "state": ok_state, "questions": {"q": choice_ok}},
        },
        {
            "name": "extra_top_field",
            "body": {"model": model, "state": ok_state, "questions": {"q": choice_ok}, "extra": 1},
        },
        {"name": "missing_state", "body": {"model": model, "questions": {"q": choice_ok}}},
        {
            "name": "questions_missing",
            "body": {"model": model, "state": ok_state},
        },
        {"name": "questions_empty", "body": {"model": model, "state": ok_state, "questions": {}}},
        {
            "name": "questions_too_many",
            "body": {
                "model": model,
                "state": ok_state,
                "questions": {f"q{i}": choice_ok for i in range(9)},
            },
        },
        {
            "name": "questions_not_object",
            "body": {"model": model, "state": ok_state, "questions": [choice_ok]},
        },
        {
            "name": "question_name_empty",
            "body": {"model": model, "state": ok_state, "questions": {"": choice_ok}},
        },
        {
            "name": "question_name_too_long",
            "body": {"model": model, "state": ok_state, "questions": {"q" * 257: choice_ok}},
        },
        {
            "name": "question_unknown_field",
            "body": {
                "model": model,
                "state": ok_state,
                "questions": {"q": {**choice_ok, "bogus": 1}},
            },
        },
        {
            "name": "question_not_object",
            "body": {"model": model, "state": ok_state, "questions": {"q": "choice"}},
        },
        {
            "name": "question_bad_type",
            "body": {
                "model": model,
                "state": ok_state,
                "questions": {"q": {"type": "rank", "instructions": "x", "criteria": {"a": "b"}}},
            },
        },
        {
            "name": "question_type_missing",
            "body": {
                "model": model,
                "state": ok_state,
                "questions": {"q": {"instructions": "x", "criteria": {"a": "b"}}},
            },
        },
        {
            "name": "instructions_missing",
            "body": {
                "model": model,
                "state": ok_state,
                "questions": {"q": {"type": "choice", "criteria": {"a": "b"}}},
            },
        },
        {
            "name": "instructions_number",
            "body": {
                "model": model,
                "state": ok_state,
                "questions": {"q": {"type": "choice", "instructions": 5, "criteria": {"a": "b"}}},
            },
        },
        {
            "name": "instructions_whitespace",
            "body": {
                "model": model,
                "state": ok_state,
                "questions": {"q": {"type": "choice", "instructions": "  \t ", "criteria": {"a": "b"}}},
            },
        },
        {
            "name": "instructions_special_token",
            "body": {
                "model": model,
                "state": ok_state,
                "questions": {"q": {"type": "choice", "instructions": "Use <|audio_pad|>", "criteria": {"a": "b"}}},
            },
        },
        {
            "name": "instructions_object_ok_state_special",
            "body": {
                "model": model,
                "state": {"v": "<|vision_start|>"},
                "questions": {"q": {"type": "choice", "instructions": {"n": 1}, "criteria": {"a": "b"}}},
            },
        },
        {
            "name": "noul_criteria_bad_key",
            "body": {
                "model": model,
                "state": ok_state,
                "questions": {"q": {"type": "noul", "instructions": "x", "criteria": {"yes": "Maybe"}}},
            },
        },
        {
            "name": "noul_criteria_empty",
            "body": {"model": model, "state": ok_state, "questions": {"q": {"type": "noul", "instructions": "x", "criteria": {}}}},
        },
        {
            "name": "noul_criterion_whitespace",
            "body": {
                "model": model,
                "state": ok_state,
                "questions": {"q": {"type": "noul", "instructions": "x", "criteria": {"true": "  "}}},
            },
        },
        {
            "name": "noul_criterion_number",
            "body": {
                "model": model,
                "state": ok_state,
                "questions": {"q": {"type": "noul", "instructions": "x", "criteria": {"false": 3}}},
            },
        },
        {
            "name": "score_levels_one",
            "body": {
                "model": model,
                "state": ok_state,
                "questions": {"q": {"type": "score", "instructions": "x", "criteria": ["only"]}},
            },
        },
        {
            "name": "score_levels_eleven",
            "body": {
                "model": model,
                "state": ok_state,
                "questions": {"q": {"type": "score", "instructions": "x", "criteria": [str(i) for i in range(11)]}},
            },
        },
        {
            "name": "score_levels_not_list",
            "body": {
                "model": model,
                "state": ok_state,
                "questions": {"q": {"type": "score", "instructions": "x", "criteria": {"0": "a", "1": "b"}}},
            },
        },
        {
            "name": "score_level_empty",
            "body": {
                "model": model,
                "state": ok_state,
                "questions": {"q": {"type": "score", "instructions": "x", "criteria": ["ok", ""]}},
            },
        },
        {
            "name": "choice_criteria_missing",
            "body": {"model": model, "state": ok_state, "questions": {"q": {"type": "choice", "instructions": "x"}}},
        },
        {
            "name": "choice_criteria_empty",
            "body": {
                "model": model,
                "state": ok_state,
                "questions": {"q": {"type": "choice", "instructions": "x", "criteria": {}}},
            },
        },
        {
            "name": "choice_option_key_special_token",
            "body": {
                "model": model,
                "state": ok_state,
                "questions": {"q": {"type": "choice", "instructions": "x", "criteria": {"<|vision_start|>": "b"}}},
            },
        },
        {
            "name": "choice_option_key_whitespace",
            "body": {
                "model": model,
                "state": ok_state,
                "questions": {"q": {"type": "choice", "instructions": "x", "criteria": {" ": "b"}}},
            },
        },
        {
            "name": "choice_option_key_too_long",
            "body": {
                "model": model,
                "state": ok_state,
                "questions": {"q": {"type": "choice", "instructions": "x", "criteria": {"k" * 257: "b"}}},
            },
        },
        {
            "name": "choice_option_key_number",
            "body": {"model": model, "state": ok_state, "questions": {"q": {"type": "choice", "instructions": "x", "criteria": {"a": "b"}}}},
            "raw_override": '{"model": "%s", "state": "%s", "questions": {"q": {"type": "choice", "instructions": "x", "criteria": {5: "b"}}}}'
            % (model, ok_state),
        },
        {
            "name": "choice_criterion_whitespace",
            "body": {
                "model": model,
                "state": ok_state,
                "questions": {"q": {"type": "choice", "instructions": "x", "criteria": {"a": " "}}},
            },
        },
        {
            "name": "combined_text_too_long",
            "body": {
                "model": model,
                "state": ok_state,
                "questions": {"q": {"type": "choice", "instructions": "i" * 16380, "criteria": {"a": "b" * 20}}},
            },
        },
        {"name": "state_empty", "body": {"model": model, "state": "", "questions": {"q": choice_ok}}},
        {"name": "state_whitespace", "body": {"model": model, "state": " \t\n ", "questions": {"q": choice_ok}}},
        {"name": "state_number", "body": {"model": model, "state": 5, "questions": {"q": choice_ok}}},
        {"name": "state_null", "body": {"model": model, "state": None, "questions": {"q": choice_ok}}},
        {"name": "state_bool", "body": {"model": model, "state": True, "questions": {"q": choice_ok}}},
        {
            "name": "state_special_token",
            "body": {"model": model, "state": "has <|image_pad|> inside", "questions": {"q": choice_ok}},
        },
        {
            "name": "state_too_long",
            "body": {"model": model, "state": "s" * 16385, "questions": {"q": choice_ok}},
        },
        {
            "name": "image_not_data_url",
            "body": {"model": model, "state": {"image": "https://example.com/a.png"}, "questions": {"q": choice_ok}},
        },
        {
            "name": "image_bad_mime",
            "body": {"model": model, "state": {"image": "data:image/webp;base64,AAAA"}, "questions": {"q": choice_ok}},
        },
        {
            "name": "image_url_not_string",
            "body": {"model": model, "state": {"image": 7}, "questions": {"q": choice_ok}},
        },
        {
            "name": "image_extra_keys",
            "body": {"model": model, "state": {"image": "data:image/png;base64,AAAA", "x": 1}, "questions": {"q": choice_ok}},
        },
        {
            "name": "image_empty_payload",
            "body": {"model": model, "state": {"image": "data:image/png;base64,"}, "questions": {"q": choice_ok}},
        },
        {
            "name": "image_base64_invalid",
            "body": {"model": model, "state": {"image": "data:image/png;base64,!!!!"}, "questions": {"q": choice_ok}},
        },
    ]

    # image payloads that need bytes
    def with_image(name, data, mime="png"):
        return {
            "name": name,
            "body": {"model": model, "state": {"image": data_url(data, f"image/{mime}")}, "questions": {"q": choice_ok}},
        }

    cases.append(with_image("image_format_mismatch", png_bytes(8, 8, (0, 0, 0)), "jpeg"))
    cases.append(with_image("image_corrupt_truncated", truncated_png_bytes()))
    cases.append(with_image("image_animated_png", apng_bytes()))
    cases.append(with_image("image_aspect_ratio", png_bytes(400, 1, (0, 0, 0))))
    cases.append(with_image("image_side_too_long", png_bytes(2049, 11, (0, 0, 0))))
    cases.append(with_image("image_pixels_too_many", png_bytes(2048, 513, (0, 0, 0))))
    cases.append(
        {
            "name": "image_encoded_oversize",
            "synthesize": "image_encoded_oversize",
        }
    )
    cases.append(
        {
            "name": "image_bytes_oversize",
            "synthesize": "image_bytes_oversize",
        }
    )
    return cases


# --- compiler rejections (token-budget) --------------------------------------


def overflow_cases():
    model = "valen-preview-0923"
    dense = "订单异常需要人工复核。"  # ~6 tokens per 11 chars
    return [
        {
            "name": "overflow_base",
            "body": {
                "model": model,
                "state": dense * 1480,
                "questions": {"longq": {"type": "choice", "instructions": "x", "criteria": {"a": "b"}}},
            },
        },
        {
            "name": "overflow_later_question",
            "body": {
                "model": model,
                "state": dense * 1355,
                "questions": {
                    "ok": {"type": "choice", "instructions": "short", "criteria": {"a": "b"}},
                    "over": {"type": "choice", "instructions": "y" * 16000, "criteria": {"a": "b"}},
                },
            },
        },
        {
            "name": "overflow_score_branch",
            "body": {
                "model": model,
                "state": dense * 1355,
                "questions": {
                    "grade": {
                        "type": "score",
                        "instructions": "Rate.",
                        "criteria": ["a", "b" * 400, "c" * 400, "d" * 400, "e" * 400],
                    }
                },
            },
        },
    ]


# --- recording ---------------------------------------------------------------


def collapse(ids):
    """Token ids with each run of image placeholders stored as its negated length."""

    out, run = [], 0
    for token in ids:
        if token == IMAGE_PAD:
            run += 1
            continue
        if run:
            out.append(-run)
            run = 0
        out.append(token)
    if run:
        out.append(-run)
    return out


def record_valid(case, processor):
    from models.valen.protocol import parse_body, parse_request, TextState
    from models.valen.preprocess import prepare_request, question_record
    from tempfile import TemporaryDirectory
    from valen.data.compiler import Compiler

    raw = body_text(case["body"]).encode("utf-8")
    request = parse_request(parse_body(raw), special_tokens=processor.special_tokens)
    with TemporaryDirectory(prefix="system1-valen-fixture-") as directory:
        media_root = Path(directory)
        prepared = prepare_request(request, media_root)
        compiled = processor.compile(prepared)
        # The base sequence, extracted through the pinned compiler's own
        # message handling and processor call (the same code compile() runs).
        compiler = Compiler(
            processor.processor,
            media_root=media_root,
            max_length=processor._max_length,
            media_kwargs=processor._media_kwargs,
        )
        messages, _ = compiler._messages(prepared.record["request"]["state"])
        base_dict = processor.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs=dict(processor._media_kwargs, return_metadata=True),
        )
        base_ids = base_dict["input_ids"][0].tolist()
        base_mm = base_dict["mm_token_type_ids"][0].tolist()
    state = compiled.state
    tokenizer = processor.processor.tokenizer

    questions = []
    for question in state.questions:
        branches = []
        for branch in question.branches:
            ids = branch.inputs["input_ids"][0].tolist()
            mm = branch.inputs["mm_token_type_ids"][0].tolist()
            assert ids[: len(base_ids)] == base_ids, "branch loses base"
            assert len(ids) == branch.decision_position + 1, "length disagrees"
            assert mm[: len(base_mm)] == base_mm
            assert set(mm[len(base_mm):]) == {0}, "suffix mm not zero"
            suffix = ids[len(base_ids):]
            branches.append(
                {
                    "suffix_ids": suffix,
                    "candidate_positions": branch.candidate_positions,
                    "decision_position": branch.decision_position,
                    "length": len(ids),
                }
            )
        wire = next(q for q in request.questions if q.name == question.qid)
        record = question_record(wire)
        questions.append(
            {
                "qid": question.qid,
                "wire_kind": wire.kind,
                "record_kind": record["type"],
                "keys": question.keys,
                "descriptions": question.descriptions,
                "record": record,
                "branches": branches,
            }
        )

    if isinstance(request.state, TextState):
        content = request.state.text
        state_record = {
            "kind": "text",
            "user_content": content.strip(),
            "prompt": "<|im_start|>user\n" + content.strip() + "<|im_end|>\n",
        }
        assert state.media == [], "text state produced media"
    else:
        grid = next(m for m in state.media if "image_grid_thw" in m)
        image_state = {
            "kind": "image",
            "format": request.state.format,
            "width": request.state.width,
            "height": request.state.height,
            "grid_thw": grid["image_grid_thw"][0],
            "visual_tokens": grid["visual_tokens"],
            "user_content": "<|vision_start|><|image_pad|><|vision_end|>",
            "prompt": "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n",
        }
        state_record = image_state

    # sanity: mm marks exactly the image-pad run
    pads = [i for i, v in enumerate(base_mm) if v == 1]
    expected_pads = (
        [i for i, t in enumerate(base_ids) if t == IMAGE_PAD]
        if state_record["kind"] == "image"
        else []
    )
    assert pads == expected_pads, "mm token type ids disagree with image pads"
    assert all(v == 0 for v in base_mm) or state_record["kind"] == "image"
    logical = len(base_ids) + sum(len(b["suffix_ids"]) for q in questions for b in q["branches"])
    compute = sum(b["length"] for q in questions for b in q["branches"])
    assert logical == state.logical_tokens, (logical, state.logical_tokens)
    assert compute == state.compute_tokens, (compute, state.compute_tokens)

    return {
        "state": state_record,
        "questions": questions,
        "base_input_ids": collapse(base_ids),
        "image_pad_span": [pads[0], pads[-1] + 1] if pads else None,
        "logical_tokens": state.logical_tokens,
        "compute_tokens": state.compute_tokens,
    }


def record_rejection(case, processor, synthesize):
    from models.valen.protocol import parse_body, parse_request, RequestError
    from models.valen.preprocess import prepare_request
    from tempfile import TemporaryDirectory

    if "body_b64" in case:
        raw = base64.b64decode(case["body_b64"])
    elif case.get("synthesize"):
        raw = synthesize(case["synthesize"])
    elif "raw_override" in case:
        raw = case["raw_override"].encode("utf-8")
    else:
        raw = body_text(case["body"]).encode("utf-8")
    try:
        request = parse_request(parse_body(raw), special_tokens=processor.special_tokens)
        with TemporaryDirectory(prefix="system1-valen-fixture-") as directory:
            prepared = prepare_request(request, Path(directory))
            compiled = processor.compile(prepared)
        if case.get("expect_valid") and case.get("record_only"):
            return None  # validity probes belong to the valid corpus
        raise AssertionError(f"{case['name']}: unexpectedly accepted")
    except RequestError as exc:
        return {"name": case["name"], "status": exc.status, "detail": str(exc)}
    except ValueError as exc:  # the worker's decide() mapping, verbatim
        return {
            "name": case["name"],
            "status": 422,
            "detail": f"request rejected by the pinned compiler: {exc}",
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--valen-source", default=str(REPO / "weights/Valen"))
    parser.add_argument("--checkpoint", default=str(REPO / "weights/Valen-Preview-0923"))
    parser.add_argument("--base", default=str(REPO / "weights/Qwen3.5-2B"))
    parser.add_argument("--out", default=str(REPO / "tests/valen/data"))
    args = parser.parse_args()

    source = Path(args.valen_source).resolve()
    revision = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if revision != REFERENCE_REVISION:
        raise SystemExit(f"--valen-source must be pinned to {REFERENCE_REVISION}")
    sys.path.insert(0, str(source))

    import hashlib

    checkpoint = Path(args.checkpoint).resolve()
    digest = hashlib.sha256((checkpoint / "checkpoint.pt").read_bytes()).hexdigest()
    if digest != CHECKPOINT_SHA256:
        raise SystemExit("--checkpoint must match the pinned Preview checksum")

    from models.valen.preprocess import ValenProcessor

    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    processor = ValenProcessor(args.base, int(config["max_length"]), config.get("media_kwargs"))
    size = processor._media_kwargs["images_kwargs"]["size"]

    pad = "A" * (8 * 1024 * 1024 + 64)

    def synthesize(kind):
        model = "valen-preview-0923"
        question = {"type": "choice", "instructions": "Pick.", "criteria": {"a": "b"}}
        if kind == "body_oversize":
            return body_text(
                {"model": model, "state": pad, "questions": {"q": question}}
            ).encode("utf-8")
        if kind == "image_encoded_oversize":
            encoded = "A" * (4 * ((4 * 1024 * 1024 + 2) // 3) + 1)
            return body_text(
                {"model": model, "state": {"image": f"data:image/png;base64,{encoded}"}, "questions": {"q": question}}
            ).encode("utf-8")
        if kind == "image_bytes_oversize":
            raw = bytes(4 * 1024 * 1024 + 1)
            encoded = base64.b64encode(raw).decode("ascii")
            return body_text(
                {"model": model, "state": {"image": f"data:image/png;base64,{encoded}"}, "questions": {"q": question}}
            ).encode("utf-8")
        raise AssertionError(kind)

    cases = []
    for case in valid_cases():
        image = case.get("image")
        body = case["body"]
        if image and image["kind"] == "generated":
            width, height = image["size"]
            colour = tuple(image["colour"])
            if image["format"] == "png":
                data = png_bytes(width, height, colour)
            else:
                data = jpeg_bytes(width, height, colour)
            body["state"]["image"] = data_url(data, f"image/{image['format']}")
        cases.append(
            {
                "name": case["name"],
                "body": body_text(body),
                "image": image,
                "expected": record_valid({"name": case["name"], "body": body}, processor),
            }
        )

    rejections = []
    for case in rejection_cases():
        recorded = record_rejection(case, processor, synthesize)
        if recorded is not None:
            entry = {"name": case["name"]}
            if "body_b64" in case:
                entry["body_b64"] = case["body_b64"]
            elif case.get("synthesize"):
                entry["synthesize"] = case["synthesize"]
            elif "raw_override" in case:
                entry["body"] = case["raw_override"]
            else:
                entry["body"] = body_text(case["body"])
            entry.update(recorded)
            rejections.append(entry)
    for case in overflow_cases():
        recorded = record_rejection(case, processor, synthesize)
        entry = {"name": case["name"], "body": body_text(case["body"])}
        entry.update(recorded)
        rejections.append(entry)

    fixture = {
        "format": 1,
        "pins": {
            "valen_revision": REFERENCE_REVISION,
            "base_revision": BASE_REVISION,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "max_length": int(config["max_length"]),
            "min_pixels": size["shortest_edge"],
            "max_pixels": size["longest_edge"],
            "patch_size": processor.processor.image_processor.patch_size,
            "merge_size": processor.processor.image_processor.merge_size,
            "special_tokens": list(processor.special_tokens),
            "image_pad_token_id": IMAGE_PAD,
        },
        "cases": cases,
        "rejections": rejections,
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "native_compiler.json"
    path.write_text(json.dumps(fixture, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"recorded {len(cases)} cases and {len(rejections)} rejections -> {path}")


if __name__ == "__main__":
    main()
