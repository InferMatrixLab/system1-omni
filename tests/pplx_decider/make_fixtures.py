"""Golden fixtures for the pplx-decider-v1.1-27b decision contract, from the checkpoint's own code.

Runs perplexity-ai/pplx-decider-v1.1-27b@14c9d901's `source/src/autojev` unmodified, without the weights: the
server's request model (`server.EvaluationRequest`, then `model_dump(exclude_none=True)` as `server.predict`
does), the prompt (`model.decision_messages`), the checkpoint's processor and chat template (as
`DecisionModel.prepare` applies them), and the typed answer (`model.answer`) from fixed readout logits passed
through the checkpoint's temperature and a float32 softmax, as `server.predict` computes it. Each case pins the
rendered text, the input ids, the candidate readout rows, the logits fed in, the probabilities and the answer.

Usage: python tests/pplx_decider/make_fixtures.py <snapshot dir with source/, tokenizer.json, tokenizer_config.json,
chat_template.jinja, processor_config.json, decision_config.json> tests/pplx_decider/data
"""

import hashlib
import json
import math
import random
import sys
from pathlib import Path

SNAPSHOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
REVISION = "14c9d90137962d300f3de95633f7b6535e80c5ce"
PINNED = {
    "source/src/autojev/model.py": "02940963e9fba5736b8f625f860812f31de2e6658a49606fbed49fd7215c20f9",
    "tokenizer.json": "6f32ce20dc35f57a7f9ad1eac03525bd7d30f9df8cea6507e958279cc3657706",
    "chat_template.jinja": "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
    "decision_config.json": "68d4a4c704a0c42837f749b0287bd80f70d186ed3be4dc6a90802e78223925e8",
}
for name, digest in PINNED.items():
    assert hashlib.sha256((SNAPSHOT / name).read_bytes()).hexdigest() == digest, name

sys.path.insert(0, str(SNAPSHOT / "source" / "src"))
import torch  # noqa: E402
from pydantic import ValidationError  # noqa: E402
from transformers import AutoProcessor  # noqa: E402

from autojev import model as M  # noqa: E402
from autojev import server as S  # noqa: E402

CONFIG = json.loads((SNAPSHOT / "decision_config.json").read_text(encoding="utf-8"))
PROCESSOR = AutoProcessor.from_pretrained(str(SNAPSHOT))
PROCESSOR.tokenizer.padding_side = "left"
CODES, TOKEN_IDS, TEMPERATURE = CONFIG["codes"], CONFIG["token_ids"], CONFIG["temperature"]
# DecisionModel.__init__ derives the codes from the tokenizer and refuses a config that disagrees; do the same.
tokenizer = PROCESSOR.tokenizer
assert [tokenizer.encode(c, add_special_tokens=False) for c in CODES] == [[t] for t in TOKEN_IDS]
S.service.name = "pplx-decider-v1.1-27b"
RNG = random.Random(20261012)


def questions_of(request: dict) -> dict:
    body = S.EvaluationRequest.model_validate(request)
    return {k: q.model_dump(exclude_none=True) for k, q in body.questions.items()}, body


def readout(count: int) -> tuple[list[float], list[float]]:
    """Fixed float32 logits for `count` candidates and the probabilities `server.predict` makes of them."""
    logits = [round(RNG.uniform(-6.0, 6.0), 3) for _ in range(count)]
    row = torch.full((M.MAX_OPTIONS,), -1e9, dtype=torch.float32)
    row[:count] = torch.tensor(logits, dtype=torch.float32)
    probabilities = (row[None] / TEMPERATURE).softmax(-1)[0].tolist()[:count]
    return logits, probabilities


def case(name: str, request: dict) -> dict:
    questions, body = questions_of(request)
    calls = []
    for qid, question in questions.items():
        row = {"state": body.state, "question": question, "images": list(body.images)}
        messages = M.decision_messages(row, CODES)
        text = PROCESSOR.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        ids = PROCESSOR(text=[text], padding=True, return_tensors="pt")["input_ids"][0].tolist()
        count = len(M.options(question)[0])
        logits, probabilities = readout(count)
        calls.append(
            {
                "question": qid,
                "user_text": messages[1]["content"][-1]["text"],
                "text": text,
                "input_ids": ids,
                "candidate_token_ids": TOKEN_IDS[:count],
                "logits": logits,
                "probabilities": probabilities,
                "answer": M.answer(question, probabilities),
            }
        )
    response = {
        "answers": {c["question"]: c["answer"] for c in calls},
        "usage": {"input_tokens": sum(len(c["input_ids"]) for c in calls), "output_tokens": 0},
    }
    return {"name": name, "request": request, "calls": calls, "response": response}


def rejected(name: str, request: dict) -> dict:
    try:
        questions, body = questions_of(request)
        for question in questions.values():
            M.decision_messages({"state": body.state, "question": question}, CODES)
    except (ValidationError, ValueError) as error:
        return {"name": name, "request": request, "official": type(error).__name__}
    raise AssertionError(f"{name} was accepted")


FLIGHTS = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8")) if len(sys.argv) > 3 else None
M_ = "pplx-decider-v1.1-27b"
CASES = [
    case(
        "choice with descriptions and a null option",
        {
            "model": M_,
            "state": "My integration keeps failing. Please help.",
            "questions": {
                "team": {
                    "type": "choice",
                    "instructions": "Which team should handle this request?",
                    "criteria": {
                        "billing": "Charges and refunds",
                        "support": "Technical integration errors",
                        "sales": None,
                    },
                }
            },
        },
    ),
    case(
        "noul without criteria, default instructions",
        {"model": M_, "state": "The invoice was paid twice.", "questions": {"duplicate": {"type": "noul"}}},
    ),
    case(
        "noul with one side described and an empty side",
        {
            "model": M_,
            "state": "Wire sent on Monday.",
            "questions": {
                "paid": {
                    "type": "noul",
                    "instructions": "Was the payment sent?",
                    "criteria": {"true": "A payment left the account", "false": ""},
                }
            },
        },
    ),
    case(
        "score with four levels",
        {
            "model": M_,
            "state": "Someone took over my account and moved money.",
            "questions": {
                "urgency": {
                    "type": "score",
                    "instructions": "How urgent is this ticket?",
                    "criteria": ["not urgent", "within the week", "within the day", "critical, now"],
                }
            },
        },
    ),
    case(
        "score with object levels and an object state",
        {
            "model": M_,
            "state": {
                "ticket": {
                    "subject": "Retard de livraison",
                    "body": "Mon colis a 2 jours de retard — ça arrive ?",
                    "tags": ["fr", 2, 3.5, True, None],
                }
            },
            "questions": {
                "severity": {"type": "score", "criteria": [{"level": "low"}, {"level": "high", "note": "ünïcödé"}]}
            },
        },
    ),
    case(
        "list state, object instructions, empty instructions",
        {
            "model": M_,
            "state": ["line one", {"k": [1, 2]}],
            "questions": {
                "a": {
                    "type": "choice",
                    "instructions": {"goal": "pick", "n": 2},
                    "criteria": {"x": {"why": 1}, "y": ["z"]},
                },
                "b": {"type": "choice", "instructions": "", "criteria": {"only": None}},
            },
        },
    ),
    case(
        "thirty options use two-letter codes",
        {
            "model": M_,
            "state": "Pick the element.",
            "questions": {
                "target": {
                    "type": "choice",
                    "instructions": "Which element?",
                    "criteria": {str(i): f"element {i}" for i in range(1, 31)},
                }
            },
        },
    ),
    case(
        "several questions keep request order",
        {
            "model": M_,
            "state": "Flights page, form filled.",
            "questions": {
                "done": {"type": "noul", "instructions": "Are flights shown?"},
                "op": {"type": "choice", "criteria": {"CLICK": "press", "DONE": "finish"}},
                "progress": {"type": "score", "criteria": ["start", "form filled", "results shown"]},
            },
        },
    ),
]
if FLIGHTS:
    CASES.append(
        case("s1a Google Flights step", {"model": M_, "state": FLIGHTS["state"], "questions": FLIGHTS["questions"]})
    )

REJECTED = [
    rejected("unknown top-level field", {"model": M_, "state": "s", "questions": {"q": {"type": "noul"}}, "extra": 1}),
    rejected("unknown question field", {"model": M_, "state": "s", "questions": {"q": {"type": "noul", "hint": "x"}}}),
    rejected("missing state", {"model": M_, "questions": {"q": {"type": "noul"}}}),
    rejected("empty questions", {"model": M_, "state": "s", "questions": {}}),
    rejected("unknown type", {"model": M_, "state": "s", "questions": {"q": {"type": "rank"}}}),
    rejected(
        "score with one level", {"model": M_, "state": "s", "questions": {"q": {"type": "score", "criteria": ["only"]}}}
    ),
    rejected(
        "score with eleven levels",
        {"model": M_, "state": "s", "questions": {"q": {"type": "score", "criteria": [str(i) for i in range(11)]}}},
    ),
    rejected(
        "choice with no options", {"model": M_, "state": "s", "questions": {"q": {"type": "choice", "criteria": {}}}}
    ),
    rejected(
        "choice with 256 options",
        {
            "model": M_,
            "state": "s",
            "questions": {"q": {"type": "choice", "criteria": {str(i): None for i in range(256)}}},
        },
    ),
    rejected(
        "noul criteria key other than true and false",
        {"model": M_, "state": "s", "questions": {"q": {"type": "noul", "criteria": {"maybe": "x"}}}},
    ),
    rejected("number state", {"model": M_, "state": 3, "questions": {"q": {"type": "noul"}}}),
]

OUT.mkdir(parents=True, exist_ok=True)
fixture = {
    "source": {
        "model": "perplexity-ai/pplx-decider-v1.1-27b",
        "revision": REVISION,
        "files": PINNED,
        "temperature": TEMPERATURE,
        "max_tokens": 8192,
    },
    "cases": CASES,
    "rejected": REJECTED,
}
assert all(math.isfinite(p) for c in CASES for call in c["calls"] for p in call["probabilities"])
(OUT / "contract.json").write_text(json.dumps(fixture, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(len(CASES), "cases,", sum(len(c["calls"]) for c in CASES), "questions,", len(REJECTED), "rejected")
