"""End-to-end decide() tests with a CPU-only FakeVela2: no weights, no torch.

FakeVela2 mirrors the upstream contract the worker drives — the pinned
runtime's ``system_one_request`` validation, reimplemented just far
enough to produce upstream-shaped whole-request SystemOneErrors, with
answer construction delegated to the mirrors in models.vela.postprocess
(which lock the pinned formulas). Deeper validation paths the tests do
not exercise are pinned by the author-side fixtures against the real
checkpoint instead.
"""

from __future__ import annotations

import json

import pytest

from frontend.vela import decide
from models.vela.postprocess import choice_confidence, expected_level, score_confidence
from models.vela.protocol import MODEL_NAME, RequestError

from .test_protocol import body

TYPES = ("noul", "choice", "score", "span", "set")
STATE_KEY_ROLES = {"request": "user", "user": "user", "prompt": "user",
                   "source": "context", "context": "context", "document": "context",
                   "tools": "context", "previous_answer": "context",
                   "answer": "answer", "response": "answer"}


class FakeSystemOneError(ValueError):
    """The pinned runtime's error shape, mirrored for these tests."""

    def __init__(self, loc, msg, type_="value_error"):
        super().__init__(f"{'.'.join(str(x) for x in loc)}: {msg}")
        self.loc, self.msg, self.type = list(loc), msg, type_

    def detail(self):
        return {"loc": ["body"] + self.loc, "msg": self.msg, "type": self.type}


class FakeVela2:
    """The pinned runtime's contract, driven without its code."""

    def __init__(self, model_name=MODEL_NAME, too_long=False):
        self.model_name = model_name
        self.max_length = 8192
        self.SystemOneError = FakeSystemOneError
        self._too_long = too_long
        self.warmed = False
        self.closed = False
        self.calls = []

    @property
    def served(self):
        return {self.model_name}

    # -- upstream system_one_state, reduced to what the tests exercise --
    def _state(self, state):
        if isinstance(state, str):
            if not state.strip():
                raise FakeSystemOneError(["state"], "state must not be empty or whitespace")
            return {"request": ("user", 0, len(state))}, ["user"]
        if isinstance(state, list):
            raise FakeSystemOneError(["state"], "list states are not covered by the fake")
        if not isinstance(state, dict) or not state:
            raise FakeSystemOneError(["state"], "state must be a non-empty string, JSON object or JSON array")
        fields, roles = {}, []
        for role in ("user", "context", "answer"):
            keys = [k for k, v in state.items() if STATE_KEY_ROLES.get(str(k).lower(), "context") == role]
            if not keys:
                continue
            roles.append(role)
            for key in keys:
                fields[key] = role
        return fields, roles

    # -- upstream system_one_to_predict, reduced to what the tests exercise --
    def _plan(self, body):
        fields, roles = self._state(body["state"])
        plan = []
        for qid, question in body["questions"].items():
            loc = ["questions", qid]
            if not isinstance(qid, str) or not qid.strip():
                raise FakeSystemOneError(["questions"], "question IDs must not be empty or whitespace")
            if not isinstance(question, dict):
                raise FakeSystemOneError(loc, "a question must be an object with a type")
            typ = question.get("type")
            if typ not in TYPES:
                raise FakeSystemOneError(loc + ["type"], f"type must be one of {list(TYPES)}", "literal_error")
            allowed = {"type", "instructions", "criteria", "over"} | (
                {"threshold"} if typ in ("span", "set") else set()
            )
            extra = sorted(set(question) - allowed)
            if extra:
                raise FakeSystemOneError(loc + [extra[0]], "Extra inputs are not permitted", "extra_forbidden")
            if not isinstance(question.get("instructions"), str) or not question["instructions"].strip():
                raise FakeSystemOneError(loc + ["instructions"], "instructions must not be empty or whitespace")
            crit = question.get("criteria")
            if typ == "score":
                if not isinstance(crit, list) or not 2 <= len(crit) <= 10:
                    raise FakeSystemOneError(loc + ["criteria"], "score criteria must be a list of 2 to 10 levels")
            elif typ == "noul":
                if crit is not None and (not isinstance(crit, dict) or set(crit) - {"true", "false"}):
                    raise FakeSystemOneError(loc + ["criteria"], "noul criteria is an object with optional true / false")
            else:
                lo, hi = (2, 255) if typ == "choice" else (1, 255)
                if not isinstance(crit, dict) or not lo <= len(crit) <= hi:
                    raise FakeSystemOneError(loc + ["criteria"], f"{typ} criteria must be an object with {lo} to {hi} entries")
            if "over" in question and question["over"] is not None:
                items = question["over"] if isinstance(question["over"], list) else [question["over"]]
                for item in items:
                    if not isinstance(item, str) or (item not in fields and item not in roles
                                                     and STATE_KEY_ROLES.get(item) not in roles):
                        raise FakeSystemOneError(loc + ["over"],
                                                 f"'{item}' is not a field of the state (fields: {sorted(fields)}, parts: {roles})")
            if question.get("threshold") is not None:
                threshold = question["threshold"]
                if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0.0 <= threshold <= 1.0:
                    raise FakeSystemOneError(loc + ["threshold"], "threshold must be a number in [0, 1]")
            plan.append((qid, typ, question))
        for qid, typ, question in plan:
            if typ == "set":
                for label in question.get("criteria") or {}:
                    if f"{qid}.{label}" in body["questions"]:
                        raise FakeSystemOneError(["questions", f"{qid}.{label}"],
                                                 "collides with the Noul view of a set label")
        return plan

    def system_one(self, body):
        self.calls.append(body)
        plan = self._plan(body)
        if self._too_long:
            raise FakeSystemOneError(
                ["questions"], f"the questions alone do not fit the {self.max_length}-token window", "too_long"
            )
        answers, spans, sets, thresholds = {}, {}, {}, {}
        for qid, typ, question in plan:
            crit = question.get("criteria") or {}
            if typ == "noul":
                answers[qid] = {"type": "noul", "noul": 0.75}
            elif typ == "choice":
                names = list(crit)
                probabilities = [0.75 / len(names)] * (len(names) - 1) + [0.25 / len(names) + 0.5]
                values = [p / sum(probabilities) for p in probabilities]
                probabilities = dict(zip(names, values))
                answers[qid] = {
                    "type": "choice",
                    "choice": max(names, key=lambda name: probabilities[name]),
                    "confidence": choice_confidence(values),
                    "probabilities": probabilities,
                }
            elif typ == "score":
                values = [0.125, 0.25, 0.625][: len(crit)]
                values = values + [0.0] * (len(crit) - len(values))
                total = sum(values) or 1.0
                values = [v / total for v in values]
                answers[qid] = {
                    "type": "score",
                    "score": expected_level(values),
                    "confidence": score_confidence(values),
                    "legend": {str(i): level for i, level in enumerate(crit)},
                    "probabilities": {str(i): value for i, value in enumerate(values)},
                }
            elif typ == "set":
                label_p = {label: 0.9 - 0.3 * index for index, label in enumerate(crit)}
                threshold = float(question["threshold"]) if question.get("threshold") is not None else 0.3
                sets[qid] = {"selected": [k for k, v in label_p.items() if v > threshold],
                             "probabilities": label_p}
                thresholds[qid] = threshold
                for label, value in label_p.items():
                    answers[f"{qid}.{label}"] = {"type": "noul", "noul": value}
            else:  # span
                threshold = float(question["threshold"]) if question.get("threshold") is not None else 0.5
                spans[qid] = [{"label": "LABEL", "start": 0, "end": 4, "text": "text",
                               "probability": 0.9}]
                thresholds[qid] = threshold
                answers[qid] = {"type": "noul", "noul": 0.9}
        response = {"model": body["model"], "answers": answers,
                    "usage": {"input_tokens": 64, "output_tokens": 0}}
        if spans:
            response["spans"] = spans
        if sets:
            response["sets"] = sets
        if thresholds:
            response["thresholds"] = thresholds
        return response

    def warmup(self):
        request = {"state": "warmup", "model": self.model_name,
                   "questions": {"q": {"type": "noul", "instructions": "warm?"}}}
        self.system_one(request)
        self.warmed = True

    def close(self):
        self.closed = True


def five_kind_body():
    return {
        "model": MODEL_NAME,
        "state": {
            "request": "Hi, I'm Tom Baker (tom.baker@example.com).",
            "source": "Deliveries take 3 to 5 days.",
            "answer": "It was delivered instantly.",
        },
        "questions": {
            "kind": {"type": "choice", "instructions": "Which subject?",
                     "criteria": {"delivery": "parcels", "other": "anything else"}},
            "issues": {"type": "set", "instructions": "Which issues apply?",
                       "criteria": {"delivery": "parcels", "login": "accounts"}},
            "tone": {"type": "score", "instructions": "How formal?",
                     "criteria": ["casual", "neutral", "formal"]},
            "polite": {"type": "noul", "instructions": "Is it polite?"},
            "unsupported": {"type": "span", "instructions": "Unsupported answer spans?",
                            "criteria": {"unsupported": "not backed by the source"},
                            "over": "answer"},
        },
    }


def test_decide_answers_all_five_kinds_in_one_request():
    result = decide(json.dumps(five_kind_body()).encode("utf-8"), FakeVela2())

    assert result["model"] == MODEL_NAME
    assert list(result["answers"]) == [
        "kind", "issues.delivery", "issues.login", "tone", "polite", "unsupported",
    ]
    assert result["answers"]["polite"] == {"type": "noul", "noul": 0.75}
    assert result["answers"]["kind"]["type"] == "choice"
    assert result["answers"]["tone"]["legend"] == {"0": "casual", "1": "neutral", "2": "formal"}
    assert result["sets"]["issues"]["selected"] == ["delivery", "login"]
    assert result["sets"]["issues"]["probabilities"] == pytest.approx(
        {"delivery": 0.9, "login": 0.6}
    )
    assert result["thresholds"]["issues"] == 0.3
    assert result["spans"]["unsupported"][0]["label"] == "LABEL"
    assert result["thresholds"]["unsupported"] == 0.5
    assert result["usage"] == {"input_tokens": 64, "output_tokens": 0}


def test_decide_passes_the_admitted_body_to_the_executor():
    executor = FakeVela2()
    value = five_kind_body()
    decide(json.dumps(value).encode("utf-8"), executor)

    assert executor.calls == [value]


def test_decide_maps_a_too_long_request_to_413():
    with pytest.raises(RequestError) as excinfo:
        decide(json.dumps(body()).encode("utf-8"), FakeVela2(too_long=True))

    assert excinfo.value.status == 413
    assert excinfo.value.body == {
        "detail": [{
            "loc": ["body", "questions"],
            "msg": "the questions alone do not fit the 8192-token window",
            "type": "too_long",
        }]
    }


@pytest.mark.parametrize(
    ("mutate", "loc", "msg", "type_"),
    [
        (lambda b: b["questions"].update({"bad": {"type": "triangle"}}),
         ["questions", "bad", "type"], "type must be one of", "literal_error"),
        (lambda b: b["questions"]["billing"].update(threshold=0.5),
         ["questions", "billing", "threshold"], "Extra inputs are not permitted", "extra_forbidden"),
        (lambda b: b["questions"]["billing"].update(instructions="  "),
         ["questions", "billing", "instructions"], "instructions must not be empty", "value_error"),
        (lambda b: b["questions"]["billing"].update(criteria={"yes": "x"}),
         ["questions", "billing", "criteria"], "noul criteria is an object", "value_error"),
        (lambda b: b["questions"].update({"one": {"type": "choice", "instructions": "x",
                                                  "criteria": {"only": "one option"}}}),
         ["questions", "one", "criteria"], "choice criteria must be an object with 2 to 255 entries", "value_error"),
        (lambda b: b["questions"].update({"s": {"type": "span", "instructions": "x",
                                                "criteria": {"a": "b"}, "over": "nowhere"}}),
         ["questions", "s", "over"], "'nowhere' is not a field of the state", "value_error"),
        (lambda b: b["questions"].update({"s": {"type": "set", "instructions": "x",
                                                "criteria": {"a": "b"}, "threshold": 1.5}}),
         ["questions", "s", "threshold"], "threshold must be a number in [0, 1]", "value_error"),
        (lambda b: b["questions"].update({"s": {"type": "set", "instructions": "x",
                                                "criteria": {"a": "b"}},
                                          "s.a": {"type": "noul", "instructions": "x"}}),
         ["questions", "s.a"], "collides with the Noul view of a set label", "value_error"),
        (lambda b: b.update(state="   "),
         ["state"], "state must not be empty or whitespace", "value_error"),
    ],
)
def test_decide_maps_whole_request_422s_verbatim(mutate, loc, msg, type_):
    value = body()
    mutate(value)
    with pytest.raises(RequestError) as excinfo:
        decide(json.dumps(value).encode("utf-8"), FakeVela2())

    assert excinfo.value.status == 422
    entry = excinfo.value.body["detail"][0]
    assert entry["loc"] == ["body", *loc]
    assert msg in entry["msg"]
    assert entry["type"] == type_


def test_decide_maps_runtime_value_errors_to_422():
    class RejectingVela2(FakeVela2):
        def system_one(self, body):
            raise ValueError("questions must be a non-empty object of named questions")

    with pytest.raises(RequestError) as excinfo:
        decide(json.dumps(body()).encode("utf-8"), RejectingVela2())
    assert excinfo.value.status == 422
    assert "pinned runtime" in str(excinfo.value)
    assert "non-empty object" in str(excinfo.value)


def test_decide_rejects_responses_that_break_question_order():
    class ReorderedVela2(FakeVela2):
        def system_one(self, body):
            response = super().system_one(body)
            response["answers"] = dict(reversed(list(response["answers"].items())))
            return response

    with pytest.raises(RuntimeError, match="order"):
        decide(json.dumps(five_kind_body()).encode("utf-8"), ReorderedVela2())


def test_decide_rejects_responses_with_hybrid_only_fields():
    class HybridVela2(FakeVela2):
        def system_one(self, body):
            response = super().system_one(body)
            response["span_heads"] = {}
            return response

    with pytest.raises(RuntimeError, match="hybrid-only"):
        decide(json.dumps(body()).encode("utf-8"), HybridVela2())
