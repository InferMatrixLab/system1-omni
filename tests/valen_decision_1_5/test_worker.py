import json
import math

import pytest

from frontend.valen_decision_1_5 import decide, warmup
from models.valen_decision_1_5.engine import ExecutorOutput
from models.valen_decision_1_5.preprocess import CompiledInput
from models.valen_decision_1_5.protocol import DEFAULT_SPECIAL_TOKENS, RequestError

from .test_protocol import MODEL_NAME, body


class FakeProcessor:
    special_tokens = DEFAULT_SPECIAL_TOKENS

    def __init__(self):
        self.seen_media = None
        self.records = []

    def compile(self, prepared):
        if prepared.image_path is not None:
            assert prepared.image_path.is_file()
            self.seen_media = prepared.image_path
        self.records.append(prepared.record)
        return CompiledInput(object(), prepared.response_context)


class FakeExecutor:
    model_name = MODEL_NAME

    def __init__(self, output):
        self.output = output
        self.warmups = 0

    def execute(self, compiled):
        assert compiled.response_context.questions
        return self.output

    def warmup(self, compiled):
        self.warmups += 1

    def close(self):
        pass


def run(value, executor, processor=None):
    return decide(json.dumps(value).encode("utf-8"), processor or FakeProcessor(), executor)


def test_decide_keeps_http_adapter_outside_executor():
    processor = FakeProcessor()
    result = run(body(), FakeExecutor(ExecutorOutput(((0.0, 1.0),), 17, 17)), processor)

    assert result["model"] == MODEL_NAME
    assert result["answers"]["move"]["choice"] == "down"
    assert result["usage"] == {"input_tokens": 17, "output_tokens": 0}
    assert result["internal_usage"] == {"compute_tokens": 17}
    assert processor.seen_media is not None
    assert not processor.seen_media.exists()


def test_decide_serves_text_only_state_without_media():
    processor = FakeProcessor()
    value = body(state="The card was charged twice for one order.")
    result = run(value, FakeExecutor(ExecutorOutput(((0.0, 1.0),), 12, 12)), processor)

    assert result["answers"]["move"]["choice"] == "down"
    assert processor.seen_media is None


def test_decide_serves_the_id_of_the_loaded_checkpoint():
    executor = FakeExecutor(ExecutorOutput(((0.0, 1.0),), 1, 1))
    executor.model_name = "valen-decision-1.5-2b"

    with pytest.raises(RequestError, match="model must be valen-decision-1.5-2b"):
        run(body(), executor)
    assert run(body(model="valen-decision-1.5-2b"), executor)["model"] == (
        "valen-decision-1.5-2b"
    )


def test_decide_rejects_executor_cardinality_mismatch():
    with pytest.raises(RuntimeError, match="answer count"):
        run(body(), FakeExecutor(ExecutorOutput((), 1, 1)))
    with pytest.raises(RuntimeError, match="invalid output object"):
        run(body(), FakeExecutor(((0.0, 1.0),)))


def test_decide_maps_pinned_compiler_value_errors_to_422():
    class RejectingProcessor(FakeProcessor):
        def compile(self, prepared):
            raise ValueError("Shared state: 20000 tokens exceed max_length=16384; no truncation")

    with pytest.raises(RequestError) as excinfo:
        run(body(), FakeExecutor(ExecutorOutput(((0.0, 1.0),), 1, 1)), RejectingProcessor())

    assert excinfo.value.status == 422
    assert "pinned compiler" in str(excinfo.value)
    assert "exceed max_length" in str(excinfo.value)


def test_decide_blocks_the_processor_special_tokens():
    class StrictProcessor(FakeProcessor):
        special_tokens = (*DEFAULT_SPECIAL_TOKENS, "<|im_end|>")

    value = body(state="context <|im_end|> tail")
    with pytest.raises(RequestError, match="control token"):
        run(value, FakeExecutor(ExecutorOutput(((0.0, 1.0),), 1, 1)), StrictProcessor())


def test_decide_answers_mixed_type_questions_from_one_state():
    value = body(
        state="The card was charged twice for one order.",
        questions={
            "ask": {"type": "noul", "instructions": "Was the card charged twice?"},
            "grade": {
                "type": "score",
                "instructions": "Grade the severity.",
                "criteria": ["No error", "A minor error", "A major error"],
            },
            "act": {
                "type": "choice",
                "instructions": "Decide the refund action.",
                "criteria": {"refund": None, "wait": "Wait for review"},
            },
        },
    )
    executor = FakeExecutor(
        ExecutorOutput(
            (
                (math.log(3.0), 0.0),  # noul: true, false
                (math.log(7.0), math.log(2.0), 0.0),  # score levels 0, 1, 2
                (0.0, 1.0),  # choice: refund, wait
            ),
            61,
            61,
        )
    )

    result = run(value, executor)

    assert list(result["answers"]) == ["ask", "grade", "act"]
    assert result["answers"]["ask"] == {"type": "noul", "noul": pytest.approx(0.75)}
    score = result["answers"]["grade"]
    assert score["probabilities"]["0"] == pytest.approx(0.7)
    assert score["score"] == pytest.approx(0.4)
    assert score["legend"] == {"0": "No error", "1": "A minor error", "2": "A major error"}
    assert result["answers"]["act"]["choice"] == "wait"
    assert result["usage"] == {"input_tokens": 61, "output_tokens": 0}
    assert result["internal_usage"] == {"compute_tokens": 61}


def test_warmup_runs_image_and_text_state_with_every_question_type():
    processor = FakeProcessor()
    executor = FakeExecutor(None)

    warmup(processor, executor)

    assert executor.warmups == 2
    image, text = (record["request"] for record in processor.records)
    assert image["state"]["messages"][0]["content"][0]["type"] == "image_url"
    assert text["state"]["messages"][0]["content"][0]["type"] == "text"
    for request in (image, text):
        assert [q["type"] for q in request["questions"].values()] == [
            "choice",
            "noul",
            "score",
        ]
