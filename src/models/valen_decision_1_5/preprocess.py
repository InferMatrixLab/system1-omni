"""Valen-Decision-1.5 prepared-input and compiler processing.

The public contract carries an inline image for compatibility with Cua-S1.
The checkpoint's compiler reads media from a local path, so this layer
materializes one request-scoped file and produces the SystemOne record the
compiler takes. The worker owns the lifetime of media_root.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .pinned import Checkpoint
from .protocol import DEFAULT_SPECIAL_TOKENS, Question, Request, TextState


@dataclass(frozen=True)
class ResponseContext:
    model: str
    questions: tuple[Question, ...]


@dataclass(frozen=True)
class PreparedInput:
    record: dict[str, Any]
    media_root: Path
    image_path: Path | None
    image_sha256: str | None
    response_context: ResponseContext


@dataclass(frozen=True)
class CompiledInput:
    """The compiled shared-state sequence plus the response identity context."""

    state: Any
    response_context: ResponseContext


class ValenDecisionProcessor:
    """Own the checkpoint's tokenizer/processor and its compiler boundary."""

    def __init__(self, checkpoint: Checkpoint) -> None:
        try:
            from transformers import AutoProcessor
            from transformers.dynamic_module_utils import get_class_from_dynamic_module
        except ImportError as exc:
            raise RuntimeError(
                "install the reference dependencies before starting"
            ) from exc
        config = checkpoint.config
        # The compiler is the checkpoint's own compiler.py, verified by
        # pinned.verify_checkpoint before this import.
        self._compiler_type = get_class_from_dynamic_module(
            "compiler.Compiler", str(checkpoint.path), local_files_only=True
        )
        self._max_length = int(config["valen_max_length"])
        self._media_kwargs = config["valen_media_kwargs"]
        self._execution = config["qwen_execution"]
        self.processor = AutoProcessor.from_pretrained(
            str(checkpoint.path), trust_remote_code=True, local_files_only=True
        )
        # Resolve loudly: a silent fallback to the media-only set would reopen
        # the special-token gap the worker promises to close.
        self.special_tokens = tuple(
            dict.fromkeys(
                (*DEFAULT_SPECIAL_TOKENS, *self.processor.tokenizer.all_special_tokens)
            )
        )

    def compile(self, prepared: PreparedInput) -> CompiledInput:
        """Compile one prepared request without owning model/device state."""

        compiler = self._compiler_type(
            self.processor,
            media_root=prepared.media_root,
            max_length=self._max_length,
            media_kwargs=self._media_kwargs,
            execution=self._execution,
        )
        return CompiledInput(
            compiler.compile(prepared.record),
            prepared.response_context,
        )


def question_record(question: Question) -> dict[str, Any]:
    """Build the compiler's SystemOne question for one wire question.

    ``noul`` candidates are the runtime's fixed true/false wording; a rubric
    passes through as ``criteria`` and the compiler appends it to the
    instructions. ``score`` passes its ordered level descriptions straight
    through; the runtime assigns the level keys ``"0"``…
    """

    record: dict[str, Any] = {"type": question.kind, "instructions": question.instructions}
    if question.kind == "noul":
        if question.descriptions != (None, None):
            record["criteria"] = dict(zip(question.keys, question.descriptions))
    elif question.kind == "score":
        record["criteria"] = list(question.descriptions)
    else:
        record["criteria"] = dict(zip(question.keys, question.descriptions))
    return record


def prepare_request(request: Request, media_root: Path) -> PreparedInput:
    """Materialize one validated request into the compiler's record format."""

    media_root = Path(media_root)
    media_root.mkdir(parents=True, exist_ok=True)
    if isinstance(request.state, TextState):
        content: dict[str, Any] = {"type": "text", "text": request.state.text}
        assets: list[dict[str, str]] = []
        image_path = None
        image_sha256 = None
    else:
        image = request.state
        suffix = ".png" if image.format == "PNG" else ".jpg"
        image_path = media_root / f"input{suffix}"
        with image_path.open("xb") as stream:
            stream.write(image.data)
        image_sha256 = hashlib.sha256(image.data).hexdigest()
        content = {"type": "image_url", "image_url": {"url": image_path.name}}
        assets = [{"path": image_path.name, "sha256": image_sha256}]
    record = {
        "request": {
            "state": {"messages": [{"role": "user", "content": [content]}]},
            "questions": {
                question.name: question_record(question) for question in request.questions
            },
        },
        "assets": assets,
    }
    return PreparedInput(
        record=record,
        media_root=media_root,
        image_path=image_path,
        image_sha256=image_sha256,
        response_context=ResponseContext(request.model, request.questions),
    )
