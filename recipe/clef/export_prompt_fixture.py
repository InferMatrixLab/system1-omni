#!/usr/bin/env python3
"""Regenerate the text prompt fixture from the pinned Clef loader and tokenizer.

Run from the repository root. The loader is imported with a torch stub; this
does not download weights or run the joint head.

    python recipe/clef/export_prompt_fixture.py \
      --loader /path/to/joint_schema_model.py \
      --tokenizer /path/to/tokenizer.json \
      --output tests/clef/fixtures/text_contract.json

``--check`` compares an existing fixture and exits non-zero on drift.
The tokenizer must be the pinned ``tokenizer.json``
(SHA-256 06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523).
Token ids come from the ``tokenizers`` library
(``Tokenizer.encode(..., add_special_tokens=False)``), which is the backend
named in the pinned ``tokenizer_config.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

from models.clef.engine import LOADER_SHA256, TOKENIZER_SHA256
from models.clef.prompt import prompt_pieces

CASES = [
    {
        "name": "choice_noul_score",
        "request": {
            "model": "clef-flash",
            "state": {"invoice": {"currency": "USD", "total": 1250, "vendor": "Acme"}},
            "questions": {
                "team": {
                    "type": "choice",
                    "instructions": "Which team should handle the invoice?",
                    "criteria": {
                        "technical": "Bugs or outages",
                        "billing": "Payments or invoices",
                    },
                },
                "large": {
                    "type": "noul",
                    "instructions": "Is the total above 1000 USD?",
                    "criteria": {"true": "The total is greater than 1000 USD."},
                },
                "severity": {
                    "type": "score",
                    "instructions": "How severe is the customer impact?",
                    "criteria": ["No impact", "Minor", "Major", "Critical"],
                },
            },
        },
    },
    {
        "name": "string_state_and_default_instruction",
        "request": {
            "model": "Cloudflare/clef",
            "state": "Checkout has been failing for every customer.",
            "questions": {
                "urgent": {"type": "noul"},
                "team": {
                    "type": "choice",
                    "criteria": {"sales": "Plans and upgrades", "billing": "Payments"},
                },
            },
        },
    },
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _install_torch_stub() -> None:
    """Let ``joint_schema_model.py`` import without installing torch."""

    def module(name: str) -> types.ModuleType:
        created = types.ModuleType(name)
        sys.modules[name] = created
        return created

    torch = module("torch")
    torch.bfloat16 = "bfloat16"
    torch.dtype = object
    torch.device = object
    torch.nn = module("torch.nn")
    torch.nn.functional = module("torch.nn.functional")
    torch.nn.Module = object
    torch.nn.LayerNorm = object
    torch.nn.Linear = object
    torch.nn.MultiheadAttention = object
    torch.nn.Dropout = object
    torch.nn.Sequential = object
    torch.nn.GELU = object
    torch.nn.Embedding = object
    torch.nn.TransformerDecoderLayer = object
    torch.nn.Parameter = lambda *args, **kwargs: None

    def inference_mode(*args, **kwargs):
        def decorator(fn):
            return fn

        return decorator

    torch.inference_mode = inference_mode


def _load_encoder(path: Path):
    if _sha256(path) != LOADER_SHA256:
        raise SystemExit(f"{path} is not the pinned joint_schema_model.py")
    _install_torch_stub()
    spec = importlib.util.spec_from_file_location("pinned_clef_encoder", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _PieceTokenizer:
    """Adapt a ``tokenizers.Tokenizer`` to the pinned ``_tokens`` call."""

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, text: str, add_special_tokens: bool = False):
        if add_special_tokens:
            raise AssertionError("the pinned encoder must not add special tokens")
        ids = self.tokenizer.encode(text, add_special_tokens=False).ids

        class Batch:
            input_ids = ids

        return Batch()


def _ids(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False).ids


def build_fixture(loader_path: Path, tokenizer_path: Path) -> dict:
    if _sha256(tokenizer_path) != TOKENIZER_SHA256:
        raise SystemExit(f"{tokenizer_path} is not the pinned tokenizer.json")
    from tokenizers import Tokenizer

    encoder = _load_encoder(loader_path)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    adapted = _PieceTokenizer(tokenizer)
    cases = []
    for case in CASES:
        request = case["request"]
        pieces = prompt_pieces(request)
        encoded = encoder.encode_record(adapted, request, max_length=10**9)
        ids: list[int] = []
        piece_token_lengths = []
        for piece in pieces:
            piece_ids = _ids(tokenizer, piece)
            piece_token_lengths.append(len(piece_ids))
            ids.extend(piece_ids)
        if tuple(ids) != encoded.input_ids:
            raise SystemExit(f"{case['name']}: prompt pieces do not match encode_record")
        state_tokens = piece_token_lengths[1]
        cases.append(
            {
                "name": case["name"],
                "request": request,
                "pieces": pieces,
                "piece_token_lengths": piece_token_lengths,
                "state_tokens": state_tokens,
                "fixed_length": len(ids) - state_tokens,
                "input_ids": ids,
            }
        )
    return {
        "provenance": {
            "loader_sha256": LOADER_SHA256,
            "tokenizer_sha256": TOKENIZER_SHA256,
            "tokenizer_api": "tokenizers.Tokenizer.encode add_special_tokens=False",
            "repos": [
                "Cloudflare/clef-flash@f0b7d3cd65296abfe459e6c228b618f5d8b26d2f",
                "Cloudflare/clef@a30162b99d68a481d8bd75d38df069a0cf83df40",
            ],
            "note": (
                "Both revisions ship the same joint_schema_model.py and tokenizer.json. "
                "Ids are the untruncated concatenation. The encoder's max_length slice "
                "was not applied."
            ),
        },
        "cases": cases,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loader", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/clef/fixtures/text_contract.json"),
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    fixture = build_fixture(args.loader, args.tokenizer)
    encoded = json.dumps(fixture, ensure_ascii=False, indent=2) + "\n"
    if args.check:
        current = args.output.read_text(encoding="utf-8")
        if current != encoded:
            raise SystemExit(f"{args.output} does not match the pinned loader")
        print(f"checked {args.output}")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
