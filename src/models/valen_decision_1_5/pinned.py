"""Pinned identity of the released Valen-Decision-1.5 checkpoints.

A released checkpoint is a complete Hugging Face directory: weights, config,
tokenizer and processor files, and the runtime (compiler, model wrapper,
heads) as flat Python files that ``trust_remote_code`` executes. This module
verifies a local directory against the digests recorded for the pinned
revisions before any of that code is imported. It has no torch dependency.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .protocol import MODEL_NAMES

# Files shared byte for byte by all five released sizes: the runtime the
# worker executes, and the tokenizer and processor settings that fix the
# prompt tokens and image patches.
SHARED_SHA256 = {
    "batching.py": "ea45d4724bcafca559fc5a2a18092918b633746771c19da6554c29bc5f2cef64",
    "block_attention.py": "27b0a2b209a17afe13a7ecd0afd180d99ba6b873147e7b18bd1e028dea96e074",
    "compiler.py": "72cada57ec9a50ad688e8542f52615b82a021cc60e90e80c5e147e86a823de85",
    "configuration_valen.py": "29f930d193a75547bd5fa63715c0b5f3ac6acf4b5cc803f98d61713d0e951534",
    "data_types.py": "b06ce08081d8f7b2bf36abf302410a96086395e75a04f9e9b1c591db519b1641",
    "decisions.py": "476e998b624d7222c156e8e87e3d9c07b04dafe44c412cbe95b29f9d76b9935b",
    "grounding.py": "a40ae673819a3cc243e081bcd2020980044ad26f2ea9591d5f49b29fc855c16b",
    "heads.py": "63b5970a8bfa121eff94d50ffd35e1fa8387f3af5abccc5f5b75381fb084fc34",
    "modeling_valen.py": "15f388154a3cf78683dcb50b8c302554dc2a554fb7b247cb54d948eb6f99b71a",
    "responses.py": "a7239109a0a466966037c690040a773a03fa0ee8476a80408cfacf79ce463a7b",
    "runtime_model.py": "04a971069fbfb1a86315d52c294c9ac184f691f5223771efcca4454bc1be2915",
    "schema.py": "20277d14d11542c23b45e56b22f1c1dcb27a3853b4b06af449d72b6accd95d63",
    "processor_config.json": "d89ef49ce9cd37fbf510158e13c1ef063d9286411c1ec9049932dbe0487143b1",
    "tokenizer_config.json": "7488550a894dcf741616126e322ea31d66cf9457b59ec5825f4c7efdced421f8",
    "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
}


@dataclass(frozen=True)
class Release:
    """One released size: its Hub revision and per-size file digests."""

    size: str
    revision: str
    files: dict[str, str]
    weights: dict[str, str]

    @property
    def repo(self) -> str:
        return f"Valen-Team/Valen-Decision-1.5-{self.size}"

    @property
    def model_name(self) -> str:
        return MODEL_NAMES[self.size]


RELEASES = (
    Release(
        "0.8B",
        "e079395fe9b8e0763b384af403e636e91c1a1559",
        {
            "config.json": "13f0d828c5a39e18bd8329231e5ad259d087bd3b83326f2c3cd8e1b26e95af93",
            "chat_template.jinja": "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
        },
        {"model.safetensors": "d2808df20a2501debe47bd8bacac41d1cd33cc1a9d6f687d8dc436da9f3603bc"},
    ),
    Release(
        "2B",
        "1b890606578ba0f40bb70ed46891ab65f778e76d",
        {
            "config.json": "97faa5441752eecb304a6ec8d17cf03971d5bb363481eb2020b347fe88072694",
            "chat_template.jinja": "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
        },
        {"model.safetensors": "1c9c1112aa0c417b081b0fa757c78be77e487b97de50adb4c83eecfbe03c35da"},
    ),
    Release(
        "4B",
        "9b0cacc976d7fb72f252ee1adb1564f5f15e462c",
        {
            "config.json": "fc6c60829e353a7c66bcf21072fe4a6f8a52490604867d91226aa740ec12698a",
            "chat_template.jinja": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
            "model.safetensors.index.json": "547249c00c48c710dbc21713d412da785e17e3a01a9ebf4e8d4db6403b598706",
        },
        {
            "model-00001-of-00002.safetensors": "e79499386c15eebd0caffe0126372e17a895e722c1c6f234db549ec7ace7365f",
            "model-00002-of-00002.safetensors": "18e13d4365d7181ab5eefad8de38bf526ca83792998c7623f99d287b57f40c34",
        },
    ),
    Release(
        "9B",
        "5a512117ebc0f2b69f311becfcf47cdef8abf568",
        {
            "config.json": "25595cf89d8552fe33405fe6e8e5b44fa1a8ff379ab92f0d4dd7863038b07f59",
            "chat_template.jinja": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
            "model.safetensors.index.json": "5443f342888348373c769089ab6c62fff7411e8783a9260d4c75a7792afb8ea7",
        },
        {
            "model-00001-of-00004.safetensors": "c8febff8ad3c029006b7443624becd5394daa0bd5281df6e959da91c59662a34",
            "model-00002-of-00004.safetensors": "e2e37962c95a303c63ea9d09534ceb7cbe5a0d6a0f186ebf4304f7df7a45fdb5",
            "model-00003-of-00004.safetensors": "5988fd49da5e85fe045898d2540ce587dc53cbeee420dde45670812f7b5373b8",
            "model-00004-of-00004.safetensors": "959a26154c0f432680036bfa041a6ea0058a6b34fb39d3ee8b7e1cc5e9c785a9",
        },
    ),
    Release(
        "27B",
        "9f1b90bcb6841d9a4a44acfd84c6ecbd4c18fec6",
        {
            "config.json": "d30d71fc730984e6f0caf80a14eab83837fd88660224cf8699738b2371d3c830",
            "chat_template.jinja": "567255be85e0a113405f917423fd8ef5592b01616ce6bdfb6fc4122b9818ec0e",
            "model.safetensors.index.json": "df49cc26f6fb5b30359c0dc14cfe8716b5df40639bd809ff8aaeecedcf32be14",
        },
        {
            "model-00001-of-00011.safetensors": "64db8c6fb6a142c4da7567f3d8a5c9e4f02459920311e544c7a64816b7ec2042",
            "model-00002-of-00011.safetensors": "fafb6729ed29e70ca6e485de9bc41d869f2a7c9d0bc1867cd9658316972fdb77",
            "model-00003-of-00011.safetensors": "3d2facb794092e90854ec37764a039586a4778fdcb49b54f41953d95a14de7cc",
            "model-00004-of-00011.safetensors": "6816b9136421aba4591fe29f85dada4c0bec2d86bb5b5c095c58b7e5cce8f83c",
            "model-00005-of-00011.safetensors": "43b5d9d7b222d96e258f6562a3db5b461acf837cfcd0df4d3d10363f5d188292",
            "model-00006-of-00011.safetensors": "63434f7882aab05d1420b3900a1919c3beccbb5af4538cb775399d5e4865115d",
            "model-00007-of-00011.safetensors": "6f3e9ce07026c073a0bc7cb369af1500b3a2e3fc4687007e255fd8cffd299c32",
            "model-00008-of-00011.safetensors": "ac5a0aa0358bb29d70939235435c18398946d09e42b52bb195b149f82cc00400",
            "model-00009-of-00011.safetensors": "57e92e32c3dd183adbd477f4f7b2eb9ab8b99700938a2e02034ef5305653fcde",
            "model-00010-of-00011.safetensors": "87e5033793296d842e6cccb304492a1a323f8b614e656ae55be9549c544d8cbc",
            "model-00011-of-00011.safetensors": "796ec1999b31b70eda19138a8d66e3da2cddcb4af09e0fe7c6c58503cda12d82",
        },
    ),
)


@dataclass(frozen=True)
class Checkpoint:
    """A verified local checkpoint directory."""

    path: Path
    release: Release
    config: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(path: str | Path, weights: bool = True) -> Checkpoint:
    """Match a local directory to one pinned release, or refuse it.

    The release is identified by its ``config.json`` digest. Every runtime,
    tokenizer and processor file is then hashed in full; ``weights=False``
    skips only the weight shards. Nothing in the directory is imported.
    """

    root = Path(path).resolve()
    config_path = root / "config.json"
    if not config_path.is_file():
        raise ValueError(f"checkpoint must contain config.json: {root}")
    config_digest = _sha256(config_path)
    release = next(
        (entry for entry in RELEASES if entry.files["config.json"] == config_digest), None
    )
    if release is None:
        raise ValueError(
            "config.json does not match any pinned Valen-Decision-1.5 release;"
            " download a pinned revision or re-pin deliberately"
        )
    expected = {**SHARED_SHA256, **release.files}
    if weights:
        expected.update(release.weights)
    problems = []
    for name, digest in sorted(expected.items()):
        target = root / name
        if not target.is_file():
            problems.append(f"missing {name}")
        elif name != "config.json" and _sha256(target) != digest:
            problems.append(f"{name}: SHA-256 differs from {release.repo}@{release.revision}")
    if problems:
        raise ValueError("checkpoint verification failed: " + "; ".join(problems))
    return Checkpoint(root, release, json.loads(config_path.read_text(encoding="utf-8")))
