"""Clef executor for one pinned Cloudflare snapshot.

The executor owns checkpoint identity, the hash gate, and the loaded joint
schema model. It does not admit HTTP requests. ``load_release_model`` and
``systemone`` are the checkpoint's own functions, imported only after every
executed and tokenized file matches the pin.

Clef-flash (9B, Qwen3.5-9B) and Clef (27B, Qwen3.8-27B) share one loader and
one tokenizer. They do not share ``config.json`` or ``joint_head_config.json``.
Neither snapshot is an Open-Jev export: the decision head is
``joint_head.safetensors``, and the backbone index has no separate LoRA keys.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from models.clef.preprocess import MAX_INPUT_TOKENS, enforce_limit
from models.clef.prompt import render

DEFAULT_VARIANT = "clef-flash"
LOADER_SHA256 = "0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3"
TOKENIZER_SHA256 = "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"
TOKENIZER_CONFIG_SHA256 = "91a08f825d370d085d692e04cf117cdd7faad7bf18e996f1e6031b6dab03db72"
PROCESSOR_CONFIG_SHA256 = "d89ef49ce9cd37fbf510158e13c1ef063d9286411c1ec9049932dbe0487143b1"
SHARED_FILES = {
    "joint_schema_model.py": LOADER_SHA256,
    "tokenizer.json": TOKENIZER_SHA256,
    "tokenizer_config.json": TOKENIZER_CONFIG_SHA256,
    "processor_config.json": PROCESSOR_CONFIG_SHA256,
}
EXECUTABLE = "joint_schema_model.py"
MODELING = "transformers.models.qwen3_5.modeling_qwen3_5"
FLA_DISTRIBUTION = "flash-linear-attention"
FLA_VERSION = "0.5.2"
FLA_MODULE = "fla.ops.gated_delta_rule"
CONV_DISTRIBUTION = "causal-conv1d"
CONV_MODULE = "causal_conv1d"
CONV_MINIMUM = (1, 5)
FI_DISTRIBUTION = "flashinfer-python"
FI_VERSION = "0.6.16.post3"
FI_MODULE = "flashinfer.gdn_prefill"
GDN_PREFILL = "flashinfer"
FLA_PREFILL = "fla"
# FlashInfer 0.6.16 GDN prefill dispatches reported majors 9, 10, and 12.
FLASHINFER_ARCH_MAJORS = (9, 10, 12)
REQUIRED_LINEAR_ATTENTION = "flashinfer+fla+causal_conv1d"
FLA_LINEAR_ATTENTION = "fla+causal_conv1d"
READY_LINEAR_ATTENTION = frozenset({REQUIRED_LINEAR_ATTENTION, FLA_LINEAR_ATTENTION})


@dataclass(frozen=True)
class Variant:
    """One public Clef release. Hashes are of the file bytes at ``revision``."""

    name: str
    repo_id: str
    revision: str
    base_model: str
    hidden_size: int
    num_hidden_layers: int
    config_sha256: str
    joint_head_config_sha256: str


VARIANTS = {
    "clef-flash": Variant(
        name="clef-flash",
        repo_id="Cloudflare/clef-flash",
        revision="f0b7d3cd65296abfe459e6c228b618f5d8b26d2f",
        base_model="Qwen/Qwen3.5-9B",
        hidden_size=4096,
        num_hidden_layers=32,
        config_sha256="66f87f6fb2616b46604daf2a9c67ddc87938296d07156efa34d59b5be49e3238",
        joint_head_config_sha256="77efe959a38b5b17b241543e129e695f3c77465ece55a25985279bd8176279a0",
    ),
    "clef": Variant(
        name="clef",
        repo_id="Cloudflare/clef",
        revision="a30162b99d68a481d8bd75d38df069a0cf83df40",
        base_model="Qwen/Qwen3.8-27B",
        hidden_size=5120,
        num_hidden_layers=64,
        config_sha256="c42e88892bd3fd84e8276b2ad90df58c1c3b797676ea161035006a72ad468c58",
        joint_head_config_sha256="890be585d75b981201eb96a35f98a9967afe37bc8d220cfdea2e72d56507534f",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_hash(root: Path, name: str, expected: str) -> None:
    path = root / name
    if not path.is_file():
        raise ValueError(f"missing {name}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{name} SHA-256 {actual} is not the pinned digest")


def verify_snapshot(
    root: Path, variant: Variant, shared_files: dict[str, str] | None = None
) -> None:
    """Hash the loader, tokenizer and variant configs before importing them.

    Weight shards are not hashed here. Download the snapshot at
    ``variant.revision`` so those shards belong to the same commit as the
    hashed files. An extra ``.py`` or a ``__pycache__`` in the snapshot root
    is refused so the import cannot pick up a different module.
    """

    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"the Clef snapshot directory does not exist: {root}")
    for name, digest in (SHARED_FILES if shared_files is None else shared_files).items():
        _require_hash(root, name, digest)
    _require_hash(root, "config.json", variant.config_sha256)
    _require_hash(root, "joint_head_config.json", variant.joint_head_config_sha256)
    for entry in root.iterdir():
        if entry.name == "__pycache__":
            raise ValueError("unexpected bytecode cache __pycache__/ in the snapshot root")
        if entry.is_file() and entry.suffix == ".py" and entry.name != EXECUTABLE:
            raise ValueError(f"unexpected Python file {entry.name} in the snapshot root")
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    text = config.get("text_config") or {}
    if config.get("architectures") != ["Qwen3_5ForConditionalGeneration"]:
        raise ValueError("config.json is not a Qwen3_5ForConditionalGeneration backbone")
    if text.get("hidden_size") != variant.hidden_size:
        raise ValueError(
            f"config hidden_size {text.get('hidden_size')} is not {variant.hidden_size}"
        )
    if text.get("num_hidden_layers") != variant.num_hidden_layers:
        raise ValueError(
            f"config num_hidden_layers {text.get('num_hidden_layers')} "
            f"is not {variant.num_hidden_layers}"
        )
    head = json.loads((root / "joint_head_config.json").read_text(encoding="utf-8"))
    if head.get("hidden_size") != variant.hidden_size:
        raise ValueError(
            f"joint head hidden_size {head.get('hidden_size')} is not {variant.hidden_size}"
        )


def untruncated_budget(total: int, state_tokens: int, max_length: int) -> None:
    """Reject a prompt whose full encoding does not fit, without slicing state.

    ``state_tokens`` is the rendered state counted by the same tokenizer the
    encoder uses for that piece. A negative fixed length means the two counts
    disagree; the request is not served.
    """

    if state_tokens < 0 or state_tokens > total:
        raise RuntimeError(
            f"token accounting mismatch: encoded {total}, state piece {state_tokens}"
        )
    enforce_limit(total - state_tokens, total, max_length)


def require_token_agreement(admitted: int, response: dict[str, Any]) -> dict[str, Any]:
    """Refuse a response whose prompt length is not the untruncated admission count.

    The pinned ``systemone`` slices state to fit. Admission already rejected
    that case; a shorter encoder count means the slice still happened.
    """

    usage = response.get("usage") if isinstance(response, dict) else None
    got = usage.get("input_tokens") if isinstance(usage, dict) else None
    if got != admitted:
        raise RuntimeError(
            f"token accounting mismatch: admitted {admitted} untruncated tokens, "
            f"encoder returned {got}"
        )
    return response


def bound_linear_attention(modeling: Any) -> str:
    """Name the linear-attention implementation the modeling module exposes.

    Transformers copies ``chunk_gated_delta_rule`` and ``causal_conv1d_fn``
    onto each ``Qwen3_5GatedDeltaNet`` at ``__init__``. Ready is either the
    FlashInfer wrapper or the FLA chunk, together with ``causal_conv1d_fn``.
    ``/health`` reports whichever global is actually bound.
    """

    if modeling is None:
        return "unloaded"

    chunk = getattr(modeling, "chunk_gated_delta_rule", None)
    conv = getattr(modeling, "causal_conv1d_fn", None) is not None
    prefill = getattr(modeling, "clef_gdn_prefill", None)
    if prefill == GDN_PREFILL and chunk is flashinfer_chunk_gated_delta_rule and conv:
        return REQUIRED_LINEAR_ATTENTION
    if chunk is not None and conv:
        return FLA_LINEAR_ATTENTION
    if chunk is not None:
        return "fla"
    if conv:
        return "causal_conv1d"
    return "torch_chunk"


def select_gdn_prefill(
    capability: tuple[int, int] | None, requested: str = FLA_PREFILL
) -> str:
    """Choose the prefill bind. The default is the measured FLA chunk.

    FlashInfer is opt-in. It is returned only when ``requested`` is
    ``flashinfer`` and the reported major is 9, 10, or 12. A device that
    reports 8.9 stays on the FLA chunk, including the lab cards
    ``nvidia-smi`` names ``NVIDIA L20X``. The printed name is not a gate,
    and a capability other than 9.0 is not a reason to refuse the FLA path.
    """

    if requested not in (FLA_PREFILL, GDN_PREFILL):
        raise ValueError(f"gdn prefill must be {FLA_PREFILL!r} or {GDN_PREFILL!r}")
    if (
        requested == GDN_PREFILL
        and capability is not None
        and capability[0] in FLASHINFER_ARCH_MAJORS
    ):
        return GDN_PREFILL
    return FLA_PREFILL


def resolve_gdn_prefill(
    capability: tuple[int, int] | None, requested: str, flashinfer_loaded: bool
) -> str:
    """Return the bind, or refuse an opt-in FlashInfer request that cannot run.

    The default request stays on FLA even when FlashInfer imported and the
    reported major is 9, 10, or 12. That path has no forward and no answer
    parity against ``fla+causal_conv1d``, so it is not selected by itself.
    """

    backend = select_gdn_prefill(capability, requested)
    if requested == GDN_PREFILL and backend != GDN_PREFILL:
        printed = "unavailable" if capability is None else f"{capability[0]}.{capability[1]}"
        raise RuntimeError(
            "Clef FlashInfer GDN prefill was requested with --gdn-prefill "
            f"flashinfer, but this device reports capability {printed}. "
            "That kernel is selected only for reported majors 9, 10, and 12. "
            "The measured path is --gdn-prefill fla."
        )
    if requested == GDN_PREFILL and not flashinfer_loaded:
        raise RuntimeError(
            "Clef FlashInfer GDN prefill was requested, but "
            f"{FI_DISTRIBUTION}=={FI_VERSION} did not import. "
            "The measured path is --gdn-prefill fla."
        )
    return backend


def cuda_capability(device: str) -> tuple[int, int]:
    """Return the capability of a CUDA device. The name is not consulted."""

    import torch

    requested = torch.device(device)
    if requested.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Clef prefill requires --device cuda.")
    index = requested.index if requested.index is not None else torch.cuda.current_device()
    return torch.cuda.get_device_capability(index)


def bind_selected_prefill(modeling: Any, backend: str, flashinfer_loaded: bool) -> None:
    """Install FlashInfer when it was selected and the package loaded.

    Otherwise the FLA chunk global that Transformers bound at import stays
    in place. Layers copy that global, so this runs before
    ``load_release_model``.
    """

    if backend == GDN_PREFILL and flashinfer_loaded:
        install_flashinfer_gdn_prefill(modeling)


def activate_gdn_prefill(
    device: str, flashinfer_loaded: bool, requested: str = FLA_PREFILL
) -> None:
    """Bind the requested prefill. The default is the FLA chunk."""

    modeling = sys.modules.get(MODELING)
    if modeling is None:
        raise RuntimeError(f"{MODELING} is not imported.")
    bind_selected_prefill(
        modeling,
        resolve_gdn_prefill(cuda_capability(device), requested, flashinfer_loaded),
        flashinfer_loaded,
    )


def flashinfer_chunk_gated_delta_rule(
    q: Any,
    k: Any,
    v: Any,
    g: Any = None,
    beta: Any = None,
    scale: float | None = None,
    initial_state: Any = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Any = None,
    **kwargs: Any,
) -> tuple[Any, None]:
    """Run one-sequence GDN prefill through FlashInfer's fused SM90 kernel.

    Transformers calls this with FLA's ``[B, T, H, D]`` layout and a
    log-space decay ``g``. FlashInfer 0.6.16 wants ``[T, H, D]``, a float32
    multiplicative gate, and a non-null ``cu_seqlens``. ``use_cp`` is false
    so SM90 does not take the context-parallel path. Q/K L2 norm is FLA's
    ``l2norm_fwd``; FlashInfer's own flag stays false so the vectors are not
    normalized twice. vLLM's SM90 adapter is the source of that split.
    FLA state is ``[N, H, K, V]`` and FlashInfer state is ``[N, H, V, K]``,
    so a recurrent state is refused. Clef prefill passes neither.
    """

    del kwargs
    if initial_state is not None or output_final_state:
        raise RuntimeError(
            "Clef FlashInfer GDN prefill does not continue or return a recurrent "
            "state. This worker serves one full-prompt prefill."
        )
    if getattr(q, "shape", (0,))[0] != 1:
        raise RuntimeError(
            "Clef FlashInfer GDN prefill serves one sequence, "
            f"got batch {getattr(q, 'shape', None)}"
        )

    import torch
    from flashinfer.gdn_prefill import chunk_gated_delta_rule as chunk_gated_delta_rule_fi
    from fla.modules.l2norm import l2norm_fwd

    if use_qk_l2norm_in_kernel:
        q, _ = l2norm_fwd(q)
        k, _ = l2norm_fwd(k)
    q = q.squeeze(0).contiguous()
    k = k.squeeze(0).contiguous()
    v = v.squeeze(0).contiguous()
    fi_g = None if g is None else torch.exp(g.squeeze(0).contiguous().to(torch.float32))
    fi_beta = None if beta is None else beta.squeeze(0).contiguous().to(torch.float32)
    if cu_seqlens is None:
        cu_seqlens = torch.tensor([0, q.shape[0]], dtype=torch.int64, device=q.device)
    else:
        cu_seqlens = cu_seqlens.to(device=q.device, dtype=torch.int64)
    result = chunk_gated_delta_rule_fi(
        q=q,
        k=k,
        v=v,
        g=fi_g,
        beta=fi_beta,
        scale=scale,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=False,
        use_cp=False,
    )
    return result.unsqueeze(0), None


def install_flashinfer_gdn_prefill(modeling: Any) -> None:
    """Replace the modeling chunk global before layers copy it.

    ``Qwen3_5GatedDeltaNet.__init__`` binds ``self.chunk_gated_delta_rule``
    from this module global. Patching after ``load_release_model`` does not
    change layers that already copied the FLA function.
    """

    if getattr(modeling, "causal_conv1d_fn", None) is None:
        raise RuntimeError(
            "causal_conv1d_fn is not bound. Import causal_conv1d before "
            f"{MODELING}. F.conv1d is not served."
        )
    if getattr(modeling, "chunk_gated_delta_rule", None) is None:
        raise RuntimeError(
            "chunk_gated_delta_rule is not bound. Import "
            f"{FLA_DISTRIBUTION}=={FLA_VERSION} before {MODELING}. "
            "torch_chunk_gated_delta_rule is not served."
        )
    modeling.chunk_gated_delta_rule = flashinfer_chunk_gated_delta_rule
    modeling.clef_gdn_prefill = GDN_PREFILL


def version_tuple(text: str) -> tuple[int, ...]:
    """Return the leading numeric components of a distribution version."""

    parts: list[int] = []
    for piece in text.split("."):
        digits = ""
        for char in piece:
            if not char.isdigit():
                break
            digits += char
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def check_distribution_version(
    installed: str, *, exact: str | None = None, minimum: tuple[int, ...] | None = None
) -> None:
    """Reject a pin outside the recipe's exact version or minimum range."""

    if exact is not None and installed != exact:
        raise RuntimeError(f"required {exact}, found {installed}")
    if minimum is not None and version_tuple(installed) < minimum:
        joined = ".".join(str(part) for part in minimum)
        raise RuntimeError(f"required >={joined}, found {installed}")


def try_import_flashinfer() -> bool:
    """Return whether the pinned FlashInfer GDN prefill module imported.

    A missing package, a different version, or an import error means the
    FLA chunk remains the prefill. That is not a readiness failure.
    """

    try:
        installed = importlib.metadata.version(FI_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        return False
    try:
        check_distribution_version(installed, exact=FI_VERSION)
    except RuntimeError:
        return False
    try:
        importlib.import_module(FI_MODULE)
    except (ImportError, OSError):
        return False
    return True


def require_fast_linear_attention() -> bool:
    """Import conv and FLA before Transformers binds Qwen3.5.

    The modeling module chooses ``FusedRMSNormGated``, ``causal_conv1d_fn``,
    and the FLA chunk global once, at import. ``causal-conv1d`` and
    ``flash-linear-attention`` are required. A missing copy leaves
    ``torch_chunk_gated_delta_rule`` or ``F.conv1d``, so startup raises.
    FlashInfer is attempted and the return value says whether it loaded.
    The device choice patches the chunk global later, before
    ``load_release_model``.
    """

    if MODELING in sys.modules:
        raise RuntimeError(
            f"{MODELING} is already imported. Start a new process so "
            f"{CONV_DISTRIBUTION}>={'.'.join(str(part) for part in CONV_MINIMUM)} "
            f"and {FLA_DISTRIBUTION}=={FLA_VERSION} import first. "
            "torch_chunk_gated_delta_rule and F.conv1d are not served."
        )
    for distribution, exact, minimum in (
        (CONV_DISTRIBUTION, None, CONV_MINIMUM),
        (FLA_DISTRIBUTION, FLA_VERSION, None),
    ):
        try:
            installed = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"{distribution} is not installed. The PyTorch linear-attention "
                "fallback is not served."
            ) from exc
        try:
            check_distribution_version(installed, exact=exact, minimum=minimum)
        except RuntimeError as exc:
            raise RuntimeError(f"{distribution} {exc}") from exc
    for distribution, module in (
        (CONV_DISTRIBUTION, CONV_MODULE),
        (FLA_DISTRIBUTION, FLA_MODULE),
    ):
        try:
            importlib.import_module(module)
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                f"{distribution} did not import {module} ({type(exc).__name__}). "
                "torch_chunk_gated_delta_rule and F.conv1d are not served."
            ) from exc
    flashinfer_loaded = try_import_flashinfer()
    importlib.import_module(MODELING)
    return flashinfer_loaded


def require_bound_linear_attention(modeling: Any) -> str:
    """Refuse ready on ``torch_chunk`` or ``F.conv1d``.

    Both ``flashinfer+fla+causal_conv1d`` and ``fla+causal_conv1d`` are
    ready. The string is the bind the process actually installed.
    """

    bound = bound_linear_attention(modeling)
    if bound in READY_LINEAR_ATTENTION:
        return bound
    raise RuntimeError(
        f"Clef linear attention is {bound}, so the worker is not ready. "
        f"Ready is {FLA_LINEAR_ATTENTION} or {REQUIRED_LINEAR_ATTENTION} "
        f"({FLA_DISTRIBUTION}=={FLA_VERSION} and "
        f"{CONV_DISTRIBUTION}>={'.'.join(str(part) for part in CONV_MINIMUM)}). "
        "torch_chunk_gated_delta_rule and F.conv1d are not served."
    )


def load_joint_schema_module(path: Path) -> Any:
    """Import one verified ``joint_schema_model.py`` without putting its directory on ``sys.path``."""

    spec = importlib.util.spec_from_file_location("clef_joint_schema_model", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ClefExecutor:
    """Load and run one immutable pinned Clef snapshot."""

    def __init__(
        self,
        snapshot: str | Path,
        variant: str = DEFAULT_VARIANT,
        device: str = "cuda",
        gdn_prefill: str = FLA_PREFILL,
    ) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {sorted(VARIANTS)}")
        if gdn_prefill not in (FLA_PREFILL, GDN_PREFILL):
            raise ValueError(f"gdn_prefill must be {FLA_PREFILL!r} or {GDN_PREFILL!r}")
        self.variant = VARIANTS[variant]
        self.root = Path(snapshot)
        self.device = device
        self.gdn_prefill_request = gdn_prefill
        self.max_length = MAX_INPUT_TOKENS
        self._warmed = False
        verify_snapshot(self.root, self.variant)
        flashinfer_loaded = require_fast_linear_attention()
        activate_gdn_prefill(device, flashinfer_loaded, gdn_prefill)
        self.module = load_joint_schema_module(self.root / EXECUTABLE)
        self.model, self.processor = self.module.load_release_model(self.root, device=device)
        self.linear_attention = require_bound_linear_attention(sys.modules.get(MODELING))
        if self.processor.tokenizer.pad_token_id is None:
            raise ValueError("tokenizer pad_token_id is required")

    def count_tokens(self, text: str) -> int:
        return len(self.processor.tokenizer(text, add_special_tokens=False).input_ids)

    def serve(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Encode once, refuse an over-long prompt, then score that encoding.

        The pinned ``systemone`` helper tokenizes and then slices state to
        ``max_length``. Calling it after a separate admission pass would
        tokenize the schema twice and could still slice if the counts drifted.
        This path keeps the full ``encode_record`` result and runs the joint
        head on it. The duplicate BPE pass is the only serve-path change; the
        head and backbone are the pinned modules. No latency delta is claimed.
        """

        encoded = self.module.encode_record(
            self.processor.tokenizer,
            payload,
            max_length=10**9,
            processor=self.processor,
        )
        total = len(encoded.input_ids)
        state_tokens = self.count_tokens(render(payload["state"]))
        untruncated_budget(total, state_tokens, self.max_length)
        response = self._forward(payload, encoded)
        return require_token_agreement(total, response)

    def _forward(self, payload: dict[str, Any], encoded: Any) -> dict[str, Any]:
        import torch

        pad = self.processor.tokenizer.pad_token_id
        if pad is None:
            raise RuntimeError("tokenizer pad_token_id is required")
        device = next(self.model.parameters()).device
        with torch.inference_mode():
            logits = self.model(
                self.module.collate_records([encoded], pad, device)
            )[0]
        questions = payload["questions"]
        answers = {
            question.question_id: self.module.systemone_answer(
                questions[question.question_id],
                dict(
                    zip(
                        question.option_ids,
                        question_logits.float().softmax(-1).tolist(),
                    )
                ),
            )
            for question, question_logits in zip(encoded.questions, logits)
        }
        return {
            "model": payload["model"],
            "answers": answers,
            "usage": {"input_tokens": len(encoded.input_ids), "output_tokens": 0},
        }

    def warmup(self) -> None:
        """Run one text forward before this process reports ready.

        The HTTP server is not bound until this returns. A CUDA context or
        kernel failure exits startup instead of the first live request.
        The answer is discarded and is not a latency sample.
        """

        self.serve(
            {
                "model": self.variant.name,
                "state": "warmup",
                "questions": {
                    "ready": {
                        "type": "noul",
                        "instructions": "Is warmup complete?",
                    }
                },
            }
        )
        self._warmed = True

    def health(self) -> dict[str, Any]:
        return {
            "status": "ready" if self._warmed else "loading",
            "model": self.variant.repo_id,
            "variant": self.variant.name,
            "revision": self.variant.revision,
            "base_model": self.variant.base_model,
            "max_input_tokens": self.max_length,
            "modalities": ["text"],
            "prompt_mode": "non-thinking",
            "linear_attention": self.linear_attention,
        }

    def close(self) -> None:
        self.model = None
        self.processor = None
        self.module = None
