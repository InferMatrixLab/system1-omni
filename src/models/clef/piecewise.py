"""Opt-in piecewise inductor CUDA graphs for the Qwen3.5 backbone.

The default Clef path is eager FLA. ``--backbone piecewise`` compiles the
token mixer and the MLP inside each ``Qwen3_5DecoderLayer`` with inductor
``reduce-overhead``. The residual add stays in the eager layer forward:
``hidden_states.clone()`` runs outside ``torch.compile`` so a CUDAGraph
output is not aliased into the next replay. Two other call sites also stay
outside every captured graph:

1. The FLA chunk. Each ``Qwen3_5GatedDeltaNet.chunk_gated_delta_rule`` is
   wrapped and marked ``torch.compiler.disable``. That wrapper is the split
   around ``ChunkGatedDeltaRuleFunction``. A warmup that runs the chunk
   while CUDA is capturing, or while Dynamo is tracing, is not a bind.
2. The joint head. Modules whose class name contains ``Joint``, and any
   ``joint_head`` attribute, keep an eager ``forward``. They are not
   compiled.

``torch.compile`` of the root Clef model, of ``Qwen3_5TextModel``, or of
``Qwen3_5ForConditionalGeneration`` is refused. Those callables would put
the joint head, or an unbroken stack of chunks, in one graph. An operator
negative control of ``torch.compile(ClefModel)`` was +773 ms and was
attributed to ``.item()`` inside the joint head. That control was not
remeasured here.

Full attention stays the checkpoint implementation. This module does not
select FA3 and does not replace the FLA chunk with FlashInfer.

``/health`` ``backbone`` is ``piecewise+inductor`` only after warmup has
executed the chunk split outside capture and inductor has recorded at
least one ``CUDAGraphNode``. Otherwise startup raises. The eager default
publishes ``backbone`` ``eager`` and ``backbone_graphs`` 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

BACKBONE_EAGER = "eager"
BACKBONE_PIECEWISE = "piecewise"
BACKBONE_HEALTH_EAGER = "eager"
BACKBONE_HEALTH_PIECEWISE = "piecewise+inductor"
FLA_CHUNK_SPLIT = "ChunkGatedDeltaRuleFunction"
DECODER_LAYER = "Qwen3_5DecoderLayer"
TEXT_MODEL = "Qwen3_5TextModel"
CAPTURE_FORWARDS = 3

CompileRegion = Callable[..., Any]
DisableFn = Callable[[Callable[..., Any]], Callable[..., Any]]
Probe = Callable[[], bool]


@dataclass
class PiecewiseInstall:
    """What ``install_piecewise_backbone`` actually changed.

    ``ready_backbone`` refuses to publish ``piecewise+inductor`` from this
    record unless warmup also observed real captures.
    """

    split_boundaries: int = 0
    joint_modules: int = 0
    compiled_owners: list[str] = field(default_factory=list)
    chunk_calls: int = 0
    chunk_inside_capture: int = 0
    chunk_while_compiling: int = 0


def resolve_backbone(requested: str) -> str:
    """Return ``eager`` or ``piecewise``. Any other token is refused.

    ``full`` is not a mode. Compiling the whole Clef model is the negative
    control, not a flag value.
    """

    if requested == BACKBONE_EAGER:
        return BACKBONE_EAGER
    if requested == BACKBONE_PIECEWISE:
        return BACKBONE_PIECEWISE
    raise ValueError(
        "backbone must be eager or piecewise. "
        "torch.compile of the full Clef model is not a backbone mode."
    )


def assert_compile_owner(owner: str) -> None:
    """Allow a compile only for one Qwen3.5 decoder layer."""

    if owner == DECODER_LAYER:
        return
    raise RuntimeError(
        f"refusing to torch.compile {owner}. "
        "Clef compiles Qwen3_5DecoderLayer only. "
        "torch.compile(ClefModel), Qwen3_5TextModel, "
        "Qwen3_5ForConditionalGeneration, and the joint head are forbidden."
    )


def ready_backbone(
    requested: str,
    graphs: int,
    install: PiecewiseInstall | None,
) -> tuple[str, int]:
    """Return the ``/health`` backbone string and graph count.

    Piecewise is published only when the install record and ``graphs`` both
    say a split ran outside capture and a CUDA graph was recorded. A zero
    graph count does not fall back to advertising eager.
    """

    mode = resolve_backbone(requested)
    if mode == BACKBONE_EAGER:
        if install is not None or graphs:
            raise RuntimeError(
                "eager Clef backbone has a piecewise install or a CUDA graph. "
                "Refusing to advertise eager."
            )
        return BACKBONE_HEALTH_EAGER, 0
    if install is None:
        raise RuntimeError(
            "piecewise backbone was requested but no install record exists. "
            "Refusing to advertise piecewise+inductor."
        )
    if any(owner != DECODER_LAYER for owner in install.compiled_owners):
        raise RuntimeError(
            "piecewise compile owner is not Qwen3_5DecoderLayer. "
            "Refusing to advertise piecewise+inductor."
        )
    if not install.compiled_owners:
        raise RuntimeError(
            "piecewise compile installed no Qwen3_5DecoderLayer. "
            "Refusing to advertise piecewise+inductor."
        )
    if install.split_boundaries < 1:
        raise RuntimeError(
            "piecewise capture has no ChunkGatedDeltaRuleFunction split. "
            "Refusing to advertise piecewise+inductor."
        )
    if install.joint_modules < 1:
        raise RuntimeError(
            "joint head was not kept outside the graph. "
            "Refusing to advertise piecewise+inductor."
        )
    if install.chunk_calls < 1:
        raise RuntimeError(
            "warmup never executed the FLA chunk split. "
            "Refusing to advertise piecewise+inductor."
        )
    if install.chunk_inside_capture or install.chunk_while_compiling:
        raise RuntimeError(
            "FLA ChunkGatedDeltaRuleFunction ran inside CUDA graph capture "
            f"({install.chunk_inside_capture}) or a compiled region "
            f"({install.chunk_while_compiling}). "
            "Refusing to advertise piecewise+inductor."
        )
    if graphs < 1:
        raise RuntimeError(
            "piecewise was requested but no CUDA graph was captured. "
            "Refusing to advertise piecewise+inductor."
        )
    return BACKBONE_HEALTH_PIECEWISE, graphs


def walk_modules(root: Any) -> list[Any]:
    """Depth-first ``_modules`` walk. ``None`` children are skipped."""

    found: list[Any] = []
    seen: set[int] = set()

    def visit(module: Any) -> None:
        if module is None or id(module) in seen:
            return
        seen.add(id(module))
        found.append(module)
        children = getattr(module, "_modules", None)
        if isinstance(children, dict):
            for child in children.values():
                visit(child)

    visit(root)
    return found


def _find_one(modules: list[Any], name: str) -> Any:
    matches = [module for module in modules if type(module).__name__ == name]
    if len(matches) != 1:
        raise RuntimeError(
            f"piecewise backbone expected one {name}, found {len(matches)}"
        )
    return matches[0]


def _joint_modules(root: Any) -> list[Any]:
    found: list[Any] = []
    seen: set[int] = set()

    def add(module: Any) -> None:
        if module is None or id(module) in seen or not hasattr(module, "forward"):
            return
        seen.add(id(module))
        found.append(module)

    for module in walk_modules(root):
        if "Joint" in type(module).__name__:
            add(module)
        head = getattr(module, "joint_head", None)
        if head is not None and head is not module:
            add(head)
    return found


def move_residual(hidden_states: Any) -> Any:
    """Copy a decoder residual out of CUDAGraph output storage.

    ``residual = hidden_states`` aliases a graph output. The next replay
    overwrites that storage and PyTorch raises ``accessing tensor output of
    CUDAGraphs that has been overwritten by a subsequent run``. The clone
    has to run outside ``torch.compile``.
    """

    return hidden_states.clone()


def _bind_decoder_forward(
    layer: Any,
    compile_region: CompileRegion,
    compiled_owners: list[str],
) -> None:
    """Compile the mixer and the MLP. Clone both residuals in the eager forward."""

    owner = type(layer).__name__
    assert_compile_owner(owner)
    # Bind this layer in a local so the compiled closures do not share the loop variable.
    bound = layer

    def token_mixer(
        hidden_states: Any,
        position_embeddings: Any,
        attention_mask: Any,
        position_ids: Any,
        past_key_values: Any,
        **kwargs: Any,
    ) -> Any:
        hidden_states = bound.input_layernorm(hidden_states)
        if bound.layer_type == "linear_attention":
            return bound.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                attention_mask=attention_mask,
                **kwargs,
            )
        if bound.layer_type == "full_attention":
            mixed, _weights = bound.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            return mixed
        raise RuntimeError(
            f"unsupported Qwen3_5DecoderLayer layer_type {bound.layer_type!r}"
        )

    def mlp_block(hidden_states: Any) -> Any:
        hidden_states = bound.post_attention_layernorm(hidden_states)
        return bound.mlp(hidden_states)

    token_mixer.clef_region = "mixer"  # type: ignore[attr-defined]
    mlp_block.clef_region = "mlp"  # type: ignore[attr-defined]
    compiled_mixer = compile_region(token_mixer, owner)
    compiled_mlp = compile_region(mlp_block, owner)
    compiled_owners.append(owner)
    compiled_owners.append(owner)

    def forward(
        hidden_states: Any,
        position_embeddings: Any = None,
        attention_mask: Any = None,
        position_ids: Any = None,
        past_key_values: Any = None,
        **kwargs: Any,
    ) -> Any:
        residual = move_residual(hidden_states)
        hidden_states = compiled_mixer(
            hidden_states,
            position_embeddings,
            attention_mask,
            position_ids,
            past_key_values,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        residual = move_residual(hidden_states)
        hidden_states = compiled_mlp(hidden_states)
        return residual + hidden_states

    forward.clef_residual = "outside-compile"  # type: ignore[attr-defined]
    layer.forward = forward


def make_chunk_split(
    original: Callable[..., Any],
    install: PiecewiseInstall,
    *,
    capturing: Probe,
    compiling: Probe,
) -> Callable[..., Any]:
    """Eager boundary around the FLA chunk. The original callable still runs."""

    def chunk_split(*args: Any, **kwargs: Any) -> Any:
        install.chunk_calls += 1
        if capturing():
            install.chunk_inside_capture += 1
        if compiling():
            install.chunk_while_compiling += 1
        return original(*args, **kwargs)

    chunk_split.clef_split = FLA_CHUNK_SPLIT  # type: ignore[attr-defined]
    return chunk_split


def install_piecewise_backbone(
    root: Any,
    *,
    compile_region: CompileRegion,
    disable_fn: DisableFn,
    capturing: Probe,
    compiling: Probe,
) -> PiecewiseInstall:
    """Split the loaded model. Does not publish a health string by itself."""

    if type(root).__name__ == DECODER_LAYER:
        raise RuntimeError("piecewise install expected the Clef root, not one decoder layer")
    modules = walk_modules(root)
    text = _find_one(modules, TEXT_MODEL)
    text_modules = walk_modules(text)
    layers = [module for module in text_modules if type(module).__name__ == DECODER_LAYER]
    if not layers:
        raise RuntimeError("Qwen3_5TextModel has no Qwen3_5DecoderLayer to compile")
    deltas = [module for module in text_modules if type(module).__name__ == "Qwen3_5GatedDeltaNet"]
    if not deltas:
        raise RuntimeError(
            "Qwen3_5TextModel has no Qwen3_5GatedDeltaNet. "
            "An unbroken backbone graph is not the piecewise bind."
        )
    joints = _joint_modules(root)
    if not joints:
        raise RuntimeError(
            "no joint head module was found. "
            "Refusing to compile without an explicit joint exclusion."
        )

    install = PiecewiseInstall(joint_modules=len(joints))
    for joint in joints:
        joint.forward = disable_fn(joint.forward)
    for delta in deltas:
        current = getattr(delta, "chunk_gated_delta_rule", None)
        if not callable(current):
            raise RuntimeError(
                "Qwen3_5GatedDeltaNet.chunk_gated_delta_rule is not callable. "
                "The FLA chunk split cannot be installed."
            )
        if getattr(current, "clef_split", None) == FLA_CHUNK_SPLIT:
            raise RuntimeError("FLA chunk split is already installed")
        disabled = disable_fn(
            make_chunk_split(current, install, capturing=capturing, compiling=compiling)
        )
        try:
            disabled.clef_split = FLA_CHUNK_SPLIT
        except (AttributeError, TypeError):
            pass
        delta.chunk_gated_delta_rule = disabled
        install.split_boundaries += 1
    for layer in layers:
        if not callable(getattr(layer, "forward", None)):
            raise RuntimeError("Qwen3_5DecoderLayer.forward is not callable")
        _bind_decoder_forward(layer, compile_region, install.compiled_owners)
    return install


def _cuda_capturing() -> bool:
    import torch

    return bool(torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())


def _compiler_compiling() -> bool:
    import torch

    return bool(torch.compiler.is_compiling())


def _enable_inductor_cudagraphs() -> None:
    import torch._inductor.config as inductor_config

    triton = getattr(inductor_config, "triton", None)
    if triton is not None:
        if hasattr(triton, "cudagraphs"):
            triton.cudagraphs = True
        if hasattr(triton, "cudagraph_trees"):
            triton.cudagraph_trees = True


def _torch_disable(fn: Callable[..., Any]) -> Callable[..., Any]:
    import torch

    return torch.compiler.disable(fn)


def _torch_compile_region(fn: Callable[..., Any], owner: str) -> Callable[..., Any]:
    assert_compile_owner(owner)
    import torch

    _enable_inductor_cudagraphs()
    return torch.compile(
        fn,
        backend="inductor",
        fullgraph=False,
        dynamic=False,
        mode="reduce-overhead",
    )


def bind_piecewise_backbone(root: Any) -> PiecewiseInstall:
    """Install the production split. Compilation runs on the first forward."""

    return install_piecewise_backbone(
        root,
        compile_region=_torch_compile_region,
        disable_fn=_torch_disable,
        capturing=_cuda_capturing,
        compiling=_compiler_compiling,
    )


def _walk_graph_nodes(nodes: Any, count: list[int]) -> None:
    for node in nodes:
        if type(node).__name__ == "CUDAGraphNode" and getattr(node, "graph", None) is not None:
            count[0] += 1
        children = getattr(node, "children", None) or {}
        for kids in children.values():
            _walk_graph_nodes(kids, count)


def count_captured_cuda_graphs(device_index: int | None = None) -> int:
    """Count inductor ``CUDAGraphNode`` recordings on one device.

    Warmup nodes are not graphs. A manager that was never created counts
    as zero, which fails the piecewise health gate.
    """

    import torch
    from torch._inductor.cudagraph_trees import get_manager

    if not torch.cuda.is_available():
        return 0
    index = torch.cuda.current_device() if device_index is None else device_index
    manager = get_manager(index, create_if_none_exists=False)
    if manager is None:
        return 0
    roots = getattr(manager, "roots", None)
    if not roots:
        return 0
    count = [0]
    for group in roots.values():
        _walk_graph_nodes(group, count)
    return count[0]
