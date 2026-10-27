"""Offline contract for the opt-in piecewise backbone split."""

from pathlib import Path

from models.clef.piecewise import (
    BACKBONE_HEALTH_EAGER,
    BACKBONE_HEALTH_PIECEWISE,
    DECODER_LAYER,
    FLA_CHUNK_SPLIT,
    PiecewiseInstall,
    assert_compile_owner,
    install_piecewise_backbone,
    ready_backbone,
    resolve_backbone,
)


def _mod(name, modules=None, **attrs):
    obj = type(name, (), {})()
    obj._modules = modules or {}
    for key, value in attrs.items():
        setattr(obj, key, value)
    return obj


def _forward(label):
    def forward(*_args, **_kwargs):
        return label

    return forward


def _tree():
    chunk_a = lambda *_args, **_kwargs: "a"  # noqa: E731
    chunk_b = lambda *_args, **_kwargs: "b"  # noqa: E731
    gdn_a = _mod("Qwen3_5GatedDeltaNet", chunk_gated_delta_rule=chunk_a)
    gdn_b = _mod("Qwen3_5GatedDeltaNet", chunk_gated_delta_rule=chunk_b)
    layer_a = _mod(
        DECODER_LAYER, {"linear_attn": gdn_a}, forward=_forward("layer-a")
    )
    layer_b = _mod(
        "Qwen3_5DecoderLayer", {"linear_attn": gdn_b}, forward=_forward("layer-b")
    )
    layers = _mod("ModuleList", {"0": layer_a, "1": layer_b})
    text = _mod("Qwen3_5TextModel", {"layers": layers}, forward=_forward("text"))
    joint = _mod("JointSchemaHead", forward=_forward("joint"))
    root = _mod(
        "ClefModel",
        {"language_model": text, "joint_head": joint},
        forward=_forward("clef"),
    )
    return root, text, joint, gdn_a, gdn_b, chunk_a, chunk_b, layer_a


def _compile(fn, owner):
    assert_compile_owner(owner)
    seen.append(owner)

    def wrapped(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapped


def _disable(fn):
    fn.clef_disabled = True
    return fn


seen: list[str] = []


def _install(root, *, capturing=None, compiling=None):
    seen.clear()
    return install_piecewise_backbone(
        root,
        compile_region=_compile,
        disable_fn=_disable,
        capturing=capturing or (lambda: False),
        compiling=compiling or (lambda: False),
    )


def test_default_mode_is_eager_and_full_compile_is_not_a_mode():
    assert resolve_backbone("eager") == "eager"
    assert resolve_backbone("piecewise") == "piecewise"
    for rejected in ("full", "compile", "inductor", "fa3", "flashinfer"):
        try:
            resolve_backbone(rejected)
        except ValueError as exc:
            assert "not a backbone mode" in str(exc)
        else:
            raise AssertionError(rejected)
    for owner in (
        "ClefModel",
        "Qwen3_5TextModel",
        "Qwen3_5ForConditionalGeneration",
        "JointSchemaHead",
    ):
        try:
            assert_compile_owner(owner)
        except RuntimeError as exc:
            assert "forbidden" in str(exc)
        else:
            raise AssertionError(owner)


def test_install_compiles_decoder_layers_and_splits_the_chunk_and_joint():
    root, text, joint, gdn_a, gdn_b, chunk_a, chunk_b, layer_a = _tree()
    root_forward = root.forward
    text_forward = text.forward
    install = _install(root)
    assert seen == [DECODER_LAYER, DECODER_LAYER, DECODER_LAYER, DECODER_LAYER]
    assert install.compiled_owners == [DECODER_LAYER, DECODER_LAYER, DECODER_LAYER, DECODER_LAYER]
    assert install.split_boundaries == 2
    assert install.joint_modules == 1
    assert gdn_a.chunk_gated_delta_rule is not chunk_a
    assert gdn_b.chunk_gated_delta_rule.clef_split == FLA_CHUNK_SPLIT
    assert gdn_a.chunk_gated_delta_rule(1) == "a"
    assert install.chunk_calls == 1
    assert joint.forward.clef_disabled is True
    assert root.forward is root_forward
    assert text.forward is text_forward
    assert layer_a.forward.clef_residual == "outside-compile"
    assert layer_a.forward is not root_forward
    engine_source = (
        Path(__file__).resolve().parents[2] / "src" / "models" / "clef" / "engine.py"
    ).read_text(encoding="utf-8")
    assert "torch.compile" not in engine_source


def test_ready_string_requires_a_real_capture_outside_the_chunk():
    root, *_rest = _tree()
    install = _install(root)
    install.chunk_calls = 4
    assert ready_backbone("eager", 0, None) == (BACKBONE_HEALTH_EAGER, 0)
    try:
        ready_backbone("eager", 1, None)
    except RuntimeError as exc:
        assert "advertise eager" in str(exc)
    else:
        raise AssertionError("eager with a graph")
    try:
        ready_backbone("piecewise", 0, install)
    except RuntimeError as exc:
        assert "advertise piecewise+inductor" in str(exc)
    else:
        raise AssertionError("zero graphs")
    assert ready_backbone("piecewise", 2, install) == (BACKBONE_HEALTH_PIECEWISE, 2)

    captured_root, _text, _joint, gdn, *_ignored = _tree()
    captured = _install(captured_root, capturing=lambda: True)
    assert gdn.chunk_gated_delta_rule() == "a"
    assert captured.chunk_inside_capture == 1
    try:
        ready_backbone("piecewise", 3, captured)
    except RuntimeError as exc:
        assert "inside CUDA graph capture" in str(exc)
    else:
        raise AssertionError("chunk inside capture")


def test_install_refuses_a_missing_joint_or_a_missing_chunk():
    root, text, joint, gdn_a, *_rest = _tree()
    del root._modules["joint_head"]
    try:
        _install(root)
    except RuntimeError as exc:
        assert "joint head" in str(exc)
    else:
        raise AssertionError("missing joint")
    root._modules["joint_head"] = joint
    del gdn_a.chunk_gated_delta_rule
    try:
        _install(root)
    except RuntimeError as exc:
        assert "not callable" in str(exc)
    else:
        raise AssertionError("missing chunk")
    text._modules = {}
    try:
        _install(root)
    except RuntimeError as exc:
        assert "no Qwen3_5DecoderLayer" in str(exc)
    else:
        raise AssertionError("missing layers")


def test_residual_clone_runs_outside_the_compiled_regions():
    inside = {"on": False}
    compiled: list[str] = []

    class Hidden:
        def __init__(self, name: str, cloned: bool = False):
            self.name = name
            self.cloned = cloned

        def clone(self):
            assert not inside["on"]
            return Hidden(self.name + "*", cloned=True)

        def __add__(self, other):
            assert not inside["on"]
            assert self.cloned
            return Hidden(f"sum({self.name},{other.name})")

    def record(label):
        def fn(hidden, *_args, **_kwargs):
            assert inside["on"]
            assert not getattr(hidden, "cloned", False)
            return Hidden(label)

        return fn

    class Delta:
        def __init__(self):
            self._modules = {}
            self.chunk_gated_delta_rule = lambda *_a, **_k: "chunk"

        def __call__(self, hidden_states=None, **_kwargs):
            assert inside["on"]
            assert not hidden_states.cloned
            return Hidden("attn")

    Delta.__name__ = "Qwen3_5GatedDeltaNet"

    class FullAttn:
        def __call__(self, hidden_states=None, **_kwargs):
            assert inside["on"]
            assert not hidden_states.cloned
            return Hidden("full"), Hidden("weights")

    delta = Delta()
    linear = _mod(
        DECODER_LAYER,
        {"linear_attn": delta},
        forward=_forward("lin"),
        layer_type="linear_attention",
    )
    linear.linear_attn = delta
    linear.input_layernorm = record("ln")
    linear.post_attention_layernorm = record("post")
    linear.mlp = record("mlp")
    full = _mod(
        DECODER_LAYER,
        {},
        forward=_forward("full"),
        layer_type="full_attention",
    )
    full.self_attn = FullAttn()
    full.input_layernorm = record("fln")
    full.post_attention_layernorm = record("fpost")
    full.mlp = record("fmlp")
    text = _mod(
        "Qwen3_5TextModel",
        {"layers": _mod("ModuleList", {"0": linear, "1": full})},
        forward=_forward("text"),
    )
    root = _mod(
        "ClefModel",
        {
            "language_model": text,
            "joint_head": _mod("JointSchemaHead", forward=_forward("joint")),
        },
        forward=_forward("clef"),
    )

    def compile_region(fn, owner):
        assert_compile_owner(owner)
        compiled.append(fn.clef_region)

        def wrapped(*args, **kwargs):
            inside["on"] = True
            try:
                return fn(*args, **kwargs)
            finally:
                inside["on"] = False

        return wrapped

    install_piecewise_backbone(
        root,
        compile_region=compile_region,
        disable_fn=_disable,
        capturing=lambda: False,
        compiling=lambda: False,
    )
    assert compiled == ["mixer", "mlp", "mixer", "mlp"]
    assert linear.forward.clef_residual == "outside-compile"
    assert full.forward(Hidden("in"), position_embeddings=("cos", "sin")).name.startswith("sum(")
    assert linear.forward(Hidden("in")).name.startswith("sum(")


def test_piecewise_install_record_alone_is_not_a_health_string():
    install = PiecewiseInstall(
        split_boundaries=1,
        joint_modules=1,
        compiled_owners=[DECODER_LAYER],
        chunk_calls=1,
    )
    try:
        ready_backbone("piecewise", 0, install)
    except RuntimeError as exc:
        assert "no CUDA graph was captured" in str(exc)
    else:
        raise AssertionError("install without capture")
