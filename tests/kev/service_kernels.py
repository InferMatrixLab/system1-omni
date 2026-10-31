"""Reserved-GPU regression for Kev's optional Torch 2.8 arithmetic.

Use the pinned reference requirements and gpu run. --legacy-library compares
unchanged Qwen entry points against a previously built library, bit for bit.
"""

import argparse
import ctypes as C
import inspect
import json
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5RMSNormGated,
    torch_chunk_gated_delta_rule,
)

P, I, F = C.c_void_p, C.c_int, C.c_float


def bind(lib, name, signature, result=I):
    fn = getattr(lib, name)
    fn.argtypes, fn.restype = signature, result
    return fn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--legacy-library", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists(), "preserve prior evidence"
    assert torch.__version__ == "2.8.0+cu128"
    torch.set_grad_enabled(False)
    torch.manual_seed(27)
    torch.backends.cuda.matmul.allow_tf32 = False
    lib, old = C.CDLL(str(args.library)), C.CDLL(str(args.legacy_library))
    stream = torch.cuda.current_stream().cuda_stream
    ptr = lambda t: t.data_ptr()
    rand = lambda shape: torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    records = []

    def check(name, actual, expected, tolerance=0):
        torch.cuda.synchronize()
        diff = actual.float() - expected.float()
        relative = float(diff.norm() / expected.float().norm().clamp_min(1e-30))
        records.append(
            {
                "name": name,
                "different": int((actual != expected).sum()),
                "max_abs": float(diff.abs().max()),
                "relative_l2": relative,
            }
        )
        assert relative <= tolerance, records[-1]

    # Default entry points stay bit exact; the new profile follows Torch 2.8.
    for rows in (1, 16, 80, 160):
        x, delta, w = rand((rows, 2560)), rand((rows, 2560)), rand((2560,))
        for added in (False, True):
            outputs = []
            for library, prefix in (
                (old, "reference"),
                (lib, "reference"),
                (lib, "torch28"),
            ):
                residual, out = x.clone(), torch.empty_like(x)
                name = "add_rms_norm" if added else "rms_norm"
                fn = bind(
                    library,
                    "cs1_" + prefix + "_" + name,
                    [P] * (4 if added else 3) + [I, I, F, P],
                )
                values = (
                    [ptr(residual), ptr(delta), ptr(w), ptr(out)]
                    if added
                    else [ptr(residual), ptr(w), ptr(out)]
                )
                assert fn(*values, rows, 2560, 1e-6, stream) == 0
                outputs.append(out)
            check(f"legacy-norm-{rows}-{added}", outputs[1], outputs[0])
            z = (x + delta).float() if added else x.float()
            expected = (
                z
                * torch.rsqrt(z.square().mean(-1, keepdim=True) + 1e-6)
                * (1 + w.float())
            ).bfloat16()
            check(f"torch28-norm-{rows}-{added}", outputs[2], expected)
        z, gate, weight = rand((rows, 32, 128)), rand((rows, 32, 128)), rand((128,))
        outputs = []
        for library, prefix in ((old, ""), (lib, ""), (lib, "torch28_")):
            out = torch.empty_like(z)
            fn = bind(
                library,
                "cs1_" + prefix + "gated_rms_norm",
                [P, P, I, P, P, I, I, I, F, P],
            )
            assert (
                fn(
                    ptr(z),
                    ptr(gate),
                    4096,
                    ptr(weight),
                    ptr(out),
                    rows,
                    32,
                    128,
                    1e-6,
                    stream,
                )
                == 0
            )
            outputs.append(out)
        check(f"legacy-gated-norm-{rows}", outputs[1], outputs[0])
        norm = Qwen3_5RMSNormGated(128).cuda().bfloat16()
        norm.weight.copy_(weight)
        expected = norm(z, gate)
        check(f"torch28-gated-norm-{rows}", outputs[2], expected)

    # The existing reference attention-prep ABI retains its arithmetic.
    # JEMM uses the same norm entry points at width 5120; Kev only exercises 2560.
    for rows in (1, 16, 80, 160):
        x, delta, w = rand((rows, 5120)), rand((rows, 5120)), rand((5120,))
        for added in (False, True):
            outputs = []
            for library in (old, lib):
                residual, out = x.clone(), torch.empty_like(x)
                fn = bind(
                    library,
                    "cs1_reference_" + ("add_rms_norm" if added else "rms_norm"),
                    [P] * (4 if added else 3) + [I, I, F, P],
                )
                values = (
                    [ptr(residual), ptr(delta), ptr(w), ptr(out)]
                    if added
                    else [ptr(residual), ptr(w), ptr(out)]
                )
                assert fn(*values, rows, 5120, 1e-6, stream) == 0
                outputs.append((out, residual))
            for name, before, after in zip(("out", "residual"), outputs[0], outputs[1]):
                check(f"legacy-norm-5120-{rows}-{added}-{name}", after, before)

    # The existing reference attention-prep ABI retains its arithmetic.
    for rows in (1, 32, 80):
        packed = rand((rows, 2 * 16 * 256 + 4 * 256))
        qw, kw = rand((256,)), rand((256,))
        cos, sin = rand((rows, 32)), rand((rows, 32))
        outputs = []
        for library in (old, lib):
            q, gate, k = (
                rand((rows, 16, 256)),
                rand((rows, 16, 256)),
                rand((rows, 4, 256)),
            )
            fn = bind(
                library,
                "cs1_reference_attn_prep",
                [P, P, I] + [P] * 7 + [I] * 5 + [F, P],
            )
            assert (
                fn(
                    ptr(packed),
                    ptr(packed) + 2 * 16 * 256 * 2,
                    packed.shape[1],
                    ptr(qw),
                    ptr(kw),
                    ptr(cos),
                    ptr(sin),
                    ptr(q),
                    ptr(gate),
                    ptr(k),
                    rows,
                    16,
                    4,
                    256,
                    32,
                    1e-6,
                    stream,
                )
                == 0
            )
            outputs.append((q, gate, k))
        for name, a, b in zip(("q", "gate", "k"), outputs[0], outputs[1]):
            check(f"legacy-attn-prep-{rows}-{name}", b, a)

    # FP32 GDN: chunk edges, nonzero initial state, final state and unrelated replay.
    reference = inspect.unwrap(torch_chunk_gated_delta_rule)
    for t in (1, 31, 32, 33, 63, 64, 65, 129, 233):
        q, k, v = rand((1, t, 16, 128)), rand((1, t, 16, 128)), rand((1, t, 32, 128))
        g = -torch.rand(1, t, 32, device="cuda") * 0.25
        beta = torch.rand(1, t, 32, device="cuda", dtype=torch.bfloat16)
        initial = torch.randn(1, 32, 128, 128, device="cuda") * 0.01
        outputs = []
        for library, profile in ((old, ""), (lib, ""), (lib, "_torch28")):
            create = bind(library, "cs1_gdn_fp32_create" + profile, [C.c_size_t], P)
            destroy = bind(library, "cs1_gdn_fp32_destroy", [P], None)
            size = bind(library, "cs1_gdn_fp32_workspace_floats", [I, I], C.c_size_t)
            handle = create(32 << 20)
            assert handle
            try:
                work = torch.empty(size(t, 32), device="cuda")
                out = torch.empty_like(v)
                fn = bind(library, "cs1_gdn_prefill_fp32", [P] * 8 + [I, I, I, F, P])
                call = [
                    handle,
                    ptr(q),
                    ptr(k),
                    ptr(v),
                    ptr(g),
                    ptr(beta),
                    ptr(out),
                    ptr(work),
                    t,
                    32,
                    16,
                    128**-0.5,
                    stream,
                ]
                assert fn(*call) == 0
                outputs.append(out.clone())
                if profile:
                    expected, _ = reference(
                        q.repeat_interleave(2, 2),
                        k.repeat_interleave(2, 2),
                        v,
                        g,
                        beta,
                        use_qk_l2norm_in_kernel=True,
                    )
                    check(f"torch28-gdn-{t}", out, expected, 1e-4)
                    state = torch.empty_like(initial)
                    fnx = bind(
                        library,
                        "cs1_gdn_prefill_fp32_x",
                        [P] * 8 + [I, I, I, F, P, P, P],
                    )
                    xcall = call[:-1] + [ptr(initial), ptr(state), stream]
                    assert fnx(*xcall) == 0
                    expected, expected_state = reference(
                        q.repeat_interleave(2, 2),
                        k.repeat_interleave(2, 2),
                        v,
                        g,
                        beta,
                        initial_state=initial,
                        output_final_state=True,
                        use_qk_l2norm_in_kernel=True,
                    )
                    check(f"torch28-gdn-seeded-{t}", out, expected, 1e-4)
                    check(f"torch28-gdn-final-{t}", state, expected_state, 1e-4)
                    work.fill_(float("nan"))
                    assert fn(*call) == 0
                    check(f"torch28-gdn-reset-{t}", out, outputs[-1])
            finally:
                torch.cuda.synchronize()
                destroy(handle)
        check(f"legacy-gdn-{t}", outputs[1], outputs[0])

    # Non-aligned key extents regress the 255-level Score's bias-stride failure.
    size = bind(lib, "cs1_service_attention_bytes", [I, I, I], C.c_size_t)
    attention = bind(
        lib, "cs1_service_attention", [P, P, P, I, P, P] + [I] * 7 + [P, P]
    )
    for nq, prefix, left, real in [
        (1, 16, 15, 1),
        (3, 7, 0, 2),
        (32, 32, 3, 27),
        (80, 64, 12, 69),
        (1679, 7, 0, 1679),
    ]:
        nk = prefix + nq
        h, hk = 16, 4
        q, k, v, gate = (
            rand((nq, h, 256)),
            rand((nk, hk, 256)),
            rand((nk, hk, 256)),
            rand((nq, h, 256)),
        )
        out = torch.empty_like(q)
        work = torch.empty(size(nq, nk, h), device="cuda", dtype=torch.uint8)
        assert (
            attention(
                ptr(q),
                ptr(k),
                ptr(v),
                hk * 256,
                ptr(gate),
                ptr(out),
                nq,
                nk,
                h,
                hk,
                prefix,
                left,
                real,
                ptr(work),
                stream,
            )
            == 0
        )
        qi = torch.arange(nq, device="cuda")[:, None]
        ki = torch.arange(nk, device="cuda")[None, :]
        allow = (
            (ki >= left) & (ki <= prefix + qi) & ((ki < prefix) | (ki - prefix < real))
        ) | (ki == prefix + qi)
        # Torch also requires a multiple-of-8 bias stride for efficient attention.
        mask = torch.full(
            (nq, (nk + 7) // 8 * 8),
            torch.finfo(torch.bfloat16).min,
            device="cuda",
            dtype=torch.bfloat16,
        )
        mask[:, :nk].masked_fill_(allow, 0)
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            expected = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(0, 1)[None],
                k.repeat_interleave(h // hk, 1).transpose(0, 1)[None],
                v.repeat_interleave(h // hk, 1).transpose(0, 1)[None],
                attn_mask=mask[:, :nk][None, None],
            )[0].transpose(0, 1)
        check(f"service-attention-{nq}-{nk}", out, expected * gate.sigmoid())
    args.out.write_text(
        json.dumps(
            {"torch": torch.__version__, "records": records, "passed": True}, indent=2
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
