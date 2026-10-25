"""Require the CUDA causal conv before Transformers binds its fallback.

Transformers 5.17 resolves ``causal_conv1d_fn`` and ``causal_conv1d_update``
when ``modeling_qwen3_5`` is imported. If ``causal_conv1d`` is missing, the
first forward warns and calls ``F.conv1d``. That warning is the baseline
L20X trace. Importing the package first binds its functions, which is the
condition that skips the warning.

``causal-conv1d`` 1.7.0 is the release that satisfies
``flash-linear-attention==0.5.2``'s ``conv1d`` extra (``causal-conv1d>=1.4.0``).
Its ``setup.py`` gencode list has no ``sm_89``. The recipe installer adds
``arch=compute_89,code=sm_89`` before building. CUDA startup reads the
compiled extension and refuses a binary that lacks the visible device's
architecture.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

CAUSAL_CONV1D_VERSION = "1.7.0"
_SM87_LINE = 'cc_flag.append("arch=compute_87,code=sm_87")\n'
_SM89_BLOCK = (
    '        cc_flag.append("-gencode")\n'
    '        cc_flag.append("arch=compute_89,code=sm_89")\n'
)


def version_matches(installed: str, required: str = CAUSAL_CONV1D_VERSION) -> bool:
    return installed.split("+", 1)[0] == required


def ensure_sm89_gencode(setup_text: str) -> str:
    """Insert an sm_89 cubin next to the upstream sm_87 gencode line."""

    if "compute_89" in setup_text:
        return setup_text
    if _SM87_LINE not in setup_text:
        raise RuntimeError("causal-conv1d setup.py has no sm_87 gencode line")
    return setup_text.replace(_SM87_LINE, _SM87_LINE + _SM89_BLOCK, 1)


def extension_supports_arch(blob: bytes, major: int, minor: int) -> bool:
    arch = f"{major}{minor}".encode()
    return b"sm_" + arch in blob or b"compute_" + arch in blob


def binds_package(exported: Any, package_fn: Any) -> bool:
    """Whether ``exported`` reaches ``package_fn`` through wrappers or closures."""

    seen: set[int] = set()
    stack = [exported]
    while stack:
        current = stack.pop()
        if current is package_fn:
            return True
        ident = id(current)
        if ident in seen:
            continue
        seen.add(ident)
        forward = getattr(current, "forward", None)
        if forward is not None and forward is not current:
            stack.append(forward)
        wrapped = getattr(current, "__wrapped__", None)
        if wrapped is not None and wrapped is not current:
            stack.append(wrapped)
        closure = getattr(current, "__closure__", None) or ()
        for cell in closure:
            try:
                value = cell.cell_contents
            except ValueError:
                continue
            if value is package_fn:
                return True
            if callable(value):
                stack.append(value)
    return False


def prepare_causal_conv1d(*, required: bool) -> str | None:
    """Import ``causal_conv1d`` before Qwen3.5 modeling.

    CUDA refuses to continue when the package, version, or device cubin is
    missing. CPU keeps the checkpoint's own float32 path when the package
    is absent.
    """

    try:
        import causal_conv1d
        import causal_conv1d_cuda
        from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    except Exception as exc:
        if not required:
            return None
        raise RuntimeError(
            "causal_conv1d is not installed, so Transformers would warn and use "
            "F.conv1d. Run recipe/pplx_decider/install_causal_conv1d.py in this "
            "environment before starting the CUDA worker."
        ) from exc
    version = getattr(causal_conv1d, "__version__", "")
    if not isinstance(version, str) or not version_matches(version):
        raise RuntimeError(
            f"causal_conv1d {version!r} is loaded; expected {CAUSAL_CONV1D_VERSION}"
        )
    if causal_conv1d_fn is None or causal_conv1d_update is None:
        raise RuntimeError("causal_conv1d does not export both conv functions")
    if required:
        import torch

        major, minor = torch.cuda.get_device_capability()
        blob = Path(causal_conv1d_cuda.__file__).read_bytes()
        if not extension_supports_arch(blob, major, minor):
            raise RuntimeError(
                f"causal_conv1d_cuda has no sm_{major}{minor} image for this GPU. "
                "Rebuild with recipe/pplx_decider/install_causal_conv1d.py."
            )
    return version


def assert_qwen_conv_uses_package() -> None:
    """Fail if modeling still closed over the PyTorch conv reference."""

    import transformers.models.qwen3_5.modeling_qwen3_5 as modeling
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update

    pairs = (
        ("causal_conv1d_fn", causal_conv1d_fn),
        ("causal_conv1d_update", causal_conv1d_update),
    )
    missing = [
        name
        for name, package_fn in pairs
        if not binds_package(getattr(modeling, name), package_fn)
    ]
    if missing:
        raise RuntimeError(
            "Transformers still has the PyTorch conv fallback for "
            + ", ".join(missing)
            + ". The load warning is `falling back to its reference PyTorch "
            "implementation`. Restart after installing causal_conv1d into a "
            "fresh process."
        )
