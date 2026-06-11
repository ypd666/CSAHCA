from __future__ import annotations

import os
from pathlib import Path
from types import ModuleType

import torch


_CUDA_MODULE: ModuleType | None = None


def load_cuda_extension() -> ModuleType:
    """Load the installed CUDA extension, or JIT build when explicitly requested."""
    global _CUDA_MODULE
    if _CUDA_MODULE is not None:
        return _CUDA_MODULE

    try:
        import hybrid_attention_cuda  # type: ignore

        _CUDA_MODULE = hybrid_attention_cuda
        return _CUDA_MODULE
    except ImportError as exc:
        if os.environ.get("HYBRID_ATTENTION_JIT", "0") != "1":
            raise RuntimeError(
                "CUDA extension is not installed. Run `python3 -m pip install -e .` "
                "on the H100 machine, or set HYBRID_ATTENTION_JIT=1 for local JIT build."
            ) from exc

    from torch.utils.cpp_extension import load

    root = Path(__file__).resolve().parents[1]
    _CUDA_MODULE = load(
        name="hybrid_attention_cuda",
        sources=[str(root / "csrc" / "bindings.cpp"), str(root / "csrc" / "csa_attention.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
        extra_cflags=["-O3"],
        verbose=True,
    )
    return _CUDA_MODULE


def csa_decode_forward(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    selected_chunks: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    module = load_cuda_extension()
    return module.csa_decode_forward(
        q.contiguous(),
        k_cache.contiguous(),
        v_cache.contiguous(),
        selected_chunks.contiguous(),
        int(chunk_size),
    )

