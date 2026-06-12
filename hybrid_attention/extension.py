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
        sources=[
            str(root / "csrc" / "bindings.cpp"),
            str(root / "csrc" / "csa_attention.cu"),
            str(root / "csrc" / "dsv4_attention.cu"),
        ],
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


def csa_decode_forward_tiled(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    selected_chunks: torch.Tensor,
    chunk_size: int,
    tile_size: int = 8,
) -> torch.Tensor:
    module = load_cuda_extension()
    return module.csa_decode_forward_tiled(
        q.contiguous(),
        k_cache.contiguous(),
        v_cache.contiguous(),
        selected_chunks.contiguous(),
        int(chunk_size),
        int(tile_size),
    )


def dsv4_swa_decode_forward(
    q: torch.Tensor,
    paged_k_cache: torch.Tensor,
    token_indices: torch.Tensor,
    topk_lengths: torch.Tensor | None,
    attn_sink: torch.Tensor | None,
    page_size: int,
    softmax_scale: float,
) -> torch.Tensor:
    """Prototype DSV4 SWA decode over SGLang-style paged FP8 cache.

    This covers the first DSV4 ABI slice, equivalent to the ``compress_ratio=0``
    cache path. ``paged_k_cache`` may be a uint8 byte tensor or a float8 view of
    the same bytes; the CUDA entry point consumes the byte view.
    """

    module = load_cuda_extension()
    cache_u8 = paged_k_cache.contiguous().view(torch.uint8)
    if topk_lengths is None:
        topk_lengths = torch.empty(0, dtype=torch.int32, device=q.device)
    if attn_sink is None:
        attn_sink = torch.empty(0, dtype=torch.float32, device=q.device)
    return module.dsv4_swa_decode_forward(
        q.contiguous(),
        cache_u8,
        token_indices.contiguous().to(torch.int32),
        topk_lengths.contiguous().to(torch.int32),
        attn_sink.contiguous().to(torch.float32),
        int(page_size),
        float(softmax_scale),
    )


def dsv4_sparse_decode_forward(
    q: torch.Tensor,
    paged_k_cache: torch.Tensor,
    token_indices: torch.Tensor,
    topk_lengths: torch.Tensor | None,
    extra_paged_k_cache: torch.Tensor,
    extra_token_indices: torch.Tensor,
    extra_topk_lengths: torch.Tensor | None,
    attn_sink: torch.Tensor | None,
    page_size: int,
    extra_page_size: int,
    softmax_scale: float,
) -> torch.Tensor:
    """DSV4 decode over SWA cache plus C4/C128 compressed cache pages."""

    module = load_cuda_extension()
    cache_u8 = paged_k_cache.contiguous().view(torch.uint8)
    extra_cache_u8 = extra_paged_k_cache.contiguous().view(torch.uint8)
    if topk_lengths is None:
        topk_lengths = torch.empty(0, dtype=torch.int32, device=q.device)
    if extra_topk_lengths is None:
        extra_topk_lengths = torch.empty(0, dtype=torch.int32, device=q.device)
    if attn_sink is None:
        attn_sink = torch.empty(0, dtype=torch.float32, device=q.device)
    return module.dsv4_sparse_decode_forward(
        q.contiguous(),
        cache_u8,
        token_indices.contiguous().to(torch.int32),
        topk_lengths.contiguous().to(torch.int32),
        extra_cache_u8,
        extra_token_indices.contiguous().to(torch.int32),
        extra_topk_lengths.contiguous().to(torch.int32),
        attn_sink.contiguous().to(torch.float32),
        int(page_size),
        int(extra_page_size),
        float(softmax_scale),
    )
