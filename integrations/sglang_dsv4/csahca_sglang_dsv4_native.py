"""Native SGLang DeepSeek-V4 CSAHCA decode entry point.

This module is intentionally thin: SGLang owns metadata construction and CUDA
graph replay address stability; this file only maps those prepared tensors to
the CSAHCA DSV4 CUDA extension.
"""

from __future__ import annotations

import os
from typing import Any

import torch


_FALSE_VALUES = {"", "0", "false", "no", "off"}


def _enabled(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() not in _FALSE_VALUES


def _strict_fail(message: str) -> None:
    if _enabled("CSAHCA_DSV4_NATIVE_STRICT"):
        raise RuntimeError(message)


def _squeeze_q(q: torch.Tensor) -> torch.Tensor | None:
    if q.ndim == 3:
        return q
    if q.ndim == 4 and q.shape[1] == 1:
        return q.squeeze(1)
    _strict_fail(f"unsupported q shape for native CSAHCA DSV4 decode: {tuple(q.shape)}")
    return None


def _squeeze_indices(name: str, x: torch.Tensor | None) -> torch.Tensor | None:
    if x is None:
        _strict_fail(f"missing {name} for native CSAHCA DSV4 decode")
        return None
    if x.ndim == 2:
        return x
    if x.ndim == 3 and x.shape[1] == 1:
        return x.squeeze(1)
    _strict_fail(f"unsupported {name} shape for native CSAHCA DSV4 decode: {tuple(x.shape)}")
    return None


def _topk_lengths(name: str, x: torch.Tensor | None) -> torch.Tensor | None:
    if x is None:
        _strict_fail(f"missing {name} for native CSAHCA DSV4 decode")
        return None
    if x.ndim == 1:
        return x
    _strict_fail(f"unsupported {name} shape for native CSAHCA DSV4 decode: {tuple(x.shape)}")
    return None


def _flatten_flashmla_cache(name: str, x: torch.Tensor | None) -> torch.Tensor | None:
    if x is None:
        _strict_fail(f"missing {name} for native CSAHCA DSV4 decode")
        return None
    if x.ndim == 2:
        return x
    if x.ndim == 4 and x.shape[2] == 1:
        return x.reshape(x.shape[0], x.shape[1] * x.shape[3])
    _strict_fail(f"unsupported {name} shape for native CSAHCA DSV4 decode: {tuple(x.shape)}")
    return None


def _forward_mode_allowed(forward_batch: Any) -> bool:
    forward_mode = getattr(forward_batch, "forward_mode", None)
    if forward_mode is None:
        return False
    if not getattr(forward_mode, "is_decode_or_idle")():
        return False

    allowed = os.getenv("CSAHCA_DSV4_NATIVE_FORWARD_MODES", "DECODE,IDLE")
    allowed_modes = {item.strip().upper() for item in allowed.split(",") if item.strip()}
    mode_name = getattr(forward_mode, "name", str(forward_mode)).upper()
    return mode_name in allowed_modes or "DECODE_OR_IDLE" in allowed_modes


def maybe_dsv4_decode_forward(
    *,
    q: torch.Tensor,
    layer_id: int,
    compress_ratio: int,
    forward_batch: Any,
    token_to_kv_pool: Any,
    swa_k_cache: torch.Tensor,
    swa_page_indices: torch.Tensor,
    swa_topk_lengths: torch.Tensor,
    extra_k_cache: torch.Tensor | None,
    extra_indices: torch.Tensor | None,
    extra_topk_lengths: torch.Tensor | None,
    attn_sink: torch.Tensor | None,
    page_size: int,
    softmax_scale: float,
) -> torch.Tensor | None:
    """Return CSAHCA output for graph-safe DSV4 decode, or ``None`` to delegate."""

    if not _enabled("CSAHCA_DSV4_NATIVE"):
        return None
    if not _forward_mode_allowed(forward_batch):
        return None
    if compress_ratio not in (0, 4, 128):
        _strict_fail(f"unsupported native CSAHCA DSV4 compress_ratio={compress_ratio}")
        return None
    if attn_sink is None:
        _strict_fail("native CSAHCA DSV4 decode requires attn_sink")
        return None

    q3 = _squeeze_q(q)
    swa_cache = _flatten_flashmla_cache("swa_k_cache", swa_k_cache)
    token_indices = _squeeze_indices("swa_page_indices", swa_page_indices)
    topk_lengths = _topk_lengths("swa_topk_lengths", swa_topk_lengths)
    if q3 is None or swa_cache is None or token_indices is None or topk_lengths is None:
        return None

    if compress_ratio == 0:
        from hybrid_attention.extension import dsv4_swa_decode_forward

        return dsv4_swa_decode_forward(
            q3,
            swa_cache,
            token_indices,
            topk_lengths,
            attn_sink,
            int(page_size),
            float(softmax_scale),
        )

    if extra_k_cache is None:
        _strict_fail(f"missing extra_k_cache for native CSAHCA DSV4 ratio={compress_ratio}")
        return None
    extra_cache = _flatten_flashmla_cache("extra_k_cache", extra_k_cache)
    extra_token_indices = _squeeze_indices("extra_indices", extra_indices)
    extra_lengths = _topk_lengths("extra_topk_lengths", extra_topk_lengths)
    if extra_cache is None or extra_token_indices is None or extra_lengths is None:
        return None

    from hybrid_attention.extension import dsv4_sparse_decode_forward

    return dsv4_sparse_decode_forward(
        q3,
        swa_cache,
        token_indices,
        topk_lengths,
        extra_cache,
        extra_token_indices,
        extra_lengths,
        attn_sink,
        int(page_size),
        int(token_to_kv_pool.get_extra_key_page_size(layer_id)),
        float(softmax_scale),
    )
