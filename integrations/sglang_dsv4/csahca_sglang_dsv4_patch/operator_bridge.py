from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch


@dataclass(frozen=True)
class BridgeDecision:
    use_csahca: bool
    reason: str


@dataclass(frozen=True)
class TensorABI:
    shape: tuple[int, ...]
    dtype: str
    device: str
    contiguous: bool


@dataclass(frozen=True)
class DSV4BridgeInputs:
    """Runtime ABI at the SGLang DeepSeek-V4 FlashMLA replacement boundary."""

    layer_id: int
    forward_mode: str
    compress_ratio: int
    q: torch.Tensor
    current_k: torch.Tensor
    current_v: torch.Tensor
    swa_k_cache: torch.Tensor
    swa_page_indices: torch.Tensor
    swa_topk_lengths: torch.Tensor
    attn_sink: Optional[torch.Tensor]
    extra_k_cache: Optional[torch.Tensor] = None
    extra_page_indices: Optional[torch.Tensor] = None
    extra_topk_lengths: Optional[torch.Tensor] = None


def tensor_abi(x: Any) -> TensorABI:
    if not isinstance(x, torch.Tensor):
        return TensorABI((), "", "", False)
    return TensorABI(
        shape=tuple(int(v) for v in x.shape),
        dtype=str(x.dtype),
        device=str(x.device),
        contiguous=bool(x.is_contiguous()),
    )


def decide_dsv4_kernel_support(
    *,
    inputs: DSV4BridgeInputs,
) -> BridgeDecision:
    """Return whether the CSAHCA DSV4 prototype can run for this live call."""

    if inputs.q.ndim not in (3, 4):
        return BridgeDecision(False, f"unsupported q rank {inputs.q.ndim}")
    if inputs.q.shape[-1] != 512:
        return BridgeDecision(False, f"expected DSV4 q head_dim=512, got {inputs.q.shape[-1]}")
    if inputs.compress_ratio not in (0, 4, 128):
        return BridgeDecision(False, f"unsupported compress_ratio={inputs.compress_ratio}")
    if inputs.swa_k_cache.dtype not in (torch.float8_e4m3fn, torch.uint8):
        return BridgeDecision(
            False,
            f"expected packed FP8/SWA cache, got {inputs.swa_k_cache.dtype}",
        )
    if inputs.swa_page_indices.ndim not in (2, 3):
        return BridgeDecision(False, f"unsupported SWA index rank {inputs.swa_page_indices.ndim}")
    if inputs.swa_topk_lengths.ndim != 1:
        return BridgeDecision(
            False,
            f"unsupported SWA topk length rank {inputs.swa_topk_lengths.ndim}",
        )
    if inputs.attn_sink is None:
        return BridgeDecision(False, "missing DSV4 attention sink")
    if inputs.compress_ratio in (4, 128):
        if inputs.extra_k_cache is None:
            return BridgeDecision(False, f"missing extra cache for compress_ratio={inputs.compress_ratio}")
        if inputs.extra_k_cache.dtype not in (torch.float8_e4m3fn, torch.uint8):
            return BridgeDecision(
                False,
                f"expected packed FP8 extra cache, got {inputs.extra_k_cache.dtype}",
            )
        if inputs.extra_page_indices is None:
            return BridgeDecision(False, f"missing extra indices for compress_ratio={inputs.compress_ratio}")
        if inputs.extra_page_indices.ndim not in (2, 3):
            return BridgeDecision(
                False,
                f"unsupported extra index rank {inputs.extra_page_indices.ndim}",
            )
        if inputs.extra_topk_lengths is None:
            return BridgeDecision(False, f"missing extra topk lengths for compress_ratio={inputs.compress_ratio}")
        if inputs.extra_topk_lengths.ndim != 1:
            return BridgeDecision(
                False,
                f"unsupported extra topk length rank {inputs.extra_topk_lengths.ndim}",
            )
    return BridgeDecision(
        True,
        f"compress_ratio={inputs.compress_ratio} DSV4 sparse prototype is available for shadow compare",
    )


def describe_inputs(inputs: DSV4BridgeInputs) -> dict[str, Any]:
    return {
        "layer_id": inputs.layer_id,
        "forward_mode": inputs.forward_mode,
        "compress_ratio": inputs.compress_ratio,
        "q": tensor_abi(inputs.q).__dict__,
        "current_k": tensor_abi(inputs.current_k).__dict__,
        "current_v": tensor_abi(inputs.current_v).__dict__,
        "swa_k_cache": tensor_abi(inputs.swa_k_cache).__dict__,
        "swa_page_indices": tensor_abi(inputs.swa_page_indices).__dict__,
        "swa_topk_lengths": tensor_abi(inputs.swa_topk_lengths).__dict__,
        "attn_sink": tensor_abi(inputs.attn_sink).__dict__,
        "extra_k_cache": tensor_abi(inputs.extra_k_cache).__dict__,
        "extra_page_indices": tensor_abi(inputs.extra_page_indices).__dict__,
        "extra_topk_lengths": tensor_abi(inputs.extra_topk_lengths).__dict__,
    }
