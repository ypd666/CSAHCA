from __future__ import annotations

import functools
import logging
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import torch

from .operator_bridge import DSV4BridgeInputs, decide_dsv4_kernel_support

LOGGER = logging.getLogger("csahca.sglang.dsv4")

_PATCHED = False
_CALL_COUNTS: Counter[tuple[str, int, str, tuple[int, ...]]] = Counter()
_ABI_COUNTS: Counter[tuple[int, str, tuple[int, ...]]] = Counter()
_COMPARE_COUNTS: Counter[tuple[int, str, tuple[int, ...]]] = Counter()
_DELEGATE_COUNTS: Counter[tuple[int, int, str, tuple[int, ...], str]] = Counter()
_CAPTURE_COUNTS: Counter[tuple[int, int, str, tuple[int, ...]]] = Counter()
_COMPARE_TOTAL = 0
_CAPTURE_TOTAL = 0
_LAST_SUMMARY_TS = 0.0


def _enabled(name: str, default: str = "0") -> bool:
    value = os.getenv(name, default)
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _cuda_stream_capturing(x: Any) -> bool:
    if not isinstance(x, torch.Tensor) or not x.is_cuda:
        return False
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def _mode() -> str:
    return os.getenv("CSAHCA_DSV4_MODE", "trace").strip().lower()


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _shape(x: Any) -> tuple[int, ...]:
    if isinstance(x, torch.Tensor):
        return tuple(int(v) for v in x.shape)
    return ()


def _dtype(x: Any) -> str:
    if isinstance(x, torch.Tensor):
        return str(x.dtype)
    return ""


def _forward_mode_name(forward_batch: Any) -> str:
    mode = getattr(forward_batch, "forward_mode", None)
    if mode is None:
        return "unknown"
    return getattr(mode, "name", str(mode))


def _forward_mode_allowed(forward_batch: Any, env_name: str, default: str) -> bool:
    mode_filter = os.getenv(env_name, default).strip()
    if not mode_filter or mode_filter.lower() in {"*", "all"}:
        return True
    allowed_modes = {item.strip().upper() for item in mode_filter.split(",") if item.strip()}
    return _forward_mode_name(forward_batch).upper() in allowed_modes


def _tensor_all_finite(x: torch.Tensor) -> bool:
    return bool(torch.isfinite(x).all().item())


def _tensor_absmax(x: torch.Tensor) -> float:
    if x.numel() == 0:
        return 0.0
    y = torch.nan_to_num(
        x.detach().float(),
        nan=0.0,
        posinf=float("inf"),
        neginf=float("-inf"),
    )
    return float(y.abs().max().item())


def _q_guard(
    q: torch.Tensor,
    *,
    require_finite_env: str,
    require_finite_default: str,
    max_abs_env: str,
    max_abs_default: float,
) -> tuple[bool, str]:
    if _enabled(require_finite_env, require_finite_default) and not _tensor_all_finite(q):
        return False, "non-finite q"

    max_abs_limit = _float_env(max_abs_env, max_abs_default)
    if max_abs_limit > 0.0:
        q_absmax = _tensor_absmax(q)
        if q_absmax > max_abs_limit:
            return False, f"q absmax {q_absmax:.6g} exceeds {max_abs_limit:.6g}"

    return True, ""


def _int_filter_allows(value: int, filter_text: str, default: str) -> bool:
    text = (filter_text or default).strip()
    if not text or text.lower() in {"*", "all"}:
        return True
    try:
        return int(value) in {int(item) for item in text.split(",") if item.strip()}
    except ValueError:
        try:
            return int(value) == int(default)
        except ValueError:
            return False


def _mode_filter_allows(forward_batch: Any, filter_text: str, default: str) -> bool:
    text = (filter_text or default).strip()
    if not text or text.lower() in {"*", "all"}:
        return True
    allowed = {item.strip().upper() for item in text.split(",") if item.strip()}
    return _forward_mode_name(forward_batch).upper() in allowed


def _dist_rank() -> str:
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return str(torch.distributed.get_rank())
    except Exception:
        pass
    return os.getenv("RANK", "na")


def _capture_tensor(x: torch.Tensor | None) -> torch.Tensor | None:
    if x is None:
        return None
    y = x.detach().contiguous()
    if str(y.dtype).startswith("torch.float8"):
        y = y.view(torch.uint8)
    return y.cpu()


def _capture_budget_available(
    *,
    layer_id: int,
    compress_ratio: int,
    forward_batch: Any,
    q: torch.Tensor,
) -> bool:
    global _CAPTURE_TOTAL

    if not os.getenv("CSAHCA_DSV4_CAPTURE_DIR", "").strip():
        return False
    max_total = _int_env("CSAHCA_DSV4_CAPTURE_MAX_CALLS", 0)
    if max_total <= 0 or _CAPTURE_TOTAL >= max_total:
        return False
    if not _int_filter_allows(layer_id, os.getenv("CSAHCA_DSV4_CAPTURE_LAYER_IDS", "0"), "0"):
        return False
    if not _int_filter_allows(compress_ratio, os.getenv("CSAHCA_DSV4_CAPTURE_RATIOS", "all"), "all"):
        return False
    mode_default = os.getenv("CSAHCA_DSV4_COMPARE_FORWARD_MODES", "DECODE")
    if not _mode_filter_allows(forward_batch, os.getenv("CSAHCA_DSV4_CAPTURE_FORWARD_MODES", mode_default), mode_default):
        return False
    max_q = _int_env("CSAHCA_DSV4_CAPTURE_MAX_Q", _int_env("CSAHCA_DSV4_COMPARE_MAX_Q", 64))
    if max_q > 0 and q.shape[0] > max_q:
        return False

    key = (int(layer_id), int(compress_ratio), _forward_mode_name(forward_batch), _shape(q))
    _CAPTURE_COUNTS[key] += 1
    if _CAPTURE_COUNTS[key] > _int_env("CSAHCA_DSV4_CAPTURE_FIRST_N", 1):
        return False
    _CAPTURE_TOTAL += 1
    return True


def _record_call(
    *,
    q: torch.Tensor,
    layer_id: int,
    compress_ratio: int,
    forward_batch: Any,
) -> None:
    global _LAST_SUMMARY_TS

    mode = _mode()
    key = (mode, int(compress_ratio), _forward_mode_name(forward_batch), _shape(q))
    _CALL_COUNTS[key] += 1

    first_n = int(os.getenv("CSAHCA_DSV4_TRACE_FIRST_N", "12"))
    count = _CALL_COUNTS[key]
    if count <= first_n:
        LOGGER.info(
            "[CSAHCA][DSV4] hook call layer=%s mode=%s compress_ratio=%s "
            "forward_mode=%s q_shape=%s count=%s",
            layer_id,
            mode,
            compress_ratio,
            key[2],
            key[3],
            count,
        )

    interval_s = float(os.getenv("CSAHCA_DSV4_SUMMARY_INTERVAL_S", "30"))
    now = time.monotonic()
    if interval_s > 0 and now - _LAST_SUMMARY_TS >= interval_s:
        _LAST_SUMMARY_TS = now
        top = _CALL_COUNTS.most_common(8)
        LOGGER.info("[CSAHCA][DSV4] hook summary top=%s", top)


def _record_abi(
    *,
    self_obj: Any,
    q: torch.Tensor,
    k: torch.Tensor,
    layer: Any,
    forward_batch: Any,
    compress_ratio: int,
    attn_sink: torch.Tensor | None,
) -> None:
    if not _enabled("CSAHCA_DSV4_TRACE_ABI"):
        return
    if _cuda_stream_capturing(q):
        return

    key = (int(compress_ratio), _forward_mode_name(forward_batch), _shape(q))
    _ABI_COUNTS[key] += 1
    if _ABI_COUNTS[key] > int(os.getenv("CSAHCA_DSV4_TRACE_ABI_FIRST_N", "4")):
        return

    try:
        core = self_obj.forward_metadata.core_attn_metadata
        token_to_kv_pool = self_obj.token_to_kv_pool
        layer_id = layer.layer_id
        swa_cache = token_to_kv_pool.get_swa_key_buffer_radix(layer_id)
        extra_cache = (
            token_to_kv_pool.get_extra_key_buffer(layer_id)
            if compress_ratio in (4, 128)
            else None
        )
        abi = {
            "layer": int(layer_id),
            "compress_ratio": int(compress_ratio),
            "forward_mode": key[1],
            "q": (_shape(q), _dtype(q)),
            "current_k": (_shape(k), _dtype(k)),
            "attn_sink": (_shape(attn_sink), _dtype(attn_sink)),
            "swa_cache_raw": (_shape(swa_cache), _dtype(swa_cache)),
            "extra_cache_raw": (_shape(extra_cache), _dtype(extra_cache)),
            "swa_page_indices": (_shape(core.swa_page_indices), _dtype(core.swa_page_indices)),
            "swa_topk_lengths": (_shape(core.swa_topk_lengths), _dtype(core.swa_topk_lengths)),
            "c4_sparse_page_indices": (
                _shape(getattr(core, "c4_sparse_page_indices", None)),
                _dtype(getattr(core, "c4_sparse_page_indices", None)),
            ),
            "c128_page_indices": (
                _shape(getattr(core, "c128_page_indices", None)),
                _dtype(getattr(core, "c128_page_indices", None)),
            ),
        }
        LOGGER.warning("[CSAHCA][DSV4] ABI %s", abi)
    except Exception as exc:
        LOGGER.warning("[CSAHCA][DSV4] failed to record ABI: %r", exc)


class _NvtxRange:
    def __init__(self, name: str, tensor: torch.Tensor):
        self.name = name
        self.enabled = _enabled("CSAHCA_DSV4_NVTX") and tensor.is_cuda

    def __enter__(self):
        if self.enabled:
            torch.cuda.nvtx.range_push(self.name)

    def __exit__(self, exc_type, exc, tb):
        if self.enabled:
            torch.cuda.nvtx.range_pop()
        return False


def _match_num_queries(x: torch.Tensor | None, num_queries: int, value: int) -> torch.Tensor | None:
    if x is None or x.shape[0] == num_queries:
        return x
    if x.shape[0] > num_queries:
        return x[:num_queries]
    pad_shape = (num_queries - x.shape[0], *x.shape[1:])
    tail = torch.full(pad_shape, value, dtype=x.dtype, device=x.device)
    return torch.cat((x, tail), dim=0)


def _prepare_swa_indices(
    *,
    core: Any,
    q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_indices = _match_num_queries(core.swa_page_indices, q.shape[0], value=0)
    topk_lengths = _match_num_queries(core.swa_topk_lengths, q.shape[0], value=1)
    if token_indices is None or topk_lengths is None:
        raise RuntimeError("missing DSV4 SWA indices/topk lengths")
    if token_indices.ndim == 3:
        if token_indices.shape[1] != 1:
            raise RuntimeError(f"unsupported DSV4 SWA index shape {tuple(token_indices.shape)}")
        token_indices = token_indices.squeeze(1)
    if token_indices.ndim != 2:
        raise RuntimeError(f"unsupported DSV4 SWA index rank {token_indices.ndim}")
    if topk_lengths.ndim != 1:
        topk_lengths = topk_lengths.reshape(-1)
    return token_indices, topk_lengths


def _prepare_extra_indices(
    *,
    core: Any,
    q: torch.Tensor,
    compress_ratio: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if compress_ratio == 4:
        token_indices = getattr(core, "c4_sparse_page_indices", None)
        topk_lengths = getattr(core, "c4_sparse_topk_lengths", None)
    elif compress_ratio == 128:
        token_indices = getattr(core, "c128_page_indices", None)
        topk_lengths = getattr(core, "c128_topk_lengths_clamp1", None)
    else:
        raise RuntimeError(f"extra indices requested for compress_ratio={compress_ratio}")

    token_indices = _match_num_queries(token_indices, q.shape[0], value=-1)
    topk_lengths = _match_num_queries(topk_lengths, q.shape[0], value=1)
    if token_indices is None or topk_lengths is None:
        raise RuntimeError(f"missing DSV4 extra indices/topk lengths for compress_ratio={compress_ratio}")
    if token_indices.ndim == 3:
        if token_indices.shape[1] != 1:
            raise RuntimeError(f"unsupported DSV4 extra index shape {tuple(token_indices.shape)}")
        token_indices = token_indices.squeeze(1)
    if token_indices.ndim != 2:
        raise RuntimeError(f"unsupported DSV4 extra index rank {token_indices.ndim}")
    if topk_lengths.ndim != 1:
        topk_lengths = topk_lengths.reshape(-1)
    return token_indices, topk_lengths


def _swa_page_size(token_to_kv_pool: Any, core: Any) -> int:
    for owner, attr in (
        (token_to_kv_pool, "swa_window_size"),
        (getattr(token_to_kv_pool, "swa_kv_pool", None), "page_size"),
        (core, "page_size"),
    ):
        value = getattr(owner, attr, None) if owner is not None else None
        if value:
            return int(value)
    raise RuntimeError("could not infer DSV4 SWA page size")


def _extra_page_size(token_to_kv_pool: Any, layer_id: int, compress_ratio: int) -> int:
    value = token_to_kv_pool.get_extra_key_page_size(layer_id)
    if not value:
        raise RuntimeError(f"could not infer DSV4 extra page size for compress_ratio={compress_ratio}")
    return int(value)


def _compare_budget_available(
    *,
    layer_id: int,
    forward_batch: Any,
    q: torch.Tensor,
) -> bool:
    global _COMPARE_TOTAL

    layer_filter = os.getenv("CSAHCA_DSV4_COMPARE_LAYER_IDS", "0").strip()
    if layer_filter and layer_filter.lower() not in {"*", "all"}:
        try:
            allowed_layers = {int(item) for item in layer_filter.split(",") if item.strip()}
        except ValueError:
            allowed_layers = {0}
        if int(layer_id) not in allowed_layers:
            return False

    mode_filter = os.getenv("CSAHCA_DSV4_COMPARE_FORWARD_MODES", "").strip()
    if mode_filter:
        allowed_modes = {item.strip().upper() for item in mode_filter.split(",") if item.strip()}
        if _forward_mode_name(forward_batch).upper() not in allowed_modes:
            return False

    max_q = _int_env("CSAHCA_DSV4_COMPARE_MAX_Q", 64)
    if max_q > 0 and q.shape[0] > max_q:
        return False

    max_total = _int_env("CSAHCA_DSV4_COMPARE_MAX_CALLS", 8)
    if max_total <= 0 or _COMPARE_TOTAL >= max_total:
        return False

    key = (int(layer_id), _forward_mode_name(forward_batch), _shape(q))
    _COMPARE_COUNTS[key] += 1
    if _COMPARE_COUNTS[key] > _int_env("CSAHCA_DSV4_COMPARE_FIRST_N", 2):
        return False

    _COMPARE_TOTAL += 1
    return True


def _log_delegate(
    *,
    layer_id: int,
    compress_ratio: int,
    forward_batch: Any,
    q: torch.Tensor,
    reason: str,
) -> None:
    key = (
        int(layer_id),
        int(compress_ratio),
        _forward_mode_name(forward_batch),
        _shape(q),
        reason,
    )
    _DELEGATE_COUNTS[key] += 1
    if _DELEGATE_COUNTS[key] > _int_env("CSAHCA_DSV4_DELEGATE_LOG_FIRST_N", 1):
        return
    LOGGER.warning(
        "[CSAHCA][DSV4] delegating to original FlashMLA path for layer=%s "
        "compress_ratio=%s forward_mode=%s q_shape=%s: %s",
        layer_id,
        compress_ratio,
        key[2],
        key[3],
        reason,
    )


def _run_dsv4_swa_prototype(
    *,
    self_obj: Any,
    q: torch.Tensor,
    token_to_kv_pool: Any,
    core: Any,
    layer_id: int,
    compress_ratio: int,
    attn_sink: torch.Tensor | None,
) -> torch.Tensor:
    from hybrid_attention.extension import dsv4_sparse_decode_forward, dsv4_swa_decode_forward

    q3 = q.squeeze(1) if q.ndim == 4 and q.shape[1] == 1 else q
    if q3.ndim != 3:
        raise RuntimeError(f"prototype expects q rank 3 or [T,1,H,D], got {tuple(q.shape)}")

    swa_k_cache = token_to_kv_pool.get_swa_key_buffer_radix(layer_id)
    token_indices, topk_lengths = _prepare_swa_indices(core=core, q=q3)
    page_size = _swa_page_size(token_to_kv_pool, core)
    if compress_ratio in (4, 128):
        extra_k_cache = token_to_kv_pool.get_extra_key_buffer(layer_id)
        if extra_k_cache is None:
            raise RuntimeError(f"missing DSV4 extra cache for compress_ratio={compress_ratio}")
        extra_indices, extra_topk_lengths = _prepare_extra_indices(
            core=core,
            q=q3,
            compress_ratio=compress_ratio,
        )
        return dsv4_sparse_decode_forward(
            q3,
            swa_k_cache,
            token_indices,
            topk_lengths,
            extra_k_cache,
            extra_indices,
            extra_topk_lengths,
            attn_sink,
            page_size,
            _extra_page_size(token_to_kv_pool, layer_id, compress_ratio),
            float(self_obj.softmax_scale),
        )

    return dsv4_swa_decode_forward(
        q3,
        swa_k_cache,
        token_indices,
        topk_lengths,
        attn_sink,
        page_size,
        float(self_obj.softmax_scale),
    )


def _log_compare_input_stats(
    *,
    layer_id: int,
    compress_ratio: int,
    forward_batch: Any,
    q: torch.Tensor,
    token_to_kv_pool: Any,
    core: Any,
) -> None:
    if not _enabled("CSAHCA_DSV4_COMPARE_INPUT_STATS", "1"):
        return
    try:
        q3 = q.squeeze(1) if q.ndim == 4 and q.shape[1] == 1 else q
        swa_k_cache = token_to_kv_pool.get_swa_key_buffer_radix(layer_id)
        token_indices, topk_lengths = _prepare_swa_indices(core=core, q=q3)
        page_size = _swa_page_size(token_to_kv_pool, core)
        cache_slots = int(swa_k_cache.shape[0]) * int(page_size)
        idx_i64 = token_indices.to(torch.int64)
        valid = idx_i64 >= 0
        out_of_range = valid & (idx_i64 >= cache_slots)
        q_nonfinite = ~torch.isfinite(q3)
        q_abs = torch.nan_to_num(
            q3.detach().float(),
            nan=0.0,
            posinf=float("inf"),
            neginf=float("-inf"),
        ).abs()
        q_absmax = float(q_abs.max().item()) if q_abs.numel() else 0.0
        q_bad_heads = 0
        q_large_heads = 0
        if q3.ndim == 3:
            q_bad_heads = int(q_nonfinite.any(dim=-1).sum().item())
            q_large_limit = _float_env("CSAHCA_DSV4_Q_SUSPICIOUS_ABS", 1.0e6)
            if q_large_limit > 0.0:
                q_large_heads = int((q_abs.amax(dim=-1) > q_large_limit).sum().item())
        sample_indices: list[int] = []
        sample_valid_positions: list[int] = []
        if token_indices.shape[0] > 0 and token_indices.shape[1] <= 256:
            row0 = idx_i64[0].detach().cpu()
            sample_indices = [int(v) for v in row0[: min(24, row0.numel())].tolist()]
            sample_valid_positions = [
                int(pos) for pos in torch.nonzero(row0 >= 0, as_tuple=False).flatten()[:24].tolist()
            ]
        extra_summary = None
        if compress_ratio in (4, 128):
            extra_k_cache = token_to_kv_pool.get_extra_key_buffer(layer_id)
            extra_indices, extra_topk_lengths = _prepare_extra_indices(
                core=core,
                q=q3,
                compress_ratio=compress_ratio,
            )
            extra_page_size = _extra_page_size(token_to_kv_pool, layer_id, compress_ratio)
            extra_cache_slots = int(extra_k_cache.shape[0]) * int(extra_page_size)
            extra_i64 = extra_indices.to(torch.int64)
            extra_valid = extra_i64 >= 0
            extra_oor = extra_valid & (extra_i64 >= extra_cache_slots)
            extra_summary = {
                "cache_shape": _shape(extra_k_cache),
                "page_size": extra_page_size,
                "cache_slots": extra_cache_slots,
                "idx_min": int(extra_i64.min().item()) if extra_i64.numel() else None,
                "idx_max": int(extra_i64.max().item()) if extra_i64.numel() else None,
                "idx_neg": int((extra_i64 < 0).sum().item()),
                "idx_oor": int(extra_oor.sum().item()),
                "topk_min": int(extra_topk_lengths.min().item()) if extra_topk_lengths.numel() else None,
                "topk_max": int(extra_topk_lengths.max().item()) if extra_topk_lengths.numel() else None,
            }
        LOGGER.warning(
            "[CSAHCA][DSV4] compare inputs layer=%s compress_ratio=%s "
            "forward_mode=%s q_shape=%s cache_shape=%s page_size=%s "
            "cache_slots=%s idx_min=%s idx_max=%s idx_neg=%s idx_oor=%s "
            "topk_min=%s topk_max=%s q_nan=%s q_inf=%s q_bad_heads=%s "
            "q_absmax=%s q_large_heads=%s "
            "row0_first=%s row0_valid_pos=%s extra=%s",
            layer_id,
            compress_ratio,
            _forward_mode_name(forward_batch),
            _shape(q),
            _shape(swa_k_cache),
            page_size,
            cache_slots,
            int(idx_i64.min().item()) if idx_i64.numel() else None,
            int(idx_i64.max().item()) if idx_i64.numel() else None,
            int((idx_i64 < 0).sum().item()),
            int(out_of_range.sum().item()),
            int(topk_lengths.min().item()) if topk_lengths.numel() else None,
            int(topk_lengths.max().item()) if topk_lengths.numel() else None,
            int(torch.isnan(q3).sum().item()),
            int(torch.isinf(q3).sum().item()),
            q_bad_heads,
            q_absmax,
            q_large_heads,
            sample_indices,
            sample_valid_positions,
            extra_summary,
        )
    except Exception as exc:
        LOGGER.warning("[CSAHCA][DSV4] failed to log compare input stats: %r", exc)


def _log_live_compare(
    *,
    layer_id: int,
    compress_ratio: int,
    forward_batch: Any,
    reference: torch.Tensor,
    candidate: torch.Tensor,
    q: torch.Tensor,
) -> None:
    prefix = (
        "[CSAHCA][DSV4] live compare "
        f"layer={layer_id} compress_ratio={compress_ratio} "
        f"forward_mode={_forward_mode_name(forward_batch)} q_shape={_shape(q)}"
    )
    if tuple(reference.shape) != tuple(candidate.shape):
        LOGGER.warning(
            "%s shape_mismatch flashmla_shape=%s csahca_shape=%s "
            "flashmla_dtype=%s csahca_dtype=%s",
            prefix,
            _shape(reference),
            _shape(candidate),
            _dtype(reference),
            _dtype(candidate),
        )
        return

    ref = reference.float()
    got = candidate.float()
    diff = (got - ref).abs()
    ref_finite = torch.isfinite(ref)
    got_finite = torch.isfinite(got)
    both_finite = ref_finite & got_finite
    both_nan = torch.isnan(ref) & torch.isnan(got)
    nonfinite_mismatch = ref_finite != got_finite
    ref_nan = int(torch.isnan(ref).sum().item())
    got_nan = int(torch.isnan(got).sum().item())
    ref_inf = int(torch.isinf(ref).sum().item())
    got_inf = int(torch.isinf(got).sum().item())
    bad_finite = 0
    if diff.numel() and bool(both_finite.any().item()):
        finite_diff = diff[both_finite]
        finite_ref = ref[both_finite]
        max_abs = float(finite_diff.max().item())
        max_rel = float((finite_diff / finite_ref.abs().clamp_min(1.0e-6)).max().item())
        rms = float(torch.sqrt(torch.mean(finite_diff * finite_diff)).item())
        atol = float(os.getenv("CSAHCA_DSV4_COMPARE_ATOL", "0.05"))
        rtol = float(os.getenv("CSAHCA_DSV4_COMPARE_RTOL", "0.05"))
        bad_finite = int((finite_diff > (atol + rtol * finite_ref.abs())).sum().item())
    else:
        max_abs = float("nan")
        max_rel = float("nan")
        rms = float("nan")
        atol = float(os.getenv("CSAHCA_DSV4_COMPARE_ATOL", "0.05"))
        rtol = float(os.getenv("CSAHCA_DSV4_COMPARE_RTOL", "0.05"))
    close = bool(torch.allclose(candidate, reference, atol=atol, rtol=rtol))
    LOGGER.warning(
        "%s flashmla_shape=%s csahca_shape=%s max_abs=%.6g max_rel=%.6g "
        "rms=%.6g allclose=%s ref_nan=%s csahca_nan=%s ref_inf=%s csahca_inf=%s "
        "finite_pairs=%s/%s both_nan=%s nonfinite_mismatch=%s bad_finite=%s atol=%s rtol=%s",
        prefix,
        _shape(reference),
        _shape(candidate),
        max_abs,
        max_rel,
        rms,
        close,
        ref_nan,
        got_nan,
        ref_inf,
        got_inf,
        int(both_finite.sum().item()),
        int(diff.numel()),
        int(both_nan.sum().item()),
        int(nonfinite_mismatch.sum().item()),
        bad_finite,
        atol,
        rtol,
    )


def _capture_live_compare(
    *,
    self_obj: Any,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    token_to_kv_pool: Any,
    core: Any,
    layer_id: int,
    compress_ratio: int,
    forward_batch: Any,
    attn_sink: torch.Tensor | None,
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> None:
    capture_dir = Path(os.environ["CSAHCA_DSV4_CAPTURE_DIR"]).expanduser()
    capture_dir.mkdir(parents=True, exist_ok=True)

    q3 = q.squeeze(1) if q.ndim == 4 and q.shape[1] == 1 else q
    token_indices, topk_lengths = _prepare_swa_indices(core=core, q=q3)
    page_size = _swa_page_size(token_to_kv_pool, core)
    tensors: dict[str, torch.Tensor | None] = {
        "q": _capture_tensor(q3),
        "reference": _capture_tensor(reference),
        "candidate": _capture_tensor(candidate),
        "swa_token_indices": _capture_tensor(token_indices),
        "swa_topk_lengths": _capture_tensor(topk_lengths),
        "attn_sink": _capture_tensor(attn_sink),
    }
    if _enabled("CSAHCA_DSV4_CAPTURE_CURRENT_KV"):
        tensors["current_k"] = _capture_tensor(k)
        tensors["current_v"] = _capture_tensor(v)

    capture_caches = _enabled("CSAHCA_DSV4_CAPTURE_CACHES", "1")
    if capture_caches:
        tensors["swa_k_cache"] = _capture_tensor(token_to_kv_pool.get_swa_key_buffer_radix(layer_id))

    extra_page_size = None
    if compress_ratio in (4, 128):
        extra_indices, extra_topk_lengths = _prepare_extra_indices(
            core=core,
            q=q3,
            compress_ratio=compress_ratio,
        )
        extra_page_size = _extra_page_size(token_to_kv_pool, layer_id, compress_ratio)
        tensors["extra_token_indices"] = _capture_tensor(extra_indices)
        tensors["extra_topk_lengths"] = _capture_tensor(extra_topk_lengths)
        if capture_caches:
            tensors["extra_k_cache"] = _capture_tensor(token_to_kv_pool.get_extra_key_buffer(layer_id))

    metadata = {
        "schema": "csahca.dsv4.capture.v1",
        "created_unix_s": time.time(),
        "pid": os.getpid(),
        "rank": _dist_rank(),
        "layer_id": int(layer_id),
        "compress_ratio": int(compress_ratio),
        "forward_mode": _forward_mode_name(forward_batch),
        "q_shape": _shape(q),
        "q3_shape": _shape(q3),
        "q_dtype": _dtype(q),
        "page_size": int(page_size),
        "extra_page_size": extra_page_size,
        "softmax_scale": float(self_obj.softmax_scale),
        "capture_caches": capture_caches,
    }
    filename = (
        f"dsv4_l{int(layer_id):03d}_c{int(compress_ratio)}_"
        f"{metadata['forward_mode'].lower()}_q{q3.shape[0]}_"
        f"r{metadata['rank']}_p{metadata['pid']}_{int(time.time() * 1_000_000)}.pt"
    )
    path = capture_dir / filename
    torch.save({"metadata": metadata, "tensors": tensors}, path)
    LOGGER.warning(
        "[CSAHCA][DSV4] captured live compare path=%s layer=%s compress_ratio=%s forward_mode=%s q_shape=%s",
        path,
        layer_id,
        compress_ratio,
        metadata["forward_mode"],
        metadata["q3_shape"],
    )


def _maybe_forward_csahca(
    *,
    original_forward: Callable[..., torch.Tensor],
    self_obj: Any,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer: Any,
    forward_batch: Any,
    compress_ratio: int,
    save_kv_cache: bool,
    attn_sink: torch.Tensor | None,
    kwargs: dict[str, Any],
) -> torch.Tensor:
    metadata = self_obj.forward_metadata
    core = metadata.core_attn_metadata
    token_to_kv_pool = self_obj.token_to_kv_pool
    layer_id = layer.layer_id

    swa_k_cache = token_to_kv_pool.get_swa_key_buffer_radix(layer_id)
    extra_k_cache = None
    extra_page_indices = None
    extra_topk_lengths = None
    if compress_ratio in (4, 128):
        extra_k_cache = token_to_kv_pool.get_extra_key_buffer(layer_id)
        if compress_ratio == 4:
            extra_page_indices = getattr(core, "c4_sparse_page_indices", None)
            extra_topk_lengths = getattr(core, "c4_sparse_topk_lengths", None)
        else:
            extra_page_indices = getattr(core, "c128_page_indices", None)
            extra_topk_lengths = getattr(core, "c128_topk_lengths_clamp1", None)

    bridge_inputs = DSV4BridgeInputs(
        layer_id=int(layer_id),
        forward_mode=_forward_mode_name(forward_batch),
        compress_ratio=int(compress_ratio),
        q=q,
        current_k=k,
        current_v=v,
        swa_k_cache=swa_k_cache,
        swa_page_indices=core.swa_page_indices,
        swa_topk_lengths=core.swa_topk_lengths,
        attn_sink=attn_sink,
        extra_k_cache=extra_k_cache,
        extra_page_indices=extra_page_indices,
        extra_topk_lengths=extra_topk_lengths,
    )
    decision = decide_dsv4_kernel_support(inputs=bridge_inputs)

    if _cuda_stream_capturing(q) and not _enabled("CSAHCA_DSV4_ALLOW_CAPTURE_HOOK"):
        return original_forward(
            self_obj,
            q,
            k,
            v,
            layer,
            forward_batch,
            compress_ratio=compress_ratio,
            save_kv_cache=save_kv_cache,
            attn_sink=attn_sink,
            **kwargs,
        )

    replace_output = _enabled("CSAHCA_DSV4_REPLACE_OUTPUT")
    replace_forward_mode_allowed = _forward_mode_allowed(
        forward_batch,
        "CSAHCA_DSV4_REPLACE_FORWARD_MODES",
        "DECODE",
    )
    q_ok_for_replace = True
    q_guard_reason = ""
    if decision.use_csahca and replace_output and replace_forward_mode_allowed:
        q_ok_for_replace, q_guard_reason = _q_guard(
            q,
            require_finite_env="CSAHCA_DSV4_REPLACE_REQUIRE_FINITE_Q",
            require_finite_default="1",
            max_abs_env="CSAHCA_DSV4_REPLACE_MAX_ABS_Q",
            max_abs_default=1.0e6,
        )
    if (
        decision.use_csahca
        and replace_output
        and replace_forward_mode_allowed
        and q_ok_for_replace
    ):
        if save_kv_cache:
            self_obj.store_cache(layer_id, k, forward_batch)
        return _run_dsv4_swa_prototype(
            self_obj=self_obj,
            q=q,
            token_to_kv_pool=token_to_kv_pool,
            core=core,
            layer_id=int(layer_id),
            compress_ratio=int(compress_ratio),
            attn_sink=attn_sink,
        )

    reference = original_forward(
        self_obj,
        q,
        k,
        v,
        layer,
        forward_batch,
        compress_ratio=compress_ratio,
        save_kv_cache=save_kv_cache,
        attn_sink=attn_sink,
        **kwargs,
    )

    live_compare = _enabled("CSAHCA_DSV4_LIVE_COMPARE", "1")
    compare_allowed = True
    if decision.use_csahca and live_compare:
        compare_allowed, _ = _q_guard(
            q,
            require_finite_env="CSAHCA_DSV4_COMPARE_REQUIRE_FINITE_Q",
            require_finite_default="0",
            max_abs_env="CSAHCA_DSV4_COMPARE_MAX_ABS_Q",
            max_abs_default=0.0,
        )
    if decision.use_csahca and live_compare and compare_allowed and _compare_budget_available(
        layer_id=int(layer_id),
        forward_batch=forward_batch,
        q=q,
    ):
        try:
            _log_compare_input_stats(
                layer_id=int(layer_id),
                compress_ratio=int(compress_ratio),
                forward_batch=forward_batch,
                q=q,
                token_to_kv_pool=token_to_kv_pool,
                core=core,
            )
            candidate = _run_dsv4_swa_prototype(
                self_obj=self_obj,
                q=q,
                token_to_kv_pool=token_to_kv_pool,
                core=core,
                layer_id=int(layer_id),
                compress_ratio=int(compress_ratio),
                attn_sink=attn_sink,
            )
            _log_live_compare(
                layer_id=int(layer_id),
                compress_ratio=int(compress_ratio),
                forward_batch=forward_batch,
                reference=reference,
                candidate=candidate,
                q=q,
            )
            if _capture_budget_available(
                layer_id=int(layer_id),
                compress_ratio=int(compress_ratio),
                forward_batch=forward_batch,
                q=q,
            ):
                _capture_live_compare(
                    self_obj=self_obj,
                    q=q,
                    k=k,
                    v=v,
                    token_to_kv_pool=token_to_kv_pool,
                    core=core,
                    layer_id=int(layer_id),
                    compress_ratio=int(compress_ratio),
                    forward_batch=forward_batch,
                    attn_sink=attn_sink,
                    reference=reference,
                    candidate=candidate,
                )
        except Exception as exc:
            LOGGER.warning(
                "[CSAHCA][DSV4] live compare failed layer=%s compress_ratio=%s: %r",
                layer_id,
                compress_ratio,
                exc,
            )

    if decision.use_csahca:
        if replace_output and replace_forward_mode_allowed and not q_ok_for_replace:
            _log_delegate(
                layer_id=int(layer_id),
                compress_ratio=int(compress_ratio),
                forward_batch=forward_batch,
                q=q,
                reason=f"{q_guard_reason}; q guard kept FlashMLA output",
            )
        return reference

    if _enabled("CSAHCA_DSV4_REQUIRE_KERNEL"):
        raise RuntimeError(
            "[CSAHCA][DSV4] CSAHCA_DSV4_REQUIRE_KERNEL=1 but no compatible "
            f"kernel is available for layer={layer_id}, compress_ratio={compress_ratio}: "
            f"{decision.reason}"
        )

    _log_delegate(
        layer_id=int(layer_id),
        compress_ratio=int(compress_ratio),
        forward_batch=forward_batch,
        q=q,
        reason=decision.reason,
    )
    return reference


def install() -> bool:
    """Install an env-gated wrapper around SGLang's DSV4 attention backend."""

    global _PATCHED
    if _PATCHED:
        return True

    from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend

    original_forward = DeepseekV4AttnBackend.forward
    if getattr(original_forward, "_csahca_dsv4_patched", False):
        _PATCHED = True
        return True

    @functools.wraps(original_forward)
    def wrapped_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        forward_batch,
        compress_ratio,
        save_kv_cache: bool = True,
        attn_sink: torch.Tensor | None = None,
        **kwargs,
    ):
        _record_call(
            q=q,
            layer_id=layer.layer_id,
            compress_ratio=int(compress_ratio),
            forward_batch=forward_batch,
        )
        _record_abi(
            self_obj=self,
            q=q,
            k=k,
            layer=layer,
            forward_batch=forward_batch,
            compress_ratio=int(compress_ratio),
            attn_sink=attn_sink,
        )

        mode = _mode()
        nvtx_name = (
            f"CSAHCA_DSV4/{mode}/layer_{layer.layer_id}/"
            f"c{int(compress_ratio)}/{_forward_mode_name(forward_batch)}"
        )
        if mode in {"trace", "delegate", "off"}:
            with _NvtxRange(nvtx_name, q):
                return original_forward(
                    self,
                    q,
                    k,
                    v,
                    layer,
                    forward_batch,
                    compress_ratio=compress_ratio,
                    save_kv_cache=save_kv_cache,
                    attn_sink=attn_sink,
                    **kwargs,
                )
        if mode in {"csahca", "require-kernel"}:
            require_kernel = mode == "require-kernel"
            previous = os.getenv("CSAHCA_DSV4_REQUIRE_KERNEL")
            if require_kernel:
                os.environ["CSAHCA_DSV4_REQUIRE_KERNEL"] = "1"
            try:
                with _NvtxRange(nvtx_name, q):
                    return _maybe_forward_csahca(
                        original_forward=original_forward,
                        self_obj=self,
                        q=q,
                        k=k,
                        v=v,
                        layer=layer,
                        forward_batch=forward_batch,
                        compress_ratio=int(compress_ratio),
                        save_kv_cache=save_kv_cache,
                        attn_sink=attn_sink,
                        kwargs=kwargs,
                    )
            finally:
                if require_kernel:
                    if previous is None:
                        os.environ.pop("CSAHCA_DSV4_REQUIRE_KERNEL", None)
                    else:
                        os.environ["CSAHCA_DSV4_REQUIRE_KERNEL"] = previous
        raise ValueError(f"unknown CSAHCA_DSV4_MODE={mode!r}")

    wrapped_forward._csahca_dsv4_patched = True  # type: ignore[attr-defined]
    DeepseekV4AttnBackend.forward = wrapped_forward
    _PATCHED = True
    LOGGER.warning(
        "[CSAHCA][DSV4] installed DeepseekV4AttnBackend.forward hook; mode=%s",
        _mode(),
    )
    return True
