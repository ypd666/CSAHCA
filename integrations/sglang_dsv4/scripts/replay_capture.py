#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hybrid_attention.extension import dsv4_sparse_decode_forward, dsv4_swa_decode_forward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay CSAHCA DSV4 live-compare tensor captures.")
    parser.add_argument("captures", nargs="+", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--atol", type=float, default=5.0e-2)
    parser.add_argument("--rtol", type=float, default=5.0e-2)
    parser.add_argument("--json", action="store_true", help="Emit one JSON object per capture.")
    return parser.parse_args()


def to_device(x: torch.Tensor | None, device: torch.device) -> torch.Tensor | None:
    if x is None:
        return None
    return x.to(device=device, non_blocking=True)


def compare(got: torch.Tensor, ref: torch.Tensor, *, atol: float, rtol: float) -> dict[str, Any]:
    got_f = got.float()
    ref_f = ref.float()
    diff = (got_f - ref_f).abs()
    got_finite = torch.isfinite(got_f)
    ref_finite = torch.isfinite(ref_f)
    both_finite = got_finite & ref_finite
    nonfinite_mismatch = got_finite != ref_finite
    if bool(both_finite.any().item()):
        finite_diff = diff[both_finite]
        finite_ref = ref_f[both_finite]
        max_abs = float(finite_diff.max().item())
        max_rel = float((finite_diff / finite_ref.abs().clamp_min(1.0e-6)).max().item())
        rms = float(torch.sqrt(torch.mean(finite_diff * finite_diff)).item())
        bad_finite = int((finite_diff > (atol + rtol * finite_ref.abs())).sum().item())
    else:
        max_abs = math.nan
        max_rel = math.nan
        rms = math.nan
        bad_finite = 0
    return {
        "allclose": bool(torch.allclose(got, ref, atol=atol, rtol=rtol)),
        "max_abs": max_abs,
        "max_rel": max_rel,
        "rms": rms,
        "got_nan": int(torch.isnan(got_f).sum().item()),
        "ref_nan": int(torch.isnan(ref_f).sum().item()),
        "got_inf": int(torch.isinf(got_f).sum().item()),
        "ref_inf": int(torch.isinf(ref_f).sum().item()),
        "finite_pairs": int(both_finite.sum().item()),
        "total_pairs": int(diff.numel()),
        "nonfinite_mismatch": int(nonfinite_mismatch.sum().item()),
        "bad_finite": bad_finite,
    }


def replay_one(path: Path, *, device: torch.device, atol: float, rtol: float) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    metadata = payload["metadata"]
    tensors = payload["tensors"]
    missing = [name for name in ("q", "swa_k_cache", "swa_token_indices", "swa_topk_lengths") if tensors.get(name) is None]
    if missing:
        raise RuntimeError(f"{path} cannot be replayed because tensors are missing: {missing}")

    q = to_device(tensors["q"], device)
    swa_k_cache = to_device(tensors["swa_k_cache"], device)
    token_indices = to_device(tensors["swa_token_indices"], device)
    topk_lengths = to_device(tensors["swa_topk_lengths"], device)
    attn_sink = to_device(tensors.get("attn_sink"), device)
    assert q is not None and swa_k_cache is not None and token_indices is not None and topk_lengths is not None

    compress_ratio = int(metadata["compress_ratio"])
    with torch.no_grad():
        if compress_ratio in (4, 128):
            for name in ("extra_k_cache", "extra_token_indices", "extra_topk_lengths"):
                if tensors.get(name) is None:
                    raise RuntimeError(f"{path} cannot replay compress_ratio={compress_ratio}; missing {name}")
            got = dsv4_sparse_decode_forward(
                q,
                swa_k_cache,
                token_indices,
                topk_lengths,
                to_device(tensors["extra_k_cache"], device),
                to_device(tensors["extra_token_indices"], device),
                to_device(tensors["extra_topk_lengths"], device),
                attn_sink,
                int(metadata["page_size"]),
                int(metadata["extra_page_size"]),
                float(metadata["softmax_scale"]),
            )
        else:
            got = dsv4_swa_decode_forward(
                q,
                swa_k_cache,
                token_indices,
                topk_lengths,
                attn_sink,
                int(metadata["page_size"]),
                float(metadata["softmax_scale"]),
            )
        if device.type == "cuda":
            torch.cuda.synchronize()

    result: dict[str, Any] = {
        "path": str(path),
        "layer_id": int(metadata["layer_id"]),
        "compress_ratio": compress_ratio,
        "forward_mode": metadata["forward_mode"],
        "q_shape": list(metadata["q3_shape"]),
    }
    if tensors.get("candidate") is not None:
        result["vs_saved_candidate"] = compare(got, to_device(tensors["candidate"], device), atol=atol, rtol=rtol)
    if tensors.get("reference") is not None:
        result["vs_flashmla_reference"] = compare(got, to_device(tensors["reference"], device), atol=atol, rtol=rtol)
    return result


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    for capture in args.captures:
        result = replay_one(capture, device=device, atol=args.atol, rtol=args.rtol)
        if args.json:
            print(json.dumps(result, sort_keys=True))
        else:
            print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
