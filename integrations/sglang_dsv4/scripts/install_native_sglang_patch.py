#!/usr/bin/env python3
"""Install the native CSAHCA DSV4 branch into SGLang's DeepSeek-V4 backend."""

from __future__ import annotations

import argparse
from pathlib import Path


MARKER = "# CSAHCA native decode integration"


def _backend_path(sglang_src: Path) -> Path:
    return (
        sglang_src
        / "python"
        / "sglang"
        / "srt"
        / "layers"
        / "attention"
        / "deepseek_v4_backend.py"
    )


def install(sglang_src: Path) -> bool:
    backend = _backend_path(sglang_src)
    text = backend.read_text()
    if MARKER in text:
        return False

    if "\nimport os\n" not in text[:500]:
        text = text.replace(
            "from __future__ import annotations\n\nimport enum\n",
            "from __future__ import annotations\n\nimport enum\nimport os\n",
            1,
        )

    anchor = "            extra_topk_lengths = match_num_queries(extra_topk_lengths, value=1)\n\n"
    insert = f"""            extra_topk_lengths = match_num_queries(extra_topk_lengths, value=1)\n\n\
            {MARKER}\n\
            if os.getenv(\"CSAHCA_DSV4_NATIVE\", \"0\").strip().lower() not in {{\"\", \"0\", \"false\", \"no\", \"off\"}}:\n\
                from csahca_sglang_dsv4_native import maybe_dsv4_decode_forward\n\n\
                csahca_o = maybe_dsv4_decode_forward(\n\
                    q=q,\n\
                    layer_id=layer_id,\n\
                    compress_ratio=compress_ratio,\n\
                    forward_batch=forward_batch,\n\
                    token_to_kv_pool=token_to_kv_pool,\n\
                    swa_k_cache=swa_k_cache,\n\
                    swa_page_indices=swa_page_indices,\n\
                    swa_topk_lengths=swa_topk_lengths,\n\
                    extra_k_cache=extra_k_cache,\n\
                    extra_indices=extra_indices,\n\
                    extra_topk_lengths=extra_topk_lengths,\n\
                    attn_sink=attn_sink,\n\
                    page_size=token_to_kv_pool.swa_window_size,\n\
                    softmax_scale=self.softmax_scale,\n\
                )\n\
                if csahca_o is not None:\n\
                    return csahca_o\n\n"""
    if anchor not in text:
        raise RuntimeError(f"could not find insertion anchor in {backend}")
    text = text.replace(anchor, insert, 1)
    backend.write_text(text)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sglang-src",
        type=Path,
        required=True,
        help="Path to the SGLang source checkout.",
    )
    args = parser.parse_args()
    changed = install(args.sglang_src)
    print(f"native_patch_installed={str(changed).lower()}")


if __name__ == "__main__":
    main()
