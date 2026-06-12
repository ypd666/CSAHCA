"""Auto-loader for the CSAHCA SGLang DeepSeek-V4 integration.

Python imports a top-level ``sitecustomize`` module during interpreter start
when it is present on ``PYTHONPATH``. Keep this file tiny and fail-soft so that
adding the integration directory to ``PYTHONPATH`` never breaks unrelated
commands.
"""

from __future__ import annotations

import os
import sys


def _enabled(value: str | None) -> bool:
    return value is not None and value.strip().lower() not in {"", "0", "false", "no", "off"}


if _enabled(os.getenv("CSAHCA_SGLANG_DSV4_PATCH")):
    try:
        from csahca_sglang_dsv4_patch.patch import install

        install()
    except Exception as exc:  # pragma: no cover - defensive startup path
        print(f"[CSAHCA][DSV4] failed to install patch: {exc!r}", file=sys.stderr)
        if _enabled(os.getenv("CSAHCA_SGLANG_DSV4_PATCH_STRICT")):
            raise
