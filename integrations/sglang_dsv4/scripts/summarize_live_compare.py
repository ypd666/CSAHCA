#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
LIVE_RE = re.compile(
    r"live compare layer=(?P<layer>\d+) "
    r"compress_ratio=(?P<ratio>\d+) "
    r"forward_mode=(?P<mode>\S+) "
    r"q_shape=(?P<q_shape>\([^)]*\)).*?"
    r"max_abs=(?P<max_abs>\S+) "
    r"max_rel=(?P<max_rel>\S+) "
    r"rms=(?P<rms>\S+) "
    r"allclose=(?P<allclose>True|False).*?"
    r"ref_nan=(?P<ref_nan>\d+) "
    r"csahca_nan=(?P<csahca_nan>\d+) "
    r"ref_inf=(?P<ref_inf>\d+) "
    r"csahca_inf=(?P<csahca_inf>\d+) "
    r"finite_pairs=(?P<finite_pairs>\d+)/(?P<total_pairs>\d+) "
    r"both_nan=(?P<both_nan>\d+) "
    r"nonfinite_mismatch=(?P<nonfinite_mismatch>\d+) "
    r"bad_finite=(?P<bad_finite>\d+)"
)


@dataclass
class Agg:
    calls: int = 0
    allclose: int = 0
    max_abs: float = 0.0
    max_rel: float = 0.0
    max_rms: float = 0.0
    ref_nan: int = 0
    csahca_nan: int = 0
    ref_inf: int = 0
    csahca_inf: int = 0
    finite_pairs: int = 0
    total_pairs: int = 0
    both_nan: int = 0
    nonfinite_mismatch: int = 0
    bad_finite: int = 0

    def add(self, row: dict[str, str]) -> None:
        self.calls += 1
        self.allclose += 1 if row["allclose"] == "True" else 0
        self.max_abs = max(self.max_abs, parse_float(row["max_abs"]))
        self.max_rel = max(self.max_rel, parse_float(row["max_rel"]))
        self.max_rms = max(self.max_rms, parse_float(row["rms"]))
        self.ref_nan += int(row["ref_nan"])
        self.csahca_nan += int(row["csahca_nan"])
        self.ref_inf += int(row["ref_inf"])
        self.csahca_inf += int(row["csahca_inf"])
        self.finite_pairs += int(row["finite_pairs"])
        self.total_pairs += int(row["total_pairs"])
        self.both_nan += int(row["both_nan"])
        self.nonfinite_mismatch += int(row["nonfinite_mismatch"])
        self.bad_finite += int(row["bad_finite"])

    def as_row(self, key: tuple[str, ...], fields: list[str]) -> dict[str, object]:
        out: dict[str, object] = {field: value for field, value in zip(fields, key)}
        out.update(
            {
                "calls": self.calls,
                "allclose": self.allclose,
                "fail": self.calls - self.allclose,
                "allclose_rate": f"{(self.allclose / self.calls) if self.calls else 0.0:.4f}",
                "max_abs": f"{self.max_abs:.6g}",
                "max_rel": f"{self.max_rel:.6g}",
                "max_rms": f"{self.max_rms:.6g}",
                "ref_nan": self.ref_nan,
                "csahca_nan": self.csahca_nan,
                "ref_inf": self.ref_inf,
                "csahca_inf": self.csahca_inf,
                "finite_pairs": self.finite_pairs,
                "total_pairs": self.total_pairs,
                "both_nan": self.both_nan,
                "nonfinite_mismatch": self.nonfinite_mismatch,
                "bad_finite": self.bad_finite,
            }
        )
        return out


def parse_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        return 0.0
    return 0.0 if math.isnan(value) else value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize [CSAHCA][DSV4] live compare log lines.")
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument(
        "--group-by",
        default="ratio,mode,layer",
        help="Comma-separated fields from: ratio,mode,layer,q_shape,file",
    )
    parser.add_argument("--csv", action="store_true", help="Emit CSV instead of a Markdown table.")
    parser.add_argument("--fail-only", action="store_true", help="Only print groups with failed allclose or mismatch counts.")
    return parser.parse_args()


def iter_rows(paths: list[Path]):
    for path in paths:
        with path.open(errors="replace") as handle:
            for line_no, line in enumerate(handle, 1):
                clean = ANSI_RE.sub("", line)
                match = LIVE_RE.search(clean)
                if not match:
                    continue
                row = match.groupdict()
                row["file"] = str(path)
                row["line"] = str(line_no)
                yield row


def group_key(row: dict[str, str], fields: list[str]) -> tuple[str, ...]:
    return tuple(row[field] for field in fields)


def sort_key(item: tuple[tuple[str, ...], Agg]) -> tuple:
    key, agg = item
    parsed: list[object] = []
    for value in key:
        parsed.append(int(value) if value.isdigit() else value)
    return (*parsed, -agg.calls)


def print_markdown(rows: list[dict[str, object]], headers: list[str]) -> None:
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        print("| " + " | ".join(str(row[h]) for h in headers) + " |")


def main() -> None:
    args = parse_args()
    fields = [field.strip() for field in args.group_by.split(",") if field.strip()]
    allowed = {"ratio", "mode", "layer", "q_shape", "file"}
    unknown = set(fields) - allowed
    if unknown:
        raise SystemExit(f"unknown --group-by field(s): {sorted(unknown)}")

    groups: dict[tuple[str, ...], Agg] = {}
    parsed = 0
    for row in iter_rows(args.logs):
        parsed += 1
        key = group_key(row, fields)
        groups.setdefault(key, Agg()).add(row)

    rows = [agg.as_row(key, fields) for key, agg in sorted(groups.items(), key=sort_key)]
    if args.fail_only:
        rows = [
            row
            for row in rows
            if int(row["fail"]) or int(row["nonfinite_mismatch"]) or int(row["bad_finite"])
        ]

    headers = fields + [
        "calls",
        "allclose",
        "fail",
        "allclose_rate",
        "max_abs",
        "max_rel",
        "max_rms",
        "ref_nan",
        "csahca_nan",
        "nonfinite_mismatch",
        "bad_finite",
    ]
    if args.csv:
        writer = csv.DictWriter(sys.stdout, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    else:
        print(f"Parsed live-compare rows: {parsed}")
        print_markdown(rows, headers)


if __name__ == "__main__":
    main()
