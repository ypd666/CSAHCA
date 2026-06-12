from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tiny OpenAI-compatible chat benchmark.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL_PATH", os.path.expanduser("~/checkpoints/DeepSeek-V4-Flash-HF")),
    )
    parser.add_argument("--num-prompts", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--input-words", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def make_prompt(input_words: int, request_id: int) -> str:
    words = ["profile", "attention", "kernel", "latency", "throughput", "cache", "decode", "token"]
    body = " ".join(words[i % len(words)] for i in range(max(1, input_words)))
    return f"Request {request_id}. {body}\nReply with one concise sentence."


def send_one(args: argparse.Namespace, request_id: int) -> dict[str, Any]:
    url = f"http://{args.host}:{args.port}/v1/chat/completions"
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": make_prompt(args.input_words, request_id)}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "ignore_eos": args.ignore_eos,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    begin = time.perf_counter()
    with urllib.request.urlopen(req, timeout=args.timeout) as resp:
        raw = resp.read()
        status = resp.status
    latency_s = time.perf_counter() - begin
    decoded = json.loads(raw.decode("utf-8"))
    usage = decoded.get("usage") or {}
    choice = (decoded.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    return {
        "request_id": request_id,
        "status": status,
        "latency_s": latency_s,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
        "finish_reason": choice.get("finish_reason"),
        "content_preview": str(message.get("content") or "")[:120],
    }


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[idx]


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        futures = [pool.submit(send_one, args, i) for i in range(args.num_prompts)]
        for future in as_completed(futures):
            rows.append(future.result())
    duration_s = time.perf_counter() - started

    latencies = [float(row["latency_s"]) for row in rows]
    completion_tokens = sum(int(row["completion_tokens"]) for row in rows)
    prompt_tokens = sum(int(row["prompt_tokens"]) for row in rows)
    result = {
        "host": args.host,
        "port": args.port,
        "model": args.model,
        "num_prompts": args.num_prompts,
        "concurrency": args.concurrency,
        "input_words": args.input_words,
        "max_tokens": args.max_tokens,
        "ignore_eos": args.ignore_eos,
        "completed": len(rows),
        "duration_s": duration_s,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "output_tokens_per_s": completion_tokens / duration_s if duration_s > 0 else 0.0,
        "request_throughput": len(rows) / duration_s if duration_s > 0 else 0.0,
        "mean_latency_s": statistics.mean(latencies) if latencies else 0.0,
        "p50_latency_s": percentile(latencies, 50),
        "p90_latency_s": percentile(latencies, 90),
        "p99_latency_s": percentile(latencies, 99),
    }
    print(json.dumps(result, indent=2, sort_keys=True))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as handle:
            handle.write(json.dumps({"summary": result, "requests": rows}, indent=2, sort_keys=True))
            handle.write("\n")


if __name__ == "__main__":
    main()
