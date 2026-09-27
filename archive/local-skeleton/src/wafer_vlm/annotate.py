"""Resume-safe multimodal teacher annotation through OpenAI-compatible APIs."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI

from .prompts import STRUCTURED_TEACHER_SYSTEM, make_structured_teacher_prompt
from .utils import extract_json_object, image_data_url, read_jsonl


PROVIDERS = {
    "dashscope": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY"),
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY"),
}


def select_records(records: list[dict], split: str, per_class: int, seed: int, ids_file: str | None) -> list[dict]:
    allowed_ids: set[str] | None = None
    if ids_file:
        allowed_ids = {line.strip() for line in Path(ids_file).read_text(encoding="utf-8").splitlines() if line.strip()}
    filtered = [
        r for r in records
        if (split == "all" or r.get("split") == split)
        and (allowed_ids is None or r["sample_id"] in allowed_ids)
    ]
    if per_class < 0:
        return filtered
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in filtered:
        groups[row["failure_type"]].append(row)
    rng = random.Random(seed)
    selected: list[dict] = []
    for label in sorted(groups):
        group = groups[label]
        rng.shuffle(group)
        selected.extend(group[:per_class])
    rng.shuffle(selected)
    return selected


def existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
                if row.get("status") == "ok":
                    ids.add(row["sample_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return ids


async def run(args: argparse.Namespace) -> None:
    records = select_records(read_jsonl(args.manifest), args.split, args.per_class, args.seed, args.ids_file)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    done = existing_ids(output)
    pending = [row for row in records if row["sample_id"] not in done]
    print(json.dumps({"selected": len(records), "resumed": len(done), "pending": len(pending)}, ensure_ascii=False))
    if args.dry_run:
        return

    load_dotenv(args.env_file, override=False)
    import os

    default_url, key_name = PROVIDERS[args.provider]
    api_key = os.environ.get(key_name)
    if not api_key:
        raise RuntimeError(f"Missing {key_name}; load it through --env-file")
    client = AsyncOpenAI(api_key=api_key, base_url=args.base_url or default_url, timeout=args.timeout)

    queue: asyncio.Queue[dict | None] = asyncio.Queue(maxsize=args.concurrency * 2)
    write_lock = asyncio.Lock()

    async def write(row: dict[str, Any]) -> None:
        async with write_lock:
            with output.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    async def annotate_one(row: dict[str, Any]) -> dict[str, Any]:
        facts = json.dumps(row.get("features", {}), ensure_ascii=False, separators=(",", ":"))
        prompt = make_structured_teacher_prompt(facts, row["failure_type"] if args.include_label else None)
        last_error = "unknown"
        for attempt in range(1, args.retries + 1):
            started = time.time()
            try:
                response = await client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": STRUCTURED_TEACHER_SYSTEM},
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": image_data_url(row["image_path"])}},
                                {"type": "text", "text": prompt},
                            ],
                        },
                    ],
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    extra_body={"enable_thinking": args.enable_thinking},
                )
                message = response.choices[0].message
                content = message.content or ""
                parsed = extract_json_object(content)
                usage = response.usage.model_dump() if response.usage else None
                return {
                    "sample_id": row["sample_id"],
                    "status": "ok",
                    "provider": args.provider,
                    "model": args.model,
                    "prompt_version": "structured_teacher_v1",
                    "result": parsed,
                    "raw_content": content,
                    "reasoning_content": getattr(message, "reasoning_content", None),
                    "usage": usage,
                    "latency_seconds": round(time.time() - started, 3),
                    "attempt": attempt,
                }
            except Exception as exc:  # response errors must be recorded, not halt the batch
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < args.retries:
                    await asyncio.sleep(min(2 ** (attempt - 1), 20))
        return {
            "sample_id": row["sample_id"],
            "status": "error",
            "provider": args.provider,
            "model": args.model,
            "prompt_version": "structured_teacher_v1",
            "error": last_error[:2000],
        }

    async def worker() -> None:
        while True:
            row = await queue.get()
            if row is None:
                queue.task_done()
                return
            result = await annotate_one(row)
            await write(result)
            print(json.dumps({"sample_id": row["sample_id"], "status": result["status"]}, ensure_ascii=False))
            queue.task_done()

    workers = [asyncio.create_task(worker()) for _ in range(args.concurrency)]
    for row in pending:
        await queue.put(row)
    for _ in workers:
        await queue.put(None)
    await queue.join()
    await asyncio.gather(*workers)
    await client.close()


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--env-file", required=True)
    parser.add_argument("--provider", choices=sorted(PROVIDERS), default="dashscope")
    parser.add_argument("--base-url")
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="train")
    parser.add_argument("--per-class", type=int, default=-1)
    parser.add_argument("--ids-file")
    parser.add_argument("--include-label", action="store_true")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--dry-run", action="store_true", help="Show selected/resumed/pending counts without API calls")
    return parser


def main() -> None:
    asyncio.run(run(make_parser().parse_args()))


if __name__ == "__main__":
    main()
