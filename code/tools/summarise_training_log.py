"""Turn a finished training run's summary block into a record.

Every RL run in this project writes a *_train_result.json, but the SFT run never
did: its outcome exists only as a block printed at the end of logs/08_train.log.
That asymmetry showed up when auditing the documents for numbers with no source
-- "train_loss 0.3891, 32.92 GiB" could not be traced to any record, even though
both are correct.

Transcribing the block into a record is not the same as inventing one. The value
is read out of the log the run actually produced, and the record cites the file
and line it came from, so the claim can be re-checked later instead of trusted.

Refuses to write anything if no summary block is found, so a failed parse leaves
the absence visible rather than replacing it with an empty record.
"""
import argparse
import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path("/root/autodl-fs/wafer-vlm")
SUMMARY = re.compile(r"^\{.*'train_loss'.*\}$")


def coerce(value):
    """ms-swift prints every field as a string; keep the numbers as numbers so
    the record can be compared against the prose without re-parsing text."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    if re.fullmatch(r"-?\d*\.\d+([eE][-+]?\d+)?", text):
        return float(text)
    return value


def find_summary(log):
    """The last matching line wins: a log that was resumed prints the block more
    than once, and only the final block describes the finished run."""
    found = None
    for number, line in enumerate(log.read_text(errors="replace").splitlines(), 1):
        stripped = line.strip()
        if SUMMARY.match(stripped):
            found = (number, stripped)
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path)
    ap.add_argument("--name", required=True, help="run name, e.g. sft")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--note", default=None)
    args = ap.parse_args()

    if not args.log.is_file():
        sys.exit(f"no such log: {args.log}")
    hit = find_summary(args.log)
    if hit is None:
        sys.exit(f"no training summary block in {args.log}; nothing written")
    line_number, raw = hit

    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError) as exc:
        sys.exit(f"summary block on line {line_number} is not parseable: {exc}")
    if not isinstance(parsed, dict) or "train_loss" not in parsed:
        sys.exit(f"summary block on line {line_number} is not a training summary")

    record = {
        "run": args.name,
        "summary": {k: coerce(v) for k, v in parsed.items()},
        "source_log": str(args.log.relative_to(ROOT) if args.log.is_relative_to(ROOT)
                          else args.log),
        "source_line": line_number,
        "source_note": ("Transcribed verbatim from the run's own log by "
                        "tools/summarise_training_log.py; the summary block prints "
                        "every field as a string and the numbers are cast here."),
    }
    if args.note:
        record["note"] = args.note

    out = args.output or (ROOT / "outputs/reports" / f"{args.name}_train_result.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"wrote {out}")
    for k, v in record["summary"].items():
        print(f"  {k:28s} {v!r}")


if __name__ == "__main__":
    main()
