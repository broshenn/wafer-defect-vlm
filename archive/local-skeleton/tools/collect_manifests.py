"""Copy run-configuration records into manifests/, where they are committed.

The training run manifest lives in ``outputs/checkpoints/<run>/run_manifest.json``
— a directory ``.gitignore`` excludes, because its siblings are multi-gigabyte
adapter weights. The consequence is that the one file recording *how the model
was trained* would not be in the repository, and `FINAL_REPORT.md` cites it by
name. Excluding a directory in git also makes its files impossible to re-include
with a later ``!`` rule, so the fix is to copy the record out rather than to
fiddle with ignore patterns.

Every copied document keeps its source path and sha256, so a copy can be checked
against the original with ``sha256sum`` instead of being taken on trust.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    runs = []

    # The SFT-style runs write a hand-built manifest.
    for source in sorted((root / "outputs/checkpoints").glob("*/run_manifest.json")):
        document = load(source)
        if document is None:
            continue
        runs.append({
            "run": document.get("run") or source.parent.name,
            "kind": "sft",
            "source": str(source.relative_to(root)),
            "source_sha256": sha256(source),
            "document": document,
        })

    # The GRPO wrapper records its configuration in the result file rather than
    # in a run manifest; that file already lives under outputs/reports/, which
    # is committed, but it is repeated here so one document holds every run.
    grpo = load(root / "outputs/reports/grpo_train_result.json")
    if grpo:
        runs.append({
            "run": "qwen35_9b_grpo_v1",
            "kind": "grpo",
            "source": "outputs/reports/grpo_train_result.json",
            "source_sha256": sha256(root / "outputs/reports/grpo_train_result.json"),
            "document": {
                "outcome": grpo.get("outcome"),
                "reason": grpo.get("reason"),
                "elapsed_seconds": grpo.get("elapsed_seconds"),
                "gpu_peak_mib": grpo.get("gpu_peak_mib"),
                "max_steps_requested": grpo.get("max_steps_requested"),
                "steps_logged": grpo.get("steps_logged"),
                "reward_signal": grpo.get("reward_signal"),
                "config": grpo.get("config"),
            },
        })

    payload = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": (
            "Configuration records copied out of outputs/checkpoints/, which is "
            "git-ignored because it holds adapter weights. Each entry carries the "
            "sha256 of the file it was copied from; verify with sha256sum."
        ),
        "runs": runs,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    for entry in runs:
        print(f"{entry['kind']:5s} {entry['run']:28s} <- {entry['source']}")
    print(f"wrote {out} ({len(runs)} runs)")
    return 0 if runs else 1


if __name__ == "__main__":
    raise SystemExit(main())
