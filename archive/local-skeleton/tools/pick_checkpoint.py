"""Pick the checkpoint with the lowest validation loss.

Taking the last checkpoint is the easy default, but with a 604-step budget the
final weights are not necessarily the best ones, and the headline comparison
would inherit that accident. This reads the eval losses ms-swift recorded and
names the winner, printing the evidence so the choice can be checked.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def from_trainer_state(run_dir: Path) -> list[tuple[int, float, Path]]:
    found = []
    for state_path in sorted(run_dir.glob("*/checkpoint-*/trainer_state.json")):
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        losses = [entry["eval_loss"] for entry in state.get("log_history", [])
                  if isinstance(entry.get("eval_loss"), (int, float))]
        if not losses:
            continue
        found.append((int(state.get("global_step", -1)), float(min(losses)), state_path.parent))
    return found


def from_log(log_path: Path) -> list[tuple[int, float, Path]]:
    """Fallback: ms-swift prints eval_loss and global_step into the run log."""
    if not log_path.is_file():
        return []
    text = log_path.read_text(encoding="utf-8", errors="replace")
    found = []
    for match in re.finditer(r"\{'eval_loss':\s*([0-9.eE+-]+),\s*'eval_[^']*':\s*[0-9.eE+-]+"
                             r"(?:,\s*'eval_[^']*':\s*[0-9.eE+-]+)*,\s*'epoch':\s*[0-9.eE+-]+"
                             r"(?:,\s*'global_step':\s*(\d+))?", text):
        found.append((int(match.group(2) or -1), float(match.group(1)), Path("")))
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--log")
    parser.add_argument("--output")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    candidates = from_trainer_state(run_dir)
    source = "trainer_state"
    if not candidates and args.log:
        candidates = from_log(Path(args.log))
        source = "log"

    # Keep the newest record per checkpoint when a log repeats itself.
    best: dict[int, tuple[int, float, Path]] = {}
    for step, loss, path in candidates:
        if step not in best or loss < best[step][1]:
            best[step] = (step, loss, path)
    ranked = sorted(best.values(), key=lambda item: item[1])

    report = {
        "run_dir": str(run_dir),
        "source": source,
        "evaluations": [{"global_step": step, "eval_loss": loss, "path": str(path)}
                        for step, loss, path in sorted(best.values())],
        "best_global_step": ranked[0][0] if ranked else None,
        "best_eval_loss": ranked[0][1] if ranked else None,
        "best_checkpoint": str(ranked[0][2]) if ranked else None,
        "note": "empty result means no eval_loss was recorded; check eval_steps before assuming a checkpoint exists",
    }
    if args.output:
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if ranked else 1


if __name__ == "__main__":
    raise SystemExit(main())
