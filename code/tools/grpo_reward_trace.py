"""Print the per-step reward breakdown of a GRPO run from its training log.

The point of looking is the standard deviation, not the mean. With four
generations per prompt, a reward whose per-group std is zero gives every
completion in the group the same score, so the advantage is zero and that
reward contributes no gradient at all. A run where three of four rewards are
flat is a run that only trained on the fourth, however good its mean looks.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROW = re.compile(r"\{'loss':.*?\}")


def value(name: str, row: str) -> float | None:
    match = re.search(r"'%s': '([-0-9.eE]+)'" % re.escape(name), row)
    return float(match.group(1)) if match else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True)
    parser.add_argument("--every", type=int, default=10)
    parser.add_argument("--output", default=None,
                        help="write the per-reward table to this JSON record")
    args = parser.parse_args()

    rows = ROW.findall(Path(args.log).read_text(encoding="utf-8", errors="replace"))
    print(f"logged steps: {len(rows)}\n")

    header = (f"{'step':>5} {'reward':>8} {'rew_std':>8} {'zero_std':>9} {'kl':>7} "
              f"{'class':>13} {'radial':>13} {'clock':>13}")
    print(header)
    print("-" * len(header))

    for index, row in enumerate(rows):
        if index % args.every and index != len(rows) - 1:
            continue
        def pair(prefix: str) -> str:
            return f"{value(f'rewards/{prefix}/mean', row)}/{value(f'rewards/{prefix}/std', row)}"
        print(f"{index + 1:>5} {value('reward', row):>8} {value('reward_std', row):>8} "
              f"{value('frac_reward_zero_std', row):>9} {value('kl', row):>7} "
              f"{pair('WaferClass'):>13} {pair('WaferRadial'):>13} {pair('WaferClock'):>13}")

    per_reward = {}
    print("\nmean/std per reward over all logged steps:")
    for name in ("WaferClass", "WaferFormat", "WaferRadial", "WaferClock"):
        means = [v for v in (value(f"rewards/{name}/mean", r) for r in rows) if v is not None]
        stds = [v for v in (value(f"rewards/{name}/std", r) for r in rows) if v is not None]
        carries = any(s > 0 for s in stds)
        per_reward[name] = {"mean": sum(means) / len(means),
                            "mean_of_std": sum(stds) / len(stds),
                            "carries_group_signal": carries}
        print(f"  {name:14s} mean={per_reward[name]['mean']:.4f}  "
              f"mean_of_std={per_reward[name]['mean_of_std']:.4f}  "
              f"carries_group_signal={carries}")

    idle = None
    zero = [v for v in (value("frac_reward_zero_std", r) for r in rows) if v is not None]
    if zero:
        idle = sum(1 for v in zero if v >= 1) / len(zero)
        print(f"\nfraction of steps whose groups ALL had identical rewards: {idle:.3f}")

    if args.output:
        # The means are quoted in LIMITATIONS to argue about which reward
        # carries the signal. Writing them down costs nothing and means the
        # quote can be checked against the log instead of being taken on trust.
        Path(args.output).write_text(json.dumps({
            "log": args.log,
            "logged_steps": len(rows),
            "per_reward": per_reward,
            "frac_steps_all_identical": idle,
            "note": ("Computed by tools/grpo_reward_trace.py from the per-step "
                     "rows in the run's own training log. mean_of_std is the "
                     "mean of each step's within-group standard deviation; a "
                     "reward whose std is always zero contributes no gradient."),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
