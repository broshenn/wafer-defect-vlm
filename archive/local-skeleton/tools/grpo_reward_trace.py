"""Print the per-step reward breakdown of a GRPO run from its training log.

The point of looking is the standard deviation, not the mean. With four
generations per prompt, a reward whose per-group std is zero gives every
completion in the group the same score, so the advantage is zero and that
reward contributes no gradient at all. A run where three of four rewards are
flat is a run that only trained on the fourth, however good its mean looks.
"""

from __future__ import annotations

import argparse
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

    print("\nmean/std per reward over all logged steps:")
    for name in ("WaferClass", "WaferFormat", "WaferRadial", "WaferClock"):
        means = [v for v in (value(f"rewards/{name}/mean", r) for r in rows) if v is not None]
        stds = [v for v in (value(f"rewards/{name}/std", r) for r in rows) if v is not None]
        carries = any(s > 0 for s in stds)
        print(f"  {name:14s} mean={sum(means)/len(means):.4f}  "
              f"mean_of_std={sum(stds)/len(stds):.4f}  carries_group_signal={carries}")

    zero = [v for v in (value("frac_reward_zero_std", r) for r in rows) if v is not None]
    if zero:
        print(f"\nfraction of steps whose groups ALL had identical rewards: "
              f"{sum(1 for v in zero if v >= 1) / len(zero):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
