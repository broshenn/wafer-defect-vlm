"""Make the reward breakdown persist, not just print.

LIMITATIONS quotes the per-reward means (WaferFormat 0.9998, WaferRadial 0.8650,
WaferClock 0.6283) as the evidence that the clock reward carries the training
signal. tools/audit_report_numbers.py could not trace any of them to a record:
the tool that computes them prints a table and exits, so the only copy of those
numbers was in the prose that cited them.

This adds an optional --output that writes the same numbers as a record, so the
claim can be re-checked against the log it came from. The printed table is
unchanged, so nothing downstream reads differently.
"""
import ast
import sys
from pathlib import Path

TOOLS = Path("/root/autodl-fs/wafer-vlm/tools")
p = TOOLS / "grpo_reward_trace.py"
s = p.read_text(encoding="utf-8")
edits = []

old = '''import argparse
import re
from pathlib import Path
'''
new = '''import argparse
import json
import re
from pathlib import Path
'''
edits.append(("import json", old, new))

old = '''    parser.add_argument("--every", type=int, default=10)
    args = parser.parse_args()
'''
new = '''    parser.add_argument("--every", type=int, default=10)
    parser.add_argument("--output", default=None,
                        help="write the per-reward table to this JSON record")
    args = parser.parse_args()
'''
edits.append(("add --output", old, new))

old = '''    print("\\nmean/std per reward over all logged steps:")
    for name in ("WaferClass", "WaferFormat", "WaferRadial", "WaferClock"):
        means = [v for v in (value(f"rewards/{name}/mean", r) for r in rows) if v is not None]
        stds = [v for v in (value(f"rewards/{name}/std", r) for r in rows) if v is not None]
        carries = any(s > 0 for s in stds)
        print(f"  {name:14s} mean={sum(means)/len(means):.4f}  "
              f"mean_of_std={sum(stds)/len(stds):.4f}  carries_group_signal={carries}")

    zero = [v for v in (value("frac_reward_zero_std", r) for r in rows) if v is not None]
    if zero:
        print(f"\\nfraction of steps whose groups ALL had identical rewards: "
              f"{sum(1 for v in zero if v >= 1) / len(zero):.3f}")
    return 0
'''
new = '''    per_reward = {}
    print("\\nmean/std per reward over all logged steps:")
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
        print(f"\\nfraction of steps whose groups ALL had identical rewards: {idle:.3f}")

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
        print(f"\\nwrote {args.output}")
    return 0
'''
edits.append(("persist the table", old, new))

for tag, old, new in edits:
    n = s.count(old)
    if n != 1:
        sys.exit(f"FAILED at {tag}: {n} occurrences (need exactly 1)")
    s = s.replace(old, new)
    print(f"  {tag}: ok")

ast.parse(s)
p.write_text(s, encoding="utf-8")
print("reward breakdown is now recorded, not only printed")
