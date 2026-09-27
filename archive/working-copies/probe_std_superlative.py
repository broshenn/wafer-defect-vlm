"""Read-only probe: does the doc's per-reward-std superlative still hold?

The doc says, of the paper cell, that its group-internal reward std is the lowest of
the eight groups in 3 of the 4 rewards. That is a superlative over a population that
grew from seven RL runs to eight when run 41 landed, which is the exact shape of the
defect found an hour ago in the same document (a sentence claiming run 45 held the
highest macro-F1 after another run had taken that maximum). So it is computed here from
the records rather than read from the table, so that a table which agrees with its own
sentence cannot hide a sentence that no longer agrees with the runs.

The run -> result-file mapping is taken from tools/idle_step_table.py's RUNS, which is
the one place those two are declared together.
"""
import ast
import json
import pathlib
import sys

ROOT = pathlib.Path("/root/autodl-fs/wafer-vlm")
IDLE = ROOT / "tools/idle_step_table.py"
REP = ROOT / "outputs/reports"

# The doc's table column order, longest-name-first so prefix matching cannot collide.
DOC_ORDER = [
    ("GRPO(G=4) lr1e-5", "GRPO G=4 lr1e-5"),
    ("GRPO(G=4) lr5e-5", "GRPO G=4 lr5e-5"),
    ("GSPO(G=4) lr5e-5", "GSPO G=4 lr5e-5"),
    ("GSPO(G=4) lr1e-5", "GSPO G=4 lr1e-5"),
    ("GSPO(G=8) lr1e-5", "GSPO G=8 lr1e-5"),
    ("GSPO(G=8) lr5e-5", "GSPO G=8 lr5e-5"),
    ("GSPO(G=32) lr5e-5", "GSPO G=32 lr5e-5"),
    ("GSPO(G=32) lr1e-5", "GSPO G=32 lr1e-5"),
]

runs = []
for node in ast.parse(IDLE.read_text(encoding="utf-8")).body:
    if (isinstance(node, ast.Assign) and node.targets
            and getattr(node.targets[0], "id", None) == "RUNS"):
        for e in node.value.elts:
            lab, stem, _logs = ast.literal_eval(e)
            runs.append((lab, stem if stem.endswith(".json") else stem + ".json"))
by_label = dict(runs)

series = {}
for doc_lab, idle_lab in DOC_ORDER:
    stem = by_label.get(idle_lab)
    if not stem:
        sys.exit(f"no RUNS entry for {idle_lab!r}; the mapping changed, probe not run")
    p = REP / stem
    if not p.is_file():
        sys.exit(f"{stem} missing")
    series[doc_lab] = json.loads(p.read_text(encoding="utf-8"))["reward_signal"]

rewards = list(series[DOC_ORDER[0][0]].keys())
labels = [d for d, _ in DOC_ORDER]

print(f"{'reward':<14}" + "".join(f"{l.split('(')[0][-4:]+l.split('G=')[1][:2]+l[-5:]:>13}"
                                  for l in labels))
for r in rewards:
    vals = {l: series[l][r]["mean_std_across_steps"] for l in labels}
    mn = min(vals.values())
    print(f"{r:<14}" + "".join(f"{vals[l]:>13.4f}" for l in labels))

print()
for cell in ("GSPO(G=32) lr5e-5", "GSPO(G=32) lr1e-5"):
    wins = [r for r in rewards
            if abs(series[cell][r]["mean_std_across_steps"]
                   - min(series[l][r]["mean_std_across_steps"] for l in labels)) < 1e-12]
    print(f"{cell}: lowest of the {len(labels)} groups in {len(wins)} of {len(rewards)} "
          f"rewards -- {wins}")

print("\nper reward, who holds the minimum:")
for r in rewards:
    vals = {l: series[l][r]["mean_std_across_steps"] for l in labels}
    mn = min(vals.values())
    print(f"  {r:<14} {mn:.4f}  " + "、".join(l for l in labels
                                              if abs(vals[l] - mn) < 1e-12))
