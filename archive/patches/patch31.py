"""Add the two GSPO columns to the comparison table in 31_after_grpo.sh.

The table was built when GRPO was the only RL run. There are now two more:
GSPO at the paper's lr (5e-5) and GSPO at the lr the GRPO run used (1e-5), the
second existing solely to separate the algorithm from the learning rate -- the
5e-5 run changed both at once, so on its own it cannot attribute the drop.

Each `--run` is guarded by `[ -s ... ]`, so a run that has not been evaluated
yet is simply absent rather than a zero.
"""
import sys
from pathlib import Path

path = Path("/root/autodl-fs/wafer-vlm/projects/wafer-defect-vlm/scripts/31_after_grpo.sh")
src = path.read_text(encoding="utf-8")

old_header = 'step "2. comparison table (Base | SFT | GRPO)"'
new_header = 'step "2. comparison table (Base | SFT | GRPO | GSPO)"'

old_grpo = ('[ -s "$REPORTS/qwen35_9b_grpo__report.json" ]      '
            '&& RUN_ARGS+=(--run "GRPO=$REPORTS/qwen35_9b_grpo__report.json")\n')
new_grpo = old_grpo + (
    '# GSPO is not a separate algorithm in ms-swift: it is GRPO with\n'
    '# --importance_sampling_level sequence, so the recipe is otherwise identical.\n'
    '# Two runs. The first uses the paper\'s lr; the second uses the lr the GRPO run\n'
    '# used, because the 5e-5 run moved the algorithm and the lr together and so\n'
    '# cannot on its own say which one the change came from.\n'
    '[ -s "$REPORTS/gspo_lr5e5__report.json" ] && '
    'RUN_ARGS+=(--run "GSPO_lr5e5=$REPORTS/gspo_lr5e5__report.json")\n'
    '[ -s "$REPORTS/gspo_lr1e5__report.json" ] && '
    'RUN_ARGS+=(--run "GSPO_lr1e5=$REPORTS/gspo_lr1e5__report.json")\n')

for old, new in ((old_header, new_header), (old_grpo, new_grpo)):
    n = src.count(old)
    if n != 1:
        sys.exit(f"FAILED: {n} occurrences of {old[:70]!r}")
    src = src.replace(old, new)

path.write_text(src, encoding="utf-8")
print("31_after_grpo.sh patched: GSPO_lr5e5 + GSPO_lr1e5 columns added")
