"""Include run 43 in the KL/length confound record.

LIMITATIONS.md quotes this record for a claim about *all* runs ("the identity holds
on every idle step of the five runs, 54+49+23+57+63"). The run list is hard-coded, so
run 43 landing did not extend it -- the numbers stayed right and their scope silently
stopped being "all runs". That is the same defect class as the stale "全部 run" claims
in 5.2.5, so the list is extended and the record regenerated rather than reworded.
"""
import ast
import sys
from pathlib import Path

P = Path("/root/autodl-fs/wafer-vlm/tools/kl_length_confound.py")
s = P.read_text(encoding="utf-8")

OLD = '''RUNS = ["grpo", "qwen35_9b_grpo_lr5e5", "gspo_lr1e5", "qwen35_9b_gspo_v1",
        "qwen35_9b_gspo_g32"]'''
NEW = '''# Every RL run belongs here. A hard-coded list that a new run is not added to does
# not make the record wrong -- it makes it narrower than the sentences that quote it,
# which say "all runs". Adding a run means adding it here and regenerating.
RUNS = ["grpo", "qwen35_9b_grpo_lr5e5", "gspo_lr1e5", "qwen35_9b_gspo_v1",
        "qwen35_9b_gspo_g32", "qwen35_9b_gspo_g4_lr5e5"]'''
if s.count(OLD) != 1:
    sys.exit(f"run list anchor appears {s.count(OLD)} times")
s = s.replace(OLD, NEW)
ast.parse(s)
P.write_text(s, encoding="utf-8")
print("kl_length_confound.py: run 43 added to RUNS")
