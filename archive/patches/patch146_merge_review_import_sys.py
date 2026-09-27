"""The patch that adds an error message could not print it: merge_review.py never imported sys.

patch145's new tail writes to stderr, and this module's imports are argparse, csv, json,
Counter and Path -- no sys. `python -m py_compile` passes anyway, because `sys.stderr` is
a valid attribute access on a name that simply does not exist yet; the failure is at
runtime, at the moment the new error path is taken, which is the moment it is needed. The
pre-flight run hit it immediately (the report printed, then NameError), which is the whole
argument for running the branch instead of reading it: this is a defect introduced by a
patch written *while recording that lesson*, in the same hour.

The fix is the import. Then the pre-flight is run again, and this time the new outcome,
the exclusion from the statistics and the exit code are all measured.
"""
import pathlib
import shutil
import subprocess
import sys

TOOL = pathlib.Path("/root/autodl-fs/wafer-vlm/tools/merge_review.py")

OLD = """import argparse
import csv
import json
from collections import Counter
from pathlib import Path
"""
NEW = """import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
"""

s = TOOL.read_text(encoding="utf-8")
if "\nimport sys\n" in s:
    sys.exit("merge_review.py already imports sys; nothing written")
if s.count(OLD) != 1:
    sys.exit(f"the import block matches {s.count(OLD)} times; nothing written")
shutil.copy2(TOOL, "/tmp/merge_review.py.bak-patch146")
TOOL.write_text(s.replace(OLD, NEW, 1), encoding="utf-8")

r = subprocess.run([sys.executable, "-m", "py_compile", str(TOOL)],
                   capture_output=True, text=True)
if r.returncode != 0:
    shutil.copy2("/tmp/merge_review.py.bak-patch146", TOOL)
    sys.exit(f"the patched file does not compile, restored:\n{r.stderr}")

# A compile check cannot see a missing name, so the import is exercised: the module is
# loaded and its new error path is taken with an out-of-vocabulary sheet.
probe = pathlib.Path("/tmp/probe_illegal_label.py")
probe.write_text(
    "import json, pathlib, subprocess, sys\n"
    "T = pathlib.Path('/tmp/illegal_probe'); T.mkdir(exist_ok=True)\n"
    "core = [json.loads(l) for l in open('/root/autodl-fs/wafer-vlm/benchmarks/"
    "wafer_bench_v1/core.jsonl', encoding='utf-8') if l.strip()][:3]\n"
    "(T/'core.jsonl').write_text('\\n'.join(json.dumps(r, ensure_ascii=False) for r in core)"
    " + '\\n', encoding='utf-8')\n"
    "hdr = 'sample_id,image_path,defect_class,radial_zone,clock_direction,"
    "confidence_0_100,notes'\n"
    "for who, lab in (('A', 'Edgeloc'), ('B', 'Edgeloc')):\n"
    "    with (T/f'{who}.csv').open('w', encoding='utf-8', newline='') as fh:\n"
    "        fh.write(hdr + '\\n')\n"
    "        for r in core:\n"
    "            fh.write(f\"{r['sample_id']},{r['image_path']},{lab},,,,\\n\")\n"
    "r = subprocess.run([sys.executable, '/root/autodl-fs/wafer-vlm/tools/merge_review.py',\n"
    "                    '--a', str(T/'A.csv'), '--b', str(T/'B.csv'),\n"
    "                    '--core', str(T/'core.jsonl')], capture_output=True, text=True)\n"
    "print('probe exit:', r.returncode)\n"
    "print('stderr tail:', (r.stderr or '').strip().splitlines()[-1][:150] if r.stderr"
    " else '(empty)')\n"
    "d = json.loads(r.stdout)\n"
    "print('outcomes:', d['outcomes'])\n"
    "print('listed:', len(d['out_of_vocabulary_samples']), 'sample(s)')\n",
    encoding="utf-8")
p = subprocess.run([sys.executable, str(probe)], capture_output=True, text=True)
print((p.stdout or "").strip())
if p.returncode != 0 or "probe exit: 3" not in (p.stdout or ""):
    print((p.stderr or "")[-400:])
    sys.exit("the vocabulary error path still does not run")
print("\nthe error path runs and returns 3")
