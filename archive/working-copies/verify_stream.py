"""Prove the added contrast left every pre-existing recorded value untouched.

The check that matters is not "the doc's number looks right again" but "the record
the tool now produces is identical to the record it produced before the block
existed, except for the one new key". That is testable: strip the added block,
re-run, and diff.

The stripped copy is the pre-patch tool, so its output IS the original record.
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

R = Path("/root/autodl-fs/wafer-vlm")
TOOL = R / "tools/paired_significance.py"
REC = R / "outputs/reports/paired_significance.json"
PY = str(R / "venvs/wafer/bin/python")

src = TOOL.read_text(encoding="utf-8")

# Remove the whole added block: from its `for` line through its print line.
start = src.index('for a, b in (("GRPO_G4_lr5e5", "GSPO_G32_lr5e5"),):')
end = src.index('f"CI=[{lo:+.4f},{hi:+.4f}] n01={n01} n10={n10} p={p:.5f}")',
                start)
end = src.index("\n", end) + 1
stripped = src[:start] + src[end:]

if "paper_setting_paired" not in src or "paper_setting_paired" in stripped:
    sys.exit("strip failed: the marker is not confined to the added block")

Path("/tmp/ps_noblock.py").write_text(stripped, encoding="utf-8")

current = REC.read_text(encoding="utf-8")
shutil.copy(REC, "/tmp/ps_current.json")

r = subprocess.run([PY, "/tmp/ps_noblock.py"], capture_output=True, text=True)
print(f"stripped tool exit={r.returncode}")
if r.returncode != 0:
    print(r.stdout[-2000:], r.stderr[-2000:])
    sys.exit("stripped run failed")

original = json.loads(REC.read_text(encoding="utf-8"))          # pre-patch record
now = json.loads(Path("/tmp/ps_current.json").read_text(encoding="utf-8"))

print()
print("=== pre-patch record vs post-patch record ===")
print(f"  keys pre : {sorted(original)}")
print(f"  keys post: {sorted(now)}")
added = sorted(set(now) - set(original))
removed = sorted(set(original) - set(now))
changed = [k for k in sorted(set(original) & set(now))
           if json.dumps(original[k], sort_keys=True)
           != json.dumps(now[k], sort_keys=True)]
print(f"  added   : {added}")
print(f"  removed : {removed or 'none'}")
print(f"  changed : {changed or 'none'}")
print()
if changed or removed:
    for k in changed:
        print(f"  !! {k}")
        print(f"     pre : {json.dumps(original[k], sort_keys=True)[:300]}")
        print(f"     post: {json.dumps(now[k], sort_keys=True)[:300]}")
    sys.exit("FAIL: the added block perturbed pre-existing values")
if added != ["paper_setting_paired"]:
    sys.exit(f"FAIL: unexpected added keys: {added}")
print("PASS: every pre-existing key is byte-identical; only "
      "paper_setting_paired was added")

# Put the record back the way the real tool writes it.
r = subprocess.run([PY, str(TOOL)], capture_output=True, text=True)
print(f"\nreal tool re-run exit={r.returncode}")
if r.returncode != 0:
    print(r.stdout[-2000:], r.stderr[-2000:])
    sys.exit("real tool failed")
final = json.loads(REC.read_text(encoding="utf-8"))
ok = all(json.dumps(final.get(k), sort_keys=True)
         == json.dumps(now.get(k), sort_keys=True) for k in now)
print(f"record now carries both: {ok}")
print(json.dumps(final["paper_setting_paired"], ensure_ascii=False, indent=2))
