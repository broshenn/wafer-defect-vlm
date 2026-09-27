"""44_final_consolidation.sh waits for eleven reports, and the run set now has twelve.

The script's own docstring is the argument for this patch: "the run set here is not a
list" -- it exists because every queue script rebuilt the comparison from a list that was
correct when written and silently incomplete afterwards. Then it waits for a hand-written
list of eleven report names, and the twelfth (the third draw of GRPO G=4 lr 1e-5, the run
queue 43 is training tonight) is not in it. The rebuild is enumerated, so the missing
entry does not drop a column; what it does is narrower and still worth fixing: 44 will not
*wait* for the third seed's report, and nothing in the output distinguishes that from
having waited for it.

So the third seed's report is added, and the list is checked rather than trusted: every
tag `run_set.py` accounts for must appear in REQUIRED. A run added to NAMES without being
added here is then a printed failure instead of a slower wait, which is the same inversion
run_set.py makes for reports that appear in neither NAMES nor EXCLUDE -- a run is dropped
by a decision that is printed, never by a list that went stale.

The check only adds an error path. It sets the script's own FAILED flag and keeps going,
so a rebuild still happens over a partial set that says so (the script's stated
preference); what it withholds is the completion marker that the next watcher keys on,
which is how a loud stop differs from a quiet one.

Tested here in both directions, on the running machine: the guard passes against the real
run_set.py, and fails when handed a list with one entry removed. `bash -n` runs after the
edit -- never on a live file, and this script is not running (44 is stopped until the
landing restarts it).
"""
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path("/root/autodl-fs/wafer-vlm")
SH = ROOT / "projects/wafer-defect-vlm/scripts/44_final_consolidation.sh"
PY = str(ROOT / "venvs/wafer/bin/python")

OLD_ITEM = "  qwen35_9b_grpo_lr1e5_seed3408__report.json\n"
NEW_ITEM = ("  qwen35_9b_grpo_lr1e5_seed3408__report.json\n"
            "  qwen35_9b_grpo_lr1e5_seed3409__report.json\n")

OLD_END = "  qwen35_9b_gspo_g4_lr1e5__report.json\n)\n"

GUARD = '''  qwen35_9b_gspo_g4_lr1e5__report.json
)

# The list above is itself a list, which is what this script exists to stop trusting.
# What it waits for is checked against what run_set.py accounts for, so a run added to
# NAMES cannot be left out of the wait silently: it was, for the third seed. This adds an
# error path only -- a partial rebuild still happens and still says so, but the completion
# marker the next watcher keys on is withheld, which is the difference between this
# stopping loudly and this stopping quietly.
if ! "$PY" - "$ROOT" "${REQUIRED[@]}" <<'GUARDPY'
import json
import pathlib
import subprocess
import sys

root = pathlib.Path(sys.argv[1])
have = set(sys.argv[2:])
r = subprocess.run([sys.executable, str(root / "tools/run_set.py"),
                    "--root", str(root), "--emit", "json"],
                   capture_output=True, text=True)
if r.returncode != 0 and not r.stdout.strip():
    print("FATAL: run_set.py could not be asked what the run set is:")
    print((r.stderr or "").strip()[-400:])
    sys.exit(1)
d = json.loads(r.stdout)
tags = [e["tag"] for e in d["available"] + d["expected_but_absent"]]
missing = [f"{t}__report.json" for t in tags if f"{t}__report.json" not in have]
if missing:
    print(f"FATAL: run_set.py accounts for {len(tags)} run(s); this script waits for "
          f"{len(have)} and does not wait for: " + ", ".join(missing))
    print("  Add them to REQUIRED. Until then the completion marker is withheld.")
    sys.exit(1)
print(f"the wait list covers every run run_set.py accounts for ({len(tags)})")
GUARDPY
then
  echo "wait list check: covers the enumerated run set"
else
  echo "FATAL: the wait list does not cover the enumerated run set (see above)"
  FAILED=1
fi
'''

s = SH.read_text(encoding="utf-8")
if "seed3409__report.json" in s:
    sys.exit("the third seed is already in REQUIRED; nothing written")
if s.count(OLD_ITEM) != 1:
    sys.exit(f"the seed-3408 entry matches {s.count(OLD_ITEM)} times; nothing written")
if s.count(OLD_END) != 1:
    sys.exit(f"the array's end matches {s.count(OLD_END)} times; nothing written")

shutil.copy2(SH, "/tmp/44_final_consolidation.sh.bak-patch143")
s = s.replace(OLD_ITEM, NEW_ITEM, 1).replace(OLD_END, GUARD, 1)
SH.write_text(s, encoding="utf-8")

r = subprocess.run(["bash", "-n", str(SH)], capture_output=True, text=True)
if r.returncode != 0:
    shutil.copy2("/tmp/44_final_consolidation.sh.bak-patch143", SH)
    sys.exit(f"the patched 44 does not parse, restored:\n{r.stderr}")
print("44_final_consolidation.sh: the third seed is waited for, and the list is checked")

# ------------------------------------------- and run the guard, both directions
GUARD_PY = pathlib.Path("/tmp/guard_wait_list.py")
src = SH.read_text(encoding="utf-8")
i, j = src.index("<<'GUARDPY'\n") + len("<<'GUARDPY'\n"), src.index("\nGUARDPY\n")
GUARD_PY.write_text(src[i:j] + "\n", encoding="utf-8")


def run_guard(label, required):
    r = subprocess.run([PY, str(GUARD_PY), str(ROOT), *required],
                       capture_output=True, text=True)
    print(f"\n--- guard [{label}]: exit {r.returncode} ---")
    for ln in ((r.stdout or "") + (r.stderr or "")).strip().splitlines()[:4]:
        print("  " + ln[:170])
    return r


real = subprocess.run(["bash", "-c",
                       f'set -uo pipefail; R=(); while IFS= read -r l; do '
                       f'R+=("$l"); done < <(sed -n "/^REQUIRED=(/,/^)/p" {SH} '
                       f'| sed -n "s/^\\([^()]*\\.json\\)$/\\1/p"); '
                       f'printf "%s\\n" "${{R[@]}}"'],
                      capture_output=True, text=True)
required = [x for x in real.stdout.split() if x]
print(f"\nREQUIRED as the script now reads it: {len(required)} entries")

ok = run_guard("the real list", required)
if ok.returncode != 0:
    sys.exit("the guard fails against the real run set: the chain would stop here")
bad = run_guard("one entry removed", [x for x in required if "seed3409" not in x])
if bad.returncode == 0:
    sys.exit("the guard passed a list that omits a run run_set.py accounts for: it does "
             "not detect the defect it was added for")
print("\nthe guard passes the real list and fails the one-entry-short list")
