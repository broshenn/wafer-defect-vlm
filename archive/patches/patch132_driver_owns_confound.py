"""The landing must own the two steps that make the record it quotes true.

Run 42's record is now in `kl_length_confound.json` (patch130 declared the run in RUNS,
patch131 declared it a repeat of `grpo`). That happened by hand, at 17:15, because two of
the checker's facts turned red the moment the record grew -- and a red checker inside
`land_finish` stops the landing and does not restart the end-of-day chain behind it.

Doing it by hand once is not the same as the landing owning it. The checker has an item
that exists for exactly this dependency -- "the KL record covers every finished RL run" --
and it can only pass when something regenerates the record *before* the counts that read
it are re-synced. Until now the record's own docstring said it plainly: *"tools/
kl_length_confound.py has no caller; nothing regenerates it"*. So the landing calls it,
in `--pre`, which is where `land_run.sh` already puts the tools that consume a report and
must run before anything is written, and `--patch tools/patch101_kl_sentences.py` joins
the prose step so the sentences that quote the record are written by the landing too.

The driver is killed and relaunched rather than edited in place: it is a running bash
script, and bash reads a script by byte offset as it executes. This project broke two
training runs that way. It is only sitting in `while kill -0 3699; do sleep 20; done`
right now -- nothing is lost by killing it, and the relaunched one re-enters the same
wait.
"""
import pathlib
import shutil
import subprocess
import sys
import time

ROOT = pathlib.Path("/root/autodl-fs/wafer-vlm")
DRIVER = ROOT / "tools/land42_driver.sh"

OLD = '''log "step 3/3: land run 42 (no column; --pre seed_variance; --patch patch94_seed)"
if bash "$ROOT/tools/land_run.sh" --tag qwen35_9b_grpo_lr1e5_seed3408 --no-column \\
     --pre tools/seed_variance.py --patch tools/patch94_seed.py >> "$LOG" 2>&1; then'''

NEW = '''# step 3/3. The two --pre tools both have to run before anything is written, and the
# comment in land_run.sh says why --pre exists: they consume a record, and the sentences
# written below quote it. kl_length_confound is the one that was missing a caller: its own
# docstring said "no caller; nothing regenerates it", and the checker has an item that
# fails when the record is behind the run set. It is regenerated here so that this landing
# is what put the record in the state the landing's own counts read.
log "step 3/3: land run 42 (no column; --pre kl_length_confound, seed_variance; --patch patch94_seed, patch101_kl_sentences)"
if bash "$ROOT/tools/land_run.sh" --tag qwen35_9b_grpo_lr1e5_seed3408 --no-column \\
     --pre tools/kl_length_confound.py \\
     --pre tools/seed_variance.py \\
     --patch tools/patch94_seed.py \\
     --patch tools/patch101_kl_sentences.py >> "$LOG" 2>&1; then'''

# ------------------------------------------------- 1. stop the driver (it is waiting)
r = subprocess.run(["pgrep", "-f", "bash /root/autodl-fs/wafer-vlm/tools/land42_driver"],
                   capture_output=True, text=True)
pids = [p for p in r.stdout.split() if p.isdigit()]
print(f"land42_driver pid(s) before: {pids or 'none'}")
for p in pids:
    subprocess.run(["kill", p])
time.sleep(2)
r = subprocess.run(["ps", "-o", "pid=,cmd=", "-p", ",".join(pids)] if pids else ["true"],
                   capture_output=True, text=True)
alive = [l for l in r.stdout.splitlines() if "land42_driver" in l]
if alive:
    sys.exit(f"the driver is still alive after kill: {alive}; nothing written")
print("driver stopped; nothing it had done is lost (it was in its wait loop)")

# ------------------------------------------------------------- 2. the landing's steps
s = DRIVER.read_text(encoding="utf-8")
shutil.copy2(DRIVER, "/tmp/land42_driver.sh.bak-patch132")
if s.count(OLD) != 1:
    sys.exit(f"the driver anchor matches {s.count(OLD)} times; nothing written")
DRIVER.write_text(s.replace(OLD, NEW, 1), encoding="utf-8")
r = subprocess.run(["bash", "-n", str(DRIVER)], capture_output=True, text=True)
if r.returncode != 0:
    shutil.copy2("/tmp/land42_driver.sh.bak-patch132", DRIVER)
    sys.exit(f"the driver fails bash -n; restored:\n{r.stderr}")
print("land42_driver.sh: the landing now regenerates the confound record and writes the "
      "KL sentences.")

# ----------------------------------------------------------- 3. relaunch and confirm
if not (ROOT / "logs/land42_driver.log").is_file():
    sys.exit("logs/land42_driver.log is missing; relaunch by hand")
subprocess.Popen(["setsid", "nohup", "bash", str(DRIVER)],
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                 stdin=subprocess.DEVNULL, cwd=str(ROOT), start_new_session=True)
time.sleep(6)
r = subprocess.run(["pgrep", "-f", "bash /root/autodl-fs/wafer-vlm/tools/land42_driver"],
                   capture_output=True, text=True)
print(f"land42_driver pid(s) after relaunch: {r.stdout.split() or 'NOT RUNNING'}")
tail = (ROOT / "logs/land42_driver.log").read_text(encoding="utf-8").strip().splitlines()
for ln in tail[-4:]:
    print("  " + ln[:170])
r = subprocess.run(["ps", "-o", "pid=,etime=,cmd=", "-p", "3699"], capture_output=True, text=True)
print(f"run 42's queue: {r.stdout.strip()[:90] or 'GONE -- the landing will start'}")
