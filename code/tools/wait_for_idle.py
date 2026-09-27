"""Wait until the card is idle AND every expected report exists, then report state.

Why this is Python and not a shell loop: scripts are pushed and launched as a base64
blob on the command line, so the launcher's own shell holds the script's *path* in
its command line. A `pgrep -f '41_queue...'` therefore matches the shell that
launched that queue -- it never clears. /proc inspection from Python cannot match
itself, and the two matchers below are chosen so a launcher cannot impersonate work:

  * SUBSTR patterns are argument signatures that only a real worker carries
    (swift's module path, an eval script invoked as an argument). A launcher's
    command line holds a base64 blob, which cannot contain them.
  * EXACT patterns are whole command lines of the queue drivers. Those must be
    matched exactly, because a launcher's longer command line contains the driver's
    path as a substring but is not equal to it. The drivers are waited for because
    they rebuild comparison.json and FINAL_REPORT.md at the end, and racing them
    would leave whichever finished second silently overwriting the other.

Exits 0 in both the complete and the timed-out case, printing a JSON object as its
last line saying which happened. A consolidation that refuses to run because one run
is late is worse than one that runs and reports what was missing.
"""
import argparse
import json
import pathlib
import os
import time
from pathlib import Path

SUBSTR = [
    "swift/cli/rlhf.py",              # training (module invocation)
    "--gradient_accumulation_steps",  # training (either invocation)
    "24_train_grpo.sh",               # the inner training wrapper
    "30_eval_grpo.sh",                # evaluation / scoring
]

# Enumerated from disk rather than hand-written. The hand-written version held two
# command lines and was correct when written; queue 43 was launched while 44 was already
# waiting, so 44 would have started rewriting LIMITATIONS.md during that queue's own
# landing -- the race this waiter exists to prevent, reintroduced by a list that went
# stale. Both forms are generated because the launcher may use either, and a queue
# script added tomorrow is covered without anyone remembering this line.
# Both spellings: `/root/autodl-fs` is a symlink to `/autodl-fs/data` here, and a shell
# invoked through either one carries that spelling in its command line. patch139 used
# `resolve()` alone, derived `/autodl-fs/data/...` while queue 43 was running as
# `/root/autodl-fs/...`, and recognised nothing -- a waiter that enumerates the right
# script under the wrong name is the list-shaped failure it was meant to replace.
QUEUE_DIRS = []
for _f in (pathlib.Path(__file__), pathlib.Path(__file__).resolve()):
    _d = _f.parent.parent / "projects/wafer-defect-vlm/scripts"
    if _d not in QUEUE_DIRS:
        QUEUE_DIRS.append(_d)
EXACT = [
    "bash projects/wafer-defect-vlm/scripts/41_queue_gspo_g32_lr1e5.sh",
    "bash projects/wafer-defect-vlm/scripts/42_queue_grpo_seed2.sh",
]
for _d in QUEUE_DIRS:
    for _q in sorted(_d.glob("*_queue_*.sh")):
        EXACT.append(f"bash {_q}")
        try:
            EXACT.append(f"bash {_q.relative_to(_d.parent.parent)}")
        except ValueError:
            pass
EXACT = list(dict.fromkeys(EXACT))


def cmdlines():
    me = os.getpid()
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) == me:
            continue
        try:
            raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        except OSError:
            continue
        if not raw:
            continue
        yield int(pid), raw.replace(b"\0", b" ").decode("utf-8", "replace").strip()


def busy():
    """Live matches, excluding this process's own ancestry."""
    mine = {os.getpid(), os.getppid()}
    hits = []
    for pid, c in cmdlines():
        if pid in mine:
            continue
        for pat in SUBSTR:
            if pat in c:
                hits.append((pid, f"substr:{pat}", c))
                break
        else:
            if c in EXACT:
                hits.append((pid, f"exact:{c}", c))
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/root/autodl-fs/wafer-vlm")
    ap.add_argument("--timeout", type=int, default=14400)
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--require", nargs="*", default=[],
                    help="report filenames that must exist and be non-empty, "
                         "relative to outputs/reports")
    a = ap.parse_args()

    rep = Path(a.root) / "outputs/reports"
    start = time.time()
    last_note = -1
    while True:
        h = busy()
        missing = [n for n in a.require
                   if not (rep / n).is_file() or (rep / n).stat().st_size == 0]
        waited = int(time.time() - start)
        if not h and not missing:
            print(json.dumps({"idle": True, "waited_seconds": waited,
                              "missing_reports": [], "still_running": []},
                             ensure_ascii=False))
            return
        if waited >= a.timeout:
            print(f"  TIMEOUT after {waited}s: {len(h)} process(es) up, "
                  f"{len(missing)} report(s) missing", flush=True)
            for pid, pat, c in h[:6]:
                print(f"    pid {pid} [{pat}] {c[:110]}", flush=True)
            if missing:
                print(f"    missing: {missing}", flush=True)
            print(json.dumps({"idle": False, "waited_seconds": waited,
                              "missing_reports": missing,
                              "still_running": [{"pid": p, "matched": m}
                                                for p, m, _ in h]},
                             ensure_ascii=False))
            return
        if waited // 600 != last_note:
            last_note = waited // 600
            print(f"  waiting ({waited}s): {len(h)} process(es), "
                  f"{len(missing)} report(s) missing"
                  + (f" {[x[1] for x in h][:3]}" if h else ""), flush=True)
        time.sleep(a.poll)


if __name__ == "__main__":
    main()
