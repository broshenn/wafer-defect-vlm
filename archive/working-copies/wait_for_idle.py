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
import os
import time
from pathlib import Path

SUBSTR = [
    "swift/cli/rlhf.py",              # training (module invocation)
    "--gradient_accumulation_steps",  # training (either invocation)
    "24_train_grpo.sh",               # the inner training wrapper
    "30_eval_grpo.sh",                # evaluation / scoring
]

EXACT = [
    "bash projects/wafer-defect-vlm/scripts/41_queue_gspo_g32_lr1e5.sh",
    "bash projects/wafer-defect-vlm/scripts/42_queue_grpo_seed2.sh",
]


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
