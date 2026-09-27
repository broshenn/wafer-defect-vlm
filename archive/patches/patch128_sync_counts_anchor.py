"""`tools/sync_counts.py` has been a no-op, and the warning that said so was a WARNING.

Found by asking it what it would change, not by a check: it exits on its FIRST anchor
(`8.9: idle-step total and per-run list`, matched 0 times) before reaching any rewrite, so
every count it owns -- the RL record count, the verified-rate list, the per-run idle total
and its terms, 5.2.5's extremum counts -- has been frozen at whatever the last successful
run wrote. The document says 7 RL records while the audit tool checks 8, and the two
sentences never appear together in any check, which is why it stayed green.

The anchor drifted by one space. The tool's pattern is `\\d+( 个空转步均满足`, the document
now reads `347个空转步`. Spacing is not the fact here -- the number is -- so the pattern
accepts either (`\\d+\\s*`), and the rewrite re-emits one consistent form.

**Why this had to be fixed before run 42 lands.** `land_run.sh` step 6 runs
`land_finish.sh`, whose step 2 is the checker. Run 42 adds a ninth verified rate, so §9's
enumeration has to grow, and the checker compares that sentence against the tool's rows.
With sync_counts dead, nothing would grow it, the checker would report a moved fact, and
the landing would stop with the column in and the prose not -- and would not restart the
end-of-day chain, which is sequenced behind it. A frozen counter would have taken the
whole evening down, and the only trace would have been one WARNING line.

So `land_finish.sh` no longer treats that failure as a warning either: it stops. The
distinction this project keeps rediscovering is that a lagging count and a failed count
look identical in a log line, and only one of them is safe to continue past.
"""
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path("/root/autodl-fs/wafer-vlm")
SYNC = ROOT / "tools/sync_counts.py"
FINISH = ROOT / "tools/land_finish.sh"

# --------------------------------------------------- 1. the pattern accepts both spacings
OLD = '''    sub(r"[一二三四五六七八九十]+(个 run 的全部 )\\d+"
        r"( 个空转步均满足（逐 run 实测为\\s*\\n?\\s*)[\\d+]+(；)",'''
NEW = '''    # `\\d+\\s*` on purpose: the numeral is what this sentence is about, and it has been
    # written both `347 个` and `347个`. Requiring one spacing made the whole tool exit on
    # its first anchor -- writing nothing at all, including the counts below -- while
    # land_finish.sh reported that as a WARNING.
    sub(r"[一二三四五六七八九十]+(个 run 的全部 )\\d+\\s*"
        r"(个空转步均满足（逐 run 实测为\\s*\\n?\\s*)[\\d+]+(；)",'''
s = SYNC.read_text(encoding="utf-8")
shutil.copy2(SYNC, "/tmp/sync_counts.py.bak-patch128")
if s.count(OLD) != 1:
    sys.exit(f"the sync_counts anchor matches {s.count(OLD)} times; nothing written")
SYNC.write_text(s.replace(OLD, NEW, 1), encoding="utf-8")
r = subprocess.run([sys.executable, "-m", "py_compile", str(SYNC)],
                   capture_output=True, text=True)
if r.returncode != 0:
    shutil.copy2("/tmp/sync_counts.py.bak-patch128", SYNC)
    sys.exit(f"sync_counts does not compile; restored:\n{r.stderr}")
print("sync_counts.py: the idle-step anchor no longer depends on one spacing.")

# ------------------------------------------------- 2. land_finish stops rather than warns
F_OLD = '''"$PY" "$ROOT/tools/sync_counts.py" || echo "WARNING: sync_counts failed; the derived counts may lag"'''
F_NEW = '''if ! "$PY" "$ROOT/tools/sync_counts.py"; then
  printf '\\nSTOP: sync_counts refused, so none of the derived counts or the section-9\\n'
  printf 'rate list was written. It exits before writing anything if one of its anchors\\n'
  printf 'stops matching, which is what happened today: it had been failing on its first\\n'
  printf 'anchor for an unknown length of time while this line called it a WARNING. A\\n'
  printf 'count that lags and a count that failed to be written look the same in a log;\\n'
  printf 'only one of them is safe to continue past, and this is not it.\\n'
  exit 1
fi'''
f = FINISH.read_text(encoding="utf-8")
shutil.copy2(FINISH, "/tmp/land_finish.sh.bak-patch128")
if f.count(F_OLD) != 1:
    sys.exit(f"the land_finish anchor matches {f.count(F_OLD)} times; "
             f"sync_counts is fixed, land_finish is not")
FINISH.write_text(f.replace(F_OLD, F_NEW, 1), encoding="utf-8")
r = subprocess.run(["bash", "-n", str(FINISH)], capture_output=True, text=True)
if r.returncode != 0:
    shutil.copy2("/tmp/land_finish.sh.bak-patch128", FINISH)
    sys.exit(f"land_finish.sh fails bash -n; restored:\n{r.stderr}")
print("land_finish.sh: a sync_counts refusal now stops the landing.")

# ------------------------- 3. ask it what it would change; report every anchor it hits
for label in ("check", "check again"):
    r = subprocess.run([sys.executable, str(SYNC), "--check"], capture_output=True,
                       text=True, cwd=str(ROOT))
    out = (r.stdout or "").strip() + (r.stderr or "").strip()
    print(f"\n--- sync_counts --check ({label}): exit {r.returncode} ---")
    for ln in out.splitlines()[-6:]:
        print("  " + ln.strip()[:170])
