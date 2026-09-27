"""Move the item-11 table check out of the checker and into the patch that rewrites the
table, so the guard arrives with the artifact it guards.

patch123 added it, patch124 made it shape-agnostic, and it was still reporting its absent
branch -- the pattern wanted rows beginning `| G=` where the table's rows begin
`| GSPO G=8 lr1e-5`. Three attempts at the same read, each one green or absent for a
different reason, and the first two were *my* errors in the pattern rather than anything
about the document. That is the signal to stop: the item depends on a table that
patch118 replaces wholesale, so an item written against the old shape is work that
patch118 will delete, and an item written against the new shape is a checker that is red
until a patch that cannot run yet.

What stays here is the other half, because it is green and independent of the table: the
「每个 run 的梯度累积都设成了与组大小相等」 claim, now read from every run's own log. That
one is worth having today -- it is the claim the whole effective-batch argument rests on,
and it is now inside the quantifier for all nine runs rather than the seven that existed
when it was written.

What moves into patch118 is the table check, in the shape patch118 creates: the
regenerated table has a verdict column, and it is generated from the same RUNS list the
check reads, so the check is meaningful the moment the table exists.
"""
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path("/root/autodl-fs/wafer-vlm")
CHECKER = ROOT / "tools/check_quantified_claims.py"
CB = "/tmp/check_quantified_claims.py.bak-patch125"

START = "# Item 11's table of wrong records"
END = "5.2.5: the idle-step rate, its worst run, and its population"

c = CHECKER.read_text(encoding="utf-8")
shutil.copy2(CHECKER, CB)
lines = c.splitlines(keepends=True)
starts = [i for i, ln in enumerate(lines) if ln.startswith(START)]
ends = [i for i, ln in enumerate(lines) if END in ln]
if len(starts) != 1 or len(ends) != 1 or ends[0] <= starts[0]:
    sys.exit(f"block bounds ambiguous (start {len(starts)}, end {len(ends)}); nothing "
             f"written")
# Drop from the block's opening comment line up to (not including) the line that
# introduces the next block, but keep any blank separator that precedes it.
lo, hi = starts[0], ends[0]
while lo > 0 and lines[lo].strip() == "":
    lo -= 1
    break
del lines[lo:hi]
CHECKER.write_text("".join(lines), encoding="utf-8")

r = subprocess.run([sys.executable, "-m", "py_compile", str(CHECKER)],
                   capture_output=True, text=True)
if r.returncode != 0:
    shutil.copy2(CB, CHECKER)
    sys.exit(f"does not compile; restored:\n{r.stderr}")
print(f"removed {hi - lo} line(s) of the item-11 read; the accumulation claim stays.")

r = subprocess.run([sys.executable, str(CHECKER)], capture_output=True, text=True,
                   cwd=str(ROOT))
out = (r.stdout or "").strip().splitlines()
print("\n--- the accumulation item ---")
for ln in out:
    if "gradient accumulation" in ln.strip() or "grad_accum" in ln.strip():
        print("  " + ln.strip()[:150])
n_stale = sum(1 for ln in out if ln.strip().startswith("STALE"))
for i, ln in enumerate(out):
    if ln.strip().startswith("STALE"):
        for nx in out[i:i + 3]:
            print("  " + nx.rstrip())
print("--- tail ---")
for ln in out[-3:]:
    print("  " + ln.strip())
print(f"\nchecker exit {r.returncode}; {n_stale} stale item(s)")
