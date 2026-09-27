"""Check 11 has never matched anything: the pattern looks for a table at column 0.

The landing for run 42 stopped at

    1 fact(s) about the run set have moved ...
      - 11: the table of runs whose record disagrees with its own log

and the message under it is the one that matters: *the sentence this check was written
against is not in the document any more; nothing was verified*. The table is in the
document. It is indented by four spaces, because it is inside a numbered item:

    | run | 实际累积步（该 run 自己的日志）| 记录写的 | |

`P_I11` requires the header at the start of a line -- `\\| run \\| ...` with nothing before
the bar -- so `re.search` finds no match anywhere, prints STALE, and exits 1. The four
spaces are the entire difference.

**This check has never run once.** `tools/patch118_grpo_train_fixes.py` wrote the checker
and regenerated the table it reads in the same run, and its own log prints the checker
exiting 1 with this same line -- followed by `[17:36:15]   ok` from its caller. So the
shape is section 8 item 16's first bullet again, one hour later: a mechanism that exists,
whose failure is printed as a warning by its caller, and which therefore never ran and
never stopped anything. Item 11's table has been read by nothing since the day both were
written -- which is exactly the sentence item 11 is about.

The fix is four characters of pattern: `\\s*` before the header and before the separator.
What the check reads is unchanged. It is not correct-by-assertion either -- this script
runs it afterwards and prints what the check now says about the table.
"""
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path("/root/autodl-fs/wafer-vlm")
PY = str(ROOT / "venvs/wafer/bin/python")
CHK = ROOT / "tools/check_quantified_claims.py"

OLD = (r'''P_I11 = (r"\| run \| 实际累积步（该 run 自己的日志）\| 记录写的 \| \|\s*\n"
         r"\| --- \| --- \| --- \| --- \|\s*\n"''')
NEW = (r'''# The table lives inside a numbered item, so every one of its lines is indented by four
# spaces -- and this pattern was written as though the table began at column 0. It has
# therefore never matched: patch118 wrote this pattern and the table it reads in the same
# run, and its own log shows the STALE line below, followed by the caller logging `ok`.
# Nothing has read item 11's table since. `\s*` before a line is the whole fix; what the
# check reads, and what it compares the table against, are unchanged.
P_I11 = (r"\s*\| run \| 实际累积步（该 run 自己的日志）\| 记录写的 \| \|\s*\n"
         r"\s*\| --- \| --- \| --- \| --- \|\s*\n"''')

s = CHK.read_text(encoding="utf-8")
if s.count(OLD) != 1:
    sys.exit(f"check 11's pattern matches {s.count(OLD)} times; nothing written")
shutil.copy2(CHK, "/tmp/check_quantified_claims.py.bak-patch151")
CHK.write_text(s.replace(OLD, NEW, 1), encoding="utf-8")
r = subprocess.run([PY, "-m", "py_compile", str(CHK)], capture_output=True, text=True)
if r.returncode != 0:
    shutil.copy2("/tmp/check_quantified_claims.py.bak-patch151", CHK)
    sys.exit(f"the patched checker does not compile, restored:\n{r.stderr}")
print("check 11's pattern now tolerates the indentation the table actually has")

# ------------------------------------------------- what does the check say now?
r = subprocess.run([PY, str(CHK)], capture_output=True, text=True, cwd=str(ROOT))
out = (r.stdout or "") + (r.stderr or "")
mine = [l for l in out.splitlines()
        if l.strip().startswith(("ok", "STALE", "11:")) or "11:" in l]
print(f"\nchecker exit {r.returncode}; the lines for item 11:")
for l in mine:
    if "11:" in l or "STALE" in l or l.strip().startswith("ok"):
        print("  " + l.rstrip()[:150])
if "the sentence this check was written against is not in the document" in out:
    sys.exit("check 11 is still STALE, so the pattern still does not match: nothing "
             "was gained and this patch should be reverted")
print("\ncheck 11 no longer reports STALE: the anchor matches and the table is compared")
print(f"(the checker as a whole exits {r.returncode}; a non-zero exit here means a fact "
      "about the run set moved, which is a finding, not a failure of this patch)")
