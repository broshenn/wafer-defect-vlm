"""The audit read a quoted line of bash as an unfilled template field and stopped the pass.

`tools/audit_report_numbers.py` refuses when it finds an unfilled template field, on the
reasoning that a field is not a wrong number but *no* number, in a sentence that reads as
though it had one. At 17:36:45 that rule fired on LIMITATIONS.md:1028 and stopped run 42's
landing:

    LIMITATIONS.md:1028  {RUN_TAG:?…}

The line is evidence, not a defect. LIMITATIONS.md quotes the wrapper's own line --
`` `RUN_TAG="${RUN_TAG:?…}"` `` -- because that line is the reason the third seed's run was
lost, and it is worth quoting verbatim. The check saw `{RUN_TAG:?…}` and read it as a
template field nobody filled in.

**Narrowing a check so it stops firing on what is in front of it is how a check dies**, so
the rule is narrowed at two points, each with an independent reason, and the narrowing is
measured in both directions by this script's own probe:

  * `(?<!\\$)` -- a `${...}` expansion is not a template field. Bash's `${VAR:-default}` and
    `${VAR:?message}` share the braces and the colon with `str.format`, and the leading `$`
    is what says which one it is.
  * `(?::[^}?][^}]*)?` -- the first character after the colon may not be `?`. No format spec
    has ever begun with `?` (fill/align/sign/width/precision/type), so this costs the check
    nothing, while `:?` is exactly the bash operator that spans prose.

The probe writes two fake document roots and runs this same audit against them:
  * a genuine unfilled field, plain and inside a code span -- must still be counted and must
    still exit non-zero;
  * the quoted bash line, and `${VAR:-4}` -- must count zero and exit zero.
"""
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path("/root/autodl-fs/wafer-vlm")
PY = str(ROOT / "venvs/wafer/bin/python")
AUDIT = ROOT / "tools/audit_report_numbers.py"

OLD = ('PLACEHOLDER = re.compile(r"\\{[A-Za-z_][A-Za-z0-9_]*(?::[^}]*)?\\}")\n')
NEW = ('PLACEHOLDER = re.compile(r"(?<!\\$)\\{[A-Za-z_][A-Za-z0-9_]*(?::[^}?][^}]*)?\\}")\n'
       '\n'
       '# The two guards above are load-bearing and each was added for a different reason.\n'
       '# `(?<!\\$)` : a `${...}` is an expansion, not a template field. Bash\'s\n'
       '# `${VAR:-default}` and `${VAR:?message}` share the braces and the colon with\n'
       '# `str.format`, and the leading `$` is what says which one it is.\n'
       '# `[^}?]` after the colon : no format spec begins with `?`, and `:?` is the bash\n'
       '# operator whose word is prose. On 2026-09-16 this rule fired on LIMITATIONS.md:1028,\n'
       '# which quotes the wrapper line `RUN_TAG="${RUN_TAG:?…}"` because that line is why a\n'
       '# run was lost -- and stopped run 42\'s landing, on a quotation. A rule that reads\n'
       '# the document\'s evidence as a defect in the document stops being run.\n')

s = AUDIT.read_text(encoding="utf-8")
if s.count(OLD) != 1:
    sys.exit(f"the placeholder rule matches {s.count(OLD)} times; nothing written")
shutil.copy2(AUDIT, "/tmp/audit_report_numbers.py.bak-patch150")
AUDIT.write_text(s.replace(OLD, NEW, 1), encoding="utf-8")
r = subprocess.run([PY, "-m", "py_compile", str(AUDIT)], capture_output=True, text=True)
if r.returncode != 0:
    shutil.copy2("/tmp/audit_report_numbers.py.bak-patch150", AUDIT)
    sys.exit(f"the patched audit does not compile, restored:\n{r.stderr}")
print("the placeholder rule is narrowed at two points, each with its own reason")

# ------------------------------------------------------------------ the probe
PROBE = pathlib.Path("/tmp/audit_probe")
GENUINE = ("the mean reward over the run was {mean_reward} across all steps\n"
           "and the agreement the two reviewers reached was `{kappa:.4f}` on the sheets\n")
BASH = ('  the line becomes `RUN_TAG="${RUN_TAG:?…}"`, which fails immediately\n'
        'and says why; the default — a fixed name — is what let one run overwrite\n'
        'another run’s artefacts, see `${GRAD_ACCUM:-4}` for the same reading.\n')


def run_case(label, text, expect):
    d = PROBE / label
    (d / "outputs" / "reports").mkdir(parents=True, exist_ok=True)
    (d / "FINAL_REPORT.md").write_text("# report\n\nno numbers here\n", encoding="utf-8")
    (d / "LIMITATIONS.md").write_text(text, encoding="utf-8")
    (d / "outputs" / "reports" / "comparison.md").write_text("# c\n", encoding="utf-8")
    r = subprocess.run([PY, str(AUDIT), str(d)], capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    line = [l for l in out.splitlines() if "unfilled fields" in l]
    found = int(line[-1].split(":")[-1]) if line else -1
    named = [l.strip() for l in out.splitlines() if "{R" in l or "{m" in l or "{k" in l]
    ok = (found == expect) and ((r.returncode != 0) if expect else (r.returncode == 0))
    print(f"  {label:8s} unfilled fields: {found} (expected {expect}), "
          f"exit {r.returncode} -> {'ok' if ok else 'WRONG'}")
    for l in named:
        print(f"      {l[:110]}")
    return ok


print("\nboth directions, against fake document roots:")
a = run_case("genuine", GENUINE, 2)
b = run_case("bash", BASH, 0)
if not (a and b):
    sys.exit("the narrowed rule does not behave as documented; see above")
print("\na genuine unfilled field still stops the pass; the quoted bash line no longer does")
