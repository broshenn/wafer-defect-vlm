"""Check that every metric-like number in the built documents has a source.

This project has shipped twice with one fact stored in two places and corrected
in only one of them (LIMITATIONS section 8, items 5 and 6). Both times the stale
copy was a number, and both times the document still read correctly -- it just
disagreed with the measurement. That failure is silent, so it needs a check that
does not depend on anyone remembering.

The rule enforced here: every decimal in the human-readable documents that looks
like a metric -- three or more fractional digits -- must be reproducible from
some record under outputs/. A number that cannot be found there is stale, or was
computed by hand in prose, or was never measured, and each of those deserves to
be looked at rather than trusted.

Numbers with fewer than three fractional digits are deliberately not checked.
They are dominated by percentages, counts and fixed constants, where collisions
between unrelated quantities are common and the check would report noise.

Three things this tool must get right, each learned by getting it wrong first:

  * Training state counts as a record. The first version read only
    outputs/reports/*.json and flagged the SFT final train_loss as unsourced; it
    lives in the run's trainer_state.json. Widening the source set does cost
    sensitivity -- more recorded numbers means more chances for an unrelated one
    to coincide with a stale value -- so the source count is printed and the set
    is limited to files that are records of a run, not scratch output.
  * Numbers hide inside strings. Records store summaries as prose
    ("median=1.964 mean=1.572"), so every string is scanned for numbers rather
    than parsed as a whole.
  * A citation is not a metric. arXiv identifiers and dotted version strings
    would otherwise be reported as untraceable numbers.

Exit status is 0 when untraceable numbers are found: prose legitimately contains
derived arithmetic ("+0.108", "2.8x"), so that output is a list to read, not a gate
to trip.

It is NOT 0 when a template field is found. A field is not a number that failed to
match a record; it is a number that was never written, and the sentence around it
reads as complete. No legitimate prose form looks like a template field, so a hit
fails the run -- which also stops tools/sync_counts.py, since it refuses to rewrite
the document's counts when the audit fails.
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-fs/wafer-vlm")
REPORTS = ROOT / "outputs" / "reports"
CHECKPOINTS = ROOT / "outputs" / "checkpoints"
DOCS = [ROOT / "FINAL_REPORT.md", ROOT / "LIMITATIONS.md",
        REPORTS / "comparison.md"]

# A metric token. There is deliberately no guard on the right: a document that
# quotes a long float verbatim would otherwise match nothing at all, since every
# prefix of its fractional part is still followed by a digit. Long literals are
# handled by comparing against truncated renderings instead, and the dotted
# version strings the right-hand guard would have excluded are removed by
# CITATION before tokenising.
# The sign belongs to the token. Without it a document's "-0.1071" is
# tokenised as "0.1071" and can never match the record's "-0.1071", so
# negative values were systematically reported as untraceable.
TOKEN = re.compile(r"(?<![.\d])-?\d+\.\d{3,6}")
NUM_IN_TEXT = re.compile(r"-?\d+\.\d+")
CITATION = re.compile(r"arxiv[:\s]*\S+|\b\d+\.\d+\.\d+\b", re.IGNORECASE)
# An unfilled template field. Deliberately not the same kind of check as TOKEN:
# a placeholder has no digits, so no comparison against a record can find it, and
# the sentence around it reads as a complete statement about a quantity. Written in
# this file without a literal example, because this file's own text is scanned by
# nothing -- but the documents it reads are prose, and a reader who copies an
# example into one would otherwise reintroduce the defect from the fix.
PLACEHOLDER = re.compile(r"(?<!\$)\{[A-Za-z_][A-Za-z0-9_]*(?::[^}?][^}]*)?\}")

# The two guards above are load-bearing and each was added for a different reason.
# `(?<!\$)` : a `${...}` is an expansion, not a template field. Bash's
# `${VAR:-default}` and `${VAR:?message}` share the braces and the colon with
# `str.format`, and the leading `$` is what says which one it is.
# `[^}?]` after the colon : no format spec begins with `?`, and `:?` is the bash
# operator whose word is prose. On 2026-09-16 this rule fired on LIMITATIONS.md:1028,
# which quotes the wrapper line `RUN_TAG="${RUN_TAG:?…}"` because that line is why a
# run was lost -- and stopped run 42's landing, on a quotation. A rule that reads
# the document's evidence as a defect in the document stops being run.


def walk(obj, out):
    """Every number in a JSON document, including numbers written inside
    strings, which is how several records store their summary statistics."""
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, str):
        out.extend(float(m) for m in NUM_IN_TEXT.findall(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            walk(v, out)
    elif isinstance(obj, list):
        for v in obj:
            walk(v, out)


def source_files():
    """Records of a run: the report records, plus training state. Scratch
    output, caches and adapter weights are not evidence of anything."""
    files = sorted(REPORTS.glob("*.json"))
    files += sorted(CHECKPOINTS.glob("**/trainer_state.json"))
    return files


def read_source_numbers():
    values, parsed, failed = [], 0, []
    for path in source_files():
        local = []
        try:
            walk(json.loads(path.read_text(encoding="utf-8", errors="replace")), local)
        except Exception as exc:
            failed.append(f"{path.name}: {exc}")
            continue
        parsed += 1
        values.extend(local)
    for f in failed:
        print(f"  WARNING: could not parse {f}", file=sys.stderr)
    return values, parsed


def renderings(values):
    """Every way a measured number could legitimately be written: rounded and
    truncated, at 1 to 6 decimals. Truncation matters because a document may
    quote a long float verbatim, and 0.21428571428571427 truncated to six places
    is not what .6f produces."""
    out = set()
    for v in values:
        for k in range(1, 7):
            out.add(f"{v:.{k}f}")
            long_form = f"{v:.12f}"
            cut = long_form.index(".") + 1 + k
            out.add(long_form[:cut])
        # A p-value of 2.83e-26 is written in prose as "2.83196e-26", whose
        # digits the tokeniser reads as 2.83196. Every decimal rendering of
        # 2.83e-26 is "0.0000", so without the mantissa a decisive result is
        # reported as unsourced.
        if v and abs(v) < 1e-2:
            mant = float(f"{v:.6e}".split("e")[0])
            for k in range(1, 7):
                out.add(f"{mant:.{k}f}")
    return out


def classify_derived(tokens, values, tol=1.5e-4):
    """Numbers that are not in any record but ARE arithmetic on two that are.

    Returns {token: (a, b, op)} meaning the token equals a - b or a + b. Without
    this the report cannot distinguish "the document computed a difference" --
    normal and fine -- from "the document quotes a number no run produced".
    """
    import bisect
    vals = sorted(values)
    out = {}
    for tok in tokens:
        t = float(tok)
        for a in vals:
            found = None
            for cand, op in ((a - t, "−"), (t - a, "+")):
                i = bisect.bisect_left(vals, cand)
                for j in (i - 1, i, i + 1):
                    if 0 <= j < len(vals) and abs(vals[j] - cand) <= tol:
                        found = (a, vals[j], op)
                        break
                if found:
                    break
            if found:
                out[tok] = found
                break
    return out


def main():
    if not REPORTS.is_dir():
        sys.exit(f"no {REPORTS}")
    values, n_parsed = read_source_numbers()
    sourced = renderings(values)
    print(f"sources: {n_parsed} records, {len(values)} numbers, "
          f"{len(sourced)} distinct renderings")

    unreadable, report, placeholders = [], [], []
    for doc in DOCS:
        if not doc.is_file():
            unreadable.append(doc.name)
            continue
        lines = doc.read_text(encoding="utf-8", errors="replace").splitlines()
        seen, missing = set(), defaultdict(list)
        for i, line in enumerate(lines, 1):
            for ph in PLACEHOLDER.findall(line):
                placeholders.append((doc.name, i, ph))
            for tok in TOKEN.findall(CITATION.sub(" ", line)):
                seen.add(tok)
                if tok not in sourced:
                    missing[tok].append(i)
        report.append((doc.name, seen, missing, lines))

    everything = sorted({tok for _, _, m, _ in report for tok in m})
    derived = classify_derived(everything, values)

    total_tokens = total_derived = total_missing = 0
    for name, seen, missing, lines in report:
        total_tokens += len(seen)
        d_here = {t: v for t, v in derived.items() if t in missing}
        u_here = {t: v for t, v in missing.items() if t not in derived}
        total_derived += len(d_here)
        total_missing += len(u_here)
        print(f"\n=== {name}: {len(seen)} distinct metric numbers, "
              f"{len(d_here)} derived from records, {len(u_here)} untraceable ===")
        if not missing:
            print("  all traceable")
            continue
        if d_here:
            print("  derived arithmetic -- consistent with being a difference or sum of")
            print("  two recorded numbers. NOTE: this is a SEARCH, not a proof. One pair")
            print("  is shown; it may be a coincidence, and when several pairs fit it is")
            print("  not necessarily the pair the document intends. Read it as 'not an")
            print("  unsourced measurement', never as 'this is where the number came from'.")
            for tok, (a, b, op) in sorted(d_here.items()):
                print(f"    {tok:>10} = {a:.4f} {op} {b:.4f}")
        for tok, linenos in sorted(u_here.items(), key=lambda kv: -len(kv[1])):
            lead = lines[linenos[0] - 1].strip()
            if len(lead) > 140:
                lead = lead[:137] + "..."
            print(f"  {tok:>10}  x{len(linenos):<3} line {linenos[0]:<4} {lead}")

    if placeholders:
        print("\n=== unfilled template fields ===")
        print("  A field here is not a wrong number -- it is no number, in a sentence")
        print("  that reads as though it had one. There is no legitimate prose form")
        print("  that looks like one, so this is a failure, not a list to review.")
        for name, lineno, ph in placeholders:
            print(f"    {name}:{lineno}  {ph}")
    print("\n=== summary ===")
    print(f"  documents checked : {len(report)}"
          + (f"  (MISSING: {', '.join(unreadable)})" if unreadable else ""))
    print(f"  metric numbers    : {total_tokens}")
    print(f"  derived arithmetic: {total_derived}")
    print(f"  untraceable       : {total_missing}")
    print(f"  unfilled fields   : {len(placeholders)}")
    if placeholders:
        sys.exit(f"{len(placeholders)} unfilled template field(s) in the documents; "
                 "an unfilled field is a missing number, not a small one")


if __name__ == "__main__":
    main()
