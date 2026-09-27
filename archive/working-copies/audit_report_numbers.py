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

Exit status is 0 even when untraceable numbers are found: prose legitimately
contains derived arithmetic ("+0.108", "2.8x"), so the output is a list to read,
not a gate to trip.
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
TOKEN = re.compile(r"(?<![.\d])\d+\.\d{3,6}")
NUM_IN_TEXT = re.compile(r"-?\d+\.\d+")
CITATION = re.compile(r"arxiv[:\s]*\S+|\b\d+\.\d+\.\d+\b", re.IGNORECASE)


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
    return out


def main():
    if not REPORTS.is_dir():
        sys.exit(f"no {REPORTS}")
    values, n_parsed = read_source_numbers()
    sourced = renderings(values)
    print(f"sources: {n_parsed} records, {len(values)} numbers, "
          f"{len(sourced)} distinct renderings")

    unreadable, report = [], []
    for doc in DOCS:
        if not doc.is_file():
            unreadable.append(doc.name)
            continue
        lines = doc.read_text(encoding="utf-8", errors="replace").splitlines()
        seen, missing = set(), defaultdict(list)
        for i, line in enumerate(lines, 1):
            for tok in TOKEN.findall(CITATION.sub(" ", line)):
                seen.add(tok)
                if tok not in sourced:
                    missing[tok].append(i)
        report.append((doc.name, seen, missing, lines))

    total_tokens = total_missing = 0
    for name, seen, missing, lines in report:
        total_tokens += len(seen)
        total_missing += len(missing)
        print(f"\n=== {name}: {len(seen)} distinct metric numbers, "
              f"{len(missing)} not traceable to any record ===")
        if not missing:
            print("  all traceable")
            continue
        for tok, linenos in sorted(missing.items(), key=lambda kv: -len(kv[1])):
            lead = lines[linenos[0] - 1].strip()
            if len(lead) > 140:
                lead = lead[:137] + "..."
            print(f"  {tok:>10}  x{len(linenos):<3} line {linenos[0]:<4} {lead}")

    print("\n=== summary ===")
    print(f"  documents checked : {len(report)}"
          + (f"  (MISSING: {', '.join(unreadable)})" if unreadable else ""))
    print(f"  metric numbers    : {total_tokens}")
    print(f"  untraceable       : {total_missing}")


if __name__ == "__main__":
    main()
