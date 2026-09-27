"""Three defects in the number-provenance audit, found by reading its output.

  * TOKEN had no sign group, so a document's "-0.1071" tokenised as "0.1071"
    while the record's rendering is "-0.1071". Every negative value was therefore
    unfindable, and the tool reported them as untraceable -- four of this round's
    fourteen "failures" are that, not stale numbers.
  * A p-value written as 2.83196e-26 tokenises to the mantissa 2.83196, but the
    only renderings produced for 2.83e-26 were decimal ones ("0.0000").
  * Arithmetic written in prose ("+0.108", a difference of two recorded values)
    landed in the same bucket as a number nobody ever measured. The tool's own
    docstring says derived arithmetic is legitimate; it just could not tell the
    two apart, so the summary count conflated them.

The third is the one worth fixing properly. A count that mixes "this is a
difference of two recorded numbers, here is the pair" with "this number is in no
record anywhere" is a count nobody can act on. Separating them makes the
remaining number the actual signal.
"""
import ast
import sys
from pathlib import Path

p = Path("/root/autodl-fs/wafer-vlm/tools/audit_report_numbers.py")
s = p.read_text(encoding="utf-8")

OLD1 = 'TOKEN = re.compile(r"(?<![.\\d])\\d+\\.\\d{3,6}")'
NEW1 = ('# The sign belongs to the token. Without it a document\'s "-0.1071" is\n'
        '# tokenised as "0.1071" and can never match the record\'s "-0.1071", so\n'
        '# negative values were systematically reported as untraceable.\n'
        'TOKEN = re.compile(r"(?<![.\\d])-?\\d+\\.\\d{3,6}")')

OLD2 = '''            cut = long_form.index(".") + 1 + k
            out.add(long_form[:cut])
    return out'''
NEW2 = '''            cut = long_form.index(".") + 1 + k
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
    return out'''

OLD3 = '''    total_tokens = total_missing = 0
    for name, seen, missing, lines in report:
        total_tokens += len(seen)
        total_missing += len(missing)
        print(f"\\n=== {name}: {len(seen)} distinct metric numbers, "
              f"{len(missing)} not traceable to any record ===")
        if not missing:
            print("  all traceable")
            continue
        for tok, linenos in sorted(missing.items(), key=lambda kv: -len(kv[1])):
            lead = lines[linenos[0] - 1].strip()
            if len(lead) > 140:
                lead = lead[:137] + "..."
            print(f"  {tok:>10}  x{len(linenos):<3} line {linenos[0]:<4} {lead}")

    print("\\n=== summary ===")
    print(f"  documents checked : {len(report)}"
          + (f"  (MISSING: {', '.join(unreadable)})" if unreadable else ""))
    print(f"  metric numbers    : {total_tokens}")
    print(f"  untraceable       : {total_missing}")'''

NEW3 = '''    everything = sorted({tok for _, _, m, _ in report for tok in m})
    derived = classify_derived(everything, values)

    total_tokens = total_derived = total_missing = 0
    for name, seen, missing, lines in report:
        total_tokens += len(seen)
        d_here = {t: v for t, v in derived.items() if t in missing}
        u_here = {t: v for t, v in missing.items() if t not in derived}
        total_derived += len(d_here)
        total_missing += len(u_here)
        print(f"\\n=== {name}: {len(seen)} distinct metric numbers, "
              f"{len(d_here)} derived from records, {len(u_here)} untraceable ===")
        if not missing:
            print("  all traceable")
            continue
        if d_here:
            print("  derived arithmetic (a recorded number minus/plus another):")
            for tok, (a, b, op) in sorted(d_here.items()):
                print(f"    {tok:>10} = {a:.4f} {op} {b:.4f}")
        for tok, linenos in sorted(u_here.items(), key=lambda kv: -len(kv[1])):
            lead = lines[linenos[0] - 1].strip()
            if len(lead) > 140:
                lead = lead[:137] + "..."
            print(f"  {tok:>10}  x{len(linenos):<3} line {linenos[0]:<4} {lead}")

    print("\\n=== summary ===")
    print(f"  documents checked : {len(report)}"
          + (f"  (MISSING: {', '.join(unreadable)})" if unreadable else ""))
    print(f"  metric numbers    : {total_tokens}")
    print(f"  derived arithmetic: {total_derived}")
    print(f"  untraceable       : {total_missing}")'''

for old, new, tag in ((OLD1, NEW1, "TOKEN"), (OLD2, NEW2, "renderings/classify"),
                      (OLD3, NEW3, "report loop")):
    n = s.count(old)
    if n != 1:
        sys.exit(f"FAILED: {tag} appears {n} times, expected 1")
    s = s.replace(old, new)

ast.parse(s)
p.write_text(s, encoding="utf-8")
print("audit_report_numbers.py: sign, scientific notation, derived-vs-untraceable")
