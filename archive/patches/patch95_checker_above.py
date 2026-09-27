"""Let the 5.2.4 check accept the sentence's second shape instead of going STALE.

`check_quantified_claims.py` verifies 5.2.4's claim about which RL runs beat SFT with
one fixed pattern, written for the sentence as it reads today:

    "六个 RL run 里唯一 macro-F1 点估计高于 SFT 的是 GRPO lr1e-5（0.6197 对 0.6114）"

That sentence has exactly one shape only while exactly one run is above SFT. Run 45 is
GSPO at G=4 with lr 1e-5, and lr 1e-5 is the learning rate that lifted GSPO at G=8 from
0.5014 to 0.6089; if it lifts this one past SFT's 0.6114, then "唯一" is false and the
honest sentence lists two runs. At that point the fixed pattern matches nothing and the
tool reports STALE -- which does not mean the sentence is wrong. It means the tool has
stopped looking, and a landing gate that stops looking is the failure this whole tool
set exists to prevent.

So the check gets a second shape, and `anchor_any` tries both. Crucially the *check* is
unchanged by which shape is used: both are verified against the same computed set of
runs and the same values. The wording is free; the claim is not.

Deliberately not done: making the pattern loose enough to match almost anything, which
would let a sentence drop the claim entirely and still pass. Each shape is spelled out
in full, the set of names is compared as a set, and a shape that matches neither is
still STALE.

Order matters in shape 2: the document lists the runs high to low, and the check
compares the list in order, so a list that names the right runs in the wrong order is
still caught.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path("/root/autodl-fs/wafer-vlm")
SRC = ROOT / "tools/check_quantified_claims.py"
PY = sys.executable

ANCHOR_FN = '''def anchor(name: str, pattern: str):'''

ANCHOR_ANY = '''def anchor_any(name: str, patterns: tuple):
    """Match the first of several shapes a sentence may legitimately take.

    A single fixed pattern makes this tool a second author of the document: the prose
    cannot change shape without the check going STALE, and STALE reads as "nothing was
    verified" and stops the landing. Some of these sentences have more than one honest
    shape -- 5.2.4's uniqueness claim stops being writable as soon as a second run
    clears SFT -- so each shape is spelled out and every one of them is verified the
    same way: against the computed set of runs and the computed values, never against
    the wording.
    """
    for pat in patterns:
        m = re.search(pat, text, re.S)
        if m:
            return m, pat
    print(f"  STALE  {name}")
    print("         the sentence this check was written against is not in the "
          "document any more, in any of the shapes it has; nothing was verified")
    facts.append(name)
    return None, None


def anchor(name: str, pattern: str):'''

OLD = '''# ------------------------------------------------- 5.2.4: which RL runs beat SFT
# "N 个 RL run 里唯一 macro-F1 点估计高于 SFT 的是 X（A 对 B）"
P_ABOVE = (r"(" + NUM + r")\\s*个 RL run 里唯一 macro-F1 点估计高于 SFT 的是 "
           r"([^（\\n]+)（([\\d.]+) 对 ([\\d.]+)）")
m = anchor("5.2.4: the only RL run whose macro-F1 is above SFT", P_ABOVE)
if m:
    n, who, val, base = (cn2int(m.group(1)), doc_name(m.group(2).strip()),
                         float(m.group(3)), float(m.group(4)))
    above = sorted((n_ for n_ in rl if present[n_]["macro_f1"] > sft_f1),
                   key=lambda n_: -present[n_]["macro_f1"])
    cmp("5.2.4: the RL-run count", n, len(rl), kind="count",
        pattern=P_ABOVE, group=1)
    cmp("5.2.4: the set of RL runs above SFT, by name",
        [who], [doc_name(x) for x in above] or ["（none: no RL run beats SFT）"])
    cmp("5.2.4: the value quoted for it and for SFT", (val, base),
        (round(present[above[0]]["macro_f1"], 4), round(sft_f1, 4)) if above
        else (None, round(sft_f1, 4)))'''

NEW = '''# ------------------------------------------------- 5.2.4: which RL runs beat SFT
# Two shapes, because the claim has two shapes: one run above SFT is a uniqueness
# claim, two or more is a list. A pattern that knew only the first would go STALE the
# moment the second became true.
#   shape 1: "N 个 RL run 里唯一 macro-F1 点估计高于 SFT 的是 X（A 对 B）"
#   shape 2: "N 个 RL run 里有 K 个 macro-F1 点估计高于 SFT：X（A 对 B）与 Y（A 对 B）"
P_ABOVE_ONE = (r"(" + NUM + r")\\s*个 RL run 里唯一 macro-F1 点估计高于 SFT 的是 "
               r"([^（\\n]+)（([\\d.]+) 对 ([\\d.]+)）")
P_ABOVE_MANY = (r"(" + NUM + r")\\s*个 RL run 里有\\s*(" + NUM + r")\\s*个 macro-F1 点估计"
                r"高于 SFT：([^\\n]+)")
m, used = anchor_any("5.2.4: which RL runs' macro-F1 is above SFT",
                     (P_ABOVE_ONE, P_ABOVE_MANY))
if m:
    above = sorted((n_ for n_ in rl if present[n_]["macro_f1"] > sft_f1),
                   key=lambda n_: -present[n_]["macro_f1"])
    want = [doc_name(x) for x in above]
    cmp("5.2.4: the RL-run count", cn2int(m.group(1)), len(rl), kind="count",
        pattern=used, group=1)
    if used is P_ABOVE_ONE:
        cmp("5.2.4: the set of RL runs above SFT, by name",
            [doc_name(m.group(2).strip())], want or ["（none: no RL run beats SFT）"])
        cmp("5.2.4: the value quoted for it and for SFT",
            (float(m.group(3)), float(m.group(4))),
            (round(present[above[0]]["macro_f1"], 4), round(sft_f1, 4)) if above
            else (None, round(sft_f1, 4)))
    else:
        # The list is "name（value 对 SFT）" repeated, separated by 、. The name may
        # contain ASCII parens (it is a column label like GSPO(G=4) lr1e-5) but never
        # the full-width ones the pairs are wrapped in.
        listed = re.findall(r"([^（、\\n]+)（([\\d.]+) 对 ([\\d.]+)）", m.group(3))
        cmp("5.2.4: the number of RL runs above SFT", cn2int(m.group(2)), len(above),
            why="how many runs clear SFT is a fact and not a count: the sentence is "
                "about which runs, not about the size of the list")
        cmp("5.2.4: the set of RL runs above SFT, by name, in the order written",
            [doc_name(x.strip()) for x, _, _ in listed], want)
        cmp("5.2.4: the value quoted for each of them and for SFT",
            [(float(v), float(b)) for _, v, b in listed],
            [(round(present[x]["macro_f1"], 4), round(sft_f1, 4)) for x in above])'''

s = SRC.read_text(encoding="utf-8")
shutil.copy2(SRC, "/tmp/check_quantified_claims.py.bak-patch95")

if "def anchor_any(" in s:
    sys.exit("anchor_any is already there; nothing written")
if s.count(ANCHOR_FN) != 1:
    sys.exit(f"the anchor() definition appears {s.count(ANCHOR_FN)} times; "
             f"nothing written")
if s.count(OLD) != 1:
    sys.exit(f"the 5.2.4 check block appears {s.count(OLD)} times, expected 1; "
             f"nothing written")

s = s.replace(ANCHOR_FN, ANCHOR_ANY, 1)
s = s.replace(OLD, NEW, 1)
SRC.write_text(s, encoding="utf-8")

r = subprocess.run([PY, "-m", "py_compile", str(SRC)], capture_output=True, text=True)
if r.returncode != 0:
    shutil.copy2("/tmp/check_quantified_claims.py.bak-patch95", SRC)
    sys.exit(f"the patched checker does not compile, restored:\\n{r.stderr}")

# It has to still pass on the document as it stands, or the change is not additive.
# The current state is rc 0 (everything matches), so anything else means this edit
# changed a verdict rather than adding a shape.
r = subprocess.run([PY, str(SRC)], capture_output=True, text=True,
                   env={**os.environ, "WAFER_DOC": str(ROOT / "LIMITATIONS.md")})
print(r.stdout[-2500:])
if r.returncode != 0:
    print(r.stderr[-1500:])
    shutil.copy2("/tmp/check_quantified_claims.py.bak-patch95", SRC)
    sys.exit(f"the patched checker exits {r.returncode} on the current document, but "
             f"exits 0 before the patch; restored, so the landing gate is intact")
print("5.2.4's check now accepts both shapes and still passes on the current document")
