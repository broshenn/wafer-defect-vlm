"""Make paired_significance.py enumerate its cells instead of listing them.

Run 45 (GSPO, G=4, lr 1e-5) completes the 2x2 at G=4. The tool that produces the
paired record could not have noticed: the lr pairs were a two-element tuple written
by hand ((GRPO G=4), (GSPO G=8)), the group-size lr blocks were another hand-written
tuple ((8, 32)), and the macro-F1 lr blocks were a third copy of the first tuple. A
run that lands and changes nothing in the record is the defect this project keeps
finding (LIMITATIONS.md section 8, item 9) -- and here it would have been invisible,
because the record would still look complete.

Three more hand-typed things go with it:

  * `paper_claim_status.missing_cell` names GSPO at G=4 / lr 1e-5 as the missing
    cell. That is exactly run 45, so the sentence goes false the moment it lands.
  * `what_is_established` quotes "GRPO p=0.04356, GSPO G=8 p=0.000324" -- two
    p-values typed into a record that computes them a few blocks earlier.
  * `reading` asserts "the lr effect inside each algorithm is significant" and "the
    lr 1e-5 runs are indistinguishable from SFT", both of which are claims about a
    set of cells that is about to grow by one.

So this patch: adds the new run to the run table, adds a guard that fails on any
report on disk that is in neither the run table nor an explicit exclusion (so the
*next* run that lands cannot pass unremarked either), and replaces every hand-written
cell list and every typed-in p-value with a read of the data.

The grid cells are enumerated from the run names, which already encode algorithm,
group size and learning rate. Existing cell labels are reproduced exactly ("GRPO G=4",
"GSPO G=8"), so the numbers other documents quote keep their keys.
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path("/root/autodl-fs/wafer-vlm")
P = ROOT / "tools/paired_significance.py"
s = P.read_text(encoding="utf-8")
n_edits = 0


def sub(old, new, why):
    global s, n_edits
    n = s.count(old)
    if n != 1:
        sys.exit(f"anchor for '{why}' appears {n} times, expected 1; nothing written")
    s = s.replace(old, new, 1)
    n_edits += 1


# ------------------------------------------------ 1. the new run, and the guard
sub(
    """OPTIONAL = [("GSPO_G32_lr1e5", "qwen35_9b_gspo_g32_lr1e5"),
            ("GSPO_G4_lr5e5", "qwen35_9b_gspo_g4_lr5e5")]""",
    """OPTIONAL = [("GSPO_G32_lr1e5", "qwen35_9b_gspo_g32_lr1e5"),
            ("GSPO_G4_lr5e5", "qwen35_9b_gspo_g4_lr5e5"),
            ("GSPO_G4_lr1e5", "qwen35_9b_gspo_g4_lr1e5")]

# Runs on disk that are deliberately not cells of this record, each with its reason.
# Written down rather than left out: a run that lands and changes nothing here is
# indistinguishable, from the record's side, from a run that was never evaluated.
EXCLUDE = {
    "qwen35_9b_grpo_lr1e5_seed3408":
        "the second seed of GRPO_G4_lr1e5. Same configuration, so it is not a new "
        "cell in any contrast here; its purpose is the replication check in "
        "tools/seed_variance.py, and it carries a column of its own in the "
        "comparison tables.",
}""",
    "the new run in OPTIONAL and the exclusion table")

sub(
    """ABSENT = [n for n, t in OPTIONAL if not (BASE / (t + ".jsonl")).is_file()]""",
    """ABSENT = [n for n, t in OPTIONAL if not (BASE / (t + ".jsonl")).is_file()]

# Every report on disk must be one of RUNS, one of OPTIONAL, or a named exclusion.
# Without this, adding a run to the project changes nothing here and nothing says so.
_known = {t for _, t in RUNS} | {t for _, t in OPTIONAL} | set(EXCLUDE)
_unaccounted = sorted(p.name[: -len("__report.json")]
                      for p in REPORTS.glob("*__report.json")
                      if p.name[: -len("__report.json")] not in _known)
if _unaccounted:
    sys.exit("report(s) on disk that are neither a run of this record nor a named "
             f"exclusion: {_unaccounted}. Add each to RUNS/OPTIONAL, or to EXCLUDE "
             "with the reason it is not a contrast here. A run that lands and "
             "changes nothing in this record is the defect this guard exists for.")

# The grid is enumerated from the run names, which already carry algorithm, group
# size and learning rate. A hand-written list of cells was here before and would
# have kept the G=4 cell of the lr axis out of a record that still looked complete.
CELL = re.compile(r"^(GRPO|GSPO)_G(\\d+)_lr(1e5|5e5)$")


def lr_cells(names):
    \"\"\"(label, lr1e-5 run, lr5e-5 run) for every algorithm x group size that has
    both learning rates present, ordered by algorithm then group size. The label is
    the one this record has always used, so existing keys are unchanged.\"\"\"
    have = {}
    for n in names:
        m = CELL.match(n)
        if m:
            have.setdefault((m.group(1), int(m.group(2))), set()).add(m.group(3))
    return [(f"{a} G={g}", f"{a}_G{g}_lr1e5", f"{a}_G{g}_lr5e5")
            for (a, g), lrs in sorted(have.items()) if {"1e5", "5e5"} <= lrs]""",
    "the enumeration helper and the unaccounted-report guard")

# ------------------------------------- 2. the lr effect: enumerate, not list
sub(
    """print("paired lr effect within each algorithm (lr5e-5 minus lr1e-5):")
for algo, a, b in (("GRPO G=4", "GRPO_G4_lr1e5", "GRPO_G4_lr5e5"),
                   ("GSPO G=8", "GSPO_G8_lr1e5", "GSPO_G8_lr5e5")):""",
    """print("paired lr effect within each algorithm (lr5e-5 minus lr1e-5):")
for algo, a, b in lr_cells(correct):""",
    "the lr_effect_paired cell list")

sub(
    """for algo, a, b in (("GRPO G=4", "GRPO_G4_lr1e5", "GRPO_G4_lr5e5"),
                   ("GSPO G=8", "GSPO_G8_lr1e5", "GSPO_G8_lr5e5")):
    out["macro_f1_marginal"].setdefault("lr_effect", {})[algo] = {""",
    """for algo, a, b in lr_cells(correct):
    out["macro_f1_marginal"].setdefault("lr_effect", {})[algo] = {""",
    "the macro_f1_marginal cell list")

# ------------------------------------- 3. the same change at each group size
sub(
    """    print("lr effect at each group size (GSPO, sequence-level): the same change "
          "of learning rate measured at two group sizes.")
    lr_blocks = {}
    for g, a, b in ((8, "GSPO_G8_lr1e5", "GSPO_G8_lr5e5"),
                    (32, "GSPO_G32_lr1e5", "GSPO_G32_lr5e5")):""",
    """    print("lr effect at each group size (GSPO, sequence-level): the same change "
          "of learning rate measured at every group size in this record.")
    lr_blocks = {}
    for _algo, g, a, b in lr_cells(correct):
        if _algo != "GSPO":
            continue""",
    "the lr_effect_both_group_sizes cell list")

sub(
    """                    "group size" if len(_sig) == 2 else""",
    """                    "group size" if len(_sig) == len(lr_blocks) else""",
    "the two-group-sizes test in that block's reading")

# ------------------------------------- 4. the typed-in p-values and the claims
sub(
    """_isl = out.get("is_level_paired")
if _isl:""",
    """# Everything the prose below asserts is read from the blocks above rather than
# typed here. The p-values used to be literals in this file, and the two sentences
# that name the missing cell and the established result were written when the G=4
# 2x2 was incomplete -- which is a state this record has now left.
_lr_all = sorted(out.get("lr_effect_paired", {}))
_lr_sig = [k for k in _lr_all if out["lr_effect_paired"][k]["excludes_zero"]]
_five = sorted(n for n in out["comparisons_vs_SFT"] if n.endswith("lr5e5"))
_five_sig = [n for n in _five if out["comparisons_vs_SFT"][n]["excludes_zero"]]
_one = sorted(n for n in out["comparisons_vs_SFT"] if n.endswith("lr1e5"))
_one_sig = [n for n in _one if out["comparisons_vs_SFT"][n]["excludes_zero"]]
_established_txt = (
    "the lr effect is significant in "
    + (f"every cell where both learning rates exist ({', '.join(_lr_all)})"
       if _lr_all and len(_lr_sig) == len(_lr_all) else
       f"{len(_lr_sig)} of the {len(_lr_all)} cells where both learning rates "
       f"exist ({', '.join(_lr_sig) or 'none'})")
    + ", so sequence-level does not visibly tolerate the higher learning rate "
      "better than token-level does at the group sizes measured")

# The 2x2 at G=4 is the only group size at which both IS levels have both learning
# rates. Its four cells are the whole reason the G=4 runs exist, so whether they are
# all present is computed, not assumed.
_G4_2X2 = ("GRPO_G4_lr1e5", "GRPO_G4_lr5e5", "GSPO_G4_lr1e5", "GSPO_G4_lr5e5")
_g4_have = [n for n in _G4_2X2 if n in correct]
_g4_done = len(_g4_have) == 4
_g4_missing_txt = (
    "none: the 2x2 at G=4 is complete -- both IS levels at both learning rates -- "
    "so the lr effect is measured separately for each IS level at one group size, "
    "which is the form in which 'tolerates a higher learning rate' becomes a "
    "comparison rather than an impression. What is still NOT measured is whether "
    "the two effects differ in SIZE: this record holds each effect and its own "
    "test, not a test of the difference between them."
    if _g4_done else
    "tolerance is a statement about the SIZE of the lr effect at each IS level, "
    "which needs the same 2x2 at G=4. Cells present at G=4: "
    + (", ".join(_g4_have) or "none") + "; the 2x2 is not complete, so the lr "
    "effect is not measured for both IS levels at one group size")
if _g4_done:
    print()
    print("the 2x2 at G=4 is complete: lr effect measured separately for each IS "
          "level at one group size.")

_isl = out.get("is_level_paired")
if _isl:""",
    "the computed reading inputs, the 2x2 state, and the established-result text")

sub(
    """        "missing_cell": "tolerance is a statement about the SIZE of the lr effect "
                        "at each IS level, which needs the same 2x2 at G=4. This "
                        "record has token at both lrs but sequence at G=4 only at "
                        "lr 5e-5, so the cell still missing is GSPO (IS=sequence) "
                        "at G=4 with lr 1e-5",""",
    """        "missing_cell": _g4_missing_txt,""",
    "the live branch's missing-cell text")

sub(
    """        "what_is_established": "both IS levels degrade at lr 5e-5 (GRPO p=0.04356, "
                               "GSPO G=8 p=0.000324), so sequence-level does not "
                               "visibly tolerate the higher lr better than "
                               "token-level does at the group sizes measured"}""",
    """        "what_is_established": _established_txt}""",
    "the live branch's established-result text")

sub(
    """        "what_is_established": "both IS levels degrade at lr 5e-5 (GRPO "
                               "p=0.04356, GSPO G=8 p=0.000324), so "
                               "sequence-level does not visibly tolerate the "
                               "higher lr better than token-level does at the "
                               "group sizes measured"}""",
    """        "what_is_established": _established_txt}""",
    "the unreachable branch's established-result text")

sub(
    """out["reading"] = (
    "The marginal-CI rule in final_report.py reports that every RL run's "
    "interval overlaps SFT's. Paired on the same 252 rows, the lr 5e-5 drop is "
    "significant in both algorithms, and the lr effect inside each algorithm is "
    "significant, while the lr 1e-5 runs are indistinguishable from SFT. The "
    "overlap rule is conservative for independent samples and these are paired, "
    "so it understates differences it should see; the paired test does not "
    "overturn the direction, it supplies the significance the other rule could "
    "not. macro-F1 is not covered by this test.")""",
    """out["reading"] = (
    "The marginal-CI rule in final_report.py reports that every RL run's "
    "interval overlaps SFT's. Paired on the same 252 rows: "
    + (f"all {len(_five)} runs at lr 5e-5 are significantly below SFT"
       if _five and len(_five_sig) == len(_five) else
       f"{len(_five_sig)} of the {len(_five)} runs at lr 5e-5 are significantly "
       f"below SFT ({', '.join(_five_sig) or 'none'})")
    + "; the lr effect inside a fixed algorithm and group size is significant in "
    + (f"all {len(_lr_all)} cells where both learning rates exist"
       if _lr_all and len(_lr_sig) == len(_lr_all) else
       f"{len(_lr_sig)} of the {len(_lr_all)} cells where both learning rates "
       f"exist ({', '.join(_lr_sig) or 'none'})")
    + "; and "
    + (f"no run at lr 1e-5 is separable from SFT ({len(_one)} tested)"
       if not _one_sig else
       f"{len(_one_sig)} of the {len(_one)} runs at lr 1e-5 differ significantly "
       f"from SFT ({', '.join(_one_sig)})")
    + ". The overlap rule is conservative for independent samples and these are "
      "paired, so it understates differences it should see; the paired test does "
      "not overturn the direction, it supplies the significance the other rule "
      "could not. macro-F1 is not covered by this test.")""",
    "the top-level reading")

shutil.copy2(P, "/tmp/paired_significance.py.bak-patch95")
P.write_text(s, encoding="utf-8")

r = subprocess.run([sys.executable, "-m", "py_compile", str(P)],
                   capture_output=True, text=True)
if r.returncode != 0:
    shutil.copy2("/tmp/paired_significance.py.bak-patch95", P)
    sys.exit(f"the patched file does not compile, restored from backup:\n{r.stderr}")

print(f"{n_edits} edit(s) applied to {P.name}; it compiles.")
print("backup: /tmp/paired_significance.py.bak-patch95")
