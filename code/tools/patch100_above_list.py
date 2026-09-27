"""Recompute 5.2.4's "which RL runs clear SFT" list from the reports, or leave it alone.

That sentence is the one claim in the document whose *shape* changes with the run set:
at one run above SFT it was a uniqueness claim, at two it became a list, and every run
that lands can add a third. patch98 wrote the two-run form by hand when run 45 landed,
which is correct today and stale the moment another run clears SFT -- and the claims
checker will not repair it, on purpose: which runs are in the list is a fact, not a
count, and auto-editing facts is how this document would end up asserting something
nobody chose.

So the sentence is rebuilt here from the reports, in the same shape the checker reads,
and the paired-test clause beside it is rebuilt too. That clause was the part patch98
could not generalise: it said "两个都仍与 SFT 无法区分", which is true of exactly two
runs and false the moment the list has three. Here each run in the list is placed by
its own paired test against SFT, so the clause says what the record says about however
many runs there are.

Idempotent: if the document already reads the way the records do, it writes nothing and
says so. Every numeral and name comes from `outputs/reports/*__report.json` and
`paired_significance.json`; nothing is typed in.
"""
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path("/root/autodl-fs/wafer-vlm")
DOC = ROOT / "LIMITATIONS.md"
REP = ROOT / "outputs/reports"

RL = {
    "GRPO(G=4) lr1e-5": "qwen35_9b_grpo",
    "GRPO(G=4) lr5e-5": "qwen35_9b_grpo_lr5e5",
    "GSPO(G=8) lr5e-5": "qwen35_9b_gspo_v1",
    "GSPO(G=8) lr1e-5": "gspo_lr1e5",
    "GSPO(G=32) lr5e-5": "qwen35_9b_gspo_g32",
    "GSPO(G=4) lr5e-5": "qwen35_9b_gspo_g4_lr5e5",
    "GSPO(G=4) lr1e-5": "qwen35_9b_gspo_g4_lr1e5",
    "GSPO(G=32) lr1e-5": "qwen35_9b_gspo_g32_lr1e5",
}
# The runs that exist are whatever has a report on disk; the map above is the naming,
# not the claim. A mismatch is reported rather than papered over.
ON_DISK = sorted(n for n in RL if (REP / f"{RL[n]}__report.json").is_file())
MISSING = sorted(set(RL) - set(ON_DISK))
if MISSING:
    print(f"note: no report yet for {'、'.join(MISSING)} -- not part of the run set")


def clf(tag):
    p = REP / f"{tag}__report.json"
    return json.loads(p.read_text(encoding="utf-8"))["classification"]


F = {n: clf(RL[n])["macro_f1"] for n in ON_DISK}
FSFT = clf("qwen35_9b_adapter")["macro_f1"]

PS = json.loads((REP / "paired_significance.json").read_text(encoding="utf-8"))
CVS = PS.get("comparisons_vs_SFT") or {}


def cvs_key(label):
    """'GSPO(G=4) lr1e-5' -> 'GSPO_G4_lr1e5', the key the paired record uses."""
    k = re.sub(r"\(G=(\d+)\)\s*", r"_G\1_", label).strip("_")
    return k.replace("lr1e-5", "lr1e5").replace("lr5e-5", "lr5e5")


_unmapped = [n for n in ON_DISK if cvs_key(n) not in CVS]
if _unmapped:
    sys.exit(f"the paired record has no comparison for {_unmapped} "
             f"(looked for {[cvs_key(n) for n in _unmapped]}); nothing written")

ABOVE = [n for n in sorted(ON_DISK, key=lambda n: -F[n]) if F[n] > FSFT]
CN = "一二三四五六七八九十"

if not ABOVE:
    sys.exit("no RL run clears SFT any more: this sentence would have to go back to its "
             "uniqueness form, which is a rewrite and not a recomputation. Nothing "
             "written.")
if len(ON_DISK) > 10 or len(ABOVE) > 10:
    sys.exit(f"{len(ON_DISK)} runs / {len(ABOVE)} above SFT would need a numeral this "
             f"function cannot spell; nothing written")

_LIST = "、".join(f"{n}（{F[n]:.4f} 对 {FSFT:.4f}）" for n in ABOVE)
LINE1 = (f"{CN[len(ON_DISK) - 1] if 0 < len(ON_DISK) <= 10 else len(ON_DISK)}个 RL run 里有 "
         f"{len(ABOVE)} 个 macro-F1 点估计高于 SFT：{_LIST}，")

# Each run in the list, placed by its own paired test rather than by how many there are.
def quoted(names):
    return "；".join(
        f"{n} Δ准确率 {CVS[cvs_key(n)]['delta_vs_sft']:+.4f}、"
        f"`p` = {CVS[cvs_key(n)]['mcnemar']['p_exact_two_sided']}" for n in names)


def subj(names):
    """'这两个都' when there are several, the bare name (space-separated, the way the
    document writes a column label) when there is one."""
    if len(names) == 1:
        return names[0] + " "
    return "这%s个都" % (CN[len(names) - 1] if 0 < len(names) <= 10 else str(len(names)))


_sig = [n for n in ABOVE
        if CVS[cvs_key(n)]["excludes_zero"] and CVS[cvs_key(n)]["delta_vs_sft"] > 0]
_rest = [n for n in ABOVE if n not in _sig]
if not ABOVE:
    LINE2 = "成对检验下没有任何一个与 SFT 分得开；"
elif not _sig:
    LINE2 = f"成对检验下{subj(_rest)}仍与 SFT 无法区分（{quoted(_rest)}）；"
elif not _rest:
    _s = subj(_sig)
    LINE2 = (f"成对检验下{_s}**高于 SFT 且排除 0**（{quoted(_sig)}）；"
             if len(_sig) > 1 else
             f"成对检验下 {_s}**高于 SFT 且排除 0**（{quoted(_sig)}）；")
else:
    LINE2 = (f"成对检验下{subj(_sig)}**高于 SFT 且排除 0**（{quoted(_sig)}），"
             f"而{subj(_rest)}仍与 SFT 无法区分（{quoted(_rest)}）；")

s = DOC.read_text(encoding="utf-8")
NEEDLE = "个 macro-F1 点估计高于 SFT"
if s.count(NEEDLE) != 1:
    sys.exit(f"the 5.2.4 list sentence appears {s.count(NEEDLE)} times, expected 1; "
             f"nothing written")

_i = s.find(NEEDLE)
_start = s.rfind("\n", 0, _i) + 1
_j = s.find("成对检验下", _i)
if _j < 0:
    sys.exit("the sentence has no paired-test clause to rebuild; nothing written")
_end = s.find("\n", _j)
_old = s[_start:_end]
if not re.match(r"^(?:[一二三四五六七八九十两]+|\d+)个 RL run 里有", _old):
    sys.exit(f"the region to replace does not start at the list sentence; nothing "
             f"written. It starts: {_old[:40]!r}")

_new = LINE1 + "\n" + LINE2
if _old == _new:
    print(f"5.2.4's list already reads the way the records do ({len(ABOVE)} above SFT: "
          f"{'、'.join(ABOVE)}); nothing written")
    sys.exit(0)

shutil.copy2(DOC, "/tmp/LIMITATIONS.md.bak-patch100")
before = s.count("**")
s = s[:_start] + _new + s[_end:]
after = s.count("**")
if (after - before) % 2:
    sys.exit(f"bold markers went {before} -> {after}, an odd change; nothing written")
DOC.write_text(s, encoding="utf-8")

print(f"5.2.4's list rebuilt ({len(s.splitlines())} lines, bold {before} -> {after})")
print(f"  run set   : {len(ON_DISK)} RL runs, SFT {FSFT:.4f}")
print(f"  above SFT : {[(n, round(F[n], 4)) for n in ABOVE]}")
print(f"  paired    : separated={_sig or 'none'}, not separated={_rest or 'none'}")
