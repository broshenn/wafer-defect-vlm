"""Keep internal record keys out of the report's prose.

The derived "which RL run is worst on macro-F1" sentence printed the comparison
column's key verbatim: "分类最差的 RL run 不是 G=32 而是 GSPO_G4_lr5e5". The key is
correct and traceable, but a report sentence should name the configuration
(GSPO(G=4) lr5e-5), not the column it was read from.

The label helper already existed, but nested inside the paired-significance block,
so it was only in scope there. It moves to module level next to _geom -- that is
where the other name-parsing lives -- and both call sites use it.
"""
import ast
import sys
from pathlib import Path

P = Path("/root/autodl-fs/wafer-vlm/tools/final_report.py")
s = P.read_text(encoding="utf-8")

# ---------------------------------------------------- 1. the helper, at module level
A1 = '''# ------------------------------------------------------------------ idle steps'''
A1_NEW = '''def run_label(k):
    """A record / comparison-column key -> a prose label, e.g.
    "GSPO_G4_lr5e5" -> "GSPO(G=4) lr5e-5". Any suffix the key carries beyond
    algorithm/G/lr (a seed run, say) is kept, so two runs of the same cell can
    never print the same label."""
    alg = "GRPO" if str(k).upper().startswith("GRPO") else "GSPO"
    g, lr = _geom(k)
    rest = re.sub(r"^(GRPO|GSPO)", "", str(k), flags=re.I)
    rest = re.sub(r"^_?G\\d+", "", rest)
    rest = re.sub(r"^_?lr[0-9.eE+-]+", "", rest).strip("_")
    tail = (f"（{rest}）" if rest
            else "（论文设定）" if k == "GSPO_G32_lr5e5" else "")
    return f"{alg}(G={g}) lr{lr}{tail}"


# ------------------------------------------------------------------ idle steps'''
if s.count(A1) != 1:
    sys.exit(f"A1 anchor appears {s.count(A1)} times")

# ------------------------------------------- 2. the paired section uses it
A2 = '''        def _vs_sft_label(k):
            """A record key -> a label. Any suffix the key carries beyond
            algorithm/G/lr (a seed run, say) is kept, so two runs of the same cell
            can never print the same label."""
            alg = "GRPO" if str(k).upper().startswith("GRPO") else "GSPO"
            g, lr = _geom(k)
            rest = re.sub(r"^(GRPO|GSPO)", "", str(k), flags=re.I)
            rest = re.sub(r"^_?G\\d+", "", rest)
            rest = re.sub(r"^_?lr[0-9.eE+-]+", "", rest).strip("_")
            tail = (f"（{rest}）" if rest
                    else "（论文设定）" if k == "GSPO_G32_lr5e5" else "")
            return f"{alg}(G={g}) lr{lr}{tail}"
'''
A2_NEW = '''        _vs_sft_label = run_label
'''
if s.count(A2) != 1:
    sys.exit(f"A2 anchor appears {s.count(A2)} times")

# ------------------------------------- 3. the worst-RL-run sentence names it too
A3 = '''            out.append(f"- **分类最差的 RL run 不是 G=32 而是 {_worst_f1}**"
                       f"（macro-F1 {_f1row[_worst_f1]:.4f} 对 G=32 的 {_g32f1:.4f}），"'''
A3_NEW = '''            out.append(f"- **分类最差的 RL run 不是 G=32 而是 "
                       f"{run_label(_worst_f1)}**"
                       f"（macro-F1 {_f1row[_worst_f1]:.4f} 对 G=32 的 {_g32f1:.4f}），"'''
if s.count(A3) != 1:
    sys.exit(f"A3 anchor appears {s.count(A3)} times")

for old, new in ((A1, A1_NEW), (A2, A2_NEW), (A3, A3_NEW)):
    s = s.replace(old, new, 1)

try:
    ast.parse(s)
except SyntaxError as e:
    sys.exit(f"edited file does not parse: {e}; nothing written")
if s.count("def run_label") != 1 or "_vs_sft_label = run_label" not in s:
    sys.exit("helpers did not land as expected; nothing written")

P.write_text(s, encoding="utf-8")
print(f"final_report.py: run_label lifted to module level, {len(s.splitlines())} lines, parses")
