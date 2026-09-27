"""Do a patch's literal anchors still occur exactly once in the live document?

The anchors are read out of the patch's own source with `ast`, so nothing is
transcribed by hand -- a check whose input is a copy is a check that can pass while the
patch fails. Only plain string literals are checked; the f-strings a patch builds at run
time are skipped, because they are not the thing that goes stale.

    verify_patch_anchors.py <patch.py> [CONST_NAME ...]
"""
import ast
import json
import sys
from pathlib import Path

TOOLS = Path("/root/autodl-fs/wafer-vlm/tools")
DOC = Path("/root/autodl-fs/wafer-vlm/LIMITATIONS.md")

# Any constant whose name looks like an anchor is checked unless names are given.
WANT = set(sys.argv[2:])
patch = TOOLS / sys.argv[1]
src = patch.read_text(encoding="utf-8")
s = DOC.read_text(encoding="utf-8")
tree = ast.parse(src)

found, skipped, bad = 0, [], []
for node in ast.walk(tree):
    if not isinstance(node, ast.Assign):
        continue
    for t in node.targets:
        if not isinstance(t, ast.Name):
            continue
        if WANT and t.id not in WANT:
            continue
        if not WANT and not (t.id.endswith("_OLD") or t.id.endswith("_ANCHOR")
                             or t.id.startswith("P_") or t.id.endswith("_NEEDLE")):
            continue
        try:
            v = ast.literal_eval(node.value)
        except Exception:
            skipped.append(t.id)
            continue
        if not isinstance(v, str):
            skipped.append(t.id)
            continue
        found += 1
        n = s.count(v)
        head = json.dumps(v[:44], ensure_ascii=False)
        if n != 1:
            bad.append(t.id)
        print("%-16s hits=%d  %s  %s" % (t.id, n, "ok " if n == 1 else "!! ", head))

print("checked %d literal anchor(s); f-strings/expressions skipped: %s"
      % (found, ", ".join(skipped) if skipped else "none"))
if bad:
    print("NOT PRESENT EXACTLY ONCE:", ", ".join(bad))
    sys.exit(1)
print("every literal anchor is present exactly once")
