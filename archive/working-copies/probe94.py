# Do patch94's two anchors still land, and would the note go where it is meant to?
# The same expressions the patch uses, run read-only against the live document.
import json
import re
from pathlib import Path

s = Path("/root/autodl-fs/wafer-vlm/LIMITATIONS.md").read_text(encoding="utf-8")

LR_PARA = re.compile(
    r"(算法内部的学习率效应（同算法、同组大小、只差学习率）同样显著：\n.+?。)", re.S)
m = LR_PARA.search(s)
print("LR_PARA matches:", bool(m), "->", "would append after \"%s\""
      % (m.group(1)[-12:] if m else "N/A"))

NEEDLE = "个 macro-F1 点估计高于 SFT"
print("NEEDLE hits:", s.count(NEEDLE))
if s.count(NEEDLE) == 1:
    i = s.find(NEEDLE)
    j = s.find("\n\n", i)
    print("paragraph spans lines %d..%d" % (s[:i].count("\n") + 1,
                                            s[:j].count("\n") + 1))
    print("last line of the paragraph:", json.dumps(s[s.rfind("\n", 0, j) + 1:j],
                                                    ensure_ascii=False))
    probe = s[:j] + "\n  但短名单里 GRPO(G=4) lr1e-5 的那个点估计是**单次抽样**：…" + s[j:]
    print("after insertion: bold parity %s, section 6 still once: %s"
          % (("OK" if (probe.count("**") - s.count("**")) % 2 == 0 else "ODD"),
             probe.count("## 6. 评测与指标解释") == 1))
    print("5.2.7 anchor '## 6. 评测与指标解释' hits:", s.count("## 6. 评测与指标解释"))
    print("5.2.7 already present:", "#### 5.2.7" in s)

# Would the insertion land inside any line the claims checker reads? The shape-2 list
# capture is [^\n]+ on the list line, so a note on its own line must not matter.
P_ABOVE_MANY = (r"((?:[一二三四五六七八九十两]+|\d+))\s*个 RL run 里有\s*"
                r"((?:[一二三四五六七八九十两]+|\d+))\s*个 macro-F1 点估计高于 SFT：([^\n]+)")
mm = re.search(P_ABOVE_MANY, s)
print("checker shape-2 still matches:", bool(mm),
      "| list ends with:", json.dumps(mm.group(3)[-22:], ensure_ascii=False) if mm else "")
