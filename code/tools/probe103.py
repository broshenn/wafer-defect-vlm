# Every numeral in the document that looks like it counts runs / columns / groups,
# with the line it is on. The question is which of them the checker reads.
import re, pathlib
s = pathlib.Path("/root/autodl-fs/wafer-vlm/LIMITATIONS.md").read_text(encoding="utf-8")
lines = s.split("\n")
NUM = r"[一二三四五六七八九十两]+|\d+"
# a numeral followed by a counting unit, or a unit-word followed by a numeral
PATS = [
    ("N个run",   re.compile(r"(" + NUM + r")\s*个\s*(?:RL\s*)?run")),
    ("N组",      re.compile(r"(" + NUM + r")\s*组")),
    ("N列",      re.compile(r"(" + NUM + r")\s*列")),
    ("N个run(前后)", re.compile(r"(" + NUM + r")\s*个")),
    ("N个奖励",  re.compile(r"(" + NUM + r")\s*个奖励")),
]
# what the checker's count items are anchored on, so those can be marked as covered
COVERED = [
    r"个 RL run 里唯一 macro-F1 点估计高于 SFT 的是",
    r"个 RL run 里有\s*" + NUM + r"\s*个 macro-F1 点估计",
    r"G=32 的 [\d.]+ 在(" + NUM + r")个 run 里最高，",
    r"G=32 的 [\d.]+ 在(" + NUM + r")个 run 里最高（即最差）",
    r"在(" + NUM + r")个 RL run 里，\*\*恰好只有",
    r"是(" + NUM + r")个 RL run 里最低的",
    r"在 4 个奖励里有 (" + NUM + r") 个是(" + NUM + r")组最低",
]
seen = {}
for i, ln in enumerate(lines):
    if not re.search(r"run|RL|组|列|奖励", ln):
        continue
    for nm, p in PATS:
        for m in p.finditer(ln):
            key = (i + 1, m.group(0).strip())
            cov = any(re.search(c, ln) for c in COVERED)
            seen.setdefault(key, (nm, ln.strip()[:110], cov))
for (ln, txt), (nm, ctx, cov) in sorted(seen.items()):
    print("%-5s %-14s %-6s %s" % ("L%d" % ln, nm, "COVER" if cov else "----", ctx))
print()
print("total distinct numeral occurrences:", len(seen))
