"""Record the centroid-radius evidence instead of typing it into prose.

tools/audit_report_numbers.py flagged two numbers in LIMITATIONS.md that no
record backs: the per-pattern median centroid radius (Donut 0.157, Edge_Ring
0.063) and the share of the benchmark whose published radial_zone label is
"center" (83.7%).

Both turned out to be *correct* -- recomputed from the manifest they are 0.1574,
0.0631 and 210/251 = 83.67% -- so this is not a wrong-number bug. It is worse in
one narrow way: 0.063 was a literal typed into two different source files
(final_report.py's prose and radial_zone_audit.py's docstring), so the argument
would keep reading correctly after the data changed underneath it. That is the
same shape as the two failures already logged in LIMITATIONS section 8.

So the audit tool now computes both quantities and writes them into
radial_zone_audit.json, and final_report.py reads them back instead of restating
them. While computing the share, a real ambiguity surfaced: the prose's 83.7%
uses 251 scored rows, while the record's majority_floor uses all 252. Both are
defensible and they disagree, so the record now carries both denominators under
explicit names rather than leaving the choice implicit in whatever number
someone typed.

Neither edit depends on a re-run to be meaningful: the computation reproduces
values identical to the literals it replaces, and running it confirms that.
"""
import ast
import sys
from pathlib import Path

TOOLS = Path("/root/autodl-fs/wafer-vlm/tools")
edits = []

# ---------------------------------------------------------------- 1. record it
p = TOOLS / "radial_zone_audit.py"
s = p.read_text(encoding="utf-8")

old = '''    golds, special = {}, {}
    for row in gold_rows:
        sid = row["sample_id"]
        record = manifest[sid]
        golds[sid] = definitions(record["features"], record["failure_type"])
        special[sid] = golds[sid].pop("_special")
'''
new = '''    golds, special = {}, {}
    radii_by_class: dict[str, list[float]] = collections.defaultdict(list)
    for row in gold_rows:
        sid = row["sample_id"]
        record = manifest[sid]
        golds[sid] = definitions(record["features"], record["failure_type"])
        special[sid] = golds[sid].pop("_special")
        # The report argues that the published radial_zone label collapses to
        # "center" by arithmetic, citing the median centroid radius of each
        # pattern as the evidence. Those medians used to be literals typed into
        # prose in two different files; recording them here makes the argument
        # checkable and stops the two copies from drifting apart.
        radius = record["features"].get("centroid_radius_r")
        if radius is not None:
            radii_by_class[str(record["failure_type"])].append(radius)
'''
edits.append(("audit:collect-radii", p, s, old, new))

old = '''    result: dict[str, object] = {
        "manifest_reproduces_gold": True,
        "gold_distributions": {
            name: {"counts": dict(collections.Counter(v[name] for v in golds.values())),
                   "majority_floor": max(collections.Counter(
                       v[name] for v in golds.values()).values()) / len(golds)}
            for name in ("centroid", "mean_radius", "outer_extent")},
        "runs": {},
    }
'''
new = '''    # How often the published rule answers "center", by both denominators.
    # The prose and the record previously each used a different one -- 210/252
    # is 83.33% but 210/251, dropping the single unlabelled row, is 83.67% --
    # so both are written down under explicit names rather than left to
    # whichever number happened to be typed.
    centroid_zones = collections.Counter(v["centroid"] for v in golds.values())
    center_hits = centroid_zones.get("center", 0)
    n_all = len(golds)
    n_scored = n_all - centroid_zones.get("none", 0)

    result: dict[str, object] = {
        "manifest_reproduces_gold": True,
        "gold_distributions": {
            name: {"counts": dict(collections.Counter(v[name] for v in golds.values())),
                   "majority_floor": max(collections.Counter(
                       v[name] for v in golds.values()).values()) / len(golds)}
            for name in ("centroid", "mean_radius", "outer_extent")},
        "centroid_radius_by_class": {
            cls: {"n": len(vals), "median": statistics.median(vals)}
            for cls, vals in sorted(radii_by_class.items())},
        "center_share": {
            "definition": ("gold rows whose published radial_zone is 'center' "
                           "under the _zone(centroid_r) rule"),
            "count": center_hits,
            "n_all": n_all,
            "n_scored": n_scored,
            "of_all": center_hits / n_all,
            "of_scored": center_hits / n_scored,
        },
        "runs": {},
    }
'''
edits.append(("audit:record-share", p, s, old, new))

# --------------------------------------------------------------- 2. read it back
p = TOOLS / "final_report.py"
s = p.read_text(encoding="utf-8")

old = '''    # The radial_zone floor is not a fair bar: the label is _zone(centroid_r),
    # which for any pattern symmetric about the wafer centre collapses to
    # "center" regardless of where the defect actually sits -- an edge ring has
    # a median centroid radius of 0.063. So re-grade the same predictions against
    # a location-based reading of the field and report both numbers.
'''
new = '''    # The radial_zone floor is not a fair bar: the label is _zone(centroid_r),
    # which for any pattern symmetric about the wafer centre collapses to
    # "center" regardless of where the defect actually sits. So re-grade the
    # same predictions against a location-based reading of the field and report
    # both numbers. The medians quoted below are read from the audit record
    # rather than restated here, so this paragraph cannot go stale on its own.
'''
edits.append(("report:comment", p, s, old, new))

old = '''        out.append("现状标签是 `_zone(centroid_r)`，即**缺陷质心**所在的一带。"
                   "对称图案（边缘环、随机散布、满片）的质心必在圆心，"
                   "于是标签被算术地压成 `center`——全基准 83.7% 如此，"
                   "`Edge_Ring` 的中位质心半径只有 0.063。"
                   "下面对**同一批预测**换用「缺陷落在哪一带」重新评分：")
'''
new = '''        share = audit.get("center_share") or {}
        edge_ring = (audit.get("centroid_radius_by_class") or {}).get("Edge_Ring") or {}
        # The fraction itself is quoted, not just its percentage: pct() renders
        # four decimals rather than a percentage despite its name, and a
        # percentage alone would be a number no record holds. Printing "210 of
        # 251 (83.7%)" keeps both the recorded counts and the arithmetic visible.
        # An absent record degrades the wording rather than reinstating the old
        # value, so the sentence cannot silently go stale.
        share_text = (f"已评分的 {share['n_scored']} 行里有 {share['count']} 行"
                      f"（{share['of_scored'] * 100:.1f}%）如此"
                      if share.get("n_scored") and share.get("of_scored") is not None
                      else "该比例见 radial_zone_audit.json（记录缺失，此处不引数字）")
        radius_text = (f"{edge_ring['median']:.3f}" if edge_ring.get("median") is not None
                       else "见 radial_zone_audit.json（记录缺失）")
        out.append("现状标签是 `_zone(centroid_r)`，即**缺陷质心**所在的一带。"
                   "对称图案（边缘环、随机散布、满片）的质心必在圆心，"
                   f"于是标签被算术地压成 `center`——{share_text}，"
                   f"`Edge_Ring` 的中位质心半径只有 {radius_text}。"
                   "下面对**同一批预测**换用「缺陷落在哪一带」重新评分：")
'''
edits.append(("report:prose", p, s, old, new))

by_file: dict[Path, str] = {}
for tag, path, src, old, new in edits:
    n = src.count(old)
    if n != 1:
        sys.exit(f"FAILED at {tag}: {n} occurrences (need exactly 1)")
    by_file.setdefault(path, src)
    by_file[path] = by_file[path].replace(old, new)
    print(f"  {tag}: ok")

for path, src in by_file.items():
    ast.parse(src)
    path.write_text(src, encoding="utf-8")
print("centroid-radius evidence is now computed, recorded, and read back")
