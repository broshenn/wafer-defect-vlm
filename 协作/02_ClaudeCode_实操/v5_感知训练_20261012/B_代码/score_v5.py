#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v5 打分（合同 8.2；B 卡 7）。
用法：python score_v5.py --raw <原答.jsonl> --ref <评测数据.jsonl> --kind main|L
      --out <metrics.json> [--rot-pair <原图 TEST 原答.jsonl>] [--per-sample <per_sample.jsonl>]
- main：parse_v5 后计算几何/类别/格式/一致性全部指标；每样本逐项写 per_sample.jsonl（带 sample_id/lot_name）。
- L：从回答里抽 defect_class（common_v5.parse），只算类别指标。
- 缺失的原答计"未完成"单列，不算对也不算错；有原答但格式不合法计错（进入比率分母）。
- --rot-pair：旋转等变（ROT200 按 src_id 与原图预测配对）写进 metrics.json。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common_v5 import (CLASSES, COV_ORDER, FIELDS, band_of_ratio, parse_raw_strict,  # noqa: E402
                       parse_v5, schema_errors, strip_envelope, zones_of_frac)


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def six(ref):
    return {k: ref[k] for k in FIELDS}


def score_main(raw_rows, ref_rows):
    raw_by_sid = {r["sample_id"]: r for r in raw_rows}
    per = []
    counters = defaultdict(int)
    mae_sum = {"red_ratio": 0.0, "zone_frac_l1": 0.0, "R1": 0.0, "R2": 0.0}
    mae_n = 0
    cons = []
    for ref_row in ref_rows:
        sid = ref_row["sample_id"]
        ref = parse_raw_strict(ref_row["reference_json"])
        assert not schema_errors(six(ref)), sid
        lot = ref.get("lot_name", "")
        present = sid in raw_by_sid
        parsed = parse_v5(raw_by_sid[sid]["raw"]) if present else None
        row = {"sample_id": sid, "lot_name": lot, "present": present}
        ok = bool(parsed and parsed["ok"])
        row["ok"] = ok
        if parsed is not None:
            row["raw_ok"] = parsed["raw_ok"]
        if not present:
            counters["missing"] += 1
            per.append(row)
            continue
        counters["attempted"] += 1
        # 几何
        if ref["coverage_band"] != "unknown":
            counters["cov_n"] += 1
            if ok:
                band = parsed["obj"]["coverage_band"]
                if band == ref["coverage_band"]:
                    counters["cov_exact"] += 1
                    row["cov_exact"] = True
                else:
                    row["cov_exact"] = False
                if band in COV_ORDER and abs(COV_ORDER.index(band) - COV_ORDER.index(ref["coverage_band"])) <= 1:
                    counters["cov_adj"] += 1
                    row["cov_adj"] = True
                else:
                    row["cov_adj"] = False
            else:
                row["cov_exact"] = row["cov_adj"] = False
        if ref["red_mass_zones"] != "unknown":
            counters["zones_n"] += 1
            if ok:
                mz = parsed["obj"]["red_mass_zones"]
                if mz == "unknown":
                    row["zones_exact"] = False
                    row["zones_jaccard"] = 0.0
                else:
                    ms, rs = set(mz), set(ref["red_mass_zones"])
                    row["zones_exact"] = ms == rs
                    row["zones_jaccard"] = len(ms & rs) / len(ms | rs) if (ms | rs) else 1.0
            else:
                row["zones_exact"] = False
                row["zones_jaccard"] = 0.0
            if row.get("zones_exact"):
                counters["zones_exact"] += 1
            counters["zones_jac_sum"] = counters.get("zones_jac_sum", 0.0) + row["zones_jaccard"]
        if ref["angular_type"] != "unknown":
            counters["ang_n"] += 1
            if ok and parsed["obj"]["angular_type"] == ref["angular_type"]:
                counters["ang_ok"] += 1
                row["ang_ok"] = True
            else:
                row["ang_ok"] = False
            if ref["angular_type"] in ("single", "opposed"):
                counters["sec_n"] += 1
                if ok and (parsed["obj"]["angular_type"] == ref["angular_type"]
                           and set(parsed["obj"]["clock_sectors"]) == set(ref["clock_sectors"])):
                    counters["sec_ok"] += 1
                    row["sector_ok"] = True
                else:
                    row["sector_ok"] = False
        # measure 误差（解析成功才计）
        if ok:
            m = parsed["measure"]
            ex = ref["measure_exact"]
            mae_sum["red_ratio"] += abs(m["red_ratio"] - ex["red_ratio"])
            mae_sum["zone_frac_l1"] += sum(abs(a - b) for a, b in zip(m["zone_frac"], ex["zone_frac"]))
            mae_sum["R1"] += abs(m["R1"] - ex["R1"])
            mae_sum["R2"] += abs(m["R2"] - ex["R2"])
            mae_n += 1
            c1 = 1.0 if band_of_ratio(m["red_ratio"]) == parsed["obj"]["coverage_band"] else 0.0
            if parsed["obj"]["red_mass_zones"] == "unknown":
                c2 = 0.0
            else:
                c2 = 1.0 if zones_of_frac(m["zone_frac"]) == list(parsed["obj"]["red_mass_zones"]) else 0.0
            row["consistency"] = (c1 + c2) / 2
            cons.append(row["consistency"])
        # 类别
        truth = ref["defect_class"]
        row["true_class"] = truth
        if ok and parsed["obj"]["defect_class"] == truth:
            counters["class_ok"] += 1
            row["class_ok"] = True
            row["class_pred"] = truth
        else:
            row["class_ok"] = False
            row["class_pred"] = (parsed["obj"]["defect_class"] if ok else None)
        per.append(row)
    geo_parts = []
    if counters["cov_n"]:
        geo_parts.append(counters["cov_exact"] / counters["cov_n"])
    if counters["zones_n"]:
        geo_parts.append(counters["zones_exact"] / counters["zones_n"])
    if counters["ang_n"]:
        geo_parts.append(counters["ang_ok"] / counters["ang_n"])
    geoscore = sum(geo_parts) / len(geo_parts) if geo_parts else None
    # 类 F1
    tp = Counter()
    fp = Counter()
    fn = Counter()
    support = Counter()
    for row in per:
        if not row["present"]:
            continue
        ref_row = next(r for r in ref_rows if r["sample_id"] == row["sample_id"])
        truth = parse_raw_strict(ref_row["reference_json"])["defect_class"]
        support[truth] += 1
        pred = row.get("class_pred")
        if pred == truth:
            tp[truth] += 1
        else:
            fn[truth] += 1
            if pred in CLASSES:
                fp[pred] += 1
    per_class = {}
    f1s = []
    for c in CLASSES:
        p = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        r = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        per_class[c] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
                        "support": support[c]}
        f1s.append(f1)
    n_attempted = counters["attempted"] or 1
    metrics = {
        "kind": "main", "n_ref": len(ref_rows), "n_attempted": counters["attempted"],
        "n_missing": counters["missing"],
        "geoscore": round(geoscore, 5) if geoscore is not None else None,
        "coverage": {"exact_rate": round(counters["cov_exact"] / counters["cov_n"], 5) if counters["cov_n"] else None,
                     "adj_rate": round(counters["cov_adj"] / counters["cov_n"], 5) if counters["cov_n"] else None,
                     "n": counters["cov_n"]},
        "zones": {"exact_rate": round(counters["zones_exact"] / counters["zones_n"], 5) if counters["zones_n"] else None,
                  "jaccard_mean": round(counters.get("zones_jac_sum", 0.0) / counters["zones_n"], 5) if counters["zones_n"] else None,
                  "n": counters["zones_n"]},
        "angular_type": {"acc": round(counters["ang_ok"] / counters["ang_n"], 5) if counters["ang_n"] else None,
                         "n": counters["ang_n"]},
        "sector": {"acc": round(counters["sec_ok"] / counters["sec_n"], 5) if counters["sec_n"] else None,
                   "n": counters["sec_n"]},
        "measure_mae": {k: round(v / mae_n, 5) for k, v in mae_sum.items()} if mae_n else None,
        "class": {"acc": round(counters["class_ok"] / n_attempted, 5),
                  "macro_f1": round(sum(f1s) / len(f1s), 5),
                  "per_class": per_class,
                  "unknown_rate": round(sum(1 for r in per if r.get("class_pred") == "unknown") / n_attempted, 5)},
        "format": {"stripped_legal_rate": round(sum(1 for r in per if r.get("ok")) / n_attempted, 5),
                   "raw_legal_rate": round(sum(1 for r in per if r.get("raw_ok")) / n_attempted, 5)},
        "consistency_mean": round(sum(cons) / len(cons), 5) if cons else None,
    }
    return metrics, per


def score_L(raw_rows, ref_rows):
    raw_by_sid = {r["sample_id"]: r for r in raw_rows}
    per = []
    for ref_row in ref_rows:
        sid = ref_row["sample_id"]
        truth = ref_row.get("label")
        if truth is None:
            truth = parse_raw_strict(ref_row["reference_json"])["defect_class"]
        present = sid in raw_by_sid
        pred = None
        if present:
            try:
                obj = parse_raw_strict(strip_envelope(raw_by_sid[sid]["raw"]))
                cand = obj.get("defect_class")
                pred = cand if cand in CLASSES else (cand if cand == "unknown" else None)
            except Exception:
                pred = None
        per.append({"sample_id": sid, "lot_name": "", "present": present,
                    "class_pred": pred, "class_ok": pred == truth, "true_class": truth})
    tp = Counter(); fp = Counter(); fn = Counter(); support = Counter()
    n_att = sum(1 for r in per if r["present"]) or 1
    for row in per:
        if not row["present"]:
            continue
        ref_row = next(x for x in ref_rows if x["sample_id"] == row["sample_id"])
        truth = ref_row.get("label") or parse_raw_strict(ref_row["reference_json"])["defect_class"]
        support[truth] += 1
        if row["class_pred"] == truth:
            tp[truth] += 1
        else:
            fn[truth] += 1
            if row["class_pred"] in CLASSES:
                fp[row["class_pred"]] += 1
    per_class = {}
    f1s = []
    for c in CLASSES:
        p = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        r = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        per_class[c] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
                        "support": support[c]}
        f1s.append(f1)
    metrics = {"kind": "L", "n_ref": len(ref_rows),
               "n_attempted": sum(1 for r in per if r["present"]),
               "n_missing": sum(1 for r in per if not r["present"]),
               "class": {"acc": round(sum(1 for r in per if r["class_ok"]) / n_att, 5),
                         "macro_f1": round(sum(f1s) / len(f1s), 5), "per_class": per_class,
                         "unknown_rate": round(sum(1 for r in per if r["class_pred"] == "unknown") / n_att, 5)}}
    return metrics, per


def rot_metrics(rot_raw, rot_ref, orig_raw):
    rot_by_sid = {r["sample_id"]: r for r in rot_raw}
    orig_by_sid = {r["sample_id"]: r for r in orig_raw}
    cov_n = cov_same = zone_n = zone_same = 0
    sec_n = sec_ok = 0
    for ref_row in rot_ref:
        sid = ref_row["sample_id"]
        if sid not in rot_by_sid or sid not in orig_by_sid:
            continue
        pr = parse_v5(rot_by_sid[sid]["raw"])
        po = parse_v5(orig_by_sid[sid]["raw"])
        if not (pr["ok"] and po["ok"]):
            continue
        cov_n += 1
        cov_same += pr["obj"]["coverage_band"] == po["obj"]["coverage_band"]
        zone_n += 1
        zr = "unknown" if pr["obj"]["red_mass_zones"] == "unknown" else set(pr["obj"]["red_mass_zones"])
        zo = "unknown" if po["obj"]["red_mass_zones"] == "unknown" else set(po["obj"]["red_mass_zones"])
        zone_same += zr == zo
        if po["obj"]["angular_type"] == "single" and po["obj"]["clock_sectors"]:
            sec_n += 1
            want = (po["obj"]["clock_sectors"][0] - 1 + 3) % 12 + 1
            if pr["obj"]["angular_type"] == "single" and pr["obj"]["clock_sectors"] == [want]:
                sec_ok += 1
    return {"coverage_unchanged": {"same": cov_same, "n": cov_n,
                                   "rate": round(cov_same / cov_n, 5) if cov_n else None},
            "zones_unchanged": {"same": zone_same, "n": zone_n,
                                "rate": round(zone_same / zone_n, 5) if zone_n else None},
            "sector_plus3": {"ok": sec_ok, "n": sec_n,
                             "rate": round(sec_ok / sec_n, 5) if sec_n else None}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--kind", choices=["main", "L"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rot-pair", default=None)
    ap.add_argument("--per-sample", default=None)
    a = ap.parse_args()
    raw_rows = load_jsonl(a.raw)
    ref_rows = load_jsonl(a.ref)
    if a.kind == "main":
        metrics, per = score_main(raw_rows, ref_rows)
        if a.rot_pair:
            metrics["rot_pair"] = rot_metrics(raw_rows, ref_rows, load_jsonl(a.rot_pair))
    else:
        metrics, per = score_L(raw_rows, ref_rows)
    metrics["raw_file"] = str(a.raw)
    metrics["ref_file"] = str(a.ref)
    if a.per_sample:
        with open(a.per_sample, "w", encoding="utf-8", newline="\n") as f:
            for r in per:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    Path(a.out).write_text(json.dumps(metrics, ensure_ascii=False, indent=1) + "\n",
                           encoding="utf-8", newline="\n")
    print(json.dumps({k: metrics[k] for k in metrics
                      if k in ("kind", "n_ref", "n_attempted", "n_missing", "geoscore",
                               "consistency_mean", "class", "format", "rot_pair")},
                     ensure_ascii=False)[:600])
    return 0


if __name__ == "__main__":
    sys.exit(main())
