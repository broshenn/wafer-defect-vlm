#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段 C：56 图候选上 Base / L / D 的**离线**类别与格式分析。

口径（任务书 §2.C）：
  · 三个模型**同一题面、同一解析规则**；
  · 类别真值 = 公开 `manifest.failure_type`（`ground_truth`），**只在这里读 gold**；
  · 未解析**按类别错误计**；围栏诊断对全部候选**统一实施**、另列；
  · 报七类（实际覆盖）的 Accuracy / Macro-F1 / 每类 P-R-F1 / 混淆矩阵；
    另列「固定九类」Macro-F1 时**必须写明缺 Near_full / none 及处理方式**，
    **不与旧 36/84 图成绩换分母比较**；
  · 格式与视觉事实**分列**：严格 JSON、七字段完整合法率、缺字段、约定警告、
    统一完整围栏诊断；
  · 同图配对差 + 按图 bootstrap 区间，**固定 seed 3407**；
  · 原答**不修改**后再当成绩。

跑法：python 类别与格式分析.py <原答目录> [--json 输出.json]
"""
from __future__ import annotations
import argparse, json, random, sys
from collections import Counter, defaultdict
from pathlib import Path

D = Path(__file__).resolve().parent
sys.path.insert(0, str(D))
import schema_check as SC                                    # 冻结检查器（不改）

SEVEN = ["Center", "Donut", "Edge_Loc", "Edge_Ring", "Loc", "Random", "Scratch"]
NINE = SEVEN + ["Near_full", "none"]
SEED = 3407
BOOT = 10000


def fence_strip(text: str) -> str:
    """统一围栏诊断：去掉一层 ```json ``` 围栏后再按同一检查器判。"""
    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else ""
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    return s.strip()


def classify(raw: str):
    """返回 (严格类别, 围栏诊断类别, 检查结果)。未解析 → None。"""
    chk = SC.check_answer(raw)
    cls = None
    if chk["schema_ok"]:
        obj, st = SC.parse_strict(raw)
        if st == "ok" and obj:
            cls = obj.get("defect_class")
    fs = SC.check_answer(fence_strip(raw))
    fcls = None
    if fs["schema_ok"]:
        obj, st = SC.parse_strict(fence_strip(raw))
        if st == "ok" and obj:
            fcls = obj.get("defect_class")
    return cls, fcls, chk, fs


def prf(pairs, classes):
    """pairs = [(truth, pred)]；pred None 计为错误（不计入任何类 TP/FP，只压 recall）。"""
    per = {}
    for c in classes:
        tp = sum(1 for t, p in pairs if t == c and p == c)
        fp = sum(1 for t, p in pairs if t != c and p == c)
        fn = sum(1 for t, p in pairs if t == c and p != c)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        per[c] = {"n": sum(1 for t, _ in pairs if t == c), "tp": tp, "fp": fp, "fn": fn,
                  "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}
    macro = sum(v["f1"] for v in per.values()) / len(classes) if classes else 0.0
    micro = sum(1 for t, p in pairs if t == p) / len(pairs) if pairs else 0.0
    return per, round(macro, 4), round(micro, 4)


def boot_ci(correct: list[int], n=BOOT, seed=SEED):
    """按图 bootstrap 95% 区间（对 0/1 正确序列重采样）。"""
    rnd = random.Random(seed)
    N = len(correct)
    means = []
    for _ in range(n):
        s = sum(correct[rnd.randrange(N)] for _ in range(N))
        means.append(s / N)
    means.sort()
    return round(means[int(0.025 * n)], 4), round(means[int(0.975 * n) - 1], 4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("answer_dir")
    ap.add_argument("--gold", default=str(D / "冻结清单.json"))
    ap.add_argument("--json", default=str(D / "类别与格式结果.json"))
    a = ap.parse_args()
    ans_dir = Path(a.answer_dir)
    gold = {r["sample_id"]: r for r in
            json.loads(Path(a.gold).read_text(encoding="utf-8"))["逐图"]}
    order = [r["sample_id"] for r in
             json.loads(Path(a.gold).read_text(encoding="utf-8"))["逐图"]]

    out = {"口径": {
        "真值": "公开 manifest.failure_type (ground_truth)，只在本离线脚本读，未下发模型",
        "未解析处理": "按类别错误计（不计入任何类别的 TP/FP，只压该图 recall）",
        "围栏诊断": "统一去掉一层 ```json 围栏后按同一冻结检查器再判，对全部候选实施",
        "解析器": "schema_check.py（冻结，SHA 7d46c37f72bb8fd09488dd73754ad75e6a1b711d91a6f08f82e5b34026c0c0ff）",
        "种子": SEED, "bootstrap": BOOT,
        "边界": "56 图仅覆盖七类；Near_full/none 缺候选，九类 Macro-F1 以缺类计 0 呈现，不代表九类全面覆盖",
    }, "模型": {}}

    cat = {}
    for tag in ["Base", "L-N3072-3407", "D-N3072-3407"]:
        p = ans_dir / f"{tag}_raw.jsonl"
        if not p.exists():
            print(f"**缺 {p.name}**"); return 2
        rows = [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
        by_id = {r["sample_id"]: r for r in rows}
        bundle = {"N": len(rows), "缺图": [s for s in order if s not in by_id],
                  "逐图": {}, "格式": {}, "类别": {}}
        pairs, fence_pairs, correct, fence_correct = [], [], [], []
        fmt = Counter(); missing_field = Counter(); warn_counter = Counter()
        ps = Counter(); fps = Counter()
        for sid in order:
            r = by_id.get(sid)
            if r is None:
                continue
            cls, fcls, chk, fs = classify(r["raw"])
            truth = gold[sid]["failure_type"]
            pairs.append((truth, cls)); fence_pairs.append((truth, fcls))
            ok = int(cls == truth); correct.append(ok)
            fence_correct.append(int(fcls == truth))
            fmt["严格JSON通过" if chk["strict_json"] else "严格JSON失败"] += 1
            fmt["schema全合法" if chk["schema_ok"] else "schema不合法"] += 1
            fmt["围栏诊断通过" if fs["schema_ok"] else "围栏诊断失败"] += 1
            ps[chk["parse_status"]] += 1
            if fs["schema_ok"] and not chk["schema_ok"]:
                fmt["仅围栏补救成功"] += 1
            for pr in chk["schema_problems"]:
                if pr.startswith("缺字段:"):
                    for f in pr[4:].split(","):
                        missing_field[f] += 1
            for w in chk["contract_warnings"]:
                warn_counter[w.split(":")[0]] += 1
            for f in SC.FIELDS:
                obj, st = SC.parse_strict(r["raw"])
                if st == "ok" and obj and f not in obj:
                    fps[f] += 1
            bundle["逐图"][sid] = {
                "真值": truth, "严格类别": cls, "围栏类别": fcls,
                "严格对": ok, "围栏对": int(fcls == truth),
                "parse_status": chk["parse_status"],
                "schema_ok": chk["schema_ok"], "schema问题": chk["schema_problems"],
                "警告": chk["contract_warnings"],
                "是否输出七字段": sorted(k for k in (SC.parse_strict(r["raw"])[0] or {})
                                      if k in SC.FIELDS),
                "秒": r.get("seconds"), "finish": r.get("finish_reason"),
                "raw_sha256前缀": __import__("hashlib").sha256(
                    r["raw"].encode("utf-8")).hexdigest()[:16],
            }
        per7, macro7, acc7 = prf(pairs, SEVEN)
        per9, macro9, _ = prf(pairs, NINE)
        fper7, fmacro7, facc7 = prf(fence_pairs, SEVEN)
        bundle["类别"] = {
            "N": len(pairs), "严格Acc": acc7, "严格MacroF1_七类": macro7,
            "严格每类": per7,
            "围栏Acc": facc7, "围栏MacroF1_七类": fmacro7, "围栏每类": fper7,
            "九类口径": {
                "MacroF1": macro9, "每类": per9,
                "说明": "固定九类 Macro-F1 把缺席的 Near_full / none 按各自 n=0、F1=0 计入分母；"
                        "**仅表示本 56 图七类覆盖下的口径**，不构成九类全面覆盖，"
                        "也不与旧 36 / 84 图的成绩换分母比较。",
            },
            "混淆矩阵": {t: {p or "(未解析)": sum(1 for tt, pp in pairs if tt == t and pp == p)
                          for p in sorted({pp for tt, pp in pairs if tt == t}, key=str)}
                      for t in SEVEN},
            "Acc_bootstrap95": boot_ci(correct),
            "围栏Acc_bootstrap95": boot_ci(fence_correct),
        }
        bundle["格式"] = {
            "严格JSON通过": fmt["严格JSON通过"], "schema全合法": fmt["schema全合法"],
            "围栏诊断通过": fmt["围栏诊断通过"], "仅围栏补救成功": fmt.get("仅围栏补救成功", 0),
            "parse_status分布": dict(ps), "缺字段计数": dict(missing_field),
            "约定警告计数": dict(warn_counter), "七字段出现计数": dict(fps),
        }
        out["模型"][tag] = bundle
        cat[tag] = (pairs, fence_pairs, correct, bundle)

    # 同图配对差（L vs D、L vs Base、D vs Base）
    pt = {}
    for x, y in [("L-N3072-3407", "D-N3072-3407"), ("L-N3072-3407", "Base"),
                 ("D-N3072-3407", "Base")]:
        bx, by = out["模型"][x], out["模型"][y]
        d = [bx["逐图"][s]["严格对"] - by["逐图"][s]["严格对"] for s in order
             if s in bx["逐图"] and s in by["逐图"]]
        diffs = [1 if v > 0 else -1 if v < 0 else 0 for v in d]
        rnd = random.Random(SEED); N = len(diffs); ms = []
        for _ in range(BOOT):
            ms.append(sum(diffs[rnd.randrange(N)] for _ in range(N)) / N)
        ms.sort()
        both_right = sum(1 for v, s in zip(d, order) if v == 0 and bx["逐图"][s]["严格对"] == 1)
        both_wrong = sum(1 for v, s in zip(d, order) if v == 0 and bx["逐图"][s]["严格对"] == 0)
        pt[f"{x} − {y}"] = {
            "配对N": N, "均值差(严格Acc)": round(sum(diffs) / N, 4),
            "bootstrap95": [round(ms[int(0.025 * BOOT)], 4), round(ms[int(0.975 * BOOT) - 1], 4)],
            "同对同错": both_wrong, "同对同对": both_right,
            "x优": sum(1 for v in diffs if v > 0), "y优": sum(1 for v in diffs if v < 0),
        }
    out["同图配对差"] = pt

    Path(a.json).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    print(f"写出 {a.json}")

    # ---- 控制台小结 ----
    print(f"\n{'模型':<16}{'严格Acc':>9}{'MacroF1(7)':>12}{'围栏Acc':>9}{'schema合法':>11}{'警告':>6}")
    for t in ["Base", "L-N3072-3407", "D-N3072-3407"]:
        b = out["模型"][t]
        print(f"{t:<16}{b['类别']['严格Acc']:>9.4f}{b['类别']['严格MacroF1_七类']:>12.4f}"
              f"{b['类别']['围栏Acc']:>9.4f}{b['格式']['schema全合法']:>11}"
              f"{sum(b['格式']['约定警告计数'].values()):>6}")
    print("\n同图配对差：")
    for k, v in pt.items():
        print(f"  {k}: 均值 {v['均值差(严格Acc)']:+.4f}  CI {v['bootstrap95']}  同对同错 {v['同对同错']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
