#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Part A 复算：按 Codex 验收意见重算全部指标，不改任何原答。

Codex 2026-10-08 验收（`协作/01_Codex_指挥/验收_20261008_夜间3072与下一轮.md`）要求：

1. 「七字段Acc/MF1」实际只评价 defect_class → 必须改名为
   **「七字段题面中的类别指标」**，并分别给出 JSON 解析、完整 schema、
   clock_direction 字面 "null" 等约定违例、以及未测事实字段。
2. Base 的**原严格分**与**统一去完整围栏诊断分**必须分列；
   字段缺失不能算内容空白，更不能算满分。
3. 配对区间必须 **Macro-F1 与 Accuracy 分别计算**；
   同 seed3407 的 D/L 为主配对，另两个描述 seed 只作波动。
4. 小规模交接里的 0.7941 等若分母是 102 不能标成 val84；
   从原答重算两集合，分母写清。

口径（继承原合同，未放宽）：
  · 严格 JSON：唯一合法 defect_class，允许附加字段；
    拒重复键 / 截断 / 多对象 / 尾随文本 / NaN
  · val84 = 原答里 part == "val90" 的 84 条；full102 = 全部 102 条（含 dev18 18 条）
  · 围栏剥离只作**离线诊断**，主分不动
  · 不把「格式合规」当「事实正确」；形态/径向/方向/Caption 无独立 gold → 未测
"""
from __future__ import annotations
import io, json, random, re, sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
NIGHT = HERE.parent / "夜间扩容_20261008_0244" / "结果"
OUT = HERE / "结果"
OUT.mkdir(exist_ok=True)

CLASSES = ["Center", "Donut", "Edge_Loc", "Edge_Ring", "Loc",
           "Near_full", "Random", "Scratch", "none"]
ENUM = CLASSES + ["unknown"]
EMPTY_THINK = re.compile(r"^\s*<think\b[^>]*>\s*</think\s*>\s*", re.S)
FENCE = re.compile(r"^\s*```[a-zA-Z0-9_-]*\s*\n(.*?)\n?\s*```\s*$", re.S)

# 七字段题面的 full schema 约定——**逐字照抄冻结的评测协议**，
# 定义在 `.ssh-tmp/night/eval_7f_5090.py:107 schema7()`，与各原答里存的
# `schema_ok` / `schema_problems` 字段同源，不得自行放宽或收紧：
#   七键齐备且无多余键；defect_class ∈ 九类+unknown；radial_zone ∈ 六值；
#   clock_direction 为字符串或 JSON null（**字符串 "null" 类型合法**）；
#   extent_r 必须是**真 JSON null**（任何非 null 值都算违例）；
#   morphology / caption_zh / uncertainty 必须是字符串。
FIELDS = ["defect_class", "morphology", "radial_zone", "clock_direction",
          "extent_r", "caption_zh", "uncertainty"]
ZONES = ["center", "middle", "edge", "global", "none", "unknown"]
ENUM_CLASS = CLASSES + ["unknown"]


class NonFinite(ValueError):
    pass


def parse_json(text):
    if not isinstance(text, str):
        return None, "not_str"
    s = EMPTY_THINK.sub("", text.strip())
    if not s:
        return None, "empty"

    def _nodup(pairs):
        k = [p[0] for p in pairs]
        if len(k) != len(set(k)):
            raise ValueError("dup")
        return dict(pairs)

    def _const(x):
        raise NonFinite(x)
    try:
        obj, end = json.JSONDecoder(object_pairs_hook=_nodup,
                                    parse_constant=_const).raw_decode(s)
    except NonFinite:
        return None, "non_finite"
    except ValueError:
        return None, "duplicate_key"
    except json.JSONDecodeError:
        if s.startswith("{") and not s.rstrip().endswith("}"):
            return None, "truncated"
        if s.count("{") > 1:
            return None, "multiple_objects"
        return None, "not_json"
    rest = s[end:].strip()
    if rest:
        return None, ("multiple_objects" if "{" in rest else "trailing_text")
    return (obj, "ok") if isinstance(obj, dict) else (None, "not_object")


def fence_parse(text):
    if not isinstance(text, str):
        return None, "not_str"
    m = FENCE.match(EMPTY_THINK.sub("", text.strip()))
    return parse_json(m.group(1)) if m else (None, "no_fence")


def pred_of(obj):
    if not isinstance(obj, dict):
        return None
    v = obj.get("defect_class")
    return v if isinstance(v, str) and v in ENUM else None


def schema_check(obj):
    """七字段 schema 判定，与冻结协议 `eval_7f_5090.py:schema7()` **逐条一致**。
    返回 (是否通过, [问题列表])。缺字段算**字段缺失**，不算内容空白，也不算满分。"""
    if not isinstance(obj, dict):
        return False, ["未解析"]
    p = []
    missing = [f for f in FIELDS if f not in obj]
    if missing:
        p.append("缺字段:" + ",".join(missing))
    extra = [k for k in obj if k not in FIELDS]
    if extra:
        p.append("多字段:" + ",".join(sorted(extra)))
    if "defect_class" in obj and obj["defect_class"] not in ENUM_CLASS:
        p.append(f"defect_class越界:{str(obj['defect_class'])[:20]}")
    elif "defect_class" in obj and not isinstance(obj["defect_class"], str):
        p.append("defect_class非字符串")
    if "radial_zone" in obj:
        v = obj["radial_zone"]
        if not isinstance(v, str):
            p.append("radial_zone非字符串")
        elif v not in ZONES:
            p.append(f"radial_zone越界:{v[:20]}")
    if "clock_direction" in obj:
        v = obj["clock_direction"]
        if not (v is None or isinstance(v, str)):
            p.append("clock_direction类型错")
    if "extent_r" in obj and obj["extent_r"] is not None:
        p.append(f"extent_r非null:{str(obj['extent_r'])[:12]}")
    for f in ("morphology", "caption_zh", "uncertainty"):
        if f in obj and not isinstance(obj[f], str):
            p.append(f"{f}非字符串")
    return (not p), p


def prf(pairs, classes=CLASSES):
    out, f1s = {}, []
    for c in classes:
        tp = sum(1 for g, p in pairs if g == c and p == c)
        fp = sum(1 for g, p in pairs if g != c and p == c)
        fn = sum(1 for g, p in pairs if g == c and p != c)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        out[c] = {"P": round(pr, 4), "R": round(rc, 4), "F1": round(f, 4),
                  "n": sum(1 for g, _ in pairs if g == c)}
        f1s.append(f)
    return sum(f1s) / len(f1s), out


_LOT = None


def lot_map():
    global _LOT
    if _LOT is None:
        _LOT = {}
        p = HERE.parent / "夜间扩容_20261008_0244" / "数据" / "lot_map.tsv"
        if p.exists():
            for l in io.open(p, encoding="utf-8"):
                a, _, b = l.rstrip("\n").partition("\t")
                if a:
                    _LOT[a] = b
    return _LOT


def load(fname):
    p = NIGHT / fname
    if not p.exists():
        return None
    rows = [json.loads(l) for l in io.open(p, encoding="utf-8") if l.strip()]
    for r in rows:
        obj, how = parse_json(r.get("raw"))
        r["_obj"], r["_how"] = obj, how
        r["_pred"] = (pred_of(obj) if how == "ok" else None) or "unknown"
        fo, fh = fence_parse(r.get("raw"))
        # 统一去围栏**诊断**口径：能剥掉完整外层围栏就用剥后的，剥不掉就退回原严格解析
        # （不是「没有围栏就算失败」——那样会把本来就严格可解析的行也一起丢掉）
        r["_fence_pred"] = ((pred_of(fo) if fh == "ok" else None)
                            or r["_pred"])
        r["_part"] = r.get("part", "?")
        sok, sprob = schema_check(obj if how == "ok" else None)
        r["_schema_ok"], r["_schema_prob"] = (sok and how == "ok"), sprob
        r["lot_name"] = r.get("lot_name") or lot_map().get(r["sample_id"], r["sample_id"])
    return rows


def boot_acc(diffs, n=5000, seed=3407):
    if not diffs:
        return None
    rnd = random.Random(seed)
    N = len(diffs)
    m = sorted(sum(diffs[rnd.randrange(N)] for _ in range(N)) / N for _ in range(n))
    return {"均差": round(sum(diffs) / N, 4),
            "lo": round(m[int(0.025 * n)], 4), "hi": round(m[int(0.975 * n)], 4)}


def boot_mf1(vals_a, vals_b, n=5000, seed=3407):
    """配对 Macro-F1 差：每次重抽样本后**重算两侧 MF1**再作差。"""
    if not vals_a:
        return None
    rnd = random.Random(seed)
    N = len(vals_a)
    vals = []
    for _ in range(n):
        idx = [rnd.randrange(N) for _ in range(N)]
        pa = [(vals_a[i][0], vals_a[i][1]) for i in idx]
        pb = [(vals_b[i][0], vals_b[i][1]) for i in idx]
        vals.append(prf(pa)[0] - prf(pb)[0])
    vals.sort()
    ma = prf([(g, p) for g, p in vals_a])[0]
    mb = prf([(g, p) for g, p in vals_b])[0]
    return {"均差": round(ma - mb, 4),
            "lo": round(vals[int(0.025 * n)], 4), "hi": round(vals[int(0.975 * n)], 4)}


def lot_boot_acc(a, b, n=5000, seed=3407):
    a = {r["sample_id"]: r for r in a}
    b = {r["sample_id"]: r for r in b}
    lots = defaultdict(list)
    for r in a.values():
        if r["_part"] != "val90":
            continue
        sid = r["sample_id"]
        if sid not in b:
            continue
        d = int(r["_pred"] == r["label"]) - int(b[sid]["_pred"] == b[sid]["label"])
        lots[r.get("lot_name") or sid].append(d)
    keys = sorted(lots)
    if not keys:
        return None
    rnd = random.Random(seed)
    vals = []
    for _ in range(n):
        s2 = []
        for _ in range(len(keys)):
            s2 += lots[keys[rnd.randrange(len(keys))]]
        vals.append(sum(s2) / len(s2) if s2 else 0.0)
    vals.sort()
    return {"均差": round(sum(vals) / len(vals), 4),
            "lo": round(vals[int(0.025 * n)], 4), "hi": round(vals[int(0.975 * n)], 4),
            "n_lots": len(keys)}


def val84(rows):
    return [r for r in rows if r["_part"] == "val90"]


def metrics(rows, use_fence=False):
    """返回一组口径分明的指标。use_fence=True 时用去围栏后的预测（**仅诊断**）。"""
    key = "_fence_pred" if use_fence else "_pred"
    v = val84(rows)
    acc84 = sum(1 for r in v if r[key] == r["label"]) / len(v) if v else 0.0
    mf84 = prf([(r["label"], r[key]) for r in v])[0] if v else 0.0
    acc102 = sum(1 for r in rows if r[key] == r["label"]) / len(rows) if rows else 0.0
    how = Counter(r["_how"] for r in rows)
    cd = Counter()
    for r in rows:
        o = r["_obj"]
        if isinstance(o, dict):
            cd["JSON null" if o.get("clock_direction") is None
               else f"字面 {o.get('clock_direction')!r}"] += 1
    return {
        "n_全102": len(rows), "n_val84": len(v), "n_dev18": len(rows) - len(v),
        "val84_类别Accuracy": round(acc84, 6),
        "val84_类别MacroF1": round(mf84, 6),
        "val84_正确数": sum(1 for r in v if r[key] == r["label"]),
        "全102_类别Accuracy": round(acc102, 6),
        "全102_正确数": sum(1 for r in rows if r[key] == r["label"]),
        "JSON可解析": how.get("ok", 0), "JSON可解析率": round(how.get("ok", 0) / len(rows), 4),
        "解析判定分布": dict(how),
        "完整schema通过": sum(1 for r in rows if r["_schema_ok"]),
        "完整schema通过率": round(sum(1 for r in rows if r["_schema_ok"]) / len(rows), 4),
        "schema问题分布": dict(Counter(p for r in rows for p in r["_schema_prob"])),
        "clock_direction分布": dict(cd),
        "clock字面null_约定违例": sum(1 for r in rows if isinstance(r["_obj"], dict)
                                     and r["_obj"].get("clock_direction") == "null"),
        "截断": how.get("truncated", 0), "空答": how.get("empty", 0),
    }


def main() -> int:
    res = {"口径": {
        "val84": "原答中 part=='val90' 的 84 条",
        "full102": "全部 102 条（dev18 18 + val84 84）",
        "严格主分": "唯一合法 defect_class；拒重复键/截断/多对象/尾随文本/NaN",
        "去围栏": "剥离完整外层 Markdown 围栏后重解析，**仅诊断，不改主分**",
        "未测": ["morphology / radial_zone / clock_direction / caption_zh 的**事实正确性**"
                 "（无独立 gold）", "尺寸（不作能力目标）",
                 "跨机器泛化（101/102 仅支持该检查点、题面、环境）"],
    }, "模型": {}, "配对": {}}

    # ---------- 自检：本地复算的 schema 判定必须与运行期冻结值逐条一致 ----------
    print("=" * 96)
    print("自检：本地复算 schema 判定 vs 运行期冻结 schema_ok")
    print("=" * 96)
    bad = 0
    for tag in ["L-N3072-3407", "D-N3072-3407", "D-N3072-3408", "D-N3072-3409", "Base"]:
        rows = load(f"{tag}_7f_raw.jsonl")
        if rows is None:
            continue
        d = [r for r in rows if r["_schema_ok"] != bool(r.get("schema_ok"))]
        bad += len(d)
        print(f"  {tag:<16} 不一致 {len(d)}/{len(rows)}")
        for r in d[:3]:
            print(f"     {r['sample_id']} 本地{r['_schema_prob']} 冻结{r.get('schema_problems')}")
    print(f"  合计不一致 {bad} —— {'OK，复算口径与冻结协议一致' if bad == 0 else '**需排查**'}")

    # ---------- 七字段题面（本轮被误称为「七字段准确率」的那批） ----------
    TAGS_7F = ["L-N3072-3407", "D-N3072-3407", "D-N3072-3408", "D-N3072-3409", "Base"]
    loaded7 = {}
    print("=" * 96)
    print("七字段题面 —— 各口径分列（类别指标 / 解析 / schema / 约定违例）")
    print("=" * 96)
    print(f"{'模型':<16}{'val84类别Acc':>13}{'val84类别MF1':>13}{'JSON解析':>10}"
          f"{'完整schema':>11}{'clock字面null':>13}")
    for t in TAGS_7F:
        rows = load(f"{t}_7f_raw.jsonl")
        if rows is None:
            print(f"{t:<16}  缺文件")
            continue
        loaded7[t] = rows
        m = metrics(rows)
        res["模型"].setdefault(t, {})["七字段题面"] = m
        print(f"{t:<16}{m['val84_正确数']:>4}/{m['n_val84']:<8}"
              f"{m['val84_类别MacroF1']:>13.6f}{m['JSON可解析']:>7}/{m['n_全102']:<3}"
              f"{m['完整schema通过']:>8}/{m['n_全102']:<3}{m['clock字面null_约定违例']:>13}")

    # Base 严格 vs 去围栏诊断
    print("\n" + "=" * 96)
    print("Base：原严格分 与 统一去完整围栏**诊断分** 分列（诊断分不进主榜）")
    print("=" * 96)
    for kind, fn in (("类别题面", "Base_raw.jsonl"), ("七字段题面", "Base_7f_raw.jsonl")):
        rows = load(fn)
        if rows is None:
            continue
        st, fe = metrics(rows, False), metrics(rows, True)
        res["模型"].setdefault("Base", {})[f"{kind}_严格与诊断"] = {
            "严格": {k: st[k] for k in ("val84_正确数", "n_val84", "val84_类别Accuracy",
                                       "val84_类别MacroF1", "JSON可解析",
                                       "完整schema通过", "n_全102")},
            "去围栏诊断": {k: fe[k] for k in ("val84_正确数", "n_val84", "val84_类别Accuracy",
                                          "val84_类别MacroF1", "JSON可解析")},
        }
        print(f"{kind}  严格 {st['val84_正确数']}/{st['n_val84']} "
              f"Acc={st['val84_类别Accuracy']:.4f} MF1={st['val84_类别MacroF1']:.6f} "
              f"| schema通过 {st['完整schema通过']}")
        print(f"{'':<10}去围栏诊断 {fe['val84_正确数']}/{fe['n_val84']} "
              f"Acc={fe['val84_类别Accuracy']:.4f} MF1={fe['val84_类别MacroF1']:.6f}")

    # ---------- 类别题面（对照） ----------
    print("\n" + "=" * 96)
    print("类别题面（对照）")
    print("=" * 96)
    for t in ["L-N3072-3407", "D-N3072-3407", "D-N3072-3408", "D-N3072-3409", "Base"]:
        rows = load(f"{t}_raw.jsonl")
        if rows is None:
            continue
        m = metrics(rows)
        res["模型"].setdefault(t, {})["类别题面"] = m
        print(f"{t:<16}{m['val84_正确数']:>4}/{m['n_val84']:<4} "
              f"Acc={m['val84_类别Accuracy']:.4f} MF1={m['val84_类别MacroF1']:.6f} "
              f"| 全102 {m['全102_正确数']}/{m['n_全102']}={m['全102_类别Accuracy']:.4f}")

    # ---------- N=356 小规模：分母写清 ----------
    print("\n" + "=" * 96)
    print("N=356 小规模点：**两个分母分开写**（旧交接把 102 分母误标成 val84）")
    print("=" * 96)
    for t in ["L-N-3407", "D-N-3407", "D-N-3408", "D-N-3409"]:
        rows = load(f"{t}_raw.jsonl")
        if rows is None:
            continue
        m = metrics(rows)
        res["模型"].setdefault(t, {})["类别题面"] = m
        print(f"{t:<12} val84 {m['val84_正确数']:>3}/{m['n_val84']} Acc={m['val84_类别Accuracy']:.4f} "
              f"MF1={m['val84_类别MacroF1']:.6f} | 全102 {m['全102_正确数']:>3}/{m['n_全102']} "
              f"Acc={m['全102_类别Accuracy']:.4f} | dev18 n={m['n_dev18']}")

    # ---------- 配对差：Acc 与 MF1 分别算 ----------
    print("\n" + "=" * 96)
    print("配对差（val84 逐样本）—— Accuracy 与 Macro-F1 **分别**给区间")
    print("=" * 96)
    PAIRS = [("D-N3072-3407", "L-N3072-3407", "主配对（同 seed3407）"),
             ("D-N3072-3408", "L-N3072-3407", "描述波动（异 seed）"),
             ("D-N3072-3409", "L-N3072-3407", "描述波动（异 seed）"),
             ("Base", "L-N3072-3407", "基座对照"),
             ("Base", "D-N3072-3407", "基座对照")]
    for t1, t2, why in PAIRS:
        for kind, sfx in (("类别题面", "_raw.jsonl"), ("七字段题面", "_7f_raw.jsonl")):
            a, b = load(f"{t1}{sfx}"), load(f"{t2}{sfx}")
            if not a or not b:
                continue
            bd = {r["sample_id"]: r for r in b}
            ids = [r["sample_id"] for r in val84(a) if r["sample_id"] in bd]
            if not ids:
                continue
            ad = {r["sample_id"]: r for r in a}
            dacc = [int(ad[s]["_pred"] == ad[s]["label"]) - int(bd[s]["_pred"] == bd[s]["label"])
                    for s in ids]
            va = [(ad[s]["label"], ad[s]["_pred"]) for s in ids]
            vb = [(bd[s]["label"], bd[s]["_pred"]) for s in ids]
            ca, cm = boot_acc(dacc), boot_mf1(va, vb)
            la = lot_boot_acc(a, b)
            key = f"{t1} − {t2}｜{kind}"
            res["配对"][key] = {"说明": why, "n": len(ids),
                                "Accuracy差": ca, "MacroF1差": cm, "lot重抽Accuracy": la,
                                "赢/输": [sum(1 for x in dacc if x > 0), sum(1 for x in dacc if x < 0)]}
            flag = lambda c: "跨0" if c and c["lo"] <= 0 <= c["hi"] else "**不跨0**"
            print(f"{key:<38} n={len(ids)}")
            print(f"    Acc 均差 {ca['均差']:+.4f} [{ca['lo']:+.4f}, {ca['hi']:+.4f}] {flag(ca)}"
                  f"   MF1 均差 {cm['均差']:+.4f} [{cm['lo']:+.4f}, {cm['hi']:+.4f}] {flag(cm)}")
            if la:
                print(f"    lot 重抽 Acc [{la['lo']:+.4f}, {la['hi']:+.4f}] （{la['n_lots']} 个 lot）")

    p = OUT / "更正口径复算.json"
    io.open(p, "w", encoding="utf-8", newline="\n").write(
        json.dumps(res, ensure_ascii=False, indent=2) + "\n")
    print(f"\n写出 {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
