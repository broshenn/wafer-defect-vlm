# -*- coding: utf-8 -*-
"""追加记录 · 程序B：16个自报时间矛盾的逐条分类（用审计日志机器时间+文件mtime交叉核）。

输入：程序A_审计日志时间线.json + 本轮30图各 started/observation_meta/done JSON 的自报时间。
分类规则（对每个自报字段）：
  machine_match   嵌入时间与审计事件时刻差 |d|<=120s（真实UTC、机器生成）
  local_as_Z      嵌入时间 ≈ 审计事件本地钟面（差≈+8h，即把本地时间误标为Z）
  estimate        其余：手写/估计时间，与真实写入时刻不符（给出偏差）
  missing         无对应审计事件
输出：程序B_矛盾分类.json。只读分析；不回填、不改任何原件。
"""
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

D = Path(__file__).parent
ROUND = D.parent

CODEX16 = ["wbv3_002", "wbv3_003", "wbv3_005", "wbv3_006", "wbv3_009", "wbv3_010",
           "wbv3_014", "wbv3_016", "wbv3_038", "wbv3_039", "wbv3_040", "wbv3_043",
           "wbv3_044", "wbv3_046", "wbv3_049", "wbv3_073"]

def parse_z(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None

def classify(emb, audit_ms):
    if emb is None or audit_ms is None:
        return "missing", None
    e = parse_z(emb)
    if e is None:
        return "unparseable", None
    a = datetime.fromtimestamp(audit_ms / 1000, timezone.utc)
    d = (e - a).total_seconds()
    if abs(d) <= 120:
        return "machine_match", round(d, 1)
    if abs(d - 8 * 3600) <= 120:
        return "local_as_Z", round(d, 1)
    return "estimate", round(d, 1)

def main():
    A = json.loads((D / "程序A_审计日志时间线.json").read_text(encoding="utf-8"))
    sr = {r["item_id"]: r for r in A["self_report_vs_mtime"]}
    per = {r["item_id"]: r for r in A["per_item"]}

    rows = []
    for it in sorted(per):  # wbv3_002..076
        rid = per[it]["review_id"]
        ev = per[it]
        s = sr[it]
        stages = {}
        # started
        cls, d = classify(s.get("started", {}).get("embedded"),
                          ev["started_write"][0]["ts_ms"] if ev["started_write"] else None)
        stages["started"] = {"embedded": s.get("started", {}).get("embedded"),
                             "audit_utc": datetime.fromtimestamp(ev["started_write"][0]["ts_ms"] / 1000, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if ev["started_write"] else None,
                             "mtime_utc": s.get("started", {}).get("mtime_utc"),
                             "class": cls, "delta_seconds": d}
        # observation
        cls, d = classify(s.get("observation_meta", {}).get("embedded"),
                          ev["observation_write"][0]["ts_ms"] if ev["observation_write"] else None)
        stages["observation_written"] = {"embedded": s.get("observation_meta", {}).get("embedded"),
                                         "audit_utc": datetime.fromtimestamp(ev["observation_write"][0]["ts_ms"] / 1000, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if ev["observation_write"] else None,
                                         "mtime_utc": s.get("observation_meta", {}).get("mtime_utc"),
                                         "class": cls, "delta_seconds": d}
        # done
        cls, d = classify(s.get("done", {}).get("embedded"),
                          ev["done_write"][0]["ts_ms"] if ev["done_write"] else None)
        stages["done"] = {"embedded": s.get("done", {}).get("embedded"),
                          "audit_utc": datetime.fromtimestamp(ev["done_write"][0]["ts_ms"] / 1000, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if ev["done_write"] else None,
                          "mtime_utc": s.get("done", {}).get("mtime_utc"),
                          "class": cls, "delta_seconds": d}

        flags = [k for k, v in stages.items() if v["class"] not in ("machine_match",)]
        in16 = it in CODEX16
        rows.append({"item_id": it, "review_id": rid, "in_codex16": in16,
                     "codex_flagged_sequence": in16,  # 序列级问题以Codex名单为准
                     "stage_flags": flags, "stages": stages,
                     "machine_order_ok": ev["machine_order_png_then_obs_then_oldraw_then_review"],
                     "png_read_count": ev["png_read_count"],
                     "oldraw_read_tool_count": ev["oldraw_read_tool_count"]})

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "codex16_items": CODEX16,
        "stage_class_counts": {},
        "all_machine_order_ok": all(r["machine_order_ok"] for r in rows),
        "rows": rows,
        "notes": [
            "分类依据：审计日志工具事件时刻（客户端机器钟，UTC纪元毫秒）为主，文件mtime为辅；两者独立于worker自报。",
            "local_as_Z 表示 worker 把本地钟面时间(UTC+8)误标为Z；estimate 表示手写/估计值，与真实写入时刻不符。",
            "审计事件只证明工具调用发生顺序与次数；不证明模型实际获得的像素/文本内容，也不证明后端请求。",
        ],
    }
    from collections import Counter
    cc = Counter()
    for r in rows:
        for k, v in r["stages"].items():
            cc[f"{k}:{v['class']}"] += 1
    summary["stage_class_counts"] = dict(cc)
    (D / "程序B_矛盾分类.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    # 控制台摘要
    print("stage_class_counts:", json.dumps(summary["stage_class_counts"], ensure_ascii=False))
    print("machine_order_ok_all:", summary["all_machine_order_ok"])
    print("\nCodex16 逐条：")
    for r in rows:
        if r["in_codex16"]:
            st = r["stages"]
            print(f"{r['item_id']}({r['review_id']}): started={st['started']['class']}({st['started']['delta_seconds']}s) "
                  f"obs={st['observation_written']['class']}({st['observation_written']['delta_seconds']}s) "
                  f"done={st['done']['class']}({st['done']['delta_seconds']}s)")
    print("\n非Codex16但有stage flag的：")
    for r in rows:
        if not r["in_codex16"] and r["stage_flags"]:
            st = r["stages"]
            print(f"{r['item_id']}({r['review_id']}): " +
                  " ".join(f"{k}={st[k]['class']}({st[k]['delta_seconds']}s)" for k in r["stage_flags"]))

if __name__ == "__main__":
    main()
