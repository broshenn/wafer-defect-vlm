# -*- coding: utf-8 -*-
"""追加记录 · 程序A：从客户端审计日志重建内容复核30轮的真实工具事件时间线。

输入（只读）：
  C:/Users/谌伦杰/.workbuddy/audit-log/2026-10-10.jsonl（已闭合段，manifest含fileSha256）
  C:/Users/谌伦杰/.workbuddy/audit-log/2026-10-11.jsonl（开放段）
  C:/Users/谌伦杰/.workbuddy/audit-log/manifest.jsonl / state.json
输出（本目录）：
  程序A_审计日志时间线.json —— 逐图工具事件（PNG Read / observation Write / 旧raw Read /
  review Write / started/meta/done Write）+ 会话汇总 + 时间窗内全部相关Bash命令；
  同时采集各证据文件的 NTFS mtime（UTC）与自报时间比对所需原始数据。
只读分析；不修改任何被审计对象文件；不回填任何时间。
"""
import hashlib
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

AUDIT = Path(r"C:/Users/谌伦杰/.workbuddy/audit-log")
OUTDIR = Path(r"D:/pycode/晶圆图研究/协作/04_WorkBuddy_复核/内容复核30_v3_20261010_233400/追加记录_20261011_0133")
ROUND = OUTDIR.parent
OLDDIR = Path(r"D:/pycode/晶圆图研究/协作/04_WorkBuddy_复核/方向形态84_v3_20261010_200950")

# 内容复核30轮时间窗（客户端本地 UTC+8）：2026-10-10 23:34:00 —— 2026-10-11 00:30:00
WIN_START = datetime(2026, 10, 10, 15, 34, 0, tzinfo=timezone.utc)
WIN_END = datetime(2026, 10, 11, 16, 30, 0, tzinfo=timezone.utc)  # 上限放宽，覆盖至00:30本地次日+余量

def iso(ts_ms):
    dt = datetime.fromtimestamp(ts_ms / 1000, timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z", (dt + timedelta(hours=8)).strftime("%m-%d %H:%M:%S")

def main():
    events = []
    for day in ["2026-10-10", "2026-10-11"]:
        p = AUDIT / f"{day}.jsonl"
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            ts = r.get("timestamp")
            if not isinstance(ts, int):
                continue
            dt = datetime.fromtimestamp(ts / 1000, timezone.utc)
            if not (WIN_START <= dt <= WIN_END):
                continue
            paths = ""
            mp = r.get("messageParams") or {}
            if isinstance(mp.get("paths"), str):
                paths = mp["paths"]
            cmd = mp.get("command") if isinstance(mp.get("command"), str) else ""
            mk = r.get("messageKey") or ""
            et = r.get("eventType") or ""
            if "readWithRule" in mk:
                tool = "Read"
            elif "writeWithRule" in mk:
                tool = "Write"
            elif et.startswith("command-safety"):
                tool = "Bash"
            else:
                tool = (r.get("toolCallId") or "?").split("_")[0]
            events.append({
                "ts_ms": ts, "seq": r.get("sequence"), "id": r.get("id"),
                "session": (r.get("sessionId") or "")[:8],
                "tool": tool,
                "eventType": et,
                "paths": paths.replace("\\", "/"),
                "cmd": cmd.replace("\\", "/")[:300],
            })
    events.sort(key=lambda e: (e["ts_ms"], e["seq"] or 0))

    items = []
    for s in ["shard1", "shard2", "shard3", "shard4", "shard5"]:
        items += json.loads((ROUND / f"{s}_条目.json").read_text(encoding="utf-8"))["items"]

    per_item, session_map = [], {}
    for it in items:
        rid, iid, sid = it["review_id"], it["item_id"], it["sample_id"]
        png_key = f"data/images/{sid}.png"
        old_key = f"方向形态84_v3_20261010_200950/{iid}.raw.txt"

        def evs(match):
            return [e for e in events if match(e)]

        png_reads = evs(lambda e: e["tool"] == "Read" and png_key in e["paths"])
        obs_writes = evs(lambda e: e["tool"] == "Write" and f"/{rid}.observation_raw.json" in e["paths"])
        meta_writes = evs(lambda e: e["tool"] == "Write" and f"/{rid}.observation_meta.json" in e["paths"])
        started_writes = evs(lambda e: e["tool"] == "Write" and f"/{rid}.started.json" in e["paths"])
        oldraw_reads = evs(lambda e: e["tool"] == "Read" and old_key in e["paths"])
        oldraw_bash = evs(lambda e: e["tool"] == "Bash" and iid + ".raw.txt" in e["cmd"] and "type " + iid not in e["cmd"])
        review_writes = evs(lambda e: e["tool"] == "Write" and f"/{rid}.review.json" in e["paths"])
        done_writes = evs(lambda e: e["tool"] == "Write" and f"/{rid}.done.json" in e["paths"])

        def t(lst):
            return lst[0]["ts_ms"] if lst else None

        order = None
        if all([png_reads, obs_writes, oldraw_reads, review_writes]):
            order = (t(png_reads) <= t(obs_writes) <= t(oldraw_reads) <= t(review_writes))
        rec = {
            "review_id": rid, "item_id": iid, "sample_id": sid,
            "png_read": [{"ts_ms": e["ts_ms"], "utc": iso(e["ts_ms"])[0], "local8": iso(e["ts_ms"])[1], "session": e["session"],
                          "seq": e["seq"], "id": e["id"]} for e in png_reads],
            "observation_write": [{"ts_ms": e["ts_ms"], "utc": iso(e["ts_ms"])[0], "local8": iso(e["ts_ms"])[1], "session": e["session"],
                                   "seq": e["seq"], "id": e["id"]} for e in obs_writes],
            "started_write": [{"ts_ms": e["ts_ms"], "utc": iso(e["ts_ms"])[0], "session": e["session"], "seq": e["seq"]} for e in started_writes],
            "meta_write": [{"utc": iso(e["ts_ms"])[0], "session": e["session"], "seq": e["seq"]} for e in meta_writes],
            "oldraw_read": [{"ts_ms": e["ts_ms"], "utc": iso(e["ts_ms"])[0], "local8": iso(e["ts_ms"])[1], "session": e["session"],
                             "seq": e["seq"], "id": e["id"]} for e in oldraw_reads],
            "oldraw_read_via_bash": [{"utc": iso(e["ts_ms"])[0], "session": e["session"], "seq": e["seq"],
                                      "cmd": e["cmd"][:160]} for e in oldraw_bash],
            "review_write": [{"ts_ms": e["ts_ms"], "utc": iso(e["ts_ms"])[0], "local8": iso(e["ts_ms"])[1], "session": e["session"],
                              "seq": e["seq"], "id": e["id"]} for e in review_writes],
            "done_write": [{"ts_ms": e["ts_ms"], "utc": iso(e["ts_ms"])[0], "session": e["session"], "seq": e["seq"]} for e in done_writes],
            "png_read_count": len(png_reads),
            "oldraw_read_tool_count": len(oldraw_reads),
            "machine_order_png_then_obs_then_oldraw_then_review": order,
        }
        if started_writes:
            session_map.setdefault(started_writes[0]["session"], set()).add(rid)
        per_item.append(rec)

    # 证据文件 NTFS mtime（UTC）与自报时间采集（只读）
    selfreport = []
    for it in items:
        rid, iid = it["review_id"], it["item_id"]
        row = {"review_id": rid, "item_id": iid}
        for kind in ["started", "observation_meta", "review", "done"]:
            p = ROUND / f"{rid}.{kind}.json"
            if p.exists():
                mt = datetime.fromtimestamp(p.stat().st_mtime, timezone.utc)
                try:
                    j = json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    j = {}
                emb = j.get({"started": "started_at_utc", "observation_meta": "written_at_utc",
                             "review": None, "done": "done_at_utc"}[kind])
                row[kind] = {"mtime_utc": mt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                             "mtime_local8": (mt + timedelta(hours=8)).strftime("%m-%d %H:%M:%S"),
                             "embedded": emb}
        selfreport.append(row)

    # 会话汇总
    sess = {}
    for e in events:
        s = sess.setdefault(e["session"], {"n": 0, "first": None, "last": None,
                                           "tools": {}, "writes": set(), "reads": set()})
        s["n"] += 1
        f, l = iso(e["ts_ms"])
        s["first"] = s["first"] or l
        s["last"] = l
        s["tools"][e["tool"]] = s["tools"].get(e["tool"], 0) + 1
        if e["tool"] == "Write" and "/rv30_" in e["paths"]:
            s["writes"].add(e["paths"].rsplit("/", 1)[-1].split(".")[0])
        if e["tool"] == "Read" and "条目" in e["paths"]:
            s["reads"].add(e["paths"].rsplit("/", 1)[-1])
    sessions_out = {k: {"n_events": v["n"], "first_local8": v["first"], "last_local8": v["last"],
                        "tools": v["tools"], "rv30_writes": sorted(v["writes"]), "read_shard_files": sorted(v["reads"])}
                    for k, v in sorted(sess.items())}

    # 审计日志自身完整性：已闭合段 fileSha256 现算比对
    integrity = {}
    man = [json.loads(l) for l in (AUDIT / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    for m in man:
        seg = m.get("segment")
        if seg in ("2026-10-10.jsonl", "2026-10-08.jsonl", "2026-10-01.jsonl"):
            fp = AUDIT / seg
            now_sha = hashlib.sha256(fp.read_bytes()).hexdigest()
            integrity[seg] = {"manifest_sha256": m.get("fileSha256"), "recomputed": now_sha,
                              "match": now_sha == m.get("fileSha256"),
                              "entry_count_manifest": m.get("entryCount")}

    out = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "purpose": "程序A：从客户端审计日志重建内容复核30轮逐图工具事件时间线（只读）",
        "audit_log_files": {
            "2026-10-10.jsonl": "已闭合段（manifest含fileSha256，982条）",
            "2026-10-11.jsonl": "开放段（提取时仍可追加）",
            "manifest/state": "段级hash链与序号",
        },
        "time_window_utc": [WIN_START.strftime("%Y-%m-%dT%H:%M:%SZ"), WIN_END.strftime("%Y-%m-%dT%H:%M:%SZ")],
        "audit_integrity_check": integrity,
        "sessions_in_window": sessions_out,
        "per_item": per_item,
        "self_report_vs_mtime": selfreport,
        "notes": [
            "审计日志timestamp为客户端Unix毫秒（UTC纪元）；本表同时给UTC与本地UTC+8两种读法，时区换算只用此固定偏移，不用于消解自报时间矛盾。",
            "file-safety.allow-listed 记录的是客户端放行的工具调用事件（工具、路径、时刻），不是工具返回内容，也不是后端请求记录。",
            "png_read_count/oldraw_read_tool_count 仅统计审计日志中的工具调用次数；模型是否实际获得并使用像素/文本由后端决定，本日志不能证明。",
        ],
    }
    (OUTDIR / "程序A_审计日志时间线.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    # 控制台摘要
    ok = sum(1 for r in per_item if r["machine_order_png_then_obs_then_oldraw_then_review"] is True)
    partial = sum(1 for r in per_item if r["machine_order_png_then_obs_then_oldraw_then_review"] is None)
    print(json.dumps({
        "items": len(per_item),
        "machine_order_ok": ok,
        "machine_order_incomplete_evidence": partial,
        "machine_order_false": [r["review_id"] for r in per_item if r["machine_order_png_then_obs_then_oldraw_then_review"] is False],
        "png_read_counts_not_1": {r["review_id"]: r["png_read_count"] for r in per_item if r["png_read_count"] != 1},
        "oldraw_tool_reads_not_1": {r["review_id"]: r["oldraw_read_tool_count"] for r in per_item if r["oldraw_read_tool_count"] != 1},
        "oldraw_via_bash": {r["review_id"]: len(r["oldraw_read_via_bash"]) for r in per_item if r["oldraw_read_via_bash"]},
        "integrity": integrity,
        "sessions": {k: (v["n_events"], v["first_local8"], v["last_local8"]) for k, v in sessions_out.items()},
    }, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    main()
