#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""百炼四模型 × 固定 36 图：同题面视觉对照。

**这是要花钱的脚本。默认 dry-run；只有显式加 `--execute` 才会真的发请求。**

口径（照 Codex 任务书）：
  · 只发**百炼**端点，不做其他平台付费回退；
  · 输入只给 PNG + 冻结 591 字题面；**公开标签/几何/旧教师/模型分数一律不进请求**；
  · 每模型 36 次、每图**独立上下文**（不带历史）、每图**一次请求**、**失败不重试**；
  · **共享费用守卫**：累计 ≤ `--budget`（默认 20 元），
    **在途请求按 max_tokens 预留**，完成时按实际 usage 结算；
  · 最多 **2 个在途**；
  · 实际 usage / finish_reason / 耗时 / **实际返回 ID** 完整落盘；
  · 支持 **resume**：只续「没有原答且未开始」的项；失败原件保留、不重试。

思考参数（Codex 要求事前锁死）：
  · 三个 Qwen：`enable_thinking=False`（可关思考的明确关闭）；
  · **Kimi K3：暂按原生思考跑**，`max_tokens=4096`，**另列思考/完成 token 与预算差异**，
    **不称与 9B 生成预算完全相同**。若实际路由不接受该参数，脚本会记录并在报告里如实写。

用法：
  python 百炼对照.py --dry-run                 # 只算钱、不请求（默认）
  python 百炼对照.py --execute --budget 20     # 真跑
  python 百炼对照.py --execute --only qwen3-vl-plus   # 只跑一个模型
"""
from __future__ import annotations
import argparse, base64, hashlib, io, json, os, re, sys, time, threading
import urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

D = Path(__file__).resolve().parent
REPO = D.parent.parent.parent                       # 仓库根
BLIND = REPO / "协作/01_Codex_指挥/百炼多模型对照_20261009/blind36.jsonl"
PROMPT = REPO / "协作/01_Codex_指挥/WorkBuddy外部102_执行补包_20261008/prompt_7f.txt"
SECRETS = Path("D:/pycode/.api-secrets/dashscope.key")
ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
SYSTEM = "你是半导体晶圆缺陷分析专家。"
PROMPT_SHA = "8a8315f785a2398ba5a823754c40b8a8658f89b79aad1a0dbd68058acb43cfda"
OUT = D / "原答"
LOG = D / "请求日志.jsonl"

# 目标顺序 = **便宜且风险低的先跑**（候选集固定，不因结果换模型；这只是风险次序）
MODELS = [
    {"tag": "qwen3.5-397b-a17b", "id": "qwen3.5-397b-a17b",
     "in_price": 1.2, "out_price": 7.2, "max_tokens": 256, "thinking": "disable",
     "note": "输入≤128k 档"},
    {"tag": "qwen3-vl-plus", "id": "qwen3-vl-plus",
     "in_price": 1.0, "out_price": 10.0, "max_tokens": 256, "thinking": "disable",
     "note": "专用视觉路线；官方称与快照 2025-12-19 功能等同，此处锁滚动别名"},
    {"tag": "qwen3.8-max-0902", "id": "qwen3.8-max-0902",
     "in_price": 12.0, "out_price": 36.0, "max_tokens": 256, "thinking": "disable",
     "note": "固定快照"},
    {"tag": "kimi-k3", "id": "kimi-k3",
     "in_price": 20.0, "out_price": 100.0, "max_tokens": 4096, "thinking": "native",
     "note": "**按原生思考跑**；思考 token 另列，不与 9B 同预算"},
]
MAX_BYTES = 4 * 1024 * 1024


def load_key():
    v = (os.environ.get("DASHSCOPE_API_KEY") or "").strip()
    if v:
        return v, "环境变量"
    if SECRETS.is_file():
        v = SECRETS.read_text(encoding="ascii", errors="ignore").strip()
        if v:
            return v, f"仓库外私有文件（{SECRETS.name}）"
    return None, "未取得"


class Guard:
    """共享费用守卫。在途按 max_tokens 预留，完成按实际结算。"""
    def __init__(self, budget, price):
        self.budget, self.price = budget, price
        self.actual, self.reserved = 0.0, 0.0
        self.lock = threading.Lock()

    def reserve(self, in_tok_est, max_tokens):
        with self.lock:
            need = (in_tok_est * self.price[0] + max_tokens * self.price[1]) / 1e6
            if self.actual + self.reserved + need > self.budget:
                return None
            self.reserved += need
            return need

    def settle(self, held, in_tok, out_tok):
        with self.lock:
            self.reserved -= held
            self.actual += (in_tok * self.price[0] + out_tok * self.price[1]) / 1e6
            return self.actual

    def total(self):
        with self.lock:
            return self.actual, self.reserved


def cost(in_tok, out_tok, p):
    return (in_tok * p[0] + out_tok * p[1]) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", default=True)
    ap.add_argument("--execute", action="store_true", help="**真的发请求**")
    ap.add_argument("--budget", type=float, default=20.0, help="人民币上限")
    ap.add_argument("--only", default=None, help="只跑指定 tag")
    ap.add_argument("--limit", type=int, default=None, help="每模型最多几条（调试用）")
    ap.add_argument("--in-flight", type=int, default=2)
    a = ap.parse_args()
    if a.execute:
        a.dry_run = False

    rows = [json.loads(l) for l in io.open(BLIND, encoding="utf-8") if l.strip()]
    assert len(rows) == 36, f"盲包应 36 条，实际 {len(rows)}"
    pb = PROMPT.read_bytes()
    assert hashlib.sha256(pb).hexdigest() == PROMPT_SHA, "题面哈希不一致"
    prompt = pb.decode("utf-8")
    # 图片路径是相对仓库根的；逐张核 hash
    for r in rows:
        p = REPO / r["image_path"]
        assert p.is_file(), f"缺图 {p}"
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        assert h == r["image_sha256"], f"图 hash 不符 {r['sample_id']}"
    print(f"盲包 36 条 ✓  题面 sha256 ✓  36 张图 hash 全部核对 ✓")

    models = [m for m in MODELS if (a.only is None or m["tag"] == a.only)]
    key, src = load_key()
    print(f"凭据来源：{src}")
    if not a.dry_run and not key:
        print("**未取得凭据，不能执行**"); return 3

    # ---------- dry-run：只算钱 ----------
    est_in = 2600        # 448×448 PNG 视觉 token + 591 字题面，保守估计
    if a.dry_run:
        print(f"\n{'模型':<22}{'in/out 价':>14}{'每图预留(元)':>14}{'36 图合计(元)':>16}")
        tot = 0.0
        for m in models:
            one = cost(est_in, m["max_tokens"], (m["in_price"], m["out_price"]))
            s = one * 36
            tot += s
            print(f"{m['tag']:<22}{m['in_price']:>6}/{m['out_price']:<6}"
                  f"{one:>14.4f}{s:>16.3f}")
        print(f"{'合计（按输入 %d token 估）' % est_in:<22}{'':>14}{'':>14}{tot:>16.3f}")
        print(f"\n预算 {a.budget} 元 —— 预估占 {tot/a.budget*100:.1f}%")
        if tot > a.budget:
            print("**预估已超预算：真实执行时守卫会在超限前停止派发**")
        print("\n这是 dry-run，没有发送任何请求。加 --execute 才真跑。")
        return 0

    OUT.mkdir(exist_ok=True)
    done = {}
    if LOG.exists():
        for l in io.open(LOG, encoding="utf-8"):
            if l.strip():
                d = json.loads(l)
                if d.get("status") == "ok":
                    done[(d["model"], d["sample_id"])] = d
        print(f"resume：已有 {len(done)} 条成功原答")

    print(f"\n=== 开始执行（在途 {a.in_flight}，预算 {a.budget} 元）===")
    grand = 0.0
    for m in models:
        price = (m["in_price"], m["out_price"])
        guard = Guard(a.budget - grand, price)
        todo = [r for r in rows if (m["tag"], r["sample_id"]) not in done]
        if a.limit:
            todo = todo[:a.limit]
        print(f"\n--- {m['tag']}：{len(todo)} 条待跑 ---")
        if not todo:
            continue
        fh = io.open(LOG, "a", encoding="utf-8", newline="\n")
        results = []

        def one(r):
            held = guard.reserve(est_in, m["max_tokens"])
            if held is None:
                return {"model": m["tag"], "sample_id": r["sample_id"],
                        "status": "budget_stop", "note": "费用守卫在派发前拦下"}
            t0 = time.perf_counter()
            try:
                body = {
                    "model": m["id"],
                    "messages": [{"role": "system", "content": SYSTEM},
                                 {"role": "user", "content": [
                                     {"type": "image_url", "image_url": {
                                         "url": "data:image/png;base64," +
                                                base64.b64encode(
                                                    (REPO / r["image_path"]).read_bytes()
                                                ).decode()}},
                                     {"type": "text", "text": prompt}]}],
                    "temperature": 0, "max_tokens": m["max_tokens"],
                }
                if m["thinking"] == "disable":
                    body["enable_thinking"] = False
                req = urllib.request.Request(
                    ENDPOINT, data=json.dumps(body).encode("utf-8"),
                    headers={"Authorization": f"Bearer {key}",
                             "Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=180) as resp:
                    raw = resp.read().decode("utf-8")
                j = json.loads(raw)
                u = j.get("usage") or {}
                it, ot = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
                guard.settle(held, it, ot)
                return {"model": m["tag"], "model_id": m["id"],
                        "sample_id": r["sample_id"], "item_id": r["item_id"],
                        "image_sha256": r["image_sha256"], "status": "ok",
                        "response_id": j.get("id"),
                        "created": j.get("created"),
                        "content": ((j.get("choices") or [{}])[0].get("message") or {}).get("content"),
                        "finish_reason": ((j.get("choices") or [{}])[0].get("finish_reason")),
                        "usage": u, "usage_raw": u,
                        "thinking_note": ("原生思考" if m["thinking"] == "native" else "已请求关闭"),
                        "seconds": round(time.perf_counter() - t0, 2),
                        "prompt_sha256": PROMPT_SHA}
            except Exception as e:
                guard.settle(held, 0, 0)          # 失败也释放预留（费用按平台实际计）
                code = getattr(e, "code", None)
                return {"model": m["tag"], "model_id": m["id"],
                        "sample_id": r["sample_id"], "item_id": r["item_id"],
                        "status": "error", "http_code": code,
                        "error_type": type(e).__name__, "error": str(e)[:400],
                        "seconds": round(time.perf_counter() - t0, 2)}

        with ThreadPoolExecutor(max_workers=a.in_flight) as ex:
            futs = [ex.submit(one, r) for r in todo]
            for i, f in enumerate(as_completed(futs), 1):
                d = f.result()
                fh.write(json.dumps(d, ensure_ascii=False) + "\n"); fh.flush()
                results.append(d)
                mark = "✓" if d["status"] == "ok" else ("⛔预算" if d["status"] == "budget_stop" else "✗")
                print(f"  [{i}/{len(todo)}] {mark} {d['sample_id']} "
                      f"{d.get('seconds','')}s {d.get('finish_reason','') or d.get('error_type','')}",
                      flush=True)
                if d["status"] == "error" and d.get("http_code") in (400, 401, 403):
                    print(f"  **{m['tag']} 认证/参数错误，停止该模型**")
                    for g in futs: g.cancel()
                    break
        fh.close()
        act, res = guard.total()
        print(f"  {m['tag']} 本模型实付 ≈ {act:.4f} 元")
        grand += act
        print(f"  === 累计实付 ≈ {grand:.4f} / {a.budget} 元 ===")
        ok = sum(1 for d in results if d["status"] == "ok")
        print(f"  成功 {ok} / {len(results)}")
    print(f"\n=== 全部结束，累计实付 ≈ {grand:.4f} 元 ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
