#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""三教师对照：实际执行器（默认 dry-run，不发任何请求）。

候选（平台隔离，凭据不得混用）：
  kimi     → 百炼   model=kimi/kimi-k3   env=DASHSCOPE_API_KEY
  qwen     → 百炼   model=qwen3.8-max    env=DASHSCOPE_API_KEY
  deepseek → 官网   model=deepseek-flash env=DEEPSEEK_API_KEY

固定输入（SHA256 锁定，漂移即停）：十八张盲清单、提示词 v2、检查器 v3。

## 安全设计

  * **默认 dry-run**：不带 `--run` 就只构造与校验，一个字节都不发。
  * **费用守卫失败关闭**：未核验价目 / 未给预算 / 算不出成本 → 拒绝发送。
  * **凭据只在内存**：只从环境变量读；**不打印、不落盘、不写进任何结果文件**；
    错误信息里剔除疑似密钥片段。
  * **平台隔离**：每个候选固定只认自己那个环境变量，不做跨平台回退。
  * **客户端重试 = 0**：不发第二次。
  * **原答逐字节保存**：原始响应体、状态码、耗时、usage 全部落盘。

用法：
    python run_teacher_compare.py --dry-run
    python run_teacher_compare.py --run --budget-usd 5.0 [--candidate deepseek] [--limit 1]
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import json
import os
import re
import ssl
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cost_guard import admit, admit_tokens, load_pricing   # noqa: E402

REPO = HERE.parents[2]
BLIND = REPO / "协作/02_ClaudeCode_实操/口径修正与准备_20261002-021207/blind_inputs.jsonl"
PROMPT = REPO / "协作/01_Codex_指挥/prompt_image_only_v2_20261002.txt"
CHECKER = REPO / "协作/02_ClaudeCode_实操/口径修正与准备_20261002-021207/check_answer_v3.py"
EXPECTED = {
    BLIND: "70d48e6f5a7ab3c14f4c3d675e94adc11a9cb518f814120fadb8b2e802120b20",
    PROMPT: "7df73031ee1a30dfb018dbb2417f505ea9ff139280307bce2934de604119954a",
    CHECKER: "794a254e005eaa232362c539817abde98a1c2406ecb52574b9e9edb35303ec55",
}

CANDIDATES = {
    # 2026-10-02：`kimi/kimi-k3` 返回 HTTP 400 "product is not activated"（账号未开通该 ID）。
    # 用户明确授权改用 `kimi-k3` 试一次。这是一次**有授权的 ID 变更**，不是自动回退。
    "kimi": {"provider": "bailian", "model": "kimi-k3",
             "endpoint": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
             "credential_env": "DASHSCOPE_API_KEY"},
    "qwen": {"provider": "bailian", "model": "qwen3.8-max",
             "endpoint": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
             "credential_env": "DASHSCOPE_API_KEY"},
    "deepseek": {"provider": "deepseek_official", "model": "deepseek-flash",
                 "endpoint": "https://api.deepseek.com/chat/completions",
                 "credential_env": "DEEPSEEK_API_KEY"},
}

# 事前锁定，失败后不扩额
MAX_TOKENS_LOCKED = 16384
TIMEOUT_SECONDS_LOCKED = 600
CLIENT_RETRIES_LOCKED = 0
# 输入 token 的最坏估计（图 + 提示词）；用于费用守卫，宁大不小
INPUT_TOKENS_WORST = 3000

SECRET_PAT = re.compile(r"(sk-[A-Za-z0-9_\-]{6,}|Bearer\s+\S+)")

# 仓库外的私有凭据目录。**只读，不列出内容，不回显。**
SECRETS_DIR = Path(r"D:/pycode/.api-secrets")
CRED_FILE = {"DASHSCOPE_API_KEY": "dashscope.key", "DEEPSEEK_API_KEY": "deepseek.key"}


def scrub(text: str) -> str:
    """把疑似密钥从任何要落盘/打印的文本里剔掉。"""
    return SECRET_PAT.sub("<REDACTED>", text or "")


def load_api_key(env_name: str) -> tuple[str | None, str]:
    """按**环境变量优先、仓库外私有文件兜底**的顺序取凭据。

    返回值里**只有来源描述，没有凭据本身**。绝不打印、绝不落盘。
    """
    v = os.environ.get(env_name)
    if v and v.strip():
        return v.strip(), "环境变量"
    f = SECRETS_DIR / CRED_FILE.get(env_name, "")
    if f and f.is_file():
        try:
            v = f.read_text(encoding="ascii", errors="ignore").strip()
            if v:
                return v, f"仓库外私有文件（{f.name}）"
        except OSError:
            pass
    return None, "未取得"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_inputs() -> tuple[list[dict], str]:
    for p, h in EXPECTED.items():
        got = sha256_bytes(p.read_bytes())
        assert got == h, f"固定输入漂移：{p.name} {got[:12]}… != {h[:12]}…"
    blind = [json.loads(l) for l in BLIND.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(blind) == 18 and len({r["sample_id"] for r in blind}) == 18
    prompt = PROMPT.read_text(encoding="utf-8")
    return blind, prompt


def build_body(model: str, image_b64_url: str, prompt: str) -> dict:
    """单图、单轮、无历史。内容顺序照预览：先是图，再是提示词。"""
    return {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_b64_url}},
            {"type": "text", "text": prompt},
        ]}],
        "max_tokens": MAX_TOKENS_LOCKED,
        "stream": False,
        "n": 1,
    }


def post_once(url: str, body: dict, api_key: str) -> dict:
    """**只发一次**。返回 {http_status, raw_text, elapsed, error}。"""
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {api_key}")
    ctx = ssl.create_default_context()
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS_LOCKED, context=ctx) as r:
            raw = r.read().decode("utf-8", errors="replace")
            return {"http_status": r.status, "raw_text": raw,
                    "elapsed_seconds": round(time.time() - t0, 3), "error": None}
    except urllib.error.HTTPError as e:
        body_txt = ""
        try:
            body_txt = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        # 错误体里可能回显请求内容 → 一律 scrub 后才存
        return {"http_status": e.code, "raw_text": scrub(body_txt),
                "elapsed_seconds": round(time.time() - t0, 3),
                "error": f"HTTPError {e.code}"}
    except Exception as e:                                    # noqa: BLE001
        return {"http_status": None, "raw_text": "",
                "elapsed_seconds": round(time.time() - t0, 3),
                "error": scrub(f"{type(e).__name__}: {e}")}


def already_succeeded() -> set[tuple[str, str]]:
    """扫历史运行目录，收集**已成功**的 (候选, item_id)。

    用途：续跑时不重复花钱。只认 `http_status == 200`；失败/缺凭据的不算，
    所以**失败项仍会被重跑**——那不是"自动重试"，是原任务书允许的"在预算内修复后再试"。
    """
    done: set[tuple[str, str]] = set()
    runs = HERE / "runs"
    if not runs.is_dir():
        return done
    for d in sorted(runs.iterdir()):
        if not d.is_dir():
            continue
        for f in d.glob("*__*.json"):
            try:
                o = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            if o.get("http_status") == 200:
                done.add((o.get("candidate"), o.get("item_id")))
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--budget-usd", type=float, default=None,
                    help="金额上限（USD）；单价未核验的候选不走这条")
    ap.add_argument("--budget-tokens", type=float, default=None,
                    help="token 额度，用于单价未核验但用户给了免费额度的候选")
    ap.add_argument("--candidate", choices=list(CANDIDATES), default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=1,
                    help="并发请求数。请求之间本就独立（每个都是单图单轮无历史），"
                         "并发不改变『每次独立上下文』这一要求；默认 1=串行。")
    a = ap.parse_args()

    if a.run and a.dry_run:
        print("[停止] --run 与 --dry-run 互斥")
        return 2
    will_send = a.run

    print("=" * 78)
    print(f"三教师对照执行器 —— {'**实际发送**' if will_send else 'dry-run（不发任何请求）'}")
    print("=" * 78)

    blind, prompt = load_inputs()
    print(f"\n[1] 固定输入校验通过")
    for p in EXPECTED:
        print(f"    {p.name:26s} {EXPECTED[p][:16]}…")
    print(f"    盲清单 {len(blind)} 张；提示词 {len(prompt.encode('utf-8'))} 字节")

    names = [a.candidate] if a.candidate else list(CANDIDATES)
    items = blind[: a.limit] if a.limit else blind
    print(f"\n[2] 计划：候选 {names} × 图 {len(items)} 张 = {len(names) * len(items)} 次请求")
    print(f"    锁定：max_tokens={MAX_TOKENS_LOCKED}  timeout={TIMEOUT_SECONDS_LOCKED}s  "
          f"client_retries={CLIENT_RETRIES_LOCKED}")

    pr = load_pricing()
    done = already_succeeded()
    spent_usd = 0.0
    spent_tok = 0.0
    plan, blocked, skipped = [], [], []
    for name in names:
        cand = CANDIDATES[name]
        price = pr["candidates"][name]
        use_tokens = price.get("budget_mode") == "tokens"
        for b in items:
            if (name, b["item_id"]) in done:
                skipped.append((name, b["item_id"]))
                continue
            if use_tokens:
                dec = admit_tokens(name, price, INPUT_TOKENS_WORST,
                                   MAX_TOKENS_LOCKED, spent_tok, a.budget_tokens)
                if dec["allow"]:
                    spent_tok += dec["estimate"]["worst_case_tokens"]
            else:
                dec = admit(name, price, INPUT_TOKENS_WORST,
                            MAX_TOKENS_LOCKED, spent_usd, a.budget_usd)
                if dec["allow"]:
                    spent_usd += dec["estimate"]["worst_case_cost"]
            if not dec["allow"]:
                blocked.append((name, b["item_id"], dec["reason"]))
                continue
            plan.append((name, b, dec))

    print(f"\n[3] 费用守卫（金额与 token 两条路径，均失败关闭）")
    print(f"    金额路径：--budget-usd = {a.budget_usd}；按最坏累计 {spent_usd:.6f} USD")
    print(f"    token 路径：--budget-tokens = {a.budget_tokens}；按最坏累计 {spent_tok:.0f} tok")
    if blocked:
        seen = {}
        for n, it, why in blocked:
            seen.setdefault((n, why), []).append(it)
        print(f"    被拒 {len(blocked)} 次，去重后：")
        for (n, why), its in seen.items():
            print(f"      {n:9s} ×{len(its):2d}  {why[:70]}")
    if skipped:
        print(f"    跳过已成功过的 {len(skipped)} 次（不重复花钱）；失败项不在跳过之列")
    print(f"    放行 {len(plan)} 次")

    if not will_send:
        print("\n" + "=" * 78)
        print("dry-run：未发送任何请求。要真发请加 --run 并给出 --budget-usd")
        print("=" * 78)
        return 0

    # ── 真正发送 ────────────────────────────────────────
    if a.budget_usd is None:
        print("\n[停止] 没有费用上限，拒绝发送。")
        return 3
    out_dir = HERE / "runs" / dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[4] 输出目录 {out_dir}")

    results = []
    lock = __import__("threading").Lock()

    def do_one(name, b, dec):
        cand = CANDIDATES[name]
        key, key_src = load_api_key(cand["credential_env"])
        if not key:
            return {"candidate": name, "item_id": b["item_id"], "status": "no_credential",
                    "detail": f"{cand['credential_env']} 既不在环境变量也不在私有文件"}
        raw = Path(b["image_path"]).read_bytes()
        assert sha256_bytes(raw) == b["image_sha256"], "图片指纹漂移"
        url = "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
        body = build_body(cand["model"], url, prompt)
        payload_sha = sha256_bytes(
            json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        r = post_once(cand["endpoint"], body, key)
        rec = {
            "candidate": name, "provider": cand["provider"], "model_requested": cand["model"],
            "item_id": b["item_id"], "sample_id": b["sample_id"],
            "image_sha256": b["image_sha256"], "payload_sha256": payload_sha,
            "prompt_byte_sha256": EXPECTED[PROMPT],
            "max_tokens": MAX_TOKENS_LOCKED, "timeout_seconds": TIMEOUT_SECONDS_LOCKED,
            "client_retries": CLIENT_RETRIES_LOCKED,
            "sent_at": dt.datetime.now().astimezone().isoformat(),
            "http_status": r["http_status"], "elapsed_seconds": r["elapsed_seconds"],
            "error": r["error"], "raw_text": r["raw_text"],
            "credential_source": key_src,
            "cost_estimate_worst_usd": (dec["estimate"] or {}).get("worst_case_cost"),
            "token_estimate_worst": (dec["estimate"] or {}).get("worst_case_tokens"),
            "is_peak_now": dec.get("is_peak_now"),
        }
        try:
            obj = json.loads(r["raw_text"])
            rec["returned_model"] = obj.get("model")
            rec["finish_reason"] = (obj.get("choices") or [{}])[0].get("finish_reason")
            rec["usage"] = obj.get("usage")
            rec["answer_text"] = scrub(
                (obj.get("choices") or [{}])[0].get("message", {}).get("content"))
        except Exception:
            rec["returned_model"] = rec["finish_reason"] = rec["usage"] = rec["answer_text"] = None
        (out_dir / f"{name}__{b['item_id']}.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
        with lock:
            print(f"      {name:9s} {b['item_id']}  HTTP={rec['http_status']} "
                  f"{rec['elapsed_seconds']}s", flush=True)
        return rec

    print(f"[4b] 并发度 {a.concurrency}；开始发送 {len(plan)} 次", flush=True)
    if a.concurrency <= 1:
        for name, b, dec in plan:
            results.append(do_one(name, b, dec))
    else:
        with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
            futs = [ex.submit(do_one, n, b, d) for n, b, d in plan]
            for fu in as_completed(futs):
                results.append(fu.result())

    (out_dir / "run_summary.json").write_text(
        json.dumps({"sent": len([r for r in results if r.get("http_status")]),
                    "total_planned": len(plan), "budget_usd": a.budget_usd,
                    "worst_case_total_usd": round(spent_usd, 6),
                    "results": [{k: v for k, v in r.items() if k != "raw_text"} for r in results]},
                   ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    print(f"\n[5] 完成 {len(results)} 次；结果在 {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
