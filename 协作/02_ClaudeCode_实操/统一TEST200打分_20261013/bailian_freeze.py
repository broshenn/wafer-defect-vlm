#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""百炼三模型：列目录 → 合成图连通检查 → 冻结（全部发生在任何 TEST200 调用之前）。

子命令（严格按顺序）：
  --plan     只读列出账号模型目录，按事前候选链解析出候选顺序 → candidates.json（无图片请求）
  --probe    用**新生成的中性合成 PNG** 逐个候选做连通检查 → probe.json（不读任何测试图）
  --freeze   读 probe.json，按链序取第一个通过的唯一模型填满三角色 → frozen_models.json

纪律：
  · 密钥只从环境变量 `DASHSCOPE_API_KEY` 读；不打印、不落盘、不写日志、不算指纹。
  · 合成图只让模型说左右两侧的形状与颜色；不取类别/几何答案，不用测试表现选候选。
  · 冻结只在 TEST200 第一张调用之前发生；冻结后不得因 TEST200 表现替换。
  · 只允许官方百炼 HTTPS 域名；拒绝重定向（由 bailian_client 保证）。无自动重试。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import struct
import sys
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLIENT = Path("D:/pycode/晶圆图研究/协作/01_Codex_指挥/统一TEST200_20261013/code/bailian_client.py")

spec = importlib.util.spec_from_file_location("bailian_client_pinned", CLIENT)
bc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bc)

# 事前写死的候选链（Codex `分项/百炼选型建议.md`；**不看 TEST200 分数**）
CHAINS = {
    "flagship": ["qwen3.8-max-0902", "qwen3.8-max-2026-09-02", "qwen3.8-max",
                 "qwen3.5-397b-a17b"],
    "small": ["qwen3.8-27b", "qwen3.5-27b", "qwen3.5-35b-a3b", "qwen3.5-122b-a10b"],
    "max_or_plus": ["qwen3.7-plus-2026-05-26", "qwen3.7-plus", "qwen3-vl-plus",
                    "qwen3.5-plus-2026-02-15"],
}
ROLE_ORDER = ["flagship", "small", "max_or_plus"]

PROBE_PROMPT = ("图中有左右两个图形。请只回答它们的形状和颜色，格式为一行："
                "左=<形状><颜色>；右=<形状><颜色>。不要解释。")

PROBE_PNG = HERE / "inputs" / "probe_synthetic.png"


# ---------------------------------------------------------------- 合成图（纯标准库，逐字节可复现）
def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))


def make_probe_png(path: Path, size: int = 256) -> None:
    """白底 + 左侧红色实心方块 + 右侧蓝色实心圆。与任何测试图无关。"""
    px = [[(255, 255, 255) for _ in range(size)] for _ in range(size)]
    sq = (24, size // 2 - 40, 88, size // 2 + 40)          # x0,x1,y0,y1
    for y in range(sq[2], sq[3]):
        for x in range(sq[0], sq[1]):
            px[y][x] = (220, 30, 30)
    cx, cy, r = size - 72, size // 2, 44
    for y in range(size):
        for x in range(size):
            if (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
                px[y][x] = (30, 60, 220)
    raw = b"".join(b"\x00" + bytes(v for p in row for v in p) for row in px)
    png = (b"\x89PNG\r\n\x1a\n"
           + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
           + _png_chunk(b"IDAT", zlib.compress(raw, 9))
           + _png_chunk(b"IEND", b""))
    path.parent.mkdir(parents=True, exist_ok=True)
    if not (path.exists() and path.read_bytes() == png):
        path.write_bytes(png)


def sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _key_or_die() -> str:
    if not os.environ.get("DASHSCOPE_API_KEY", "").strip():
        print("!! 环境变量 DASHSCOPE_API_KEY 未设置（值不会被打印或落盘）", file=sys.stderr)
        raise SystemExit(3)
    return "set"


def _client():
    return bc.BailianClient()


# -------------------------------------------------------------------------------- plan
def cmd_plan() -> int:
    _key_or_die()
    rec = _client().list_models()
    if rec.get("status") != "ok":
        print(json.dumps({"status": rec.get("status"), "http_status": rec.get("http_status"),
                          "error_type": rec.get("error_type"), "request_id": rec.get("request_id")},
                         ensure_ascii=False, indent=1))
        return 4
    body = rec.get("response") or {}
    ids = sorted(d.get("id") for d in body.get("data", []) if d.get("id"))
    (HERE / "bailian_models_list.json").write_text(json.dumps({
        "fetched_utc": rec.get("started_at_utc"), "request_id": rec.get("request_id"),
        "returned_model": rec.get("returned_model"), "n_models": len(ids), "models": ids,
        "note": "只读目录；未发送任何图片请求",
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    pool = set(ids)
    resolved, unresolved = {}, {}
    for role in ROLE_ORDER:
        present = [c for c in CHAINS[role] if c in pool]
        resolved[role] = present
        if not present:
            unresolved[role] = CHAINS[role]
    (HERE / "candidates.json").write_text(json.dumps({
        "schema": "bailian-candidates-v1",
        "account_model_count": len(ids), "account_models_sha256": hashlib.sha256(
            json.dumps(ids, ensure_ascii=False).encode()).hexdigest(),
        "chains": CHAINS, "roles_in_chain_order_present_in_account": resolved,
        "roles_with_no_candidate": unresolved,
        "note": "仅目录存在性；尚未做连通检查，尚未冻结。冻结须在任一 TEST200 调用之前完成。",
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"目录模型数: {len(ids)}")
    for role in ROLE_ORDER:
        print(f"  {role:12s} 链上命中: {resolved[role]}")
    if unresolved:
        print("  无候选角色:", json.dumps(unresolved, ensure_ascii=False))
    return 0


# ------------------------------------------------------------------------------- probe
def _probe_one(client, model_id: str, png_url: str) -> dict:
    payload = bc.build_payload(model_id, png_url, PROBE_PROMPT, seed=3407)
    rec = client.complete_once(payload)
    rec["seed_used"] = 3407
    # 仅合成图阶段的显式例外：路由不接受 seed 时，事前改成省略 seed 并记录理由（不在正式题上改参）
    if rec.get("status") in ("http_error", "response_error"):
        blob = (rec.get("raw_response_text_redacted") or "")
        if "seed" in blob.lower():
            payload2 = bc.build_payload(model_id, png_url, PROBE_PROMPT, seed=None)
            rec2 = client.complete_once(payload2)
            rec2["seed_used"] = None
            rec2["seed_omitted_reason"] = "合成图阶段路由拒绝 seed；事前省略并记录，正式题同此参数"
            rec2["first_attempt_status"] = rec.get("status")
            rec = rec2
    return rec


def cmd_probe() -> int:
    _key_or_die()
    cand = json.loads((HERE / "candidates.json").read_text(encoding="utf-8"))
    make_probe_png(PROBE_PNG)
    png_sha = sha_file(PROBE_PNG)
    url = bc.image_data_url(PROBE_PNG)
    prompt_sha = hashlib.sha256(PROBE_PROMPT.encode("utf-8")).hexdigest()

    client = _client()
    results, order = [], []
    for role in ROLE_ORDER:
        for cand_model in cand["roles_in_chain_order_present_in_account"].get(role, []):
            if cand_model in order:
                continue
            order.append(cand_model)
    for model_id in order:
        rec = _probe_one(client, model_id, url)
        results.append({
            "model": model_id,
            "status": rec.get("status"), "http_status": rec.get("http_status"),
            "error_type": rec.get("error_type"), "request_id": rec.get("request_id"),
            "returned_model": rec.get("returned_model"),
            "elapsed_s": rec.get("elapsed_s"), "usage": rec.get("usage"),
            "finish_reason": rec.get("finish_reason"),
            "content": rec.get("content"),
            "reasoning_content_present": bool(rec.get("reasoning_content")),
            "seed_used": rec.get("seed_used"),
            "seed_omitted_reason": rec.get("seed_omitted_reason"),
            "request_sha256": rec.get("request_sha256"),
            "raw_response_text_redacted": rec.get("raw_response_text_redacted"),
        })
        ok = rec.get("status") == "ok" and bool((rec.get("content") or "").strip())
        print(f"  {model_id:34s} {rec.get('status'):14s} "
              f"{(rec.get('elapsed_s') or 0):6.2f}s  {'PASS' if ok else 'FAIL'}")
    (HERE / "probe.json").write_text(json.dumps({
        "schema": "bailian-probe-v1",
        "probe_png": {"path": "inputs/probe_synthetic.png", "sha256": png_sha},
        "probe_prompt": PROBE_PROMPT, "probe_prompt_sha256": prompt_sha,
        "params": {"max_tokens": 384, "temperature": 0, "stream": False,
                   "enable_thinking": False, "seed": 3407},
        "note": "合成图连通检查；未读取任何 TEST200 图片，未取类别/几何答案",
        "results": results,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"合成图 sha256={png_sha[:16]}…  通过 {sum(1 for r in results if r['status']=='ok' and (r['content'] or '').strip())}/{len(results)}")
    return 0


# ------------------------------------------------------------------------------ freeze
def cmd_freeze() -> int:
    from datetime import datetime, timezone
    cand = json.loads((HERE / "candidates.json").read_text(encoding="utf-8"))
    probe = json.loads((HERE / "probe.json").read_text(encoding="utf-8"))
    ok = {r["model"]: r for r in probe["results"]
          if r["status"] == "ok" and (r["content"] or "").strip()}
    frozen, used, notes, unresolved = {}, set(), {}, {}
    for role in ROLE_ORDER:
        hit = None
        for m in cand["roles_in_chain_order_present_in_account"].get(role, []):
            if m in used:
                notes.setdefault(role, []).append(f"{m} 已分配给其他角色，跳过")
                continue
            if m in ok:
                hit = m
                break
            notes.setdefault(role, []).append(
                f"{m} 连通检查未通过（status={next((r['status'] for r in probe['results'] if r['model']==m), 'not_probed')}），按事前规则不重试、顺延下一候选")
        if hit:
            frozen[role] = hit
            used.add(hit)
            r = ok[hit]
            notes.setdefault(role, []).append(
                f"采用 {hit}；合成图返回={r['content'][:120]!r}")
        else:
            unresolved[role] = cand["roles_in_chain_order_present_in_account"].get(role, [])
    out = {
        "schema": "bailian-frozen-v1",
        "frozen_before_any_test_image": True,
        "frozen_at_utc": probe.get("probe_at_utc"),
        "roles": frozen, "unresolved_roles": unresolved,
        "selection_rule": "按事前候选链在账户目录内、且通过合成图连通检查的第一个唯一模型；不看 TEST200 分数",
        "chains": CHAINS, "decision_notes": notes,
        "params": probe["params"], "probe_png_sha256": probe["probe_png"]["sha256"],
        "account_model_count": cand["account_model_count"],
        "note": "冻结即生效：此后不得因 TEST200 表现替换、增补或挑种子",
    }
    (HERE / "frozen_models.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                             encoding="utf-8")
    print("冻结结果:", json.dumps(frozen, ensure_ascii=False, indent=1))
    for role, ns in notes.items():
        for n in ns:
            print(f"  [{role}] {n}")
    if unresolved:
        print("未解析角色:", json.dumps(unresolved, ensure_ascii=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    for f in ("plan", "probe", "freeze"):
        ap.add_argument(f"--{f}", action="store_true")
    a = ap.parse_args()
    if a.plan:
        return cmd_plan()
    if a.probe:
        return cmd_probe()
    if a.freeze:
        return cmd_freeze()
    ap.error("需要 --plan / --probe / --freeze 之一")


if __name__ == "__main__":
    raise SystemExit(main())
