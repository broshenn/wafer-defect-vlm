#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""百炼三模型 × TEST200：固定 600 槽位、单次请求、无自动重试。

纪律（来自任务书与 `分项/百炼选型建议.md`）：
  · 三个模型已在**任何 TEST200 调用之前**冻结（`frozen_models.json`）；本脚本拒绝跑冻结外的模型。
  · 600 个 (模型, sample_id) 槽位固定；派发前先写 `started` 并 fsync，结果逐条追加。
  · 恢复只允许真正**未开始**的槽位；有 started 无结果的标 `结果未知`，**不重发**。
  · 失败留空，不从 200 的主分母里删除；成功/空答/解析错误/HTTP失败/断连/未执行各自留证。
  · 图片字节逐张核 SHA，必须与本地考试同一 PNG；不一致即拒绝启动。
  · 密钥只从环境变量读，不打印、不落盘。

用法：
  python bailian_run.py --model-role flagship --out bailian_out
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLIENT = Path("D:/pycode/晶圆图研究/协作/01_Codex_指挥/统一TEST200_20261013/code/bailian_client.py")
spec = importlib.util.spec_from_file_location("bailian_client_pinned", CLIENT)
bc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bc)

NAME_FOR_ROLE = {
    "flagship": "BL_flagship",
    "small": "BL_small",
    "max_or_plus": "BL_maxplus",
}


def sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def api_text_from_frozen(path: Path) -> tuple[str, str]:
    """把冻结的完整用户消息转过成 API 侧文本。

    本地考试的首行 `<image>` 是 ms-swift 的图片占位 token，由模板展开成视觉 token，
    模型看不到字面量。百炼侧由 image 内容块承担同一角色，故发送**去掉该占位行**的正文。
    返回 (api_text, derivation_note)。
    """
    full = path.read_text(encoding="utf-8")
    if full.startswith("<image>\n"):
        body = full[len("<image>\n"):]
        note = "去掉首行 <image> 占位符；该占位在 API 侧由 image 内容块承担"
    elif full.startswith("<image>"):
        body = full[len("<image>"):]
        note = "去掉行首 <image> 占位符；该占位在 API 侧由 image 内容块承担"
    else:
        body = full
        note = "无可剥离的占位符"
    return body, note


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-role", required=True, choices=sorted(NAME_FOR_ROLE))
    ap.add_argument("--out", default="bailian_out")
    a = ap.parse_args()

    if not os.environ.get("DASHSCOPE_API_KEY", "").strip():
        print("!! 环境变量 DASHSCOPE_API_KEY 未设置（值不会被打印或落盘）", file=sys.stderr)
        return 3

    frozen = json.loads((HERE / "frozen_models.json").read_text(encoding="utf-8"))
    model_id = frozen["roles"].get(a.model_role)
    if not model_id:
        print(f"!! 角色 {a.model_role} 未冻结，拒绝运行")
        return 4
    if not (HERE / "frozen_models.json").exists() or not frozen.get("frozen_before_any_test_image"):
        print("!! 冻结件不完整，拒绝运行")
        return 4

    outdir = HERE / a.out
    outdir.mkdir(exist_ok=True)
    name = NAME_FOR_ROLE[a.model_role]
    raw_path = outdir / f"{name}_raw.jsonl"
    ledger_path = outdir / f"{name}_ledger.jsonl"

    ids = [x.strip() for x in (HERE / "inputs/TEST200_ids.txt")
           .read_text(encoding="utf-8-sig").splitlines() if x.strip()]
    images = [json.loads(l) for l in (HERE / "inputs/images200.jsonl").open(encoding="utf-8") if l.strip()]
    img_by_id = {r["sample_id"]: r for r in images}
    assert len(ids) == 200 and set(ids) == set(img_by_id), "冻结 ID 与图片清单不一致"

    api_text, note = api_text_from_frozen(HERE / "inputs/prompt_main_user.txt")
    contract = json.loads((HERE / "contract.json").read_text(encoding="utf-8"))
    srv_exam = HERE / "inputs/server_test200_main.jsonl"
    srv_sha = sha_file(srv_exam)
    api_prompt_sha = hashlib.sha256(api_text.encode("utf-8")).hexdigest()

    # 冻结件声明：供 build_sources.py 归类
    freeze_evt = {
        "事件": "冻结件", "dataset": "test200_main.jsonl", "dataset_sha256": srv_sha, "N": 200,
        "source": "百炼", "protocol": "v5_main",
        "prompt_sha256": contract["prompts"]["main"]["sha256"],
        "api_prompt_sha256": api_prompt_sha, "api_prompt_note": note,
        "model": model_id, "role": a.model_role,
        "params": frozen["params"], "frozen_models_sha256": sha_file(HERE / "frozen_models.json"),
    }

    # 已派发 / 已完成
    started = set()
    done = set()
    if ledger_path.exists():
        for line in ledger_path.open(encoding="utf-8"):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("事件") == "started" and d.get("sample_id"):
                started.add(d["sample_id"])
            if d.get("事件") == "finished" and d.get("sample_id"):
                done.add(d["sample_id"])
    if raw_path.exists():
        for line in raw_path.open(encoding="utf-8"):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("sample_id"):
                done.add(d["sample_id"])

    unknown = sorted(started - done)
    todo = [s for s in ids if s not in started]

    lf = ledger_path.open("a", encoding="utf-8")
    rf = raw_path.open("a", encoding="utf-8")
    if not started and not done:
        lf.write(json.dumps(freeze_evt, ensure_ascii=False) + "\n")
        lf.flush()
        os.fsync(lf.fileno())

    print(f"[{name}] model={model_id} 已完成 {len(done)}/200  未知 {len(unknown)}  待派发 {len(todo)}")
    if unknown:
        print(f"[{name}] !! {len(unknown)} 条已 started 无结果 → 标结果未知，不重发")

    client = bc.BailianClient(timeout_s=120)
    t0 = time.perf_counter()
    n_ok = n_fail = 0
    for i, sid in enumerate(todo, 1):
        row = img_by_id[sid]
        png = Path(row["image_path"])
        if not png.is_file() or sha_file(png) != row["image_sha256"]:
            raise SystemExit(f"图片缺失或 SHA 不符，拒绝继续: {sid}")

        # 派发前先落盘 started
        lf.write(json.dumps({"事件": "started", "sample_id": sid, "model": model_id,
                             "ts": time.time()}, ensure_ascii=False) + "\n")
        lf.flush()
        os.fsync(lf.fileno())

        payload = bc.build_payload(model_id, bc.image_data_url(png), api_text, seed=3407)
        rec = client.complete_once(payload)
        content = rec.get("content")
        ok = rec.get("status") == "ok" and isinstance(content, str) and content.strip()

        rec_written = {
            "sample_id": sid, "model": model_id, "role": a.model_role,
            "raw": content if isinstance(content, str) else "",
            "status": rec.get("status"),
            "http_status": rec.get("http_status"), "error_type": rec.get("error_type"),
            "request_id": rec.get("request_id"), "response_id": rec.get("response_id"),
            "returned_model": rec.get("returned_model"),
            "finish_reason": rec.get("finish_reason"),
            "reasoning_present": bool(rec.get("reasoning_content")),
            "elapsed_s": rec.get("elapsed_s"), "usage": rec.get("usage"),
            "image_sha256": row["image_sha256"],
        }
        if not ok:
            # 失败留空并留证；不带 status in {error,failed,failure} 之外的隐藏
            rec_written["error"] = (f"{rec.get('status')}"
                                    f"/{rec.get('error_type')}"
                                    f"/http={rec.get('http_status')}"
                                    f"/{(rec.get('raw_response_text_redacted') or '')[:200]}")
        rf.write(json.dumps(rec_written, ensure_ascii=False) + "\n")
        lf.write(json.dumps({"事件": "finished", "sample_id": sid, "model": model_id,
                             "status": rec.get("status"), "ok": bool(ok),
                             "ts": time.time()}, ensure_ascii=False) + "\n")
        rf.flush(); os.fsync(rf.fileno()); lf.flush(); os.fsync(lf.fileno())
        n_ok += bool(ok); n_fail += (not ok)
        if i % 20 == 0 or i == len(todo):
            el = time.perf_counter() - t0
            print(f"[{name}] {i}/{len(todo)}  ok={n_ok} fail={n_fail}  "
                  f"{el:.0f}s ({el/max(i,1):.2f}s/条)", flush=True)

    summary = {
        "model": model_id, "role": a.model_role, "name": name,
        "n_slots": 200, "n_done": len(done), "n_unknown_not_resent": len(unknown),
        "n_ok_this_run": n_ok, "n_fail_this_run": n_fail,
        "api_prompt_sha256": api_prompt_sha, "api_prompt_note": note,
        "frozen_prompt_sha256": contract["prompts"]["main"]["sha256"],
        "images_manifest_sha256": contract["images"]["sha256"],
        "server_exam_sha256": srv_sha, "params": frozen["params"],
        "raw_file": raw_path.name, "ledger_file": ledger_path.name,
    }
    (outdir / f"{name}_run.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                             encoding="utf-8")
    lf.close(); rf.close()
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
