"""按冻结计划串行测量图片调用成本;预算是下一次请求的停止阈值。

单次未知费用可能使累计 API 积分超过阈值,不会因此重试或扩大预算。
账户余额变化单列保存,不当作 API 或 WorkBuddy 整个任务的已归因费用。
"""
import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

import run_kimi_k3_wafer as probe


def now():
    return datetime.now(timezone.utc).astimezone().isoformat()


def save(path, record):
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def valid_credit(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    plan_path = args.plan.resolve()
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    budget = plan["api_credit_limit"]
    maximum = plan["max_requests"]
    if not valid_credit(budget) or budget <= 0:
        parser.error("API 积分阈值必须为正有限数值")
    if type(maximum) is not int or not 1 <= maximum <= 5:
        parser.error("此小测最多 5 次请求")
    expected = {"model": probe.MODEL_ID, "max_tokens": probe.MAX_TOKENS,
                "timeout_seconds": probe.TIMEOUT_SECONDS, "client_max_retries": 0}
    if any(plan["request"].get(name) != value for name, value in expected.items()):
        parser.error("计划参数与固定调用协议不一致")
    prompt_path = Path(plan["prompt_file"])
    prompt_bytes = prompt_path.read_bytes()
    prompt_text = prompt_bytes.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    if probe.sha256_hex(prompt_bytes) != plan["prompt_file_sha256"]:
        parser.error("提示词文件指纹已变化")
    if probe.sha256_hex(prompt_text.encode()) != plan["prompt_sent_text_sha256"]:
        parser.error("实际发送提示词指纹已变化")
    source = {"path": str(prompt_path), "file_sha256": plan["prompt_file_sha256"],
              "sent_text_sha256": plan["prompt_sent_text_sha256"],
              "newline_handling": "CRLF/CR 转 LF,沿用首轮文本读取"}
    results_dir = plan_path.parent / "results"
    existing_dirs = [Path(path) for path in plan.get("existing_result_dirs", [])] + [results_dir]
    prepared = []
    fingerprints = set()
    for item in plan["images"]:
        path = Path(item["image_path"])
        data = path.read_bytes()
        fingerprint = probe.sha256_hex(data)
        if fingerprint != item["sha256"] or not all(probe.png_dimensions(data)):
            parser.error(f"图片指纹变化或格式不合法: {path.name}")
        if fingerprint in fingerprints:
            parser.error("计划中有重复图片内容")
        fingerprints.add(fingerprint)
        completed = next((found for directory in existing_dirs
                          if (found := probe.find_completed(directory, fingerprint, prompt_text)) is not None), None)
        if completed is not None:
            print(f"[SKIP] {path.name}: 已有同一请求的合格输出", flush=True)
            continue
        prepared.append((item, path, data))
    prepared = prepared[:maximum]
    if args.dry_run:
        print(json.dumps({"dry_run": True, "max_requests": maximum,
                          "api_credit_stop_threshold": budget,
                          "planned_calls": [item["sample_id"] for item, _, _ in prepared],
                          "paid_requests": 0, "key_read": False}, ensure_ascii=False, indent=2))
        return 0
    if not prepared:
        print("没有待调用的新图片。", flush=True)
        return 0

    config = json.loads(probe.CONFIG_PATH.read_text(encoding="utf-8-sig"))
    base_url = f"http://127.0.0.1:{config['port']}"
    key = probe.load_proxy_key()
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = plan_path.parent / "summary.json"
    if summary_path.exists():
        parser.error("该批次已有汇总,请另建计划目录;不会覆盖旧批次")
    summary = {"started_at": now(), "status": "running", "plan_sha256": probe.sha256_hex(plan_bytes),
               "code_sha256": {path.name: probe.sha256_hex(path.read_bytes())
                               for path in (Path(__file__), Path(probe.__file__))},
               "api_credit_limit": budget, "maximum_requests": maximum,
               "api_credits_reported_total": 0.0, "credits_complete": True,
               "attempts": [], "completed_json_count": 0,
               "workbuddy_task_credits": probe.NOT_PROVIDED,
               "account_balance_delta": probe.NOT_PROVIDED,
               "balance_note": "账户区间变化可能包含其他任务或延迟扣费,未归因于本批 API。"}
    save(summary_path, summary)
    exit_code = 0
    summary["stop_reason"] = "planned_images_completed"
    for item, path, data in prepared:
        if summary["api_credits_reported_total"] >= budget:
            summary["stop_reason"] = "api_credit_threshold_reached"
            break
        result_path = probe.build_result_path(path, results_dir)
        record = {"run_at": now(), "status": "pending", "image": item,
                  "request": probe.request_signature(prompt_text), "prompt_source": source,
                  "api_credit_limit": budget, "credits_this_call": probe.NOT_PROVIDED}
        with result_path.open("x", encoding="utf-8") as out:
            json.dump(record, out, ensure_ascii=False, indent=2)
        summary["attempts"].append({"sample_id": item["sample_id"], "result_path": str(result_path),
                                    "status": "pending"})
        save(summary_path, summary)
        print(f"[CALL {len(summary['attempts'])}/{maximum}] {path.name}; 已报积分={summary['api_credits_reported_total']:.2f}", flush=True)
        try:
            record = probe.run_one(path, prompt_text, key, base_url, image_bytes=data)
            record["prompt_source"] = source
            record["plan_sha256"] = summary["plan_sha256"]
        except KeyboardInterrupt:
            record["status"] = "interrupted"
            record["error"] = "中断;请求结果和费用未知,未自动重试。"
        except Exception as error:
            record["status"] = "failed"
            record["error"] = probe.redact_secret(f"{type(error).__name__}: {error}", key)
        save(result_path, record)
        response = record.get("response", {})
        cost = record.get("cost", {})
        credit = response.get("credits_this_call", probe.NOT_PROVIDED)
        if valid_credit(credit):
            summary["api_credits_reported_total"] = round(summary["api_credits_reported_total"] + credit, 6)
        else:
            summary["credits_complete"] = False
        if len(summary["attempts"]) == 1:
            summary["balance_before"] = cost.get("balance_before", probe.NOT_PROVIDED)
        summary["balance_after_immediate"] = cost.get("balance_after", probe.NOT_PROVIDED)
        summary["attempts"][-1].update({"status": record["status"], "api_credit": credit,
                                       "elapsed_seconds": response.get("elapsed_seconds", probe.NOT_PROVIDED)})
        if record["status"] == "success":
            summary["completed_json_count"] += 1
        save(summary_path, summary)
        print(f"[{record['status'].upper()}] {path.name}; credit={credit}; 累计={summary['api_credits_reported_total']:.2f}", flush=True)
        if record["status"] != "success":
            summary["stop_reason"] = record["status"]
            exit_code = 130 if record["status"] == "interrupted" else 2
            break
        if not valid_credit(credit):
            summary["stop_reason"] = "api_credit_not_provided"
            exit_code = 3
            break

    summary["status"] = "complete" if summary["stop_reason"] == "planned_images_completed" else "stopped"
    summary["finished_at"] = now()
    save(summary_path, summary)
    # 只读补查不调用模型;等待上限 30 秒,保留查询时点而不推测扣费归因。
    delay = plan.get("balance_followup_delay_seconds", 30)
    if not isinstance(delay, (int, float)) or not 0 <= delay <= 30:
        delay = 30
    if exit_code != 130:
        print(f"[BALANCE] {delay} 秒后只读补查余额。", flush=True)
        time.sleep(delay)
        with httpx.Client(trust_env=False) as client:
            summary["balance_after_followup"] = probe.query_balance(client, base_url, key)
        def remain(observations):
            if not isinstance(observations, list):
                return None
            return next((x.get("remain") for x in observations if x.get("site") == "cn-cli"), None)
        before = remain(summary.get("balance_before"))
        after = remain(summary.get("balance_after_followup"))
        if valid_credit(before) and valid_credit(after):
            summary["account_balance_delta"] = before - after
        save(summary_path, summary)
    print(f"[DONE] {summary_path}; stop_reason={summary['stop_reason']}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
