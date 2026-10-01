# -*- coding: utf-8 -*-
"""经本机 WorkBuddy 代理调用 Kimi K3 (kimi-k3-1) 标注晶圆 BIN 图的可复用脚本。

设计约束(2026-10-01 积分小测):
- 直接读取原位置 PNG,不搬动、不复制图片;实际发送图片字节(base64 data URL)。
- 固定模型 kimi-k3-1,max_tokens=8192,客户端关闭自动重试,超时 300 秒。
- 不添加未经验证的推理参数。
- 不读取原始类别标签,不向模型提供预期答案。
- 本地代理 Key 仅在内存中读取,绝不打印或写入脚本、日志、结果文件。
- 失败、空回答或截断时记录失败结果并停止,不自动重试、不扩大额度。
- 完整 JSON 对象、七个必需字段及类型均通过,才记为成功;这不代表事实正确。
- 每次尝试独立保存;跳过时核对图片、提示词、请求参数及当前校验规则。
- 拿不到的字段一律记录为 "未提供",不推测、不编造。

用法:
  python -X utf8 run_kimi_k3_wafer.py --image "path/one.png"          # 单张
  python -X utf8 run_kimi_k3_wafer.py --image-dir "dir" --limit 5     # 目录批量
  python -X utf8 run_kimi_k3_wafer.py --image "one.png" --dry-run     # 只预览不调用
  python -X utf8 run_kimi_k3_wafer.py --image-dir "dir" --limit 5 --skip-done
"""
import argparse
import base64
import hashlib
import json
import math
import struct
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import httpx
from openai import OpenAI

NOT_PROVIDED = "未提供"
MODEL_ID = "kimi-k3-1"
MAX_TOKENS = 8192
TIMEOUT_SECONDS = 300
CONFIG_PATH = Path(r"D:\pycode\workbuddy\proxy\config.json")
DEFAULT_PROMPT_FILE = Path(
    r"D:\pycode\workbuddy\wafer-cost-test\20261001-211719\prompt-image-only.txt"
)
RESULTS_DIR = Path(__file__).resolve().parent / "results"
ANSWER_FIELDS = {
    "defect_class", "morphology", "radial_zone", "clock_direction",
    "extent_r", "caption_zh", "uncertainty",
}
DEFECT_CLASSES = {
    "Center", "Donut", "Edge_Loc", "Edge_Ring", "Loc", "Near_full",
    "Random", "Scratch", "none", "unknown",
}
RADIAL_ZONES = {"center", "middle", "edge", "global", "unknown", "none"}


def validate_answer(answer):
    """仅校验格式、字段和类型,不把模型回答认证为真实标注。"""
    result = {
        "answer_is_complete_json": False,
        "answer_schema_valid": False,
        "answer_json_error": None,
        "answer_validation_error": None,
    }
    if not isinstance(answer, str) or not answer.strip():
        result["answer_json_error"] = "回答为空或非字符串"
        return result

    def reject_constant(value):
        raise ValueError(f"JSON 不允许 {value}")

    try:
        obj = json.loads(answer, parse_constant=reject_constant)
    except (json.JSONDecodeError, ValueError) as error:
        result["answer_json_error"] = str(error)
        return result
    result["answer_is_complete_json"] = True
    errors = []
    if not isinstance(obj, dict):
        errors.append("回答必须是 JSON 对象")
    else:
        missing = ANSWER_FIELDS - obj.keys()
        extra = obj.keys() - ANSWER_FIELDS
        if missing:
            errors.append("缺少字段: " + ", ".join(sorted(missing)))
        if extra:
            errors.append("多余字段: " + ", ".join(sorted(extra)))
        for field in ("defect_class", "morphology", "radial_zone", "caption_zh", "uncertainty"):
            if field in obj and not isinstance(obj[field], str):
                errors.append(f"{field} 必须是字符串")
        for field in ("morphology", "caption_zh"):
            if isinstance(obj.get(field), str) and not obj[field].strip():
                errors.append(f"{field} 不能为空")
        if isinstance(obj.get("defect_class"), str) and obj["defect_class"] not in DEFECT_CLASSES:
            errors.append("defect_class 不在允许的类别中")
        if isinstance(obj.get("radial_zone"), str) and obj["radial_zone"] not in RADIAL_ZONES:
            errors.append("radial_zone 不在允许的区域中")
        direction = obj.get("clock_direction")
        if direction is not None and (not isinstance(direction, str) or not direction.strip()):
            errors.append("clock_direction 必须是非空字符串或 null")
        extent = obj.get("extent_r")
        if extent is not None and (
            isinstance(extent, bool) or not isinstance(extent, (int, float))
            or not math.isfinite(extent) or extent < 0
        ):
            errors.append("extent_r 必须是非负有限数值或 null")
    result["answer_schema_valid"] = not errors
    result["answer_validation_error"] = "; ".join(errors) if errors else None
    return result


def request_signature(prompt_text):
    """断点续跑使用的请求身份,不包含密钥、账号或预期标签。"""
    return {
        "model": MODEL_ID,
        "max_tokens": MAX_TOKENS,
        "timeout_seconds": TIMEOUT_SECONDS,
        "client_max_retries": 0,
        "extra_params": "无(未添加未经验证的推理参数)",
        "prompt_sha256": sha256_hex(prompt_text.encode("utf-8")),
    }


def find_completed(results_dir, image_sha256, prompt_text):
    """兼容旧结果,但重新核验回答,不单凭 success 字段跳过。"""
    signature = request_signature(prompt_text)
    for result_path in sorted(results_dir.glob("*.json")):
        try:
            record = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if not isinstance(record, dict) or record.get("status") != "success":
            continue
        image = record.get("image")
        request = record.get("request")
        response = record.get("response")
        if not all(isinstance(item, dict) for item in (image, request, response)):
            continue
        if image.get("sha256") != image_sha256:
            continue
        if any(request.get(name) != value for name, value in signature.items()):
            continue
        if response.get("actual_model") != MODEL_ID or response.get("finish_reason") != "stop":
            continue
        if validate_answer(response.get("answer_raw"))["answer_schema_valid"]:
            return result_path
    return None


def redact_secret(value, key):
    """错误消息或上游响应若回显密钥,保存前去掉。"""
    if isinstance(value, str):
        return value.replace(key, "[REDACTED]") if key else value
    if isinstance(value, dict):
        return {redact_secret(k, key): redact_secret(v, key) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_secret(item, key) for item in value]
    return value


def load_proxy_key() -> str:
    """仅在内存中读取本地代理 Key。"""
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
    key = config["apiKey"]
    if isinstance(key, list):
        key = key[0]
    return key


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def png_dimensions(data: bytes):
    """从 PNG IHDR 读取宽高,不依赖 Pillow。"""
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return None, None
    if len(data) < 33 or data[12:16] != b"IHDR":
        return None, None
    width, height = struct.unpack(">II", data[16:24])
    return width, height


def query_balance(client: httpx.Client, base_url: str, key: str):
    """查询 /status,只提取积分数值,不记录账号昵称等其它字段。"""
    try:
        resp = client.get(f"{base_url}/status", params={"site": "cn-cli"},
                          headers={"Authorization": f"Bearer {key}"}, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        sites = payload.get("sites") or []
        out = []
        for site in sites:
            credit = site.get("credit") or {}
            out.append({
                "site": site.get("site", NOT_PROVIDED),
                "logged_in": site.get("logged_in", NOT_PROVIDED),
                "remain": credit.get("remain", NOT_PROVIDED),
                "queried_at": datetime.now(timezone.utc).astimezone().isoformat(),
            })
        return out
    except Exception as error:  # noqa: BLE001 - 余额查询失败不阻断,记为未提供
        return [{"site": NOT_PROVIDED, "logged_in": NOT_PROVIDED, "remain": NOT_PROVIDED,
                 "queried_at": datetime.now(timezone.utc).astimezone().isoformat(),
                 "query_error": redact_secret(f"{type(error).__name__}: {error}", key)}]


def collect_images(args) -> list:
    images = []
    if args.image:
        images.append(Path(args.image).resolve())
    if args.image_dir:
        directory = Path(args.image_dir).resolve()
        images.extend(sorted(directory.glob("*.png")))
    return images


def build_result_path(image_path: Path, results_dir=RESULTS_DIR) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return results_dir / f"{image_path.stem}__{stamp}__{uuid4().hex}.json"


def run_one(image_path: Path, prompt_text: str, key: str, base_url: str,
            *, image_bytes=None, http_client_factory=None) -> dict:
    """对单张图片发起一次请求(不重试),返回完整结果记录。"""
    record = {
        "run_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "image": {},
        "request": {},
        "response": {},
        "cost": {},
        "status": "pending",
        "error": None,
    }

    image_bytes = image_path.read_bytes() if image_bytes is None else image_bytes
    width, height = png_dimensions(image_bytes)
    if not width or not height:
        raise ValueError("输入必须是包含有效 IHDR 的 PNG")
    record["image"] = {
        "path": str(image_path),
        "sha256": sha256_hex(image_bytes),
        "bytes": len(image_bytes),
        "width_px": width if width is not None else NOT_PROVIDED,
        "height_px": height if height is not None else NOT_PROVIDED,
    }
    record["request"] = {
        "base_url": base_url,
        "endpoint": "POST /v1/chat/completions",
        **request_signature(prompt_text),
        "prompt_text": prompt_text,
    }

    make_http_client = http_client_factory or (lambda: httpx.Client(trust_env=False))
    http_client = make_http_client()

    record["cost"]["balance_before"] = query_balance(http_client, base_url, key)

    image_data_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
    started = time.perf_counter()
    try:
        with OpenAI(
            base_url=f"{base_url}/v1",
            api_key=key,
            http_client=http_client,
            timeout=TIMEOUT_SECONDS,
            max_retries=0,
        ) as client:
            raw_response = client.chat.completions.with_raw_response.create(
                model=MODEL_ID,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                }],
                max_tokens=MAX_TOKENS,
            )
        elapsed = time.perf_counter() - started
        raw_text = raw_response.text
        try:
            raw_body = json.loads(raw_text)
            if not isinstance(raw_body, dict):
                raw_body = None
        except json.JSONDecodeError:
            raw_body = None
        # 先保存原始响应,后续解析字段失败也不能丢失证据。
        record["raw_response_body"] = raw_body if raw_body is not None else raw_text

        response_info = {
            "http_status": raw_response.http_response.status_code,
            "elapsed_seconds": round(elapsed, 3),
            "actual_model": NOT_PROVIDED,
            "finish_reason": NOT_PROVIDED,
            "usage": NOT_PROVIDED,
            "reasoning_tokens": NOT_PROVIDED,
            "credits_this_call": NOT_PROVIDED,
            "answer_raw": None,
            "answer_is_complete_json": None,
            "answer_json_error": None,
            "answer_schema_valid": False,
            "answer_validation_error": None,
        }

        if raw_body:
            response_info["actual_model"] = raw_body.get("model", NOT_PROVIDED)
            choices = raw_body.get("choices") or []
            if choices:
                choice = choices[0]
                response_info["finish_reason"] = choice.get("finish_reason", NOT_PROVIDED)
                message = choice.get("message") or {}
                answer = message.get("content")
                response_info["answer_raw"] = answer
                response_info.update(validate_answer(answer))
            usage = raw_body.get("usage")
            if usage:
                response_info["usage"] = usage
                details = usage.get("completion_tokens_details") or {}
                response_info["reasoning_tokens"] = details.get("reasoning_tokens", NOT_PROVIDED)
            # 上游积分字段位置未公开,按已知可能的键逐一检查;都没有则记未提供
            for credit_key in ("credits", "credit", "cost"):
                if credit_key in raw_body:
                    response_info["credits_this_call"] = raw_body[credit_key]
                    break
            if isinstance(raw_body.get("usage"), dict) and response_info["credits_this_call"] == NOT_PROVIDED:
                for credit_key in ("credits", "credit", "cost"):
                    if credit_key in raw_body["usage"]:
                        response_info["credits_this_call"] = raw_body["usage"][credit_key]
                        break
        else:
            response_info["raw_body_unparsed"] = raw_text

        record["response"] = response_info
        if not raw_body or not response_info["answer_raw"]:
            record["status"] = "failed"
            record["error"] = "空回答或响应无法解析"
        elif response_info["finish_reason"] != "stop":
            record["status"] = "failed"
            record["error"] = f"非正常结束: finish_reason={response_info['finish_reason']}"
        elif response_info["actual_model"] != MODEL_ID:
            record["status"] = "failed"
            record["error"] = "实际返回模型不匹配或未提供"
        elif not response_info["answer_schema_valid"]:
            record["status"] = "failed"
            record["error"] = (response_info["answer_json_error"]
                               or response_info["answer_validation_error"])
        else:
            record["status"] = "success"
    except Exception as error:  # noqa: BLE001 - 任何失败都记录并停止,不重试
        elapsed = time.perf_counter() - started
        record["status"] = "failed"
        record["error"] = f"{type(error).__name__}: {error}"
        error_response = getattr(error, "response", None)
        if error_response is not None:
            record["raw_response_body"] = error_response.text
        record["response"] = {
            "http_status": getattr(error_response, "status_code", NOT_PROVIDED),
            "elapsed_seconds": round(elapsed, 3),
            "actual_model": NOT_PROVIDED,
            "finish_reason": NOT_PROVIDED,
            "usage": NOT_PROVIDED,
            "reasoning_tokens": NOT_PROVIDED,
            "credits_this_call": NOT_PROVIDED,
            "answer_raw": None,
            "answer_is_complete_json": None,
            "answer_json_error": None,
            "answer_schema_valid": False,
            "answer_validation_error": None,
        }
    finally:
        # OpenAI 上下文管理器会关闭传入的 http_client,余额复查必须用独立新客户端
        http_client.close()
        with make_http_client() as balance_client:
            record["cost"]["balance_after"] = query_balance(balance_client, base_url, key)

    before = next((item for item in record["cost"]["balance_before"] if item.get("site") == "cn-cli"), {})
    after = next((item for item in record["cost"]["balance_after"] if item.get("site") == "cn-cli"), {})
    if (isinstance(before.get("remain"), (int, float))
            and isinstance(after.get("remain"), (int, float))):
        record["cost"]["workbuddy_balance_delta"] = round(before["remain"] - after["remain"], 4)
    else:
        record["cost"]["workbuddy_balance_delta"] = NOT_PROVIDED
    record["cost"]["workbuddy_task_credits"] = NOT_PROVIDED
    record["cost"]["balance_note"] = (
        "余额差是查询区间的账户变化,可能包含其他任务或延迟扣费;"
        "不是本次 API 或整个 WorkBuddy 任务的已归因费用。"
    )

    return redact_secret(record, key)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--image", type=str, help="单张 PNG 路径")
    inputs.add_argument("--image-dir", type=str, help="PNG 目录(取 *.png)")
    parser.add_argument("--limit", type=int, default=1, help="本次调用数量上限,默认 1;跳过的图片不占额度")
    parser.add_argument("--prompt-file", type=str, default=str(DEFAULT_PROMPT_FILE), help="标注提示词文件,原样读取")
    parser.add_argument("--dry-run", action="store_true", help="只预览将要发送的内容,不发起调用")
    parser.add_argument("--skip-done", action="store_true", help="跳过已有成功结果记录的图片")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR, help="结果目录,不会覆盖已有尝试")
    args = parser.parse_args(argv)

    if args.limit < 1:
        parser.error("--limit 必须是正整数")

    prompt_bytes = Path(args.prompt_file).read_bytes()
    # 沿用首轮 read_text 的通用换行行为,保持已实测的请求输入不变。
    # 文件字节与实际发送文本的指纹分开记录,不把二者混称同一指纹。
    prompt_text = prompt_bytes.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    prompt_source = {
        "path": str(Path(args.prompt_file).resolve()),
        "file_sha256": sha256_hex(prompt_bytes),
        "sent_text_sha256": sha256_hex(prompt_text.encode("utf-8")),
        "newline_handling": "CRLF/CR 转为 LF,沿用首轮 Python 通用换行读取",
    }
    images = collect_images(args)
    if not images:
        print("没有匹配的图片。")
        return 1

    results_dir = args.results_dir.resolve()
    if args.dry_run:
        print(f"[DRY-RUN] 模型={MODEL_ID} max_tokens={MAX_TOKENS} 超时={TIMEOUT_SECONDS}s 重试=0")
        print(f"[DRY-RUN] 提示词文件={args.prompt_file} sha256={sha256_hex(prompt_text.encode('utf-8'))}")
        print(f"[DRY-RUN] 文件字节 sha256={prompt_source['file_sha256']}; {prompt_source['newline_handling']}")
        print(f"[DRY-RUN] 提示词内容:\n{prompt_text}")
    exit_code = 0
    processed = 0
    for image_path in images:
        if processed >= args.limit:
            break
        data = image_path.read_bytes()
        w, h = png_dimensions(data)
        if not w or not h:
            parser.error(f"不是有效 PNG: {image_path}")
        fingerprint = sha256_hex(data)
        if args.skip_done:
            completed = find_completed(results_dir, fingerprint, prompt_text)
            if completed is not None:
                print(f"[SKIP] 图片、提示词、参数及回答校验一致: {image_path.name}")
                continue
        processed += 1
        if args.dry_run:
            print(f"[DRY-RUN] 将处理: {image_path} | {len(data)} bytes | {w}x{h} | sha256={fingerprint}")
            continue

        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
        base_url = f"http://127.0.0.1:{config['port']}"
        key = load_proxy_key()
        results_dir.mkdir(parents=True, exist_ok=True)
        result_path = build_result_path(image_path, results_dir)
        record = {
            "run_at": datetime.now(timezone.utc).astimezone().isoformat(),
            "status": "pending", "image": {"path": str(image_path), "sha256": fingerprint},
            "request": request_signature(prompt_text),
            "prompt_source": prompt_source,
            "cost": {"api_credits": NOT_PROVIDED, "workbuddy_task_credits": NOT_PROVIDED},
            "note": "请求前预留的记录;中断或未返回时费用与结果未知,不可据此重试。",
        }
        # 使用独占创建,先留下开始记录,保证中断不会使付费尝试完全消失。
        with result_path.open("x", encoding="utf-8") as output:
            json.dump(record, output, ensure_ascii=False, indent=2)

        print(f"[CALL] {image_path.name} ...", flush=True)
        try:
            record = run_one(image_path, prompt_text, key, base_url, image_bytes=data)
            record["prompt_source"] = prompt_source
        except KeyboardInterrupt:
            record["status"] = "interrupted"
            record["error"] = "用户中断,请求可能已发送;结果及费用未提供,未自动重试。"
            result_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[STOP] 中断记录已保存: {result_path}")
            return 130
        except Exception as error:
            record["status"] = "failed"
            record["error"] = redact_secret(f"{type(error).__name__}: {error}", key)
        result_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{record['status'].upper()}] {image_path.name} -> {result_path}")
        if record["status"] != "success":
            print(f"[STOP] 失败/空回答/截断,记录已保存,停止后续调用: {record['error']}")
            exit_code = 2
            break

    if args.dry_run:
        print(f"[DRY-RUN] 待调用 {processed} 张;未读取密钥、未发起 API 调用、未写结果。")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
