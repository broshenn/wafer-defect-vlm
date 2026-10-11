"""单GPU晶圆推理原型。GPU仅在服务器main内加载；CPU可检查输入/输出契约。"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import io
import json
import os
import re
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHECKER_HASH = "7d46c37f72bb8fd09488dd73754ad75e6a1b711d91a6f08f82e5b34026c0c0ff"
PROMPT_HASH = "8a8315f785a2398ba5a823754c40b8a8658f89b79aad1a0dbd68058acb43cfda"
PALETTE = {(0, 0, 0), (0, 255, 0), (255, 0, 0)}
MAX_BYTES = 4 * 1024 * 1024


class BusyError(RuntimeError):
    pass


def sha(data):
    return hashlib.sha256(data).hexdigest()


def validate_image(data):
    from PIL import Image
    if not data or len(data) > MAX_BYTES:
        raise ValueError("图片为空或超过4MB")
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.format != "PNG" or im.size != (448, 448):
                raise ValueError("请上传448×448的晶圆BIN PNG")
            im.load()
            if im.mode == "RGBA" and im.getchannel("A").getextrema() != (255, 255):
                raise ValueError("本原型不支持透明PNG")
            rgb = im.convert("RGB")
            colors = rgb.getcolors(maxcolors=4)
            if colors is None or not {color for _, color in colors} <= PALETTE:
                raise ValueError("本原型支持黑/绿/红三色BIN图，请使用项目渲染器")
            counts = dict((color, n) for n, color in colors)
            valid = counts.get((0, 255, 0), 0) + counts.get((255, 0, 0), 0)
            if not valid:
                raise ValueError("图片没有有效晶圆区域")
            return {"image_sha256": sha(data), "width": 448, "height": 448,
                    "red_present": counts.get((255, 0, 0), 0) > 0}
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("PNG无法解码") from exc


def load_checker(path):
    assert sha(path.read_bytes()) == CHECKER_HASH, "冻结检查器hash不一致"
    spec = importlib.util.spec_from_file_location("frozen_schema", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def adapt_answer(raw, checker):
    checked = checker.check_answer(raw)
    obj, how = checker.parse_strict(raw)
    issues = list(checked["schema_problems"])
    result, changes = None, []
    if obj is not None:
        result = dict(obj)
        # 仅展示/服务序列化修正：不覆盖原答，不把修后算原模型成绩。
        if result.get("clock_direction") == "null":
            result["clock_direction"] = None
            changes.append({"field": "clock_direction", "from": "null", "to": None,
                            "reason": "无方向的JSON类型规范化"})
        if not checker.schema7(result)[0]:
            result = None
        else:
            issues = []
            for field in ("morphology", "caption_zh"):
                if re.search(r"[%％]|百分之|\d+(?:\.\d+)?\s*(?:R\b|毫米|mm\b|μm|颗|枚|格)", result[field]):
                    issues.append("存在未经本服务核验的数量/尺寸表述")
            if not result["caption_zh"].strip():
                issues.append("描述为空")
            if result["uncertainty"].strip():
                issues.append("模型报告了不确定项")
            if result["defect_class"] == "unknown":
                issues.append("类别不确定")
    return {"result": result, "needs_review": result is None or bool(issues),
            "review_reasons": sorted(set(issues)), "format_changes": changes,
            "raw_answer": raw, "raw_sha256": sha(raw.encode("utf-8")),
            "raw_strict_json": checked["strict_json"], "raw_schema_ok": checked["schema_ok"],
            "content_validation": "未经逐图内容复核；needs_review不是经校准置信度"}


class Runtime:
    def __init__(self, args):
        import torch
        from swift import InferRequest, RequestConfig
        from swift.infer_engine import TransformersEngine
        assert os.environ.get("CUDA_VISIBLE_DEVICES"), "父shell必须先以UUID锁定GPU"
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
        self.torch, self.request_type = torch, InferRequest
        self.checker = load_checker(Path(args.checker))
        prompt_bytes = Path(args.prompt).read_bytes()
        assert sha(prompt_bytes) == PROMPT_HASH, "题面hash不一致"
        self.prompt = prompt_bytes.decode("utf-8")
        self.out = Path(args.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.tag = args.tag
        self.adapter_sha = sha((Path(args.adapter) / "adapter_model.safetensors").read_bytes())
        self.lock = threading.Lock()
        t0 = time.perf_counter()
        self.engine = TransformersEngine(args.model, model_type="qwen3_5", template_type="qwen3_5", adapters=[args.adapter])
        self.load_seconds = time.perf_counter() - t0
        self.config = RequestConfig(max_tokens=256, temperature=0.0, seed=3407)
        mapping = getattr(self.engine.model, "hf_device_map", {})
        assert all(str(v) in {"0", "cuda:0"} for v in mapping.values()), "模型不全在这一张GPU"
        assert any(p.is_cuda for p in self.engine.model.parameters()), "没有GPU模型参数"

    def predict(self, data):
        image_meta = validate_image(data)
        if not self.lock.acquire(blocking=False):
            raise BusyError("模型忙，请稍后重试")
        try:
            rid = uuid.uuid4().hex
            t0 = time.perf_counter()
            self.torch.cuda.reset_peak_memory_stats()
            with tempfile.TemporaryDirectory(prefix="wafer_request_") as folder:
                path = Path(folder) / "image.png"
                path.write_bytes(data)
                req = self.request_type(messages=[{"role": "system", "content": "你是半导体晶圆缺陷分析专家。"},
                        {"role": "user", "content": "<image>" + self.prompt}], images=[str(path)])
                reply = self.engine.infer([req], self.config)[0].choices[0]
                raw = reply.message.content
            result = adapt_answer(raw, self.checker)
            result.update({"request_id": rid, "model_tag": self.tag, "adapter_sha256": self.adapter_sha,
                           "prompt_sha256": PROMPT_HASH, "image": image_meta,
                           "server_predict_seconds": time.perf_counter() - t0,
                           "peak_allocated_gib": self.torch.cuda.max_memory_allocated() / 2**30,
                           "finish_reason": getattr(reply, "finish_reason", None)})
            (self.out / f"{rid}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return result
        finally:
            self.lock.release()


def handler(runtime):
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status, value):
            body = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self.send_json(200, {"status": "ready", "model_tag": runtime.tag,
                                     "gpu_count": 1, "load_seconds": runtime.load_seconds})
            elif self.path == "/":
                body = (HERE / "demo.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_json(404, {"error": "not_found"})

        def do_POST(self):
            if self.path != "/predict":
                self.send_json(404, {"error": "not_found"}); return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if size <= 0 or size > MAX_BYTES:
                    self.send_json(413, {"error": "请求应包含不超过4MB的PNG"}); return
                data = self.rfile.read(size)
                t0 = time.perf_counter()
                result = runtime.predict(data)
                result["request_seconds_after_upload"] = time.perf_counter() - t0
                self.send_json(200, result)
            except ValueError as exc:
                self.send_json(422, {"error": str(exc)})
            except BusyError:
                self.send_json(429, {"error": "模型忙，请稍后重试"})
            except Exception as exc:
                self.send_json(500, {"error": "推理失败", "error_type": type(exc).__name__})

        def log_message(self, *_):
            pass
    return Handler


def main():
    ap = argparse.ArgumentParser()
    for field in ("model", "adapter", "prompt", "checker", "out", "tag"):
        ap.add_argument("--" + field, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--max-runtime-seconds", type=int, default=3600)
    args = ap.parse_args()
    assert 0 < args.max_runtime_seconds <= 3600
    # 含加载与空闲占卡。该进程退出不等于平台已关机，执行方仍核计费状态。
    def deadline():
        time.sleep(args.max_runtime_seconds)
        os._exit(7)
    threading.Thread(target=deadline, daemon=True).start()
    runtime = Runtime(args)
    server = ThreadingHTTPServer((args.host, args.port), handler(runtime))
    print(json.dumps({"ready": True, "model_tag": runtime.tag, "port": args.port,
                      "load_seconds": runtime.load_seconds}, ensure_ascii=False), flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
