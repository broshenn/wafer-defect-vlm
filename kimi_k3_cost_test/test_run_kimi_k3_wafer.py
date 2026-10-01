"""离线验证:只使用临时文件和 httpx 模拟接口,不访问 WorkBuddy。"""
import base64
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

import run_kimi_k3_wafer as probe


PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jhXcAAAAASUVORK5CYII="
)
TEST_KEY = "TEST_ONLY_NOT_A_REAL_KEY"
ANSWER = {
    "defect_class": "unknown", "morphology": "无法确定主要图案",
    "radial_zone": "unknown", "clock_direction": None, "extent_r": None,
    "caption_zh": "图片无法可靠判读。", "uncertainty": "离线测试样例",
}
PROMPT = "离线测试提示词\r\n不包含类别答案。"
SENT_PROMPT = PROMPT.replace("\r\n", "\n")


def completed_record(data=PNG, prompt=PROMPT):
    return {
        "status": "success",
        "image": {"sha256": probe.sha256_hex(data)},
        "request": probe.request_signature(prompt),
        "response": {
            "actual_model": probe.MODEL_ID, "finish_reason": "stop",
            "answer_raw": json.dumps(ANSWER, ensure_ascii=False),
        },
    }


class AnswerTests(unittest.TestCase):
    def test_valid_unknown_answer(self):
        result = probe.validate_answer(json.dumps(ANSWER))
        self.assertTrue(result["answer_is_complete_json"])
        self.assertTrue(result["answer_schema_valid"])

    def test_empty_truncated_array_and_missing_field_fail(self):
        missing = {name: value for name, value in ANSWER.items() if name != "caption_zh"}
        for answer in (None, "", "{", "[]", json.dumps(missing)):
            with self.subTest(answer=answer):
                self.assertFalse(probe.validate_answer(answer)["answer_schema_valid"])

    def test_wrong_types_enums_and_nonfinite_values_fail(self):
        changes = [
            ("defect_class", "made_up"), ("radial_zone", "made_up"),
            ("morphology", 1), ("caption_zh", ""), ("uncertainty", None),
            ("clock_direction", 3), ("extent_r", True),
            ("extent_r", -1), ("extent_r", float("nan")),
        ]
        for field, value in changes:
            with self.subTest(field=field, value=value):
                self.assertFalse(probe.validate_answer(json.dumps({**ANSWER, field: value}))["answer_schema_valid"])


class ResumeTests(unittest.TestCase):
    def test_legacy_success_revalidated_and_identity_changes_not_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = directory / "legacy.json"
            record = completed_record()
            path.write_text(json.dumps(record), encoding="utf-8")
            self.assertEqual(probe.find_completed(directory, probe.sha256_hex(PNG), PROMPT), path)
            self.assertIsNone(probe.find_completed(directory, probe.sha256_hex(PNG + b"changed"), PROMPT))
            self.assertIsNone(probe.find_completed(directory, probe.sha256_hex(PNG), PROMPT + "changed"))
            record["request"]["max_tokens"] = 2048
            path.write_text(json.dumps(record), encoding="utf-8")
            self.assertIsNone(probe.find_completed(directory, probe.sha256_hex(PNG), PROMPT))

    def test_invalid_legacy_success_is_not_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            record = completed_record()
            record["response"]["answer_raw"] = "{"
            (directory / "legacy.json").write_text(json.dumps(record), encoding="utf-8")
            self.assertIsNone(probe.find_completed(directory, probe.sha256_hex(PNG), PROMPT))


class TransportTests(unittest.TestCase):
    def run_mock(self, answer=None, finish_reason="stop", http_status=200, model=probe.MODEL_ID):
        calls = []
        balance_calls = []

        def handler(request):
            if request.url.path == "/status":
                balance_calls.append(request)
                self.assertEqual(request.url.params["site"], "cn-cli")
                return httpx.Response(200, json={"sites": [{
                    "site": "cn-cli", "logged_in": True,
                    "credit": {"remain": 1000 - len(balance_calls)},
                }]})
            self.assertEqual(request.url.path, "/v1/chat/completions")
            self.assertEqual(request.method, "POST")
            payload = json.loads(request.content)
            self.assertEqual(set(payload), {"model", "messages", "max_tokens"})
            self.assertEqual(payload["model"], probe.MODEL_ID)
            self.assertEqual(payload["max_tokens"], 8192)
            content = payload["messages"][0]["content"]
            self.assertEqual(content[0]["text"], PROMPT)
            self.assertEqual(base64.b64decode(content[1]["image_url"]["url"].split(",", 1)[1]), PNG)
            calls.append(request)
            if http_status != 200:
                return httpx.Response(http_status, json={"error": {"message": TEST_KEY + " failure"}})
            return httpx.Response(200, json={
                "id": "offline-test", "object": "chat.completion", "created": 0,
                "model": model, "choices": [{"index": 0,
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": finish_reason,
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20,
                          "total_tokens": 30, "credit": 1.25,
                          "completion_tokens_details": {"reasoning_tokens": 15}},
            })

        transport = httpx.MockTransport(handler)
        record = probe.run_one(
            Path("offline-only.png"), PROMPT, TEST_KEY, "http://offline.invalid",
            image_bytes=PNG,
            http_client_factory=lambda: httpx.Client(transport=transport, trust_env=False),
        )
        self.assertEqual(len(calls), 1, "每张图片恰好一次请求,失败也不重试")
        self.assertEqual(len(balance_calls), 2)
        self.assertNotIn(TEST_KEY, json.dumps(record))
        return record

    def test_success_preserves_bytes_prompt_usage_and_unknown_task_cost(self):
        record = self.run_mock(answer=json.dumps(ANSWER))
        self.assertEqual(record["status"], "success")
        self.assertTrue(record["response"]["answer_schema_valid"])
        self.assertEqual(record["response"]["credits_this_call"], 1.25)
        self.assertEqual(record["cost"]["workbuddy_task_credits"], "未提供")
        self.assertIn("queried_at", record["cost"]["balance_before"][0])

    def test_malformed_and_missing_field_answers_fail_even_with_stop(self):
        missing = {key: value for key, value in ANSWER.items() if key != "extent_r"}
        for answer in ("{", "[]", json.dumps(missing)):
            with self.subTest(answer=answer):
                record = self.run_mock(answer=answer)
                self.assertEqual(record["status"], "failed")
                self.assertEqual(record["raw_response_body"]["choices"][0]["message"]["content"], answer)

    def test_empty_length_missing_reason_or_wrong_model_fail(self):
        for changes in (
            {"answer": ""}, {"answer": json.dumps(ANSWER), "finish_reason": "length"},
            {"answer": json.dumps(ANSWER), "finish_reason": None},
            {"answer": json.dumps(ANSWER), "model": "other-model"},
        ):
            with self.subTest(changes=changes):
                record = self.run_mock(**changes)
                self.assertEqual(record["status"], "failed")
                self.assertEqual(record["response"]["credits_this_call"], 1.25)

    def test_server_error_no_retry_and_key_redacted(self):
        record = self.run_mock(http_status=500)
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["response"]["http_status"], 500)
        self.assertIn("[REDACTED]", record["raw_response_body"])
        self.assertEqual(record["response"]["credits_this_call"], "未提供")


class MainTests(unittest.TestCase):
    def setup_files(self, directory):
        images = directory / "images"
        images.mkdir()
        # 测试续跑按内容指纹识别;这些合成文件不来自真实晶圆。
        for name in ("a", "b", "c"):
            (images / f"{name}.png").write_bytes(PNG + name.encode())
        prompt = directory / "prompt.txt"
        prompt.write_bytes(PROMPT.encode())
        config = directory / "config.json"
        config.write_text('{"port": 8788}', encoding="utf-8")
        return images, prompt, config, directory / "results"

    def test_dry_run_does_not_read_key_call_model_or_write_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            images, prompt, config, results = self.setup_files(Path(tmp))
            with patch.object(probe, "CONFIG_PATH", Path(tmp) / "absent-config"), \
                 patch.object(probe, "load_proxy_key", side_effect=AssertionError("不能读密钥")), \
                 patch.object(probe, "run_one", side_effect=AssertionError("不能发请求")), \
                 contextlib.redirect_stdout(io.StringIO()):
                code = probe.main(["--image-dir", str(images), "--limit", "2",
                                   "--prompt-file", str(prompt), "--results-dir", str(results), "--dry-run"])
            self.assertEqual(code, 0)
            self.assertFalse(results.exists())

    def test_default_limit_does_not_inspect_later_invalid_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            images, prompt, config, results = self.setup_files(Path(tmp))
            (images / "b.png").write_bytes(b"not a PNG")
            output = io.StringIO()
            with patch.object(probe, "load_proxy_key", side_effect=AssertionError("不能读密钥")), \
                 contextlib.redirect_stdout(output):
                code = probe.main(["--image-dir", str(images), "--prompt-file", str(prompt),
                                   "--results-dir", str(results), "--dry-run"])
            self.assertEqual(code, 0)
            self.assertIn("待调用 1 张", output.getvalue())
            self.assertFalse(results.exists())

    def test_skip_happens_before_limit_and_preserves_legacy_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            images, prompt, config, results = self.setup_files(Path(tmp))
            results.mkdir()
            old_path = results / "a.json"
            old_path.write_text(json.dumps(completed_record(PNG + b"a", SENT_PROMPT)), encoding="utf-8")
            old_bytes = old_path.read_bytes()
            called = []

            def fake_run(image_path, prompt_text, key, base_url, **kwargs):
                called.append(image_path.name)
                self.assertEqual(prompt_text, SENT_PROMPT, "沿用首轮的实际发送文本")
                return completed_record(kwargs["image_bytes"], prompt_text)

            with patch.object(probe, "CONFIG_PATH", config), \
                 patch.object(probe, "load_proxy_key", return_value=TEST_KEY), \
                 patch.object(probe, "run_one", side_effect=fake_run), \
                 contextlib.redirect_stdout(io.StringIO()):
                code = probe.main(["--image-dir", str(images), "--limit", "1", "--skip-done",
                                   "--prompt-file", str(prompt), "--results-dir", str(results)])
            self.assertEqual(code, 0)
            self.assertEqual(called, ["b.png"])
            self.assertEqual(old_path.read_bytes(), old_bytes)
            self.assertEqual(len(list(results.glob("*.json"))), 2)
            new_path = next(path for path in results.glob("*.json") if path != old_path)
            source = json.loads(new_path.read_text(encoding="utf-8"))["prompt_source"]
            self.assertEqual(source["file_sha256"], probe.sha256_hex(PROMPT.encode()))
            self.assertEqual(source["sent_text_sha256"], probe.sha256_hex(SENT_PROMPT.encode()))

    def test_failure_stops_and_repeated_attempts_do_not_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            images, prompt, config, results = self.setup_files(Path(tmp))

            def fake_failure(*args, **kwargs):
                return {"status": "failed", "error": "离线模拟失败"}

            with patch.object(probe, "CONFIG_PATH", config), \
                 patch.object(probe, "load_proxy_key", return_value=TEST_KEY), \
                 patch.object(probe, "run_one", side_effect=fake_failure) as run, \
                 contextlib.redirect_stdout(io.StringIO()):
                args = ["--image-dir", str(images), "--limit", "3",
                        "--prompt-file", str(prompt), "--results-dir", str(results)]
                self.assertEqual(probe.main(args), 2)
                first = next(results.glob("*.json"))
                first_bytes = first.read_bytes()
                self.assertEqual(run.call_count, 1)
                self.assertEqual(probe.main(args), 2)
                self.assertEqual(run.call_count, 2)
            self.assertEqual(first.read_bytes(), first_bytes)
            self.assertEqual(len(list(results.glob("*.json"))), 2)

    def test_interrupt_keeps_unknown_attempt_and_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            images, prompt, config, results = self.setup_files(Path(tmp))
            with patch.object(probe, "CONFIG_PATH", config), \
                 patch.object(probe, "load_proxy_key", return_value=TEST_KEY), \
                 patch.object(probe, "run_one", side_effect=KeyboardInterrupt) as run, \
                 contextlib.redirect_stdout(io.StringIO()):
                code = probe.main(["--image-dir", str(images), "--limit", "3",
                                   "--prompt-file", str(prompt), "--results-dir", str(results)])
            self.assertEqual(code, 130)
            self.assertEqual(run.call_count, 1)
            records = list(results.glob("*.json"))
            self.assertEqual(len(records), 1)
            record = json.loads(records[0].read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "interrupted")
            self.assertEqual(record["cost"]["api_credits"], "未提供")


if __name__ == "__main__":
    unittest.main()
