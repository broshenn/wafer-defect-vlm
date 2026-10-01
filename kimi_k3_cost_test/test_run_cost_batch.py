"""批次预算的离线检查;模型调用、密钥读取和余额查询均替换为模拟。"""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_cost_batch as batch
import run_kimi_k3_wafer as probe
from test_run_kimi_k3_wafer import PNG, PROMPT, SENT_PROMPT, TEST_KEY, completed_record


class BatchTests(unittest.TestCase):
    def run_case(self, credits, status="success", dry_run=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt = root / "prompt.txt"
            prompt.write_bytes(PROMPT.encode())
            config = root / "config.json"
            config.write_text('{"port":8788}', encoding="utf-8")
            images = []
            for number in range(5):
                data = PNG + str(number).encode()
                image = root / f"image-{number}.png"
                image.write_bytes(data)
                images.append({"sample_id": f"image-{number}", "image_path": str(image),
                               "sha256": probe.sha256_hex(data)})
            plan_path = root / "plan.json"
            plan_path.write_text(json.dumps({
                "api_credit_limit": 40, "max_requests": 5,
                "request": probe.request_signature(SENT_PROMPT),
                "prompt_file": str(prompt), "prompt_file_sha256": probe.sha256_hex(PROMPT.encode()),
                "prompt_sent_text_sha256": probe.sha256_hex(SENT_PROMPT.encode()),
                "images": images, "balance_followup_delay_seconds": 0,
            }), encoding="utf-8")
            called = []

            def fake_call(path, prompt_text, key, base_url, **kwargs):
                credit = credits[len(called)]
                called.append(path.name)
                record = completed_record(kwargs["image_bytes"], prompt_text)
                record["status"] = status
                record["response"]["credits_this_call"] = credit
                record["cost"] = {"balance_before": [{"site": "cn-cli", "remain": 100}],
                                  "balance_after": [{"site": "cn-cli", "remain": 90}]}
                return record

            with patch.object(probe, "CONFIG_PATH", config), \
                 patch.object(probe, "load_proxy_key", return_value=TEST_KEY) as key_loader, \
                 patch.object(probe, "run_one", side_effect=fake_call), \
                 patch.object(probe, "query_balance", return_value=[{"site": "cn-cli", "remain": 90}]), \
                 contextlib.redirect_stdout(io.StringIO()):
                code = batch.main(["--plan", str(plan_path)] + (["--dry-run"] if dry_run else []))
            if dry_run:
                self.assertEqual(key_loader.call_count, 0)
                self.assertFalse((root / "summary.json").exists())
                self.assertFalse((root / "results").exists())
                return code, called, None
            summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(len(list((root / "results").glob("*.json"))), len(called))
            self.assertNotIn(TEST_KEY, (root / "summary.json").read_text(encoding="utf-8"))
            return code, called, summary

    def test_exact_threshold_stops_before_fifth_request(self):
        code, calls, summary = self.run_case([10] * 5)
        self.assertEqual(code, 0)
        self.assertEqual(len(calls), 4)
        self.assertEqual(summary["api_credits_reported_total"], 40)
        self.assertEqual(summary["stop_reason"], "api_credit_threshold_reached")
        self.assertEqual(summary["status"], "stopped")

    def test_one_call_can_cross_threshold_but_no_following_call(self):
        code, calls, summary = self.run_case([15] * 5)
        self.assertEqual(len(calls), 3)
        self.assertEqual(summary["api_credits_reported_total"], 45)
        self.assertEqual(summary["stop_reason"], "api_credit_threshold_reached")

    def test_unknown_credit_stops_even_when_answer_successful(self):
        code, calls, summary = self.run_case(["未提供"] * 5)
        self.assertEqual(code, 3)
        self.assertEqual(len(calls), 1)
        self.assertFalse(summary["credits_complete"])
        self.assertEqual(summary["stop_reason"], "api_credit_not_provided")

    def test_failure_retains_reported_cost_and_stops(self):
        code, calls, summary = self.run_case([12] * 5, status="failed")
        self.assertEqual(code, 2)
        self.assertEqual(len(calls), 1)
        self.assertEqual(summary["api_credits_reported_total"], 12)
        self.assertEqual(summary["completed_json_count"], 0)

    def test_five_successes_keep_api_and_account_observations_separate(self):
        code, calls, summary = self.run_case([6] * 5)
        self.assertEqual(code, 0)
        self.assertEqual(len(calls), 5)
        self.assertEqual(summary["api_credits_reported_total"], 30)
        self.assertEqual(summary["account_balance_delta"], 10)
        self.assertEqual(summary["workbuddy_task_credits"], "未提供")
        self.assertEqual(summary["status"], "complete")

    def test_preview_does_not_load_key_or_create_results(self):
        code, calls, _ = self.run_case([6] * 5, dry_run=True)
        self.assertEqual(code, 0)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
