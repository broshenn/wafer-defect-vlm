"""仅CPU工程检查；不是GPU服务或视觉内容验收。"""
import io
import json
import os
import unittest
from pathlib import Path
from PIL import Image
from wafer_service import adapt_answer, load_checker, validate_image

ROOT = Path(__file__).resolve().parents[3]


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checker = load_checker(Path(os.environ.get("WAFER_CHECKER", str(ROOT / "协作/01_Codex_指挥/WorkBuddy外部102_执行补包_20261008/schema_check.py"))))
        cls.obj = {"defect_class": "unknown", "morphology": "合成文本", "radial_zone": "unknown",
                   "clock_direction": None, "extent_r": None, "caption_zh": "合成文本", "uncertainty": ""}

    def image(self, color, size=(448, 448)):
        data = io.BytesIO(); Image.new("RGB", size, color).save(data, format="PNG"); return data.getvalue()

    def test_palette_and_presence(self):
        self.assertTrue(validate_image(self.image((255, 0, 0)))["red_present"])
        self.assertFalse(validate_image(self.image((0, 255, 0)))["red_present"])

    def test_reject_invalid_input(self):
        for data in (b"", b"notpng", self.image((0, 0, 0)), self.image((0, 0, 255)), self.image((255, 0, 0), (224, 224))):
            with self.subTest(size=len(data)), self.assertRaises(ValueError): validate_image(data)

    def test_serialization_keeps_raw(self):
        raw = json.dumps({**self.obj, "clock_direction": "null"}, ensure_ascii=False)
        out = adapt_answer(raw, self.checker)
        self.assertEqual(out["raw_answer"], raw)
        self.assertIsNone(out["result"]["clock_direction"])
        self.assertEqual(len(out["format_changes"]), 1)

    def test_bad_structure_not_repaired(self):
        raw = json.dumps({**self.obj, "clock_direction": 12}, ensure_ascii=False)
        out = adapt_answer(raw, self.checker)
        self.assertIsNone(out["result"]); self.assertTrue(out["needs_review"])

    def test_unsupported_claim_remains_auditable(self):
        raw = json.dumps({**self.obj, "caption_zh": "红色缺陷宽度1.2R"}, ensure_ascii=False)
        out = adapt_answer(raw, self.checker)
        self.assertTrue(out["needs_review"]); self.assertEqual(out["raw_answer"], raw)

    def test_duplicate_key_rejected(self):
        raw = json.dumps(self.obj, ensure_ascii=False)[:-1] + ',"defect_class":"Loc"}'
        self.assertIsNone(adapt_answer(raw, self.checker)["result"])


if __name__ == "__main__": unittest.main()
