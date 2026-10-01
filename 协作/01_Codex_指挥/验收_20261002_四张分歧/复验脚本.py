"""Codex 验收用只读复跑；结果只写本目录，不执行交付脚本的写盘 main。"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DELIVERY = REPO / "协作/02_ClaudeCode_实操/核查_20261002_四张分歧"
sys.path.insert(0, str(DELIVERY))
sys.path.insert(0, str(REPO / "code/projects/wafer-defect-vlm/src"))
import wafer_png
import patterns
import check_answer_v2 as checker
from wafer_vlm.utils import calculate_static_features


def same(a, b):
    if a is None or b is None:
        return a is b
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    return bool(np.allclose(a, b, rtol=1e-9, atol=1e-12))


def main():
    target = HERE / "复验记录.json"
    if target.exists():
        raise SystemExit("记录已存在，不覆盖；另建新版目录后再运行。")
    entries = json.loads((DELIVERY / "核查结果.json").read_text(encoding="utf-8"))["items"]
    diagnostic = json.loads((DELIVERY / "图案诊断.json").read_text(encoding="utf-8"))
    manifest = {r["sample_id"]: r for r in (
        json.loads(s) for s in (REPO / "data/manifest.jsonl").read_text(encoding="utf-8").splitlines()
    )}
    result = {"author": "Codex", "date": "2026-10-02", "raw_pkl_verified": False,
              "items": [], "patterns": [], "checker": {}, "source_sha256": {}}
    for file in DELIVERY.iterdir():
        if file.is_file():
            result["source_sha256"][file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
    for e in entries:
        sid = e["sample_id"]
        row = manifest[sid]
        png = REPO / "data/images" / f"{sid}.png"
        rec = wafer_png.recover_matrix(png)
        pixels = np.asarray(Image.open(png).convert("RGB"))
        h, w = row["matrix_shape"]
        yy = np.floor((np.arange(h) + .5) * 448 / h).astype(int)
        xx = np.floor((np.arange(w) + .5) * 448 / w).astype(int)
        sampled = pixels[np.ix_(yy, xx)]
        # 独立按 manifest shape 取格中心；与交付的无标签格点拟合比较。
        matrix = ((sampled[:, :, 1] == 255).astype("uint8")
                  + 2 * (sampled[:, :, 0] == 255).astype("uint8"))
        feats = calculate_static_features(matrix)
        diff = [k for k, v in row["features"].items() if not same(v, feats.get(k))]
        result["items"].append({"item_id": e["item_id"], "sample_id": sid,
            "recovered_shape": list(rec["matrix"].shape),
            "shape_matches_manifest": list(rec["matrix"].shape) == row["matrix_shape"],
            "independent_sample_equals_recovery": bool(np.array_equal(matrix, rec["matrix"])),
            "roundtrip_pixels_differing": wafer_png.roundtrip(matrix, png)["roundtrip_pixels_differing"],
            "png_sha256_matches_delivery": rec["png_sha256"] == e["png_sha256_actual"],
            "manifest_fields_differing": diff})
    for e in diagnostic["items"]:
        matrix = wafer_png.recover_matrix(REPO / "data/images" / f"{e['sample_id']}.png")["matrix"]
        band = patterns.banding(matrix)
        radial = patterns.radial_profile(matrix)
        result["patterns"].append({"item_id": e["item_id"],
            "banding_matches_delivery": band == e["banding"],
            "radial_matches_delivery": radial == e["radial"],
            "banding": band, "radial": radial})
    for name, args in (("self_test", ["--self-test"]),
                       ("nine_raw", ["--dir", str(REPO / "协作/03_ZCode_标注/20261002-012138")])):
        proc = subprocess.run([sys.executable, "-X", "utf8", "-B", str(DELIVERY / "check_answer_v2.py"), *args],
                              capture_output=True, text=True, encoding="utf-8", timeout=15)
        result["checker"][name] = {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
    # 补测类型检查之后是否还会崩溃：只记录，绝不修改他人检查器。
    obj = json.loads(checker.GOOD_UNKNOWN)
    obj["defect_class"] = []
    try:
        result["checker"]["list_type_probe"] = checker.check_object(obj)
    except Exception as error:
        result["checker"]["list_type_probe"] = {"exception": type(error).__name__, "message": str(error)}
    # 用真实 CLI 文件验证成功、截断失败、第三张未执行。文件全在 Codex 验收目录。
    fixture = HERE / "停止验证输入"
    fixture.mkdir()
    inputs = [("a_ok_raw.json", checker.GOOD_UNKNOWN),
              ("b_truncated_raw.json", '{"defect_class":"Center"'),
              ("c_never_checked_raw.json", checker.GOOD_UNKNOWN)]
    for name, raw in inputs:
        (fixture / name).write_text(raw, encoding="utf-8")
    attempts = []
    for name, _ in inputs:
        proc = subprocess.run([sys.executable, "-X", "utf8", "-B", str(DELIVERY / "check_answer_v2.py"), str(fixture / name)],
                              capture_output=True, text=True, encoding="utf-8", timeout=15)
        attempts.append({"file": name, "exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr})
        if proc.returncode != 0:
            break
    result["checker"]["sequential_stop"] = {"attempts": attempts, "third_checked": len(attempts) == 3}
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    ok = all(i["shape_matches_manifest"] and i["independent_sample_equals_recovery"]
             and i["roundtrip_pixels_differing"] == 0 and i["png_sha256_matches_delivery"] for i in result["items"])
    ok = ok and all(i["banding_matches_delivery"] and i["radial_matches_delivery"] for i in result["patterns"])
    ok = ok and result["checker"]["self_test"]["returncode"] == 0 and result["checker"]["nine_raw"]["returncode"] == 0
    ok = ok and [a["exit_code"] for a in attempts] == [0, 2]
    print(json.dumps({"replay_pass": ok, "matrix_count": len(entries), "pattern_count": len(diagnostic["items"]),
                      "list_type_probe": result["checker"]["list_type_probe"], "evidence": str(target)}, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
