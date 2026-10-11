"""从冻结36图及已有原答另存审阅副本；不重新推理、不改变主分。"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SOURCE = REPO / "协作/02_ClaudeCode_实操/描述事实验收_20261008_1911"
RESULTS = REPO / "协作/02_ClaudeCode_实操/夜间扩容_20261008_0244/结果"
FIELDS = ("morphology", "radial_zone", "clock_direction", "caption_zh")
EMPTY_THINK = re.compile(r"^\s*<think\b[^>]*>\s*</think\s*>\s*", re.S)
FENCE = re.compile(r"^```(?:json)?\s*\n(.*?)\n?\s*```$", re.S | re.I)


def no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_key")
        result[key] = value
    return result


def nonfinite(value):
    raise ValueError("non_finite")


def read_object(text):
    if not isinstance(text, str):
        return None
    try:
        value = json.loads(text, object_pairs_hook=no_duplicates, parse_constant=nonfinite)
        return value if isinstance(value, dict) else None
    except (ValueError, TypeError):
        return None


def derive(raw):
    # 与原口径一致，只剥空think；完整围栏只用于展示副口径。
    text = EMPTY_THINK.sub("", raw.strip())
    obj = read_object(text)
    if obj is not None:
        return obj, "原严格可解析"
    match = FENCE.fullmatch(text)
    if match:
        obj = read_object(match.group(1))
        if obj is not None:
            return obj, "原格式失败；仅展示时剥完整围栏"
    return None, "原格式失败；显示完整原文"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    manifest_path = SOURCE / "C_描述事实审阅/样本清单.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    models = manifest["模型"]
    raw_by_model = {}
    inputs = {str(manifest_path.relative_to(REPO)): hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
    for model in models:
        path = RESULTS / f"{model}_7f_raw.jsonl"
        rows = [json.loads(line, object_pairs_hook=no_duplicates) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(rows) == 102 and len({row["sample_id"] for row in rows}) == 102
        raw_by_model[model] = {row["sample_id"]: row for row in rows}
        inputs[str(path.relative_to(REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()
    blocks, counts = [], {}
    for sample in manifest["样本"]:
        sid = sample["sample_id"]
        image = SOURCE / "C_描述事实审阅/图" / f"{sid}.png"
        assert hashlib.sha256(image.read_bytes()).hexdigest() == sample["图sha256"]
        answers = []
        for blind, model in sample["盲号对照"].items():
            raw = raw_by_model[model][sid]["raw"]
            assert isinstance(raw, str)
            obj, mode = derive(raw)
            counts[mode] = counts.get(mode, 0) + 1
            answers.append({"blind": blind, "display_mode": mode,
                            "fields": {key: obj[key] for key in FIELDS if obj is not None and key in obj},
                            "raw_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                            "raw": raw})
        blocks.append({"number": sample["图号"], "sample_id": sid,
                       "image_sha256": sample["图sha256"], "answers": answers})
    assert len(blocks) == 36 and sum(len(block["answers"]) for block in blocks) == 108
    evidence = {"purpose": "匿名开发审阅副本，非人工gold；原分数及原答不变", "inputs": inputs,
                "images": 36, "answer_sets": 108, "display_counts": counts,
                "raw_nonempty": sum(bool(a["raw"].strip()) for b in blocks for a in b["answers"])}
    if not args.write:
        print(json.dumps(evidence, ensure_ascii=False, indent=2))
        return
    out = Path(__file__).parent / "审阅副本_v2"
    if out.exists():
        raise SystemExit("输出已存在，拒绝覆盖")
    out.mkdir()
    (out / "图").mkdir()
    (out / "raw").mkdir()
    lines = ["<!doctype html><html lang='zh'><meta charset='utf-8'><title>匿名描述事实审阅 v2</title>",
             "<style>body{font-family:system-ui;max-width:1300px;margin:24px auto;line-height:1.5}section{border-top:2px solid #aaa;padding:16px}img{width:300px;image-rendering:pixelated}.answers{display:flex;gap:12px}.answer{flex:1;background:#f4f4f4;padding:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere}.status{font-size:13px;color:#73521b}</style>",
             "<h1>36图匿名描述事实审阅</h1><p>开发审阅，非专家金标。逐图甲乙丙随机；原格式失败与内容分别评价。原答保留且主分不变。真人复核栏留空。</p>"]
    for block in blocks:
        sid = block["sample_id"]
        shutil.copyfile(SOURCE / "C_描述事实审阅/图" / f"{sid}.png", out / "图" / f"{sid}.png")
        lines.append(f"<section><h2>图{block['number']} {sid}</h2><img src='图/{sid}.png'><div class='answers'>")
        for answer in block["answers"]:
            blind = answer["blind"]
            (out / "raw" / f"{sid}_{blind}.txt").write_text(answer["raw"], encoding="utf-8", newline="")
            lines.append(f"<div class='answer'><h3>{blind}</h3><div class='status'>{html.escape(answer['display_mode'])}</div>")
            if answer["fields"]:
                for field in FIELDS:
                    value = answer["fields"].get(field, "[缺字段]")
                    display = "JSON null" if value is None else str(value)
                    lines.append(f"<b>{field}</b><pre>{html.escape(display)}</pre>")
            else:
                lines.append(f"<pre>{html.escape(answer['raw'])}</pre>")
            lines.append("</div>")
        lines.append("</div></section>")
    lines.append("</html>")
    (out / "审阅页.html").write_text("\n".join(lines), encoding="utf-8")
    # 无公开标签或模型映射的机器可读审阅材料；原答案中的预测不是公开gold。
    (out / "匿名审阅材料.json").write_text(json.dumps(blocks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (Path(__file__).parent / "审阅副本核验.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
