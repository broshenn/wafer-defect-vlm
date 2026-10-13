#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统一 TEST200 打分：构建 contract.json 与冻结输入副本（本地，一次性）。

产出（全部落在本目录）：
  inputs/TEST200_ids.txt        200 个冻结 ID（TEST200.txt 的副本）
  inputs/reference200.jsonl     冻结考卷（本地 test200_main.jsonl 的副本）
  inputs/images200.jsonl        逐图 manifest：sample_id / image_sha256 / image_path
  inputs/prompt_main_user.txt   主列题面 = "<image>\\n" + prompt_v5.txt 的完整用户消息
  inputs/prompt_L_user.txt      类别列题面 = L 的用户消息
  inputs/common_v5.py           冻结解析器
  inputs/server_test200_main.jsonl  服务器实际消费的考卷（图片路径已改写）
  contract.json                 带 SHA 钉死的合同

纪律：只复制与哈希，不修改任何原始文件；已存在且字节相同的产出不重写。
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
V5 = REPO / "协作/02_ClaudeCode_实操/v5_感知训练_20261012"
A = V5 / "A_数据"
CODE = V5 / "B_代码"
INPUTS = HERE / "inputs"


def sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sha_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def canonical_id_sha(ids) -> str:
    return sha_text("".join(x + "\n" for x in sorted(set(ids))))


def copy_once(src: Path, dst: Path) -> None:
    if dst.exists() and dst.read_bytes() == src.read_bytes():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def main() -> int:
    INPUTS.mkdir(exist_ok=True)

    # 1) 冻结 ID
    ids_src = A / "TEST200.txt"
    ids_dst = INPUTS / "TEST200_ids.txt"
    copy_once(ids_src, ids_dst)
    ids = [l.strip() for l in ids_dst.read_text(encoding="utf-8-sig").splitlines() if l.strip()]
    assert len(ids) == 200 == len(set(ids)), f"ID 数异常 {len(ids)}"

    # 2) 冻结考卷（本地版）
    ref_src = A / "test200_main.jsonl"
    ref_dst = INPUTS / "reference200.jsonl"
    copy_once(ref_src, ref_dst)

    # 3) 逐图 manifest（本地 PNG 绝对路径 + SHA）
    img_rows = []
    for sid in ids:
        png = REPO / "data" / "images" / f"{sid}.png"
        assert png.is_file(), f"缺图 {png}"
        img_rows.append({"sample_id": sid, "image_sha256": sha_file(png),
                         "image_path": str(png.resolve())})
    img_dst = INPUTS / "images200.jsonl"
    text = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in img_rows)
    if not (img_dst.exists() and img_dst.read_text(encoding="utf-8") == text):
        img_dst.write_text(text, encoding="utf-8", newline="\n")

    # 4) 题面（打分器要的是**完整用户消息**，含 <image> 前缀）
    ref_rows = [json.loads(l) for l in ref_dst.open(encoding="utf-8") if l.strip()]
    main_user = ref_rows[0]["messages"][0]["content"]
    prompt_src = (A / "prompt_v5.txt").read_text(encoding="utf-8")
    assert main_user == "<image>\n" + prompt_src, "主列题面与 prompt_v5.txt 不一致"
    pm = INPUTS / "prompt_main_user.txt"
    if not (pm.exists() and pm.read_text(encoding="utf-8") == main_user):
        pm.write_text(main_user, encoding="utf-8", newline="")
    L_rows = [json.loads(l) for l in (A / "test_L.jsonl").open(encoding="utf-8") if l.strip()]
    L_user = L_rows[0]["messages"][0]["content"]
    assert len({json.dumps(r["messages"][0]["content"]) for r in L_rows}) == 1, "L 题面不唯一"
    pl = INPUTS / "prompt_L_user.txt"
    if not (pl.exists() and pl.read_text(encoding="utf-8") == L_user):
        pl.write_text(L_user, encoding="utf-8", newline="")

    # 5) 解析器
    copy_once(CODE / "common_v5.py", INPUTS / "common_v5.py")

    # 6) 服务器实际消费的考卷（图片路径已改写）
    srv = INPUTS / "server_test200_main.jsonl"
    if not srv.exists():
        print(f"!! 缺 {srv}（需从服务器拉取 server 版考卷后再跑本脚本）")

    # 7) 分母（从冻结参考直接算）
    import importlib.util
    spec = importlib.util.spec_from_file_location("common_v5_pin", INPUTS / "common_v5.py")
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    den = {"coverage": 0, "zones": 0, "angular_type": 0}
    for r in ref_rows:
        ref = common.parse_raw_strict(r["reference_json"])
        den["coverage"] += ref["coverage_band"] != "unknown"
        den["zones"] += ref["red_mass_zones"] != "unknown"
        den["angular_type"] += ref["angular_type"] != "unknown"

    contract = {
        "n_expected": 200,
        "ids": {"path": "inputs/TEST200_ids.txt", "sha256": sha_file(ids_dst),
                "canonical_sha256": canonical_id_sha(ids)},
        "reference": {"path": "inputs/reference200.jsonl", "sha256": sha_file(ref_dst)},
        "images": {"path": "inputs/images200.jsonl", "sha256": sha_file(img_dst)},
        "prompts": {
            "main": {"path": "inputs/prompt_main_user.txt", "sha256": sha_file(pm)},
            "L": {"path": "inputs/prompt_L_user.txt", "sha256": sha_file(pl)},
        },
        "parser": {"path": "inputs/common_v5.py", "sha256": sha_file(INPUTS / "common_v5.py")},
        "geometry_denominators": den,
        "note": "统一 TEST200 打分合同；主列题面 = <image>\\n + prompt_v5.txt；类别列 = L 的用户消息；"
                "服务器消费版另存 inputs/server_test200_main.jsonl（仅 images 路径不同，其余逐字段一致）",
    }
    cp = HERE / "contract.json"
    text = json.dumps(contract, ensure_ascii=False, indent=2) + "\n"
    if not (cp.exists() and cp.read_text(encoding="utf-8") == text):
        cp.write_text(text, encoding="utf-8", newline="\n")

    print(json.dumps({"ids": len(ids), "denominators": den,
                      "main_prompt_sha256": contract["prompts"]["main"]["sha256"],
                      "L_prompt_sha256": contract["prompts"]["L"]["sha256"],
                      "reference_sha256": contract["reference"]["sha256"],
                      "images_sha256": contract["images"]["sha256"],
                      "server_exam_present": srv.exists(),
                      "server_exam_sha256": sha_file(srv) if srv.exists() else None},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
