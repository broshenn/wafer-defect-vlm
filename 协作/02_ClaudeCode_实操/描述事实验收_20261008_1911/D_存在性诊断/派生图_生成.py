#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""D 段诊断用的**确定性派生图**生成（纯 CPU，不占 GPU）。

依据 `任务书_20261008_ClaudeCode_描述事实验收与收口.md` D 节：

  1. 原图       —— gold = 最终 RGB 图中是否存在**至少一个纯红像素**
  2. 去红       —— 所有红色变绿、黑背景不变          → gold = false
  3. 全有效区变红 —— 原绿/红有效像素全部变红、黑背景不变 → gold = true
  4. 无图       —— 不给图片、删除 `<image>` 占位（**不生成本文件**，运行期处理）

颜色定义沿用 AGENTS.md §5.2：0=黑[0,0,0] 在外、1=绿[0,255,0] 良品、
2=红[255,0,0] 失效。派生图**保留原 parent/lot**，另算新 sha256。

用法：python 派生图_生成.py <原图目录> <输出目录>
"""
from __future__ import annotations
import hashlib, io, json, sys
from pathlib import Path

import numpy as np
from PIL import Image

BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)


def classify(a: np.ndarray) -> np.ndarray:
    """把 HxWx3 uint8 图映射为 0/1/2 的类别图。未识别的颜色抛错（不许猜）。"""
    h, w, _ = a.shape
    out = np.full((h, w), -1, dtype=np.int8)
    for idx, col in enumerate((BLACK, GREEN, RED)):
        m = np.all(a == np.array(col, dtype=np.uint8), axis=-1)
        out[m] = idx
    bad = int((out < 0).sum())
    if bad:
        raise ValueError(f"出现 {bad} 个非标准颜色像素，拒绝静默处理")
    return out


def paint(cls: np.ndarray) -> np.ndarray:
    lut = np.array([BLACK, GREEN, RED], dtype=np.uint8)
    return lut[cls]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with io.open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dev18 = json.loads(io.open(sys.argv[3], encoding="utf-8").read()) if len(sys.argv) > 3 else None
    ids = dev18["sample_ids"] if dev18 else sorted(p.stem for p in src.glob("*.png"))
    dst.mkdir(parents=True, exist_ok=True)

    manifest = []
    for sid in ids:
        p = src / f"{sid}.png"
        a = np.array(Image.open(p).convert("RGB"))
        cls = classify(a)
        n_red = int((cls == 2).sum())
        n_valid = int((cls != 0).sum())

        rec = {"sample_id": sid, "parent": sid, "原图": str(p), "原图sha256": sha256_file(p),
               "红像素数": n_red, "有效像素数": n_valid,
               "gold_原图": bool(n_red > 0), "gold_去红": False, "gold_全红": True}

        # 派生 2：去红 —— 红 → 绿，黑不变
        c2 = cls.copy()
        c2[c2 == 2] = 1
        p2 = dst / f"{sid}__dered.png"
        Image.fromarray(paint(c2)).save(p2, format="PNG", optimize=False)
        rec["去红"] = str(p2)
        rec["去红sha256"] = sha256_file(p2)
        rec["去红_红像素数"] = int((c2 == 2).sum())

        # 派生 3：全有效区变红 —— 原绿或红 → 红，黑不变
        c3 = cls.copy()
        c3[c3 != 0] = 2
        p3 = dst / f"{sid}__allred.png"
        Image.fromarray(paint(c3)).save(p3, format="PNG", optimize=False)
        rec["全红"] = str(p3)
        rec["全红sha256"] = sha256_file(p3)
        rec["全红_红像素数"] = int((c3 == 2).sum())

        manifest.append(rec)
        print(f"{sid}  红{n_red:>6}  有效{n_valid:>6}  "
              f"去红后红{rec['去红_红像素数']}  全红后红{rec['全红_红像素数']}", flush=True)

    mp = dst / "派生清单.json"
    io.open(mp, "w", encoding="utf-8", newline="\n").write(
        json.dumps({"说明": "D 段存在性诊断的确定性派生图；黑背景不变，只动红/绿",
                    "条数": len(manifest), "样本": manifest},
                   ensure_ascii=False, indent=2) + "\n")
    print(f"\n写出 {mp}")

    # 自检：gold 与像素计数必须自洽
    assert all(r["去红_红像素数"] == 0 for r in manifest), "去红后仍有红像素"
    assert all(r["全红_红像素数"] == r["有效像素数"] for r in manifest), "全红后红像素数应等于有效像素数"
    print("自检通过：去红后红像素恒为 0；全红后红像素 == 有效像素数")
    return 0


if __name__ == "__main__":
    sys.exit(main())
