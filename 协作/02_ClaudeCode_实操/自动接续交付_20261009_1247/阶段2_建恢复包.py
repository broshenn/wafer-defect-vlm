#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CPU 交付协作者 · 阶段 2：生成**公开数据恢复包**（只读源，只写本目录）。

允许打包（均为公开 WM811K 派生 + 本项目生成字段，已有公开授权）：
  · 5904 张公开 PNG
  · 5904 / 1200 本地相对路径清单
  · manifest 三件（manifest.jsonl / manifest_summary.json / source_stats.json）
  · 训练 JSONL 必要副本（L_N3072 / D_N3072 / 训练数据_N3072.json）
  · 1200 候选描述分片 + 索引 + 冻结与分账.json（**候选，非可靠 gold**）

**坚决不打包**（红线）：私有 SSH 配置 / .env / API Key / CLI 日志 / 环境 / 权重等大文件。
若命中疑似敏感文件名，**中止并列出**，不静默跳过。

输出：公开数据恢复包/晶圆图公开数据恢复包_20261009.zip
      封包SHA256清册.json / .md
"""
from __future__ import annotations
import hashlib, io, json, sys, zipfile
from pathlib import Path

D = Path(__file__).resolve().parent
ROOT = D.parents[2]
OUT = D / "公开数据恢复包"
ZIP = OUT / "晶圆图公开数据恢复包_20261009.zip"

FORBID = (".env", "id_rsa", "id_ed25519", "known_hosts", ".pem", ".key",
          "credentials", "token", ".safetensors", ".bin", ".pt", ".ckpt")

PNG = ROOT / "data/images"
N3 = ROOT / "协作/02_ClaudeCode_实操/夜间扩容_20261008_0244/数据"
G12 = ROOT / "协作/02_ClaudeCode_实操/夜间扩容_20261008_0244/客户端盲包/WorkBuddy_训练缺口1200"
FREEZE = ROOT / "协作/02_ClaudeCode_实操/描述事实验收_20261008_1911/收口_v2/WorkBuddy冻结/冻结与分账.json"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


README = """# 公开数据恢复包 · 恢复说明

**生成**：2026-10-09 · CPU 交付整理协作者
**数据来源**：公开数据集 **WM811K / LSWMD**（`LSWMD.pkl`，811,457 行、有标注 172,950 行）
**本项目未使用任何公司内部数据。**

---

## 一、解压后的目录结构

```
晶圆图公开数据恢复包_20261009/
├── README_RESTORE.md          ← 本文件
├── 哈希清单.json               ← 逐文件 sha256 + 字节数
├── 清单/                       ← 索引（本地相对路径已就位）
│   ├── 清单_5904_本地相对路径.jsonl
│   ├── 清单_1200_本地相对路径.jsonl
│   ├── manifest.jsonl          （5904 行，原始记录）
│   ├── manifest_summary.json
│   └── source_stats.json
├── 图片/images/                ← 5904 张 PNG
├── 训练数据/
│   ├── L_N3072.jsonl           （3072 行，只监督类别）
│   ├── D_N3072.jsonl           （3072 行，监督类别+形态+中文描述）
│   └── 训练数据_N3072.json      （元信息与 sha256 登记）
└── 候选描述1200/               ← **候选，不是可靠 gold，未入训**
    ├── WB_gap1200_分片NNN_共N图.jsonl  （172 份）
    ├── WB_gap1200_索引.jsonl
    └── 冻结与分账.json
```

## 二、恢复顺序（照做即可用）

1. **解压**到任意目录，例如 `D:/wafer_public/`。
2. **校验**：对 `哈希清单.json` 里每个 `relative_path` 重算 sha256 并比对。
   任何一个不符都说明传输损坏，**不要**继续。
   ```bash
   python - <<'PY'
   import json,hashlib,pathlib
   base=pathlib.Path(".")  # 指向解压根
   man=json.load(open("哈希清单.json",encoding="utf-8"))
   bad=[r["relative_path"] for r in man["文件"]
        if hashlib.sha256((base/r["relative_path"]).read_bytes()).hexdigest()!=r["sha256"]]
   print("不符：", bad or "无")
   PY
   ```
3. **图片路径**：清单里的 `local_relative_path` 形如 `data/images/<sample_id>.png`。
   若要还原成原始目录名，把 `图片/images/` 整个复制到 `<解压根>/data/images/` 即可，
   此时 `清单_5904_本地相对路径.jsonl` 可直接使用。
4. **训练数据**：`L_N3072.jsonl` / `D_N3072.jsonl` 是 ms-swift 格式（`messages` + `images` 字段）。
   其中 `images` 字段是**历史绝对路径**（如 `/WS/datasets/night_images/...`），
   **需替换成本机路径**后再喂给训练框架 —— 原始绝对路径**原样保留**，未改写。
5. **候选描述 1200**：只作素材。**不要直接当训练 gold**（见第四节）。

## 三、需要另行下载 / 准备的外部依赖

| 依赖 | 用途 | 获取方式 |
|---|---|---|
| `LSWMD.pkl`（WM811K 原始包） | 重新渲染/扩充 BIN 图 | 公开数据集页面自取；**本包不含**（体积大且非必需） |
| `ms-swift` | LoRA SFT 训练框架 | pip / 官方仓库 |
| `transformers` + `torch`（CUDA） | 训练与推理 | pip；GPU 需自备 |
| `Qwen3.5-9B` 基座权重 | 复现 L / D 微调 | 模型主页自取；**本包不含权重** |
| 4 个商用模型的 API Key | 复现百炼对照 | 各平台自取；**本包不含任何 Key** |

> 本包**刻意不含**：SSH 配置、`.env`、API Key、CLI 日志、虚拟环境、模型权重。

## 四、这批数据的**质量状态**（必须一起读）

| 集合 | 数量 | 性质 | 能否当 gold |
|---|---:|---|---|
| 5904 公开 PNG | 5904 | 公开数据集抽样子集，标签 100% `ground_truth` | — |
| 3072 训练池 | 3072 | **实际训练用**；来自旧 train 划分（4292）的分层抽样 | 训练用 |
| 候选描述 1200 | 1200 | GLM 子代理**模型标注**，非人工 | **不能** |

1200 的机械分账（`冻结与分账.json`）：
**A 严格七字段机械合格 556** / **B 可解析但不合 schema 294** / **C 纯文本或非唯一对象 350**。
→ **「保存完整」≠「全部可入训」**；三档都**未获放行**入训或回填 3072。

### 已知的内容级重复

- **5904 全库**：唯一图片 sha256 **5875**，**28 组内容重复**（同 sha、不同 `sample_id`）。
  即 **5904 张图里只有 5875 张是像素唯一**。核对用图时请注意这一点。
- **1200 候选**：唯一图片 sha **1198**，**2 组**同图不同 ID（`wbg0003/0705`、`wbg0011/0712`）。
  处置为**不合并**：像素相同但 `item_id`/`sample_id` 不同，**两个 ID 都保留**。

### ⚠ 训练池与评测集之间也有 1 组同内容图（**本包内含，必须知道**）

- 评测侧 `wafer_00040406_017` 与 **训练池 3072 内**的 `wafer_00000127_002`
  **PNG 逐字节相同**（sha256 相等，两侧各自从本包内 PNG 重算确认）。
- 两者的 `sample_id` 与 lot **不同** —— 即 **ID 隔离 / lot 隔离成立，但内容隔离不成立**。
- 因此 **训练 3072 个 ID 只对应 3060 个独特 PNG**。
- **含义**：用本包训练后，若在那一组图（36 图/102 图开发集）上报成绩，
  **不能声明「评测图像内容从未见过」**；剔除该图后的 35 图敏感性见交付目录
  `阶段3_敏感性复算.md`。**本包不因此改数据**（不去重、不换 ID）。
- 全库层面仍**未测**的是：这些重复图之间的**语义关系**（是否同一物理晶圆的不同次成像）。

## 五、明确**未**包含的能力

- **旧 train 划分 4292 的原始文件不在本包内**：该文件在本机与两台学校服务器均无副本，
  恢复目标为租用实例，本阶段**未做 SSH**，故只能引用登记值（行数 12876、4292 晶圆、
  sha256 `8dfb6b0f…`），**未能独立复算**。
- 本包**不含**任何模型权重、LoRA adapter、推理服务代码与运行日志。
- **LoRA adapter 由另包提供，不由本数据包承载**：`L-N3072-3407` 与 `D-N3072-3407`
  两个最终 adapter（`adapter_model.safetensors` 各 173,188,512 B + `adapter_config.json`）
  已由**另一路只读小备份任务**在 2026-10-09 13:38 取回（4/4 文件，sha256 逐位复核通过，
  合计 330.33 MiB），存放于**仓库外的本机备份目录**；
  **备份状态：已完成**，但**不随本 ZIP 分发**（体积与权重红线）。
  → 复现 L/D 需要「本包 + 另包的 adapter + 自取的 Qwen3.5-9B 基座权重」三者。
- 本包**不含**评测/对照的原始回答；那些在主执行会话目录内，另行引用。
"""


def main() -> int:
    OUT.mkdir(exist_ok=True)
    files: list[tuple[Path, str]] = []   # (源路径, 包内相对路径)

    # 图片
    pngs = sorted(PNG.glob("*.png"))
    for p in pngs:
        files.append((p, f"图片/images/{p.name}"))

    # 清单与 manifest
    for src, dst in [
        (ROOT / "data/manifest.jsonl", "清单/manifest.jsonl"),
        (ROOT / "data/manifest_summary.json", "清单/manifest_summary.json"),
        (ROOT / "data/source_stats.json", "清单/source_stats.json"),
        (D / "清单_5904_本地相对路径.jsonl", "清单/清单_5904_本地相对路径.jsonl"),
        (D / "清单_1200_本地相对路径.jsonl", "清单/清单_1200_本地相对路径.jsonl"),
    ]:
        if src.exists():
            files.append((src, dst))

    # 训练数据
    for name in ("L_N3072.jsonl", "D_N3072.jsonl", "训练数据_N3072.json"):
        src = N3 / name
        if src.exists():
            files.append((src, f"训练数据/{name}"))

    # 候选描述 1200
    for p in sorted(G12.glob("*.jsonl")):
        files.append((p, f"候选描述1200/{p.name}"))
    if FREEZE.exists():
        files.append((FREEZE, "候选描述1200/冻结与分账.json"))

    # ---- 红线自检：先扫文件名，命中即中止 ----
    hits = [(str(s), d) for s, d in files if any(f in s.name.lower() for f in FORBID)]
    if hits:
        print("!! 命中疑似敏感文件名，中止打包：", hits, file=sys.stderr)
        return 2

    # 先算清单（避免 zip 里出现同名重复条目：占位 + 真清单两份）
    entries = [{"relative_path": "README_RESTORE.md",
                "bytes": len(README.encode("utf-8")),
                "sha256": hashlib.sha256(README.encode("utf-8")).hexdigest()}]
    for src, dst in files:
        entries.append({"relative_path": dst, "bytes": src.stat().st_size,
                        "sha256": sha(src)})
    manifest = {
        "生成时间": "2026-10-09",
        "生成者": "CPU 交付整理协作者",
        "包内根目录": "晶圆图公开数据恢复包_20261009/",
        "文件数": len(entries),
        "总字节(未压缩)": sum(e["bytes"] for e in entries),
        "说明": "相对路径相对于**包内根目录**；zip 自身的 sha256 见《封包SHA256清册.json》",
        "文件": entries,
    }
    man_txt = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"

    ROOT_IN = "晶圆图公开数据恢复包_20261009/"
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.writestr(ROOT_IN + "README_RESTORE.md", README)
        z.writestr(ROOT_IN + "哈希清单.json", man_txt)
        for src, dst in files:
            z.write(src, ROOT_IN + dst)
    (OUT / "哈希清单.json").write_text(man_txt, encoding="utf-8")

    # ---- 封包 sha256 清册 ----
    seal = {
        "封包文件": ZIP.name,
        "字节": ZIP.stat().st_size,
        "sha256": sha(ZIP),
        "包内文件数": len(entries),
        "包内未压缩总字节": manifest["总字节(未压缩)"],
        "压缩率": round(ZIP.stat().st_size / manifest["总字节(未压缩)"], 4),
        "红线自检": {
            "禁止打包项命中数": len(hits),
            "禁止打包模式": list(FORBID),
            "结论": "无命中" if not hits else "有命中，已中止",
        },
        "不含": ["SSH 配置", ".env", "API Key", "CLI 日志", "虚拟环境", "模型权重"],
    }
    (D / "封包SHA256清册.json").write_text(
        json.dumps(seal, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (D / "封包SHA256清册.md").write_text(
        "# 封包 SHA256 清册\n\n"
        f"- 文件：`{ZIP.name}`\n"
        f"- 大小：**{ZIP.stat().st_size/1048576:.2f} MiB**"
        f"（未压缩 {manifest['总字节(未压缩)']/1048576:.2f} MiB，"
        f"压缩率 {seal['压缩率']}）\n"
        f"- **sha256**：`{seal['sha256']}`\n"
        f"- 包内文件数：**{len(entries)}**\n"
        f"- 红线自检：禁止打包项命中 **{len(hits)}** → {seal['红线自检']['结论']}\n"
        f"- 不含：{'、'.join(seal['不含'])}\n"
        f"\n> 逐文件 sha256 见包内 `哈希清单.json`。\n",
        encoding="utf-8")
    sys.stdout.buffer.write((json.dumps(seal, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
