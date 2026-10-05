# 给 Codex：GLM 训练标注的数据准备 · 完成

**执行**：Claude Code，2026-10-02 19:12–19:15
**任务书**：`协作/01_Codex_指挥/任务书_20261002_ClaudeCode_GLM标注准备.md`
**本目录**：`D:\pycode\晶圆图研究\协作\02_ClaudeCode_实操\GLM训练标注准备_20261002\`
（该目录此前**不存在**，未覆盖任何旧文件）

> **一句话**：清单与校验全部完成，**独立检查 69/69 通过**；
> **真实标注一次未执行**，未生成任何 GLM 答案，未启动标注子 agent。

---

## 1. 输入（全部实测核对，逐位吻合）

| 输入 | SHA256 | 核对 |
|---|---|---|
| `学校训练准备_20261002-025848/train_180.jsonl` | `04febd99674f70bf4619c3b529fe4c1313a5b8d739e89c69c390fe686324db47` | ✅ |
| `…/smoke_20.jsonl` | `f59ae6ba5ba917fae410a9a82f6747c886abde20df189878093467242dd98eec` | ✅ |
| `…/dev_18.jsonl` | `08b267ea73def5b3119dbf6ca73d27630ef565bb4011f6e53b404d753f7972ac` | ✅ |
| `协作/01_Codex_指挥/prompt_image_only_v2_20261002.txt` | `7df73031ee1a30dfb018dbb2417f505ea9ff139280307bce2934de604119954a` | ✅ |
| `协作/02_ClaudeCode_实操/口径修正与准备_20261002-021207/check_answer_v3.py` | `794a254e005eaa232362c539817abde98a1c2406ecb52574b9e9edb35303ec55` | ✅ |

**准备器在开工前核对全部五项指纹，任一不符即 `exit 3` 停止，不静默修原数据。**

---

## 2. 分批与校验结果

**规则（照任务书 §1）**：首批 = `smoke_20` **原顺序**（20 条）；其余 160 = `train_180`
**原顺序**剔除首批 ID 后按 20 切 8 批；全局 `item_id` 按此顺序连续编号 `item_001…item_180`。
**未重新选样、未按教师表现调整名单、未按类别排序。**

| 检查 | 结果 |
|---|---|
| 九批 × 20 = 180 | ✅ |
| 九批并集 == 全量（**顺序也一致**） | ✅ |
| 九批两两不重叠 | ✅ |
| 四份子清单各 5 条、两两不重叠、并集**精确等于**首批 20 | ✅ |
| `item_id` 唯一且连续 `item_001…item_180` | ✅ |
| `sample_id` 唯一 | ✅ |
| 180 张图**全部存在**、SHA256 与清单一致、全局指纹不重复 | ✅ |
| **lot 隔离**：train∩val / train∩test / val∩test / train∩benchmark / val∩benchmark | **全 0** |
| manifest 对应：标签 / lot / split / `ground_truth` 来源 / 行号 | ✅ 无问题 |
| 盲清单每行**仅四个字段**，无任何答案性字段 | ✅ |
| `zcode_batch_01_inputs.json` 内路径与 hash **实测复核** | ✅ |

**独立检查器 `verify_glm_preparation.py` 结果：通过 69/69，全部通过。**
它**不 import 准备器** —— 字段、并集、指纹、隔离都是另写一份实现重算的。

---

## 3. 交付物与指纹

**首批（本次要执行的那批）**

| 文件 | SHA256 |
|---|---|
| `batch_01_blind.jsonl` | `0c2f476d3f099eccee62e3c927b3621757d69bced95a3421bab0db4d9c681db2` |
| `batch_01_worker_01.jsonl` | `c9a5603ee5b2feb446b7d23bb8d9721eaabe3dbfbe3de855b986a593bb257f2e` |
| `batch_01_worker_02.jsonl` | `543600f8e2e3c19229c79b0194d817ac652f54cd6d1972aa66451b60f5fb0c1e` |
| `batch_01_worker_03.jsonl` | `b41c97590325f3352f427e146500abcb52a1289d61fbc59970aa5b08cb6f8b27` |
| `batch_01_worker_04.jsonl` | `126f2e35063a5c3a85c9434c0ab2c17da87ebfa3e7e5853d88379e4bbc7085f1` |
| **`zcode_batch_01_inputs.json`** | **`641f75212839e5c847edf0d7bb53225460b03f99f4e16c10a1343f47cd97ffe7`** |

**其余批次（本次只预备，不执行）**

| 文件 | SHA256 |
|---|---|
| `blind_inputs_all_180.jsonl` | `600dbbed0b04e9b068ce7b5fc053e346a01a151bc77684e818ce8266854996af` |
| `batch_02_blind.jsonl` | `130053911f596d17bebd3154824d4dd32879ad7a41187eb66caa81ec84cc5b9f` |
| `batch_03_blind.jsonl` | `2d98bc8f7aee88a89aaec9887596affbd2f13ca43022e965a406cb8f7efbbbbf` |
| `batch_04_blind.jsonl` | `fcf1639b97ed34742d441e7b4d5e464e008e81b78df1217ceb55823fb36076f0` |
| `batch_05_blind.jsonl` | `e337161b5d4b96757c8bc716641d5fb6af03d611c6649d2adf7274163f7f2d72` |
| `batch_06_blind.jsonl` | `392f5b8e66c22ff319f48d1160b3719771095f40c8d5688a646879987ed0d5d4` |
| `batch_07_blind.jsonl` | `19c5471a572fe0becda6ddfde09eaabe8f03392e2c060aa8b6ce1e3d4bfa4e3d` |
| `batch_08_blind.jsonl` | `0299cc54d91b6e3519392045039b69b1d0630184732a51001c17cfd71308f68e` |
| `batch_09_blind.jsonl` | `732d6f0db98a77fb4b033c45926cc0ea61509f480f766aa91c2b2871e6c78539` |

**只给 Codex / Claude 的（禁止给 ZCode）**

| 文件 | SHA256 |
|---|---|
| `audit_selected.jsonl`（含标签/lot/split/行号） | `bb6b9222aa6acbf6405138bd560554701a2538286252441df0c78064284f4f54` |
| `plan.json`（含审计信息） | `beefec7644c6da372ef797eb6d671bb8d65a699a866b207ffff8d9b701af630a` |

**脚本**

| 文件 | SHA256 |
|---|---|
| `prepare_glm_annotation.py` | `dac6d7b5cb2af2805d834ca144c458cd0d93c886764193c9311e0e163ebe9e3c` |
| `verify_glm_preparation.py` | `6d49f11094817711305e0d81acbcf31197e8fd3e3fc6976ea00c1933992a20c3` |

**`zcode_batch_01_inputs.json` 内容**：`batch_id=batch_01`、`total_limit=20`、
`max_concurrency=4`、`batch_manifest`/`prompt`/`checker` 三项**真实路径 + 实测 hash**、
`workers` **四条完整记录**（各 5 条，路径与 hash 均为实测）。
**占位文字已全部替换，无任何猜值。** 索引内**不含**标签、类别数量、lot、split、几何或评价。

---

## 4. 行为验证（实测）

| 情形 | 结果 |
|---|---|
| `--dry-run`（默认） | 只算不写，打印 15 个清单文件的行数与指纹 |
| `--write` 首次 | 写出 17 个文件（15 清单 + plan + zcode 索引） |
| **重复 `--write`** | **`exit 3` 拒绝覆盖**，列出已存在的 17 个文件 |
| `--write --resume`（字节一致） | **`exit 0`**；**盲清单与子清单字节不变** |
| 独立检查 | `exit 0`，69/69 |

> **[事实] 一处需要你知道的细节**：`--resume` 会重写 `plan.json` 与
> `zcode_batch_01_inputs.json`，其中 `plan.json` 含 `prepared_at` 时间戳，
> 所以**它的 hash 每次 resume 都会变**（上面记的是当前值）。
> **盲清单、子清单、audit 的字节在 resume 前后完全不变**（已逐项核对）。
> 若你要把 `plan.json` 的 hash 当验收依据，请以**交付时的这一个**为准。

---

## 5. 未知项

| # | 未知 | 状态 |
|---|---|---|
| 1 | ZCode 实际标注质量 | **未测**（看图作答由 ZCode 完成，本轮未启动） |
| 2 | ZCode 会话内可读到的用量/额度 | 已知**读不到**（B31），仍需用户从客户端 UI 补 |
| 3 | `batch_02…09` 是否会被批准执行 | **未定**，等指挥会话 |
| 4 | 首批 20 条里类别边界难例的比例 | **不提供** —— 索引里刻意不含类别信息，也不做难度标注 |

---

## 6. 实际消耗

| 项 | 值 |
|---|---|
| 模型请求 | **0** |
| 付费调用 | **0** |
| GPU | **0** |
| 上传 | **0** |
| 生成 GLM 答案 | **0** |
| 启动标注子 agent | **0** |
| 改动已有文件 | **0**（只在新目录新建 **20** 个文件：15 个清单 + `plan.json` + `zcode_batch_01_inputs.json` + 2 个脚本 + 本回复；`ls -1 \| wc -l` 实测） |
| 改动提示词 / 图片 / 检查器 / 原标签 / 已有清单 | **0** |

---

## 7. 未做的事（避免误读）

- **真实标注未执行** —— 本轮只产出清单与索引。
- **未生成任何 GLM 答案**，也未启动标注子 agent。
- **`dev_18` 只用于隔离核对**（确认与训练集 lot 无交集），**未进入任何交给 ZCode 的清单**。
- **未改**提示词、图片、检查器、原标签与已有清单。
- **学校训练**仍由我负责，沿用既有任务与 20 步/60 步授权；本任务未在本地跑 GPU，
  也未重装环境或重下基座。跳板机离线状态未变。
