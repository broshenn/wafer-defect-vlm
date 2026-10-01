# 九类试标的离线准备：交付

**执行**：Claude Code（Fable 5），2026-10-02 01:02（北京时间）
**任务书**：`协作/01_Codex_指挥/任务书_20261002_ClaudeCode_九类试标准备.md`
**本轮模型请求**：**0 次** · **付费调用**：**0 次** · **GPU**：**0 次** · **未启动 ZCode** · **未上传服务器**

> ## ⚠️ 保密边界（先读这段）
>
> **本文件含真实标签（`audit_selected.jsonl` 的内容），禁止转交 ZCode。**
> 交给 ZCode 的**只有** `blind_inputs.jsonl`（四字段、无答案）。
>
> 项目三层读法：**[事实]** = 可核对的产物/代码/文档 · **[解释]** = 我的推理，可能错 · **[未测]** = 没有证据。

---

## 0. 一个必须先说的发现：这批图有一份「前任计划」

**[事实]** 仓库外 `D:\pycode\workbuddy\wafer-cost-test\20261001-211719\plan.json` 是 **2026-10-01 21:17 的一次九类试标**，和本次任务**几乎是同一套设计**：

| 项 | 前任计划 | 本次任务书 |
|---|---|---|
| 规模 | `"maximum_images": 9`（`plan.json:6`） | 九张 |
| 种子 | `"seed": 3407`（`plan.json:5`） | seed=3407 |
| 类别 | 九类各一张（`sampling_class_not_sent`） | 九类各一张 |
| 输入 | 仅图片 | 仅图片 |

**那一轮只跑到第 5 张就中断了**：

| 序 | 样本 | 证据 | 状态 |
|---|---|---|---|
| 1 | `wafer_00011910_009` | `call-01.json:10` `http_status=200` | ✅ 真发出去了 |
| 2 | `wafer_00017178_007` | `call-02.json` / `-retry-01` / `-geometry-01` | ✅ 三次 |
| 3 | `wafer_00012836_003` | `call-03.json:10` `http_status=200` | ✅ |
| 4 | `wafer_00015000_006` | `call-04.json:10` `http_status=200` | ✅ |
| 5 | `wafer_00025247_021` | `started-05.json:2` 存在 **但无 `call-05.json`** | ⚠️ 已发起、无记录 |
| 6–9 | Random / Near_full / Scratch / none | 只在 `plan.json` 里，**无任何 started / call 文件** | 未送出 |

**[解释]** 这说明「九类各一张 + seed 3407」这个做法会被确定性复现——我用同样的规则重跑，**抽出来的图跟前任计划高度接近**（见 §附录 C）。

**[事实]** 本次默认口径下，抽中的 9 张里有 **2 张**落在那份前任计划的名单内：

```
item_008  wafer_00017102_001   (前任: Random)
item_001  wafer_00044523_017   (前任: none)
```

**[解释]** 这两张**没有任何发送证据**（没有 `started-*`、没有 `call-*`），按任务书「**实际试标过**」的字面口径**不算已看过**，所以默认保留。但它们确实进过一份同设计的计划。

**处置**：默认集按字面口径交付；备选集已经算好（`plan.json` 的 `alternative_if_also_excluding_never_sent`），一条命令可切换：

```bash
python prepare_blind_set.py --also-exclude-never-sent --write --out <新目录>
```

备选集是：`wafer_00044627_009`(none) / `wafer_00044063_004`(Near_full) / `wafer_00025336_019`(Loc) / `wafer_00015139_006`(Edge_Ring) / `wafer_00047187_023`(Scratch) / `wafer_00013704_014`(Edge_Loc) / `wafer_00018489_020`(Donut) / `wafer_00017106_021`(Random) / `wafer_00011911_008`(Center)。

**这是路线选择，请指挥会话拍板；我不自行放宽也不自行收紧任务书给的规则。**

---

## 1. 核实了什么

### 1.1 提示词版本已锁定 **[事实]**

| 项 | 值 |
|---|---|
| 文件 | `协作/01_Codex_指挥/prompt_image_only_v2_20261002.txt` |
| 字节 SHA256 | `7df73031ee1a30dfb018dbb2417f505ea9ff139280307bce2934de604119954a` |
| 任务书 §2 签发值 | `7df73031ee1a30dfb018dbb2417f505ea9ff139280307bce2934de604119954a` |
| 一致 | **是** |
| LF 统一后文本 SHA256 | `7df73031ee1a30dfb018dbb2417f505ea9ff139280307bce2934de604119954a` |
| 说明 | 该文件**本来就是纯 LF**（CR 计数 0），所以字节指纹 = LF 文本指纹；没有"默默接受新提示词" |

### 1.2 lot 隔离：全清单三向交集都是空的 **[事实]**

| 检查 | 结果 |
|---|---|
| train ∩ val | **0** |
| train ∩ test | **0** |
| val ∩ test | **0** |
| benchmark lot ∩ train lot | **0** |
| benchmark lot ∩ val lot | **0** |

（train 3065 lot / val 372 / test 385；benchmark `metadata.json` 的 `leakage_rule` 写明"uses test split only"。）

**[事实]** 因此任务书 §2 的「若发现冲突停止」**没有触发**——没有可报的冲突，也没有需要"静默修正"的东西。

### 1.3 选出来的九张（实测，非计划）**[事实]**

| item | sample_id | 类别 | manifest 行 | lot |
|---|---|---|---|---|
| item_001 | `wafer_00044523_017` | none | 4685 | lot44523 |
| item_002 | `wafer_00043922_015` | Near_full | 4425 | lot43922 |
| item_003 | `wafer_00025336_019` | Loc | 2627 | lot25336 |
| item_004 | `wafer_00015139_006` | Edge_Ring | 1398 | lot15139 |
| item_005 | `wafer_00047178_011` | Scratch | 5724 | lot47178 |
| item_006 | `wafer_00013704_014` | Edge_Loc | 1106 | lot13704 |
| item_007 | `wafer_00018489_020` | Donut | 2030 | lot18489 |
| item_008 | `wafer_00017102_001` | Random | 1649 | lot17102 |
| item_009 | `wafer_00011911_008` | Center | 974 | lot11911 |

**九类齐全、九个不同 lot、指纹互不重复。** 九张图全部 `448×448` / PNG / 恰好 3 色 / 三色 ⊆ {黑,绿,红}。

### 1.4 独立校验：44 项全过 **[事实]**

`verify_blind_set.py` **刻意不 import 准备脚本**——抽样、排除表、lot 封禁在里面用**另一份实现**重写了一遍。两份代码得出同一结果才算证据。

```
通过 44/44  结论：全部通过
```

其中最关键的一项（§8）是**复现抽样**：独立重实现重跑一遍，得到的样本序列与交付的盲清单**逐项一致**。

### 1.5 盲清单确实没有答案 **[事实]**

`blind_inputs.jsonl` 逐行**只有四个字段**：`item_id` / `sample_id` / `image_path` / `image_sha256`。校验器另外做了三件事：

- 盲清单**原文**里搜九个类别名（大小写不敏感）→ **零命中**
- 搜 `lot\d+` → 零命中（**注意**：见 §附录 D 第 2 条，`sample_id` 本身编码了 lot）
- 图片文件名搜类别名 → 零命中（文件名形如 `wafer_XXXXXXXX_YYY.png`）

### 1.6 历史试标排除表：从产物里扫出来，不是靠举例子 **[事实]**

任务书 §2 点名的两个只是例子。扫描根目录是 `kimi_k3_cost_test/`、`zcode_glm53flash_test/`、`D:/pycode/workbuddy/wafer-cost-test/`，**只提取 `wafer_XXXXXXXX_YYY` 编号（白名单正则），不读配置、不读凭据**。

命中 14 个编号 → 判定后**排除 10 个**：

| 样本 | 判定 | 依据 |
|---|---|---|
| `wafer_00000161_006` | 已确认送出 | Kimi 批次 `20261001-232051` success |
| `wafer_00011320_018` | 已确认送出 | 同上 |
| `wafer_00013751_005` | 已确认送出 | 同上 |
| `wafer_00046291_009` | 已确认送出 | 同上 |
| `wafer_00016894_009` | 已确认送出 | 同上（空回答）+ ZCode GLM 首轮 |
| `wafer_00011910_009` | 已确认送出 | WorkBuddy `call-01.json` |
| `wafer_00017178_007` | 已确认送出 | WorkBuddy `call-02` 系列 + Kimi 单张 |
| `wafer_00012836_003` | 已确认送出 | WorkBuddy `call-03.json` |
| `wafer_00015000_006` | 已确认送出 | WorkBuddy `call-04.json` |
| `wafer_00025247_021` | **存疑·已发起无记录** | `started-05.json` 在、`call-05.json` 不在 → **保守排除** |
| `wafer_00017102_001` 等 4 张 | 计划过·未送出 | 只在 `plan.json` → **默认不排除**（见 §0） |

**另外扫过的、确认与试标无关的**：`学习记录/基准可动性诊断_20260928.md`、`交付文档/…说明书`、`archive/working-copies/show_samples*.py` 里的编号**全部是 test split / benchmark**（已被 lot 排除覆盖）；`参考文档/第一代方案复现文档_已脱敏.md:399` 的 `wafer_00017115_016` 只是**路径格式举例**（`lot17115_16 → wafer_00017115_016`），不是试标。

---

## 2. 仍未知什么

| # | 未知 | 性质 |
|---|---|---|
| 1 | `wafer_00025247_021` 那张图**到底送出去没有** | **未查明**。已发起、无结果记录；按保守口径排除。不再追究 |
| 2 | ZCode 打开这九张时的**真实单张成本** | **未提供**（`踩雷记录.md:1085` B31 已记：会话内读不到用量） |
| 3 | ZCode 后端**实际模型身份** | **未提供**（`踩雷记录.md:1291` B29 已记） |
| 4 | 这九张对**视觉能力**的区分度 | **未测**，本轮目的不是这个——是查流程稳定性。且任务书 §1 明确"不为了制造难题而选模型已答错的图片" |
| 5 | 前任计划里"只进过计划"的那 4 张，将来会不会被那一轮复用 | 不是我能决定的；本次已把冲突点摆在明面上 |

---

## 3. 改了什么

### 3.1 新建 7 个文件（全在本任务新目录内）

```
协作/02_ClaudeCode_实操/准备_20261002_九类试标/
├── prepare_blind_set.py      ← 准备脚本（支持 --dry-run / --write / 拒绝覆盖）
├── verify_blind_set.py       ← 独立校验脚本（只读，不写任何文件）
├── blind_inputs.jsonl        ← 交给 ZCode 的唯一输入
├── audit_selected.jsonl      ← 只给 Codex，含真实标签
├── plan.json                 ← 指纹与逐项检查状态
├── prepare_report.md         ← 本文件
└── 给Codex的回复.md           ← 交给指挥会话的验收请求（含唯一待裁决项）
```

### 3.2 修改 2 个已有文件（按协作规则 3「改动要记名」在此记名）

| 文件 | 改动 | 为什么 |
|---|---|---|
| `踩雷记录.md` | **纯追加** B33 一节（+3511 字节），未改任何原有内容 | 协作规则 4：发现问题要归档。本次发现「只搜仓库会漏掉 4 张真被模型看过的图」 |
| `协作/02_ClaudeCode_实操/README.md` | 更新产物清单与当前状态；更正一行过时的环境描述 | 规则：各 agent 目录的 README 由该 agent 自己维护 |

**`踩雷记录.md` 的改动可逐字节核对**：

```
改前 SHA256  9043c56c5edc6e624c26af88a7029020c0f759d7da8aba3b305328cb6460a543  (85391 B)
改后 SHA256  349c33739754c29b5d59ad1cbd415fe22a3c5b76b9454efdec7de7128d282645  (88904 B)
纯追加校验   通过（旧字节是新字节的前缀）
换行符       与原文件一致，统一 CRLF
```

**注意**：`踩雷记录.md` 在本次改动**之前**就已经处于未提交的修改状态（`git status` 显示 `M`）——那是**别的 agent** 的改动，不是我造成的，我也没有碰它。

**未触碰**：`AGENTS.md`、`data/`、`benchmark/`、`code/`、`docs/`、`results/`，以及另外两个 agent 的目录。**未复制、未搬动任何图片**——`image_path` 指向仓库内原位置。

---

## 4. 下一步最小动作（**未运行**）

> **状态：未运行。** 下面只是方案，需要指挥会话验收 + 用户预算指令。

1. **指挥会话先做两件事**：
   - 判定 §0 的路线选择：**用默认集**（含 2 张前任计划图）还是**切备选集**；
   - 验收本轮产物（`blind_inputs.jsonl` / `plan.json`，或直接跑 `verify_blind_set.py` 复校）。
2. **再由用户把 ZCode 任务书 + `blind_inputs.jsonl` 绝对路径转交标注会话。**
3. **给 ZCode 的只有 `blind_inputs.jsonl`。** `audit_selected.jsonl`、本报告、任务书本身**都不能给**（任务书 §5 末句）。
4. **ZCode 那一步仍有 B31 未解决**：单张成本读数需要用户从客户端 UI 手工补。扩大样本前应先解决。

---

## 5. 证据路径

| 内容 | 路径 |
|---|---|
| **盲清单（交 ZCode）** | `D:\pycode\晶圆图研究\协作\02_ClaudeCode_实操\准备_20261002_九类试标\blind_inputs.jsonl` |
| **计划与指纹** | 同目录 `plan.json` |
| **审计表（含标签）** | 同目录 `audit_selected.jsonl` |
| **准备脚本** | 同目录 `prepare_blind_set.py` |
| **独立校验脚本** | 同目录 `verify_blind_set.py` |
| 数据清单 | `D:\pycode\晶圆图研究\data\manifest.jsonl`（SHA256 `c2f8a9afbf7ab3bc7bac92dc2f5896f8896bf379de2ac1c9d3667afcebcbaa19`，5904 行） |
| 提示词 | `协作\01_Codex_指挥\prompt_image_only_v2_20261002.txt` |
| **前任九类计划（仓库外）** | `D:\pycode\workbuddy\wafer-cost-test\20261001-211719\plan.json` + `call-*.json` + `started-05.json` |
| Kimi 批次 | `kimi_k3_cost_test\batches\20261001-232051\` |
| ZCode 首轮 | `zcode_glm53flash_test\20261002-002240\` |

**文件指纹**

| 文件 | SHA256 |
|---|---|
| `blind_inputs.jsonl` | `fa44b215087c1a9ef0f3d42ec5ede6b93a5a96b3e239afe363f3a3e3c9687a03` |
| `audit_selected.jsonl` | `789ed58d05c00e8cb2aee291c4f5a311e173814f5c293e86ec7c6fcc9199e68a` |
| `plan.json` | `01943ad06d71a7caeb4061e8d5c4afc5a73962a8af03adaf7c1b38bcece1e1bf` |

**注意**：`plan.json` 的指纹**不在 plan.json 里**（自指不可能）。以上值由本次运行打印；`verify_blind_set.py` 会重新计算并与文件内容对照，可用于复校。

---

## 6. 实际调用与消耗

| 项 | 值 |
|---|---|
| **外部模型请求次数** | **0** |
| **付费积分消耗** | **0** |
| **GPU 训练 / 推理** | **0** |
| 启动 ZCode | **否** |
| 上传学校服务器 | **否** |
| 读过的凭据 | **0**（只按白名单正则抓 `wafer_` 编号，未读 WorkBuddy 配置、密钥、Token、环境变量） |
| **新建**文件 | **7 个**，全在本任务新目录内 |
| **修改**已有文件 | **2 个**，见 §3.2（`踩雷记录.md` 纯追加；本 agent 的 README） |
| 图片复制/搬动 | **0** |

**"没联网"怎么验证的**：两个脚本的 import 全表 = `argparse / glob / hashlib / io / json / os / random / re / sys / pathlib` + `PIL` + `numpy`；对 `requests|urllib|http|socket|openai|api_key|apiKey|token|password` 做关键字扫描 —— **零命中**。这两个数字都能当场重跑验证。

**"没动数据"怎么验证的**：`data/images/` 目录 mtime 仍是 `Sep 19 00:11`（未变）；`data/manifest.jsonl` 的 SHA256 与 `plan.json` 记录一致。

**"改动是纯追加"怎么验证的**：`踩雷记录.md` 改前/改后 SHA256 已列在 §3.2，且校验了"旧字节是新字节的前缀"。

---

## 附录 A · 隔离检查（逐项）

| 检查 | 结果 |
|---|---|
| 九张来自不同 lot | ✅ 9 个 lot 互不相同 |
| 图片指纹互不重复 | ✅ 9 个 SHA256 互不相同 |
| 全部 `split=train` | ✅ |
| 全部 `label_source=ground_truth` | ✅ |
| 未命中 benchmark 样本 | ✅ |
| lot 未落在 benchmark / val / test | ✅ |
| 未命中已试标样本或其 lot | ✅ |
| 图像 PNG / 448×448 / 三色 | ✅ 9/9 |
| 文件名不含答案 | ✅ |
| 盲清单不含类别名 / lot 字面量 | ✅ |
| 盲序不是按类别排的 | ✅ 见 §1.3（none→Near_full→Loc→Edge_Ring→Scratch→Edge_Loc→Donut→Random→Center） |

**候选池规模（实测）**：排除 767 个 lot 后，train 剩 4699 行；各类候选 Center 644 / Donut 419 / Edge_Loc 635 / Edge_Ring 632 / Loc 621 / Near_full 126 / Random 634 / Scratch 652 / none 336。**没有一类需要"选不齐"**——任务书 §2 的"选不齐如实报告"未触发。

**被 lot 规则额外挡掉的 train 行**：12 行（它们的 lot 因为别的样本被禁，而自身从未被试标）。这 12 行不是被"样本排除"挡的，是被"lot 排除"挡的——两者分开记账。

---

## 附录 B · 类覆盖（**含标签，禁止给 ZCode**）

```
item_001  none        ← 任务书点名要覆盖
item_002  Near_full
item_003  Loc         ← 任务书点名要覆盖
item_004  Edge_Ring
item_005  Scratch     ← 任务书点名要覆盖
item_006  Edge_Loc
item_007  Donut
item_008  Random
item_009  Center
```

任务书 §1 点名要覆盖的 **Scratch / Loc / none 三类都在**（item_005 / item_003 / item_001）。

**标签正规化依据**：`data/manifest.jsonl` 的 `failure_type` 取值只有九种，**与任务书 §2 列的正规名逐字相同**，无需映射。`audit_selected.jsonl` 里 `failure_type_original == failure_type_normalized`，`normalization` 字段写明 `identity`。**原清单未被改写**。

---

## 附录 C · 排除原因（为什么是这九个）

抽样规则（`plan.json` 的 `sampling_rule_version = prepare-v1@2026-10-02`）：

1. 候选 = `split=train` ∧ `label_source=ground_truth` ∧ 样本未被试标 ∧ lot 不在 {val ∪ test ∪ benchmark ∪ 试标样本的 lot}；
2. 每类候选**按 `sample_id` 排序**；
3. 类别按正规化顺序（Center→…→none）逐个抽，`random.Random(3407)`，抽中的 lot 立即占用、后续类别不再碰；
4. 九张抽齐后，**另起一个 `random.Random(3407)` 做一次 shuffle** 定盲序。

**前任计划的高度接近不是巧合**：同样的 seed、同样的类别顺序、同样的"按 sample_id 排序"，本来就会落在相邻位置。例如 Center 抽到 `wafer_00011911_008`，而前任是 `wafer_00011910_009`（后三位差 1）。**[解释]** 这说明前任计划用的很可能就是这套规则；但**没有证据说它用的是一模一样的实现**，所以两者结论不能互相替代。

---

## 附录 D · 已知限制

1. **`wafer_00025247_021` 的存疑排除会让它的 lot 也一起被禁。** 若那张其实没送出去，这是**过度排除**（多禁 1 个 lot）。方向是安全的，但记账上是保守的。

2. **`sample_id` 本身编码 lot 号**（`wafer_00017102_001` → `lot17102`）。任务书 §3 要求盲清单不含 lot，但同时又要求含 `sample_id`——**两者在字面上冲突，`sample_id` 优先**。**[事实]** 这一条有实际信息量：多行 lot 中 **59.9%** 类别唯一（619/1033），远高于随机基线 1/9≈11.1%。本次九张里 `lot17102` 有 11 行、其中 10 行是 Random。
   **[解释]** 但这要配合 `manifest.jsonl` 才能利用，而 ZCode **拿不到**清单也没有答案。**风险可控，但必须写明**：ZCode 端不得在仓库内检索这三个 `sample_id`。

3. **"格式通过 ≠ 内容正确"**，本轮一个字都没有验证标注**内容**。本轮产物是**输入**，不是结果。

4. **本机改动会污染复现**：`prepare_blind_set.py` 依赖 `data/manifest.jsonl` 的当前字节。若清单将来被改，重跑会得到不同结果。**manifest 的 SHA256 已锁进 `plan.json`**，复校时会核对。

5. **备用集未做全套图像检查**。`alternative_if_also_excluding_never_sent` 只给了 sample_id 与盲序，**没有**跑 §附录 A 的逐图检查。真要切过去，应先 `--write` 到新目录并跑一遍 `verify_blind_set.py`。

6. **`协作/02_ClaudeCode_实操/README.md:28` 有一句已过时**：「本机没装 `scipy`」。实测本机 `scipy 1.16.2` 可用（`pandas` 确实没有）。这条属于我的地盘，但**本次未改**（避免与 Codex 的未提交改动撞车），留待一声指令。
