# 交接给 Codex：v3 之后的实现权移交（2026-10-11）

> **移交声明**：从本件起，**奖励函数、数据构造、训练脚本的修改与运行由 Codex 直接做**。
> Claude Code 不再往返提交"下一轮计划"，改为**按 Codex 的指令执行或验收**。
> 用户 2026-10-11 明确："把东西都给 codex，codex 来改奖励函数之类的，不然每次都要来回折腾。"
>
> 本件**自包含**：Codex 只需读这一份 + 仓库里点名的文件，就能直接开工。
> 全部产物已推送到 `main`（最新 `3b358be`）。**本件不含任何地址、账号、口令。**

---

## 一、现状一句话

v3 六字段首轮已跑完并结项：**共同起点（M0_v3）已被框架级证据证明加载正确**；
**开发比较未显示 GRPO 优势、规则合规反而下降**；
训练侧归因结论是 **"没有发现奖励或掩码的实现错误"**，
失败主因是模型系统性答错，其中区带的一大部分是**向训练池最高频组合（45.7%）塌缩**。

## 二、权威件地图（按"你想改什么"索引）

| 你想做什么 | 打开哪个文件（仓库内路径） |
|---|---|
| **改奖励函数** | `协作/02_ClaudeCode_实操/一天半优化_20261010/core_v3_复核修复.py`（服务器上叫 `reward_v3.py`） |
| 看奖励的原始版（Codex 自己写的） | `协作/01_Codex_指挥/无卡CPU整备_20261010/core_v3.py` |
| **改训练数据组成** | `协作/01_Codex_指挥/无卡CPU整备_20261010/build_cpu_packet.py`（生成 `data_v3/*.jsonl` 的脚本） |
| 改训练脚本 | `协作/02_ClaudeCode_实操/一天半优化_20261010/server_v3_train.py`（服务器 `v3_train.py`） |
| 改评测脚本 | 同目录 `server_v3_eval.py`（服务器 `v3_eval.py`） |
| 起点/参考策略门槛 | 同目录 `runtime_guard_v3.py`（服务器同名） |
| 看题目/字段定义 | `协作/01_Codex_指挥/无卡CPU整备_20261010/题面_v3.txt` + `core_v3.py` 的 `FIELDS/COVERAGE/ANGULAR` |
| 看首轮结果 | `…/一天半优化_20261010/v3_首轮报告.md` + `out_v3/v3_指标.md` |
| 看训练侧归因 | `…/补证_20261011/训练组诊断.md` + `out_v3/训练组深挖.json` |
| 看掩码检查 | `…/补证_20261011/mask检查报告.md` + `mask_check.json` |
| 下一轮合同草案 | `…/补证_20261011/下一步合同草案.md` |
| 数据集与池 | `协作/01_Codex_指挥/无卡CPU整备_20261010/pools/*.jsonl`（adapt 512 / rl 256 / rule_dev 30 / confirmation 120） |

## 三、服务器现状与接入（**不写地址**）

- **接入**：仓库外 `D:/pycode/.ssh-tmp/wafer_rl_20261010.sh`（读取同目录 `.host`/`.pw`，脚本不打印凭据）。
  无上传脚本，上传用 `ssh "cat > 远端路径"` **并核字节数**（历史上出现过静默写 0 字节）。
- **实例状态（2026-10-11 12:xx 实测）**：无卡模式在跑；`/root/autodl-tmp` 50 G，**已用 44 G、剩 7.0 G**。
- **工作区**：
  - `/root/autodl-tmp/ws/v3run/` —— 本轮全部：`v3_train.py`、`v3_eval.py`、`reward_v3.py`、
    `runtime_guard_v3.py`、`数据/`（4 个 jsonl + reference）、`图/`（888 张）、
    `out/{M0_v3,M1_v3,M2_v3,gate_*}`、`out/评测_v3/`
  - `/root/autodl-tmp/ws/models/Qwen3.5-9B-Dmerged`（18 G，**共同基座，别删**）
  - `/root/autodl-tmp/ws/models/Qwen3.5-9B`（19 G 原始基座）
  - **磁盘只剩 7 G**：要新建合并模型（+18 G）**放不下**；走 adapter 路径可以。
- **依赖（实测）**：torch 2.8.0+cu128 / transformers 5.16.1 / peft 0.20.0 / ms-swift 4.5.3 /
  accelerate 1.15.0 / trl 0.29.1 / safetensors 0.8.0。**`msgspec` 是后补装的**（swift 没随包带，GRPO 起不来）。
- **基座指纹**：`Qwen3.5-9B/config.json` = `d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05`
  （与 56 图那轮冻结值逐字相同）；`Dmerged/config.json` = `91cd4f87…`。
  **Dmerged 四个分片的全量 SHA 始终没取到**（读 18 G 超时），别当成已验证。

## 四、要改的话，具体动哪里

### 4.1 若改**奖励函数**

- 文件：`reward_v3.py`（= 仓库 `core_v3_复核修复.py`）。**它是 Codex 原版 `core_v3.py` 的复核副本**，
  我改过**两处**（都已留反例）：
  1. `parse()`：**剥掉完整的空 `<think></think>` 包装**（非空 think 仍拒收）。
     **理由（实测）**：模板自己就发这个前缀（`non_thinking_prefix='<think>\n\n</think>\n\n'`），
     360/360 评测原答与 1024/1024 训练候选**全带**它；
     原版直接 `json.loads(全文)` 会让**每一次 GRPO 生成的奖励恒为 0**。
     用真实答案对照：**原版 0/512 可解析，复核版 512/512**。
     另存 `parse_raw_strict()` 供"原始全文严格 JSON"口径分列报告。
  2. `score()` 的角向分：`sectors_ok` 加了 `obj['angular_type'] != 'unknown'` 前置。
     **理由**：参考是 `nondirectional`（扇区为空）时，模型回 `unknown` + `[]`
     会因"扇区集合恰好相等"白拿 0.4。
- **阈值/权重/字段名一律没动**。要改这些，请连同回归一起改，并**保留旧副本**。
- 改动后必须重跑：`python core_v3_复核修复.py`（自测 **17/17**）。
- **本轮归因的结论是"奖励没有实现错误"**：42 组里"有正确候选却没拿满"五个字段**全为 0**。
  **所以不建议为追求正结果去改评分口径**。

### 4.2 若改**数据组成**（本轮唯一有证据支持的改法）

- 证据：RL 池区带参考分布 `['middle','edge']` **45.7%** / 三带 36.7% / `['edge']` **16.0%**；
  42 个未满分同分组里 28 个"四候选全同且都错"，其中 **26 个都答 `['middle','edge']`**；
  参考为 `['edge']` 的 9 组图，**middle 质量只有 0.0–7.5%（中位 4.7%）**，模型照答 middle；
  42 组的生成多样性只有 **1.31** 种答案/组（全池 2.15），**69% 的组四候选文本完全相同**。
- 做法：在 `build_cpu_packet.py` 里改**进入 256 步训练序列的组选择**，
  让区带参考的组合分布更均衡（抬高 `['edge']`、`['middle']`、`['center','edge']` 占比）。
- **只改组成，不改题面/规则/阈值/权重/步数/seed**。
- **不要用确认池 120 图做任何选择**；新组的 ID/lot/PNG 仍须与 adapt/rule_dev/confirmation 无交集。

### 4.3 不建议的两个改法（本轮证据不支持）

| 不提出 | 为什么 |
|---|---|
| G 从 4 提到 8 | 28/42 的区带失败是**四候选答法完全相同且都错**，再采 4 个大概率还是同一个错答案 |
| 动态采样 / 换算法 | 任务书明令；且证据显示失败是"一起错"而不是"都没梯度" |

## 五、我踩过的坑（**照抄可省时间**）

1. **`**extra` 没接进构造函数**（首轮）：脚本定义了 `extra` 却没传给 `SftArguments/RLHFArguments`，
   结果 M1/M2 **没有**从 M0 起步，而脚本还打印了"从已有 adapter 继续"、`v2_config.json` 也记了错的起点。
   **教训**：**只信训练器自存的 `args.json`**；打印与自写配置都不算证据。
   （`v3_train.py` 已改成：缺 `--start-adapter` 直接 `RuntimeError` 拒绝启动。）
2. **`get_model_processor(model_dir, load_model=False)` 返回 `(None, processor)` 元组**，processor 在**第二个**位置。
   不显式给 `model_type='qwen3_5'` 时，该目录能匹配 4 个候选类型而报错。
3. **collator 沙盒检查要先设模板 mode**：`Template._encode()` 结尾
   `if not self.is_training: labels=None`；`is_training` 是只读属性，由 `mode` 推导
   （`mode not in {'transformers','vllm','lmdeploy','sglang'}`）。**不设 mode 就看不到 labels。**
4. **GRPO 的模板 mode 是 `'train'`，不是 `'rlhf'`**：
   `mode_mapping = {'kto':'kto','gkd':'train','ppo':'transformers','grpo':'train'}`。
   设成 `rlhf` 会走 DPO 路径并报 `inputs.rejected is None`。
5. **collator 为 `position_ids` 会内部建 dummy model**，它调用 `get_model_processor(model_dir)` 不带 model_type
   → 报"多种候选类型"。我在检查脚本里**进程内包一层补 `model_type`**（不改框架源码）。
6. **`runtime_guard_v2fix.py` 写死了 v2 那轮的 M0 指纹**（`43b36375…`），拿 v3 的 M0 跑必报
   `M0 file fingerprint mismatch`。我另存 `runtime_guard_v3.py`：**期望指纹必须由 `WAFER_EXPECT_M0_SHA` 显式给出，
   不给就拒绝启动**（不放松、不沿用旧常量）。
7. **磁盘**：曾到 **98%（剩 1.2 G）**。安全清理顺序：① 删一次性门槛探针的 checkpoint 目录；
   ② 删已完成训练的 `optimizer.pt`（346 MB/个，仅用于续训）——
   **保留 `adapter_model.safetensors` / `args.json` / `trainer_state.json` / `scheduler.pt` / `rng_state.pth`**。
   **别动 `ws/models`、`codex_fix_20261010`、`labguard-gpu`。**
8. **打包要按任务书点名的文件逐项核**：我上轮"按手头目录顺手打"漏带了**正式 M1_v3/M2_v3 的
   `runtime_gate.json`**，导致 Codex 以为原件缺失（其实一直在）。

## 六、未决与风险（移交时仍开放）

| 项 | 状态 |
|---|---|
| 方向可判样本极稀 | 全量 5904 张里只有 **1.5%**；确认池 120 图里正向 **0** 张 → 该池**评不了方向** |
| 确认池性质 | **开发确认池**（7 类、缺 `none` 与 `Near_full`，84 张 WorkBuddy 候选全在 adapt 池、57 张被旧 D 见过）→ **不能当留出测试** |
| `coverage_band='zero'` | 四池 **0 样本**，该档无监督、无评测 |
| labels/mask 历史记录 | 训练当时**没留**；本轮补的是**事后独立检查**（4 条样本），**不回填成当时已做** |
| 累计设备分钟 | **≈1437 为暂估**（含 80 分钟估算）；已核上界另有起点修复 39.25 + v3 约 87.9。**上一阶段 120 已用尽** |
| 租机人民币 | **未知**（无账单证据） |
| Dmerged 分片全量 SHA | **未取到**（读 18 G 超时） |
| 真人复核 | 用户已定**永久不做**，真人栏永久留空 |

## 七、建议的下一步顺序（供 Codex 裁量，不是指令）

1. **先决定是否做那一个数据组成的改法**（4.2）；若不做，就按现状收口并写进结论。
2. 若做：改 `build_cpu_packet.py` 的组选择 → 重建 `group256_*` → **CPU 复核隔离与 SHA** →
   按 `下一步合同草案.md` 跑 2 步门槛（两线 `runtime_gate` + mask 复跑 + 单卡断言）。
3. 门槛过 → M1b（匹配 SFT 256 步）与 M2b（固定规则 GRPO 256 步）→ 评测 3×120。
4. **新额度须显式放行**（估算 ≈79 设备分钟）；实测步速超估算 20% 即停并回报。

## 八、Claude 之后做什么

按用户新分工：**Claude 不再主动提方案往返**。
Codex 若要 Claude 继续执行，直接在任务书里点明文件与验收条件即可；
Claude 会按本件 §二 的地图与 §五 的坑清单执行，不再重复调研。
