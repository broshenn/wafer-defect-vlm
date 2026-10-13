# 统一 TEST200 打分（2026-10-13）

不训练、不改权重、不改测试标签的一次性统一打分。考卷只用**已经冻结的 TEST200**（200 张，分层，种子 3407）。
252 / 56 / 36 / 120 张上的旧分数一律没有搬进本目录；589 张本轮没有跑。

**结果先读 [`统一TEST200总表.md`](统一TEST200总表.md)。**

## 一句话结果

在冻结的 TEST200 上，**没有任何模型在几何分上高于 `S0_rerun`、也没有任何模型在类别上高于 `L`**——
所有点估计为正的配对，95% 区间都跨 0。点位最高的分别是 `Rfullp`（几何 +0.0127）和
`D_L_test200`（类别 Macro-F1 +0.0170）。

## 目录

| 路径 | 是什么 |
|---|---|
| `统一TEST200总表.md` | **交付件**：主列表、类别列、配对区间、三句话、缺席清单、来源出处、复现性检查、限制 |
| `contract.json` | 打分合同：冻结 ID / 考卷 / 逐图 SHA / 题面 / 解析器，全部带 SHA256 钉死 |
| `inputs/` | 冻结输入副本：`TEST200_ids.txt`、`reference200.jsonl`、`images200.jsonl`、两份题面、`common_v5.py`、服务器实际消费的考卷 |
| `sources/` | 50 条来源的原答（只读；`build_sources.py` 不改写它们） |
| `evidence/` | 每条来源的台账 / 指标 / 运行摘要副本，作为「同一份考卷」的哈希证据 |
| `source_manifest.json` | 来源清单：每条声明 rawfile SHA、四份 identity SHA、evidence 列表 |
| `source_provenance_overrides.json` | 来源类别声明（复用原答 / 新推理 / 百炼），不靠文件名推断 |
| `pairs.json` | **看分之前锁定**的 95 对配对比较；SHA 写进清单 |
| `frozen_models.json` | 百炼三模型冻结件（在任一 TEST200 调用之前写入） |
| `probe.json` | 合成图连通检查的完整请求/返回记录 |
| `bailian_out/` | 百炼 600 条原答 + 逐条台账 + 运行摘要（`sources/`、`evidence/` 里的百炼件是它的副本） |
| `诊断_zone_frac刻度.json` | 题面歧义诊断（**不是成绩**） |
| `out_v2_无S3/` | **正式打分产物（当前版）**：49 条来源，`main.csv`（一行一个模型）、`metrics.json`、`models/<名>/per_sample.jsonl` |
| `excluded_from_table/` | 按用户指令从表里剔除的 `S3_fixed` 原答与台账。**没有被销毁**，只是不进清单 |
| `out_unified/` | 上一版打分产物（50 条，含 `S3_fixed`）；配对锁定件另存为 `pairs_S3included.json`、`source_manifest_S3included.json` |
| `out_prelim/` | **已作废的中间产物**：加百炼与新增三条之前的 44 模型版；保留只为留痕，不要引用 |

> **v2 与 v1（git `e793b6a`）的唯一差别**：按用户指令把 `S3_fixed` 移出清单（49 条取代 50 条，
> 配对 93 对取代 95 对）。逐行比对过 `main.csv`：**其他 49 个模型的任何数字都没有变**。
> 配对是在看过分之后重锁的，改动被验证限制为「只删 S3 那两对、不加任何一对」——
> 新旧两份 `pairs.json` 都留了档，可以直接 diff。
>
> **注意提交时的坑**：仓库根 `.gitignore` 有一条 `models/`，本意是排除 19 GB 的权重目录，
> 但它会连带命中打分产物里的 `out_*/models/`。这些逐样本记录必须用
> `git add -f` 才能进版本库，否则会静默漏掉（第一版就漏过一次，已补上）。

## 复现顺序

```bash
python build_contract.py        # 冻结输入与合同
python build_sources.py         # 扫描 sources/ 生成来源清单
python lock_pairs.py            # 看分前锁定配对（必须在 unified_score.py 之前）
python ../01_Codex_指挥/统一TEST200_20261013/code/unified_score.py \
    --contract contract.json --sources source_manifest.json \
    --pairs pairs.json --out out_v2_无S3
python make_table.py            # 生成中文总表（默认读 out_v2_无S3，可用 --out 换）
```

打分器是 Codex 写的、本轮**只读不改**。它在写任何输出前会做完整审计：
逐条核对 identity 四份 SHA、逐 ID 核对题面/图片/参考、并拒绝覆盖已有输出。

百炼部分：

```bash
export DASHSCOPE_API_KEY="$(cat <仓库外的密钥文件>)"
python bailian_freeze.py --plan     # 只读列目录
python bailian_freeze.py --probe    # 合成图连通检查（不读测试图）
python bailian_freeze.py --freeze   # 冻结三个模型
python bailian_run.py --model-role flagship      # 各 200 张
python bailian_run.py --model-role small
python bailian_run.py --model-role max_or_plus
```

## 三条必须知道的口径

1. **主列与类别列是两套题面，不能混用。** 主列 = v5 的 `prompt_v5.txt`；类别列 = L 自己的类别题面。
   第 4 节第 2 句把主列模型的类别准确率拿去和 `L` 比，是跨题面比较。
2. **程序几何与公开类别都不是人工金标。** 几何参考由程序按几何特征算出，类别参考是公开标签。
3. **点估计更高不等于显著优于。** 区间跨 0 的一律只能读成「这一卷上看不出差别」。

## 本轮的边界

- **不训练、不改权重、不改测试标签**；已经有同题原答的一律重算分，不占卡重答。
- 只有 `D_v5_test200`、`D_L_test200`、`M0_v2_test200` 三条是 2026-10-13 在**学校 5090 的空闲卡**
  上新做的推理（GPU1/2/3，未碰 GPU0 上他人的任务，未 pkill 任何进程）。
- 百炼只走**官方百炼 HTTPS 域名**，单次请求、零自动重试、拒绝重定向；密钥只从环境变量读，
  不打印、不落盘、不进日志（已扫描确认无 `sk-` 形态字符串落在本目录）。
- 合成图连通检查里，三个后备候选（`qwen3.5-27b`、`qwen3.5-35b-a3b`、`qwen3.5-122b-a10b`）
  返回 403 `AllocationQuota.FreeTierOnly`（「免费额度已耗尽，若要按量付费请关闭『仅用免费额度』」），
  **据此推断**账号处于「仅用免费额度」模式；正式 600 次请求没有出现该错误，只出现 2 次网络失败。
  所以「本轮没有产生需要单独授权的付费」是**推断**，不是账单核实；平台账单未核。
- 未做的：真人金标、完整九类分列、按难度分层、`none` 类的单独报告。
- `S3_fixed` 按用户指令从表里剔除（它不是「没训成」——学校机器上确有 checkpoint-375 和 200 条原答；
  剔的是展示范围，不是证据）。原答保存在 `excluded_from_table/`。
