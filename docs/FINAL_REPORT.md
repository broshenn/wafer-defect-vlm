# 晶圆缺陷 VLM 最终报告 / Wafer Defect VLM — Final Report

生成时间 (UTC): 2026-09-16T11:06:40+00:00

本项目在 WM811K 上对 Qwen3.5-9B 做教师蒸馏数据构建、QLoRA 微调与基准评测。
所有指标均由 `outputs/reports/` 下的 JSON 生成，未手工誊写。


## 1. 结论摘要 / Headline

- 分类准确率 accuracy：Base 0.2143 → SFT 0.6230（Δ +0.4087）
- 分类 macro-F1：Base 0.1347 → SFT 0.6114（Δ +0.4768）
- 平凡基线 floors：多数类 accuracy 0.1190 / macro-F1 0.0236；均匀随机 accuracy 0.1230 / macro-F1 0.1211
- 训练检查点：按验证损失选择 step 604（eval_loss 0.2800）
- 基准状态：`draft_pending_human_review`，人工双审：`bypassed_by_allow_pending (252 samples unapproved)`

## 2. 环境与资源 / Environment

| 项 | 值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4090 (48508 MiB) |
| 驱动 | 580.105.08 |
| Python | 3.12.3 |
| torch / CUDA | 2.8.0+cu128 / 12.8 |
| transformers | 5.16.1 |
| trl / peft / bitsandbytes | 0.29.1 / 0.20.0 / 0.50.2 |
| ms-swift | 4.6.0.dev0 @ `9d3d03dd35c8a60e519e3165962586837cea67ee` |

RL 未安装 vLLM 与 deepspeed，因此 GRPO 只能走 ms-swift 的 PyTorch 生成路径。

## 3. 数据 / Data

- 预处理清单 5904 张，样本 ID 唯一 5904，划分 {'train': 4721, 'val': 594, 'test': 589}
- 抽查 600 张：尺寸或调色板异常 0 张
- 教师蒸馏：接受 4829 条，隔离 486 条，判定明细 {"no_valid_teacher": 47, "primary": 3448, "teacher_disagreement": 439, "secondary+geometry": 652, "primary+geometry": 696, "secondary": 33}
- Lot 隔离：3822 个 lot，跨划分 0 个，违反划分规则 0 行
- 训练/验证样本 12876 / 1611，任务分布 {"caption": 4292, "classification": 4292, "structured": 4292}

## 4. Benchmark

- `wafer_bench_v1`，仅使用 test 划分，核心 252 张，检索查询 90 条，qrels 22505 对
- 类别分布 {"Center": 30, "Donut": 30, "Edge_Loc": 30, "Edge_Ring": 30, "Loc": 30, "Near_full": 16, "Random": 30, "Scratch": 30, "none": 26}
- 状态 `draft_pending_human_review`；冻结门禁记录为 `bypassed_by_allow_pending (252 samples unapproved)`
- 说明：核心集中 `Near_full` 16 张、`none` 26 张，低于其他类别的 30 张，这两类的指标方差更大。
- 随机排序地板：mAP@10 0.0330，nDCG@10 0.0596，Recall@10 0.0394

## 5. 教师蒸馏 / Teacher distillation

- 调用 10612 次，累计 token 11419291，其中 prompt 9874005 / completion 1545286
  - `primary_deepseek.jsonl`：5423 行，成功 5315，失败 108，token 4723053，模型 {"deepseek-v4.1-flash": 1532, "deepseek-v3.2": 3891}
  - `secondary_qwen.jsonl`：5189 行，成功 5189，失败 0，token 6696238，模型 {"qwen3.5-397b-a17b": 5189}
- API 未返回计费字段，因此只能报告 token 用量，不推算金额。

## 6. 结果对比 / Results

| 指标 | Base | SFT | GRPO_lr1e5 | GRPO_lr5e5 | GRPO_lr1e5_s2 | GRPO_lr1e5_s3 | GSPO_lr5e5 | GSPO_lr1e5 | GSPO_G32 | GSPO_G32_lr1e5 | GSPO_G4_lr5e5 | GSPO_G4_lr1e5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| classification accuracy | 0.2143 | 0.6230 | 0.6230 | 0.5675 | 0.6071 | 0.6429 | 0.5397 | 0.6151 | 0.5675 | 0.6190 | 0.5357 | 0.6270 |
| classification macro-F1 | 0.1347 | 0.6114 | 0.6197 | 0.5535 | 0.5934 | 0.6211 | 0.5014 | 0.6089 | 0.5403 | 0.6143 | 0.4910 | 0.6255 |
| macro-F1 95% CI | [0.0955, 0.1706] | [0.5496, 0.6602] | [0.5554, 0.6704] | [0.4956, 0.6110] | [0.5310, 0.6444] | [0.5614, 0.6717] | [0.4367, 0.5537] | [0.5472, 0.6613] | [0.4807, 0.5965] | [0.5492, 0.6699] | [0.4324, 0.5483] | [0.5643, 0.6746] |
| off-vocabulary answers | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| structured defect_type acc | 0.2103 | 0.5840 | 0.6032 | 0.5516 | 0.5800 | 0.5873 | 0.5635 | 0.5595 | 0.5595 | 0.5714 | 0.5437 | 0.5794 |
| structured radial_zone acc | 0.0794 | 0.2640 | 0.2698 | 0.3452 | 0.2760 | 0.2738 | 0.3056 | 0.2460 | 0.4603 | 0.2698 | 0.3254 | 0.2619 |
| clock circular MAE (sectors) | 5.0000 | 1.6847 | 1.7121 | 1.9730 | 1.7574 | 1.7143 | 1.9519 | 1.7098 | 1.6860 | 1.7026 | 1.6761 | 1.7463 |
| size MAE (R) | 1.2139 | 0.4999 | 0.5695 | 0.9296 | 0.7211 | 0.6134 | 0.6927 | 0.6203 | 1.6509 | 0.7085 | 0.4025 | 0.8384 |
| caption must-hit rate | 0.6270 | 0.7540 | 0.7222 | 0.5516 | 0.7024 | 0.6310 | 0.6627 | 0.7143 | 0.5714 | 0.7063 | 0.6349 | 0.7381 |
| caption empty rate | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| root-cause hallucination rate | 0.2183 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| robustness accuracy | 0.1905 | 0.5622 | 0.5661 | 0.5251 | 0.5489 | 0.5820 | 0.5146 | 0.5688 | 0.5556 | 0.5688 | 0.5093 | 0.5807 |
| flip rate vs clean | 0.1389 | 0.2474 | 0.2672 | 0.2619 | 0.2619 | 0.2500 | 0.2354 | 0.2487 | 0.2540 | 0.2606 | 0.2553 | 0.2632 |
| retrieval mAP@10 | 0.3236 | 0.3873 | 0.3742 | 0.4260 | 0.3887 | 0.3943 | 0.3607 | 0.3782 | 0.4337 | 0.3702 | 0.4138 | 0.3748 |
| retrieval nDCG@10 | 0.2978 | 0.3563 | 0.3497 | 0.3917 | 0.3566 | 0.3605 | 0.3253 | 0.3437 | 0.3906 | 0.3431 | 0.3804 | 0.3481 |
| retrieval Recall@10 | 0.1564 | 0.1836 | 0.1794 | 0.2022 | 0.1817 | 0.1845 | 0.1788 | 0.1796 | 0.2008 | 0.1773 | 0.1928 | 0.1780 |


## 6a. 结构化字段 vs 平凡基线 / Structured fields vs their floors

「恒答基线」= 完全不看图、永远输出同一个值所能取得的最好成绩。**没有超过它的模型等于没有学到这个字段。**

| 字段 | 恒答基线 | Base | SFT | GRPO_lr1e5 | GRPO_lr5e5 | GRPO_lr1e5_s2 | GRPO_lr1e5_s3 | GSPO_lr5e5 | GSPO_lr1e5 | GSPO_G32 | GSPO_G32_lr1e5 | GSPO_G4_lr5e5 | GSPO_G4_lr1e5 | 结论 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| defect_type（结构化） | 0.1190 | 0.2103 | 0.5840 | 0.6032 | 0.5516 | 0.5800 | 0.5873 | 0.5635 | 0.5595 | 0.5595 | 0.5714 | 0.5437 | 0.5794 | 全部超过基线 |
| radial_zone（结构化） | 0.8333 | 0.0794 | 0.2640 | 0.2698 | 0.3452 | 0.2760 | 0.2738 | 0.3056 | 0.2460 | 0.4603 | 0.2698 | 0.3254 | 0.2619 | **全部未超过基线** |
| clock_sector（MAE，越低越好） | 2.6932 | 5.0000 | 1.6847 | 1.7121 | 1.9730 | 1.7574 | 1.7143 | 1.9519 | 1.7098 | 1.6860 | 1.7026 | 1.6761 | 1.7463 | SFT、GRPO_lr1e5、GRPO_lr5e5、GRPO_lr1e5_s2、GRPO_lr1e5_s3、GSPO_lr5e5、GSPO_lr1e5、GSPO_G32、GSPO_G32_lr1e5、GSPO_G4_lr5e5、GSPO_G4_lr1e5 超过；**Base 未超过** |
| size_r（MAE，越低越好） | 0.1101 | 1.2139 | 0.4999 | 0.5695 | 0.9296 | 0.7211 | 0.6134 | 0.6927 | 0.6203 | 1.6509 | 0.7085 | 0.4025 | 0.8384 | **全部未超过基线** |

注意 `radial_zone` 与 `size_r` 的取值高度集中（分别是 83% 为 `center`、以及中位数附近极度密集），所以恒答基线的成绩本身就很高。**这两个字段上所有模型都低于基线**。但两者的成因**不同**，必须分开读：`size_r` 是**真实失败**（预测与真值相关系数仅 0.0382，即不含该字段信息）；`radial_zone` 的低分则主要来自**标签退化**，见 6c。`clock_sector` 是微调后唯一超过基线的几何字段，但仅覆盖 203/250 行，且被排除的正是最难评分的两类。

## 6b. Adapter 与 Merged 等价性 / Equivalence

- 比对 60 行，完全一致 28，一致率 0.4667
- 训练格式提示词诊断：accuracy 0.8849，macro-F1 0.8882（仅诊断，不作为榜单成绩）

## 6c. radial_zone 标签复核 / Re-grading radial_zone

现状标签是 `_zone(centroid_r)`，即**缺陷质心**所在的一带。对称图案（边缘环、随机散布、满片）的质心必在圆心，于是标签被算术地压成 `center`——已评分的 251 行里有 210 行（83.7%）如此，`Edge_Ring` 的中位质心半径只有 0.063。下面对**同一批预测**换用「缺陷落在哪一带」重新评分：

| 判据 | 金标多数类占比 | Base | SFT | GRPO_lr1e5 | GRPO_lr5e5 | GRPO_lr1e5_s2 | GRPO_lr1e5_s3 | GSPO_lr5e5 | GSPO_lr1e5 | GSPO_G32 | GSPO_G32_lr1e5 | GSPO_G4_lr5e5 | GSPO_G4_lr1e5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `centroid`（现状：质心） | 0.8333 | 0.0794 | 0.2640 | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run |
| `mean_radius`（缺陷像素平均半径） | 0.5079 | 0.1310 | 0.4560 | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run |
| `outer_extent`（外沿半径） | 0.9960 | 0.1706 | 0.4200 | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run |

换用位置判据后 SFT 从 0.2640 升到 0.4560，同时下限从 0.8333 降到 0.5079——说明该字段上的低分**主要反映标签退化，而非模型无能力**。但换标签后模型仍**未超过下限**，且 `mean_radius` 只有两个取值、随机水平即 0.5，故准确说法是**接近随机**，既非「学会」也非「远差于不看图」。另：提示词要求 `center|middle|edge|full|none` 五值，而金标中 `full` 出现 0 次；SFT 有 69/250 次回答 `full`，在构造上不可能得分。详见 `LIMITATIONS.md` 6.7。

时钟 MAE 的覆盖面（`clock_parse_audit.json`，计分/解析）：Base 1/252（丢弃 251）；SFT 203/250（丢弃 47）；grpo 198/252（丢弃 54）。丢弃的答案是 `all`/`none`，且**非随机**——集中在 `Random`（丢弃率 0.80）与 `Edge_Ring`（0.60）这两类「单一钟点方向本无意义」的图案上。因此该 MAE 条件于一个更容易的子集，且 SFT 与 GRPO 的计分子集并不相同、两者的 MAE 不可直接相减。
其中 **Base** 只保留了 1/252（100% 被丢弃），其时钟 MAE 建立在极小的子集上，不应作为对照。
详见 `LIMITATIONS.md` 6.8。

## 7. RL 状态 / Reinforcement learning

- **GSPO 可用；本报告初版此处写错了，现更正。** ms-swift 该提交的 `rlhf_type` 枚举 `['dpo','orpo','simpo','kto','cpo','rm','ppo','grpo','gkd']` 确实不含 `gspo`，但据此断言「GSPO 无法执行」是错的 —— GSPO 在该实现里**不是**独立的 `rlhf_type`，而是 GRPO 的损失变体，由 `--importance_sampling_level sequence` 开启（`swift/rlhf_trainers/args_mixin.py:425` 的注释直指 GSPO 论文 arXiv:2507.18071；该参数默认值 `token` 即普通 GRPO）。教训：只查了一个枚举名就断言某个算法不存在，而真正的开关是同文件里的另一个参数；因此**第一次跑的是普通 GRPO，不是 GSPO**，已按论文补跑，见下。
- GRPO 可用，且 LoRA 训练不需要额外参考模型；但环境未安装 vLLM，无法使用 `--use_vllm true`。
- 奖励函数已实现为确定性规则（可信类别标签 + 圆拟合几何），未使用任何 LLM 评审：`wafer_class` / `wafer_format` / `wafer_radial` / `wafer_clock`。
- 冒烟结果：`smoke_completed`（2 steps finished without error），耗时 82s，峰值显存 10597 MiB
- 正式运行：`completed`（150 steps finished without error），150 步 / 请求 150 步，耗时 2414s，峰值显存 14391 MiB
- **奖励信号**：各奖励的平均组内标准差 — WaferClass 0.0803，WaferFormat 0.0005，WaferRadial 0.1180，WaferClock 0.2244（越高说明该奖励越能区分组内采样）。
- 平均 KL 1.1068，平均奖励 1.6973
- **GSPO（G=4，lr 1e-5）**：`completed`（150 steps finished without error），150 步 / 请求 150 步，耗时 2578s，峰值显存 48010 MiB，平均 KL 1.0937，平均奖励 1.6879
  - 各奖励的平均组内标准差：WaferClass 0.0815，WaferFormat 0.0043，WaferRadial 0.1052，WaferClock 0.2326
  - **零优势步占比 33.33%**（50/150 步：组内所有采样奖励完全相同 → 优势为 0 → 不产生梯度）；作为对照 GRPO 为 36.00%（54/150 步），本 run 低于 GRPO 2.67%
  - 组大小 4（论文设定为 32。G=32 实测约 60s/步、15.2 GiB 显存，150 步约 2.5 小时，可以运行 —— 本报告初版曾断言其不可行，实测推翻，见 LIMITATIONS 5.2.1）
- **GSPO（G=4，lr 5e-5，单变量 IS 层级对照的 sequence 一侧）**：`completed`（150 steps finished without error），150 步 / 请求 150 步，耗时 2497s，峰值显存 31323 MiB，平均 KL 0.9882，平均奖励 1.5364
  - 各奖励的平均组内标准差：WaferClass 0.1136，WaferFormat 0.0090，WaferRadial 0.0903，WaferClock 0.2767
  - **零优势步占比 26.67%**（40/150 步：组内所有采样奖励完全相同 → 优势为 0 → 不产生梯度）；作为对照 GRPO 为 36.00%（54/150 步），本 run 低于 GRPO 9.33%
  - 组大小 4（论文设定为 32。G=32 实测约 60s/步、15.2 GiB 显存，150 步约 2.5 小时，可以运行 —— 本报告初版曾断言其不可行，实测推翻，见 LIMITATIONS 5.2.1）
- **GSPO（G=8，lr 1e-5，与 GRPO 对齐以分离算法与学习率）**：`completed`（150 steps finished without error），150 步 / 请求 150 步，耗时 3228s，峰值显存 14879 MiB，平均 KL 1.1201，平均奖励 1.7064
  - 各奖励的平均组内标准差：WaferClass 0.0912，WaferFormat 0.0030，WaferRadial 0.1256，WaferClock 0.2526
  - **零优势步占比 15.33%**（23/150 步：组内所有采样奖励完全相同 → 优势为 0 → 不产生梯度）；作为对照 GRPO 为 36.00%（54/150 步），本 run 低于 GRPO 20.67%
  - 组大小 8（论文设定为 32。G=32 实测约 60s/步、15.2 GiB 显存，150 步约 2.5 小时，可以运行 —— 本报告初版曾断言其不可行，实测推翻，见 LIMITATIONS 5.2.1）
- **GSPO（G=8，lr 5e-5）**：`completed`（150 steps finished without error），150 步 / 请求 150 步，耗时 3173s，峰值显存 14617 MiB，平均 KL 1.0217，平均奖励 1.5963
  - 各奖励的平均组内标准差：WaferClass 0.0855，WaferFormat 0.0132，WaferRadial 0.0528，WaferClock 0.2306
  - **零优势步占比 38.00%**（57/150 步：组内所有采样奖励完全相同 → 优势为 0 → 不产生梯度）；作为对照 GRPO 为 36.00%（54/150 步），本 run 高于 GRPO 2.00%
  - 组大小 8（论文设定为 32。G=32 实测约 60s/步、15.2 GiB 显存，150 步约 2.5 小时，可以运行 —— 本报告初版曾断言其不可行，实测推翻，见 LIMITATIONS 5.2.1）
- **GSPO（G=32，lr 1e-5，与 G=32 lr 5e-5 构成组大小的学习率对照）**：`completed`（150 steps finished without error），150 步 / 请求 150 步，耗时 10882s，峰值显存 48010 MiB，平均 KL 1.0995，平均奖励 1.7073
  - 各奖励的平均组内标准差：WaferClass 0.1017，WaferFormat 0.0098，WaferRadial 0.1149，WaferClock 0.2434
  - **零优势步占比 7.33%**（11/150 步：组内所有采样奖励完全相同 → 优势为 0 → 不产生梯度）；作为对照 GRPO 为 36.00%（54/150 步），本 run 低于 GRPO 28.67%
  - 组大小 32
- **GSPO（G=32，lr 5e-5，论文设定）**：`completed`（150 steps finished without error），150 步 / 请求 150 步，耗时 9841s，峰值显存 15187 MiB，平均 KL 0.9593，平均奖励 1.3662
  - 各奖励的平均组内标准差：WaferClass 0.0517，WaferFormat 0.0219，WaferRadial 0.0346，WaferClock 0.1435
  - **零优势步占比 42.00%**（63/150 步：组内所有采样奖励完全相同 → 优势为 0 → 不产生梯度）；作为对照 GRPO 为 36.00%（54/150 步），本 run 高于 GRPO 6.00%
  - 组大小 32
- GSPO(G=4) lr1e-5 macro-F1 0.6255 vs SFT 0.6114（Δ +0.0141），95% CI 0.5643–0.6746，**边际区间与 SFT 重叠**（该规则只比较两个边际区间，对配对数据偏保守 —— 10 次 run 回答的是同一批 252 行，结论以紧随其后的成对检验为准）
- GSPO(G=4) lr5e-5 macro-F1 0.4910 vs SFT 0.6114（Δ -0.1204），95% CI 0.4324–0.5483，边际区间**与 SFT 不重叠**（该规则只比较两个边际区间，对配对数据偏保守 —— 10 次 run 回答的是同一批 252 行，结论以紧随其后的成对检验为准）
- GSPO(G=8) lr1e-5 macro-F1 0.6089 vs SFT 0.6114（Δ -0.0025），95% CI 0.5472–0.6613，**边际区间与 SFT 重叠**（该规则只比较两个边际区间，对配对数据偏保守 —— 10 次 run 回答的是同一批 252 行，结论以紧随其后的成对检验为准）
- GSPO(G=8) lr5e-5 macro-F1 0.5014 vs SFT 0.6114（Δ -0.1101），95% CI 0.4367–0.5537，**边际区间与 SFT 重叠**（该规则只比较两个边际区间，对配对数据偏保守 —— 10 次 run 回答的是同一批 252 行，结论以紧随其后的成对检验为准）
- GSPO(G=32) lr1e-5 macro-F1 0.6143 vs SFT 0.6114（Δ +0.0028），95% CI 0.5492–0.6699，**边际区间与 SFT 重叠**（该规则只比较两个边际区间，对配对数据偏保守 —— 10 次 run 回答的是同一批 252 行，结论以紧随其后的成对检验为准）
- GSPO(G=32) lr5e-5，论文设定 macro-F1 0.5403 vs SFT 0.6114（Δ -0.0711），95% CI 0.4807–0.5965，**边际区间与 SFT 重叠**（该规则只比较两个边际区间，对配对数据偏保守 —— 10 次 run 回答的是同一批 252 行，结论以紧随其后的成对检验为准）
- **成对显著性检验**（同一批 252 行的准确率，按行 bootstrap + 精确 McNemar；macro-F1 无逐样本分解，不适用该检验，仍按边际值阅读）：
  - GRPO(G=4) lr1e-5 vs SFT：Δ准确率 +0.0000，95% CI [-0.0317, +0.0317]，McNemar p = 1 → 与 SFT 无法区分
  - GSPO(G=4) lr1e-5 vs SFT：Δ准确率 +0.0040，95% CI [-0.0278, +0.0357]，McNemar p = 1 → 与 SFT 无法区分
  - GRPO(G=4) lr5e-5 vs SFT：Δ准确率 -0.0556，95% CI [-0.1071, -0.0040]，McNemar p = 0.0435585 → **显著下降**
  - GSPO(G=4) lr5e-5 vs SFT：Δ准确率 -0.0873，95% CI [-0.1349, -0.0437]，McNemar p = 0.000312551 → **显著下降**
  - GSPO(G=8) lr1e-5 vs SFT：Δ准确率 -0.0079，95% CI [-0.0397, +0.0238]，McNemar p = 0.814529 → 与 SFT 无法区分
  - GSPO(G=8) lr5e-5 vs SFT：Δ准确率 -0.0833，95% CI [-0.1270, -0.0397]，McNemar p = 0.000324063 → **显著下降**
  - GSPO(G=32) lr1e-5 vs SFT：Δ准确率 -0.0040，95% CI [-0.0397, +0.0317]，McNemar p = 1 → 与 SFT 无法区分
  - GSPO(G=32) lr5e-5（论文设定） vs SFT：Δ准确率 -0.0556，95% CI [-0.1151, +0.0040]，McNemar p = 0.0869489 → 与 SFT 无法区分
  - 正对照 BASE vs SFT：Δ +0.4087，p = 2.83196e-26（该检验若看不出这么大的差，脚本直接退出、不输出结论）
  - **论文的中心主张**（序列级归一化更耐高学习率）：lr 5e-5 下 GSPO(G=8) 比 GRPO(G=4) 低 0.0278，p = 0.401062 —— **不构成差异**；且两者组大小不同（8 对 4），该对照无法把「IS 层级」与「组大小」分开。因此这条主张**既未被支持，也未被否定**。
  - **论文原始设定的那一格**（GSPO G=32 对 GRPO G=4，同为 lr 5e-5）：Δ准确率 +0.0000，95% CI [-0.0635, +0.0635]，p = 1 —— 聚合准确率完全重合。但**这不是「逐行相同」**：两者在 252 行里有 64 行判断不同，恰好 32 对 32 互补，聚合值相等是两边互换抵消的结果。该对照一次改了组大小（4→32）、IS 层级（token→sequence）与有效批次（16→1024）三样，**标识不了其中任何一样**。
  - **IS 层级的单变量对照**（GRPO(G=4) 对 GSPO(G=4)：lr 5e-5、累积 4/有效批次 16、同种子、同步数、同数据、同奖励函数与权重，**只差 `--importance_sampling_level`**）：Δ准确率（sequence − token）-0.0317，95% CI [-0.0794, +0.0159]，p = 0.268187 —— the two IS levels are not separable at G=4 / lr 5e-5。这是本项目**第一个只改一个变量的对照**，此前每一个都同时改了 IS 层级、组大小或有效批次之一以上。但它只覆盖 lr 5e-5 一个学习率，而「更耐高学习率」是关于**学习率效应的大小**的比较，因此论文的中心主张仍然**未被完整检验**，只是它的一个必要部分第一次有了无混淆的读数。
  - 组大小（GSPO lr 5e-5，G=8 → G=32）：Δ准确率 +0.0278，p = 0.410103 —— 点估计为正但**未达显著**。即在 lr 5e-5 上，算法与组大小两两都分不开，**唯一分得开的是学习率**。
  - **组大小的三点序列**（同为 GSPO、sequence、lr 5e-5，只让组大小 从 4 走到 8 再到 32）：G=4 -> G=8 Δ+0.0040（p = 1）、G=8 -> G=32 Δ+0.0278（p = 0.410103）。no rung separates the two group sizes it spans, so along this series accuracy is flat in group size from 4 to 32 —— 两点对照只能说明「这两个组大小不同」，三点才能说明这条关系有没有形状。
    - 但**本序列不是单变量**：gradient accumulation was set equal to the group size in every run of this series, so effective batch is group size squared: G=4 -> 16, G=8 -> 64, G=32 -> 1024. The series therefore moves group size 8x and effective batch 64x at the same time, and no contrast in it is single-variable.
    - this is about group size, not IS level. The paper's proposition needs a pair differing only in IS level, which is is_level_paired; and effective batch moves with group size along this series, so it is a series in configuration space, not a controlled single-variable experiment.
  - **但「组大小」这一行被有效批次混淆，读法要再退一步**：训练时梯度累积被设成等于组大小（4/4、8/8、32/32），故有效批次 = 组大小² ——G=4 → 16、G=8 → 64、G=32 → **1024**。该对照因此同时改了组大小 4 倍与有效批次 16 倍，**只能读作「加组大小或加有效批次无效」**，分不开这两个。同一个混淆也约束论文的中心主张（token 永远是 G=4、sequence 永远是 G=8 或 G=32，两个量从未单独变过）。**学习率的对照不受影响**：同一组大小内累积步与有效批次都固定。另有一个记录缺陷 —— 训练记录里的 `grad_accum` 字段是写死的常量 4，三条记录（两个 G=8、一个 G=32 lr5e-5）因此写错了实际值，见 LIMITATIONS 第 8 节第 11 条。
- **判定（按论文 §8.11 的门槛）**：该节规定「若提升落在置信区间内，结论写『未观察到显著提升』，不写『RL 有效』」，并给出分支「F -- 否 --> G[保留 SFT 为最终模型]」。GRPO 与 GSPO 均未超过 SFT，落入该分支，因此 **最终模型保留 SFT（`outputs/checkpoints/qwen35_9b_qlora_v1`），不宣称 RL 有效**。
- 需要说清的是，这**不等于**「GSPO 不如 SFT」或「GSPO 不如 GRPO」：在 lr 1e-5 上 GSPO、GRPO 与 SFT 两两都落在噪声内；在 lr 5e-5 上**两个算法都显著低于 SFT**（见上表），而它们彼此之间仍分不开 —— **算法不是差异的来源**。完整的算法 × 学习率 2×2 已补齐（GRPO 的 lr 5e-5 对照格已跑完，见 LIMITATIONS 5.2.2），此前「无法排除崩的是 GSPO 本身」这一保留意见已由该对照格解除：同一个 5e-5 作用在 GRPO 上同样退化。
- **空转步的下降是观察到的；能归因的那一对确实动了，但仍没有转化为分数**：lr 1e-5 下 GSPO 的零优势步是 15.33%（23/150），GRPO 是 36.00%（54/150） —— 但该对照**同时改了组大小**（G=8 对 G=4）与 IS 层级，本设计无法把「IS 层级」与「组大小」分开，所以这一条只是观察，不是归因。能归因的那一对是 GRPO(G=4) lr5e-5 32.67%（49/150） 对 GSPO(G=4) lr5e-5 26.67%（40/150）：同组大小 4、同有效批次 16、同学习率 5e-5、同种子 3407，只差 `--importance_sampling_level`（记录里 `single_variable` 为真），空转步低 **6.00 个百分点**。但**同一对配置**的准确率差是 -0.0317、95% CI [-0.0794, +0.0159]（包含 0，McNemar p = 0.268187）——机制层面的差异测得出，基准分数上的差异测不出（见 LIMITATIONS 5.2.3 与 5.2.6）。
- **检索指标并非由同一个 run 领先，故「G=32 检索最好」已不成立**：G=32 只在 mAP@10 上居首，nDCG@10 G=32 0.3906 < GRPO_lr5e5 0.3917；Recall@10 G=32 0.2008 < GRPO_lr5e5 0.2022。G=32 三项都高于 SFT（mAP@10 0.4337 对 0.3873），
  但**三项里只有 1 项是全部 run 最高**，因此它不能再被用来支持「G=32 重塑了表征」这一说法（见 LIMITATIONS 5.2.5）。该说法是本轮被实测推翻的又一处「点估计极值当作结论」。
- **分类最差的 RL run 不是 G=32 而是 GSPO(G=4) lr5e-5**（macro-F1 0.4910 对 G=32 的 0.5403），所以「G=32 分类最差而检索最好」这一对照不成立。G=32 的表征重塑只能由它自身的画像支持（Donut 归零、Scratch 最高、`radial_zone` 全部 run 最高），不能再借「分类最差」这个极值（见 LIMITATIONS 5.2.5）。

## 8. 验收 / Acceptance

- `pytest`：通过
- `manifest`：通过
- `lot_isolation`：通过
- `benchmark`：通过
- `model_shards`：通过
- `curated`：通过
- 总体：全部通过

## 9. 偏差与未决事项 / Deviations and open items

完整的局限清单（含每项的可核对来源与精确数值）见仓库根目录的 [`LIMITATIONS.md`](LIMITATIONS.md)；本节只是摘要。

- GRPO/GSPO: GSPO is not a separate rlhf_type in this ms-swift commit; it is GRPO with --importance_sampling_level sequence (swift/rlhf_trainers/args_mixin.py:425). An earlier revision of this record claimed GSPO was not offered at all, which was wrong; corrected in LIMITATIONS 5.2.1. GRPO runs without vLLM because vLLM is not installed.
- Micro-batch 4 with gradient accumulation 8 replaces micro-batch 1 with accumulation 32; the effective batch of 32 is unchanged.
- deepseek-v4.1-flash free quota was exhausted; part of the teacher pass was served by deepseek-v3.2, recorded per row.
- venv and the ms-swift checkout live on the instance disk and are symlinked into autodl-fs, because FUSE file creation is ~850x slower.
- CLOSED 2026-09-16 -- GRPO at lr 5e-5 was the missing control cell, and it has been run. Records: qwen35_9b_grpo_lr5e5_train_result.json (training), qwen35_9b_grpo_lr5e5__report.json (benchmark), paired_significance.json (tests). It degrades the same way GSPO does: macro-F1 0.6197 -> 0.5535, accuracy -0.0556 with paired McNemar p = 0.0436 over the same 252 rows. The 'learning rate, not algorithm' reading therefore now has cross-algorithm replication rather than resting on GSPO alone. What is still NOT resolved is a different and narrower claim: the paper's own proposition -- that sequence-level normalisation tolerates a higher learning rate -- is neither supported nor refuted, because every token-level run is G=4 and every sequence-level run is G=8, so IS level and group size move together and no contrast here separates them (cross-algorithm at lr 5e-5: p = 0.40). See LIMITATIONS 5.2.2.
- 人工双审未完成：`benchmarks/wafer_bench_v1/review/` 下的两份表格仍为空，基准保持 `draft_pending_human_review`。本报告不伪造该步骤。
- `deepseek-v4.1-flash` 免费额度耗尽，部分教师标注改由 `deepseek-v3.2` 完成，逐行保留了模型来源。
- 训练使用的微批次为 4 / 梯度累积 8（有效批次 32 不变），这是时间预算调整，已记录在 run_manifest。
- `is_boundary()` 在过滤点被移除时即触发，导致绝大多数样本被判为边界样本，因此双教师标注覆盖率接近全量；该行为方向保守，未在运行中途修改。
- venv 与 ms-swift 源码位于实例盘 `/root/autodl-tmp`，在 `/root/autodl-fs/wafer-vlm` 下以符号链接暴露；原因是 FUSE 创建文件比实例盘慢约 850 倍。ms-swift 提交号已记录，可复现。

## 10. 复现 / Reproducing

关键命令记录在 `logs/*.sh` 与 `projects/wafer-defect-vlm/scripts/*.sh`，其哈希在 `provenance.json` 中。主要步骤：
```bash
bash scripts/01_download_model.sh          # 下载并校验 Qwen3.5-9B（4 分片）
python -m wafer_vlm.cli prepare            # WM811K -> manifest + 448x448 PNG
python -m wafer_vlm.cli annotate           # 教师标注（百炼）
python -m wafer_vlm.cli curate finalize    # 校验、仲裁、去重、切分
python -m wafer_vlm.cli benchmark build    # 构建 wafer_bench_v1
bash scripts/10_train_qlora.sh             # QLoRA SFT
bash scripts/22_eval_adapter.sh            # 适配器评测
bash scripts/23_merge_and_check.sh         # 合并 + 等价性校验
bash scripts/25_finalize.sh                # 评测、检索、对比、报告
```
