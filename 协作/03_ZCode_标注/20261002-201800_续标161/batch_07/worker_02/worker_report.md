# worker_report — batch_07 / worker_02

- 日期：2026-10-03（本地时间）
- 分片清单：`shards/batch_07__worker_02.jsonl`（sha256 `8adc8fd4af79a304ac8fa378132962a96c854ee1ffafc5918b93cd73b5256da0`，7 行，核验一致）
- 规则书：`worker_instructions.md`（四个固定输入 sha256 开工前逐一核验，全部一致）

## 模型与套餐声明

本子 agent 由父会话经 ZCode 客户端原生 Agent 工具创建，运行于父会话同一模型配置（客户端声明 GLM-5.3-Flash / Start Plan）；后端实际模型标识：**未提供**。用量/扣量信息：未提供。

## 上下文披露

- 全新启动，不继承父会话对话；仅收到入口提示词、本规则书及规则书所列固定输入。
- 分片内按清单顺序处理；后图上下文包含前图与前答，**不是**相互独立的 API 调用。
- 客户端层自动注入的内容无法从子 agent 内枚举，记**未提供**。
- 每张图恰好一次视觉读取（Read 原位置 PNG），未搬图、未拼图、未用任何几何程序预读；raw.json 保存后未改写。

## 固定输入核验（sha256）

| 输入 | 期望=实测 |
|---|---|
| 分片清单 | 8adc8fd4…25da0 一致 |
| prompt_image_only_v2_20261002.txt | 7df73031…9954a 一致 |
| check_answer_v3.py | 794a254e…ec55 一致 |
| normalize_label_alias.py | 01e02d58…8f2c 一致 |

每张 PNG 指纹均与清单 image_sha256 一致（7/7 FINGERPRINT_OK）。

## 逐张状态表

| # | sample_id | checker1 | normalizer | checker2 | status | pending_review | t0 | t1 | 耗时(ms) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | wafer_00043831_010 | 0 | 0 | 0 | raw_pass | false | 01:04:08 | 01:04:36 | 27580 |
| 2 | wafer_00043836_023 | 0 | 0 | 0 | raw_pass | false | 01:04:56 | 01:05:19 | 22757 |
| 3 | wafer_00043866_025 | 0 | 0 | 0 | raw_pass | false | 01:05:32 | 01:06:02 | 30076 |
| 4 | wafer_00043944_013 | 0 | 0 | 0 | raw_pass | false | 01:06:16 | 01:06:45 | 28603 |
| 5 | wafer_00043973_022 | 0 | 0 | 0 | raw_pass | false | 01:07:04 | 01:07:24 | 20229 |
| 6 | wafer_00044063_004 | 0 | 0 | 0 | raw_pass | false | 01:07:37 | 01:08:00 | 23232 |
| 7 | wafer_00044178_021 | 0 | 0 | 0 | raw_pass | false | 01:08:12 | 01:08:34 | 21780 |

（时间均为 2026-10-03 +0800。每张耗时含作答与三轮检查。）

信息性备注（非失败、不影响退出码与 status）：wafer_00043944_013 的 checker 输出 detail 中出现 `clock_expressions_stripped: {"morphology": ["6点钟"]}`，`failures` 为空、退出码 0，原样保留在 `wafer_00043944_013_check.log`。

## 计数

- 尝试：7
- 原始通过（raw_pass）：7
- 副本补救通过（raw_fail_canonical_pass）：0
- 隔离（raw_fail / isolated_bad_json）：0
- unknown 类别作答：0（本分片 7 张均给出具体类别；unknown 是合法值，未出现不代表不可靠）
- 未执行（not_run）：0
- 别名规范化变更（alias_changes 合计）：0（7 张 changes 均为空数组）
- pending_review（content_review_warnings 非空）：0 张

## 待复核清单

无（7 张 normalization.json 的 content_review_warnings 均为空数组）。

## 停止原因与标记文件

无。未触发任何系统停止条件，未写任何 STOPPED_*.txt；同伴 STOPPED* 存在性检查（仅 `test -e`）每张开工前执行，均不存在。

## 声明

格式通过 ≠ 事实正确。以上所有"通过"仅指 check_answer_v3.py 的格式/字段检查退出码为 0 与规范化器无警告；答案内容（类别、形态、位置）未经人工复核，不构成人工 gold。模型标注与模型复核不得冒称人工审核。
