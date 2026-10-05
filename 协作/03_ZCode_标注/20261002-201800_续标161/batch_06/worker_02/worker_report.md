# worker_report — batch_06 / worker_02（续标161 · 20261002-201800）

## 模型与套餐声明

本子 agent 由父会话经 ZCode 客户端原生 Agent 工具创建，运行于父会话同一模型配置（客户端声明 GLM-5.3-Flash / Start Plan）；后端实际模型标识：未提供。

## 上下文披露

全新启动（不继承父会话对话），仅收到入口提示词与 worker_instructions.md 所列输入：分片清单、标注提示词 prompt_image_only_v2_20261002.txt、检查器 check_answer_v3.py、规范化器 normalize_label_alias.py。四个输入的 sha256 均与规则书期望值一致后才开工。分片内按清单顺序处理，后图上下文含前图与前答，不是相互独立的 API 调用；客户端层自动注入内容无法从子 agent 内枚举，记未提供。

每张图恰好一次视觉读取（Read 工具读取原位置 PNG），图片指纹与分片期望值逐一比对通过；未搬图、未拼图、未用其他模型或几何程序预读。同伴 worker_01/02/03 目录仅以 `test -e` 检查 STOPPED* 存在性（每张开工前），未读取任何内容；本轮历次检查均为 NO_STOPPED。

## 逐张状态表

| sample_id | checker1 | normalizer | checker2 | status | pending_review | t0 | t1 | 耗时 ms |
|---|---|---|---|---|---|---|---|---|
| wafer_00041974_012 | 0 | 0 | 0 | raw_pass | false | 2026-10-03 00:52:38 +0800 | 2026-10-03 00:55:27 +0800 | 168898 |
| wafer_00042079_006 | 0 | 0 | 0 | raw_pass | false | 2026-10-03 00:55:36 +0800 | 2026-10-03 00:56:20 +0800 | 43181 |
| wafer_00042086_017 | 0 | 0 | 0 | raw_pass | false | 2026-10-03 00:56:27 +0800 | 2026-10-03 00:57:00 +0800 | 33753 |
| wafer_00042225_007 | 0 | 0 | 0 | raw_pass | false | 2026-10-03 00:57:07 +0800 | 2026-10-03 00:57:43 +0800 | 35577 |
| wafer_00042243_001 | 0 | 0 | 0 | raw_pass | false | 2026-10-03 00:57:49 +0800 | 2026-10-03 00:58:28 +0800 | 39097 |
| wafer_00042262_021 | 0 | 0 | 0 | raw_pass | false | 2026-10-03 00:58:35 +0800 | 2026-10-03 00:59:13 +0800 | 38197 |
| wafer_00042284_025 | 0 | 0 | 0 | raw_pass | false | 2026-10-03 00:59:19 +0800 | 2026-10-03 00:59:56 +0800 | 37239 |

数据出处：以上各行的 sha256、退出码与时间均见本目录 worker_log.jsonl 对应行；每张的退出码与输出原文见各 `<sample_id>_check.log`。

## 计数

- 尝试：7 / 7（分片全部执行，无跳过）
- 原始通过（raw_pass）：7
- 副本补救通过（raw_fail_canonical_pass）：0
- 隔离（raw_fail / isolated_bad_json）：0
- unknown 类别：0
- 未执行（not_run）：0
- 别名规范化改动（alias_changes 总数）：0
- 首轮视觉读取失败 / 指纹漂移 / 写入冲突：0

## 待复核清单

无（全部 7 张 normalization.json 的 content_review_warnings 为空，pending_review 均为 false）。

## 停止原因与标记文件

无。未写任何 STOPPED_*.txt。

## 声明

- 格式通过不等于事实正确：checker1/checker2 退出码 0 仅表示七字段 JSON 的结构、枚举、extent_r=null 与无未验证数值等格式检查通过；全部答案未作人工复核，类别与描述的正确性待后续内容核验。
- 本轮无类别别名命中（Near-full/Edge-Loc/Edge-Ring 三个精确别名均未出现），raw 与 canonical 内容语义一致。
- extent_r 全部为 null；未写任何未经验证的数值尺寸/比例/覆盖率；未推测工艺根因。
