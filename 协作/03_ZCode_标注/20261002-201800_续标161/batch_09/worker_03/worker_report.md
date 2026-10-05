# worker_report — batch_09 / worker_03（20261002-201800 续标161）

## 模型与套餐声明

本子 agent 由父会话经 ZCode 客户端原生 Agent 工具创建，运行于父会话同一模型配置（客户端声明 GLM-5.3-Flash / Start Plan）；后端实际模型标识：未提供。

## 上下文披露

- 全新启动（不继承父会话对话），仅收到入口提示词与规则书所列输入（分片清单、标注提示词、原格式检查器、类别别名规范化器）。
- 分片内按顺序逐张处理，后图上下文含前图与前答，不是相互独立的 API 调用。
- 客户端层自动注入内容无法从子 agent 内枚举，记：未提供。
- 四个固定输入开工前逐一核验 sha256，全部匹配：
  - 分片清单 `16bdf64f…5d7b0`、提示词 `7df73031…954a`、检查器 `794a254e…ec55`、规范化器 `01e02d58…8f2c`。
- 每张恰好一次视觉读取（Read 原位置 PNG）；PNG sha256 与分片逐张比对，全部一致。

## 输入指纹核验

| sample_id | PNG sha256（分片期望=实测） |
|---|---|
| wafer_00047316_011 | 10d86fc9…29c9 一致 |
| wafer_00047327_010 | 8d043dba…feb4 一致 |
| wafer_00047395_023 | 1b45ed2b…7b48 一致 |
| wafer_00047486_001 | 58ae65b0…9287 一致 |
| wafer_00047490_005 | ec04f579…1a2b 一致 |
| wafer_00047523_009 | 7772fe47…afef 一致 |

## 逐张状态表

| sample_id | checker1 | normalizer | checker2 | status | pending_review | t0 | t1 | 耗时(ms) |
|---|---|---|---|---|---|---|---|---|
| wafer_00047316_011 | 0 | 0 | 0 | raw_pass | false | 2026-10-03 01:22:02 +0800 | 2026-10-03 01:23:58 +0800 | 115453 |
| wafer_00047327_010 | 0 | 0 | 0 | raw_pass | false | 2026-10-03 01:24:20 +0800 | 2026-10-03 01:25:26 +0800 | 66534 |
| wafer_00047395_023 | 0 | 0 | 0 | raw_pass | false | 2026-10-03 01:25:48 +0800 | 2026-10-03 01:26:40 +0800 | 52262 |
| wafer_00047486_001 | 0 | 0 | 0 | raw_pass | false | 2026-10-03 01:26:45 +0800 | 2026-10-03 01:27:10 +0800 | 24502 |
| wafer_00047490_005 | 0 | 0 | 0 | raw_pass | false | 2026-10-03 01:27:15 +0800 | 2026-10-03 01:28:21 +0800 | 65830 |
| wafer_00047523_009 | 0 | 0 | 0 | raw_pass | false | 2026-10-03 01:28:26 +0800 | 2026-10-03 01:28:51 +0800 | 24404 |

全部 6 张的 normalization 报告 `changes` 与 `content_review_warnings` 均为空（alias_changes=0）。

补充观察（非警告、非失败）：wafer_00047316_011 的 checker1/checker2 detail 中记录 `clock_expressions_stripped.caption_zh: ["9点钟"]`（检查器自报的剥离明细，退出码仍为 0，failures 为空）；其余 5 张无任何剥离明细。

## 计数

- 尝试：6
- 原始通过（raw_pass）：6
- 副本补救通过（raw_fail_canonical_pass）：0
- 隔离（raw_fail / isolated_bad_json）：0
- unknown 类别作答：0
- 未执行（not_run）：0

## 待复核清单

- pending_review 项：无（全部 6 张 content_review_warnings 为空）。

## 停止原因与标记文件

- 无。全程未触发任何系统停止条件（指纹一致、视觉读取正常、无写入冲突、无同伴 STOPPED 文件、未请求新增预算），未写入任何 STOPPED_* 文件。

## 产物清单（本目录）

每张 4 件 + 日志：`<sample_id>_raw.json`、`<sample_id>_canonical.json`、`<sample_id>_normalization.json`、`<sample_id>_check.log`；另有 `worker_log.jsonl`（6 行）。

## 声明

格式通过 ≠ 事实正确。本轮检查器与规范化器仅验证 JSON 格式、字段与类别字符串规范化，本报告中的全部答案未作人工复核；类别与形态描述均为模型标注，不冒称人工 gold。
