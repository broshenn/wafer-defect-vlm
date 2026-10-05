# worker_report · batch_07 / worker_03（20261002-201800 续标161）

## 模型与套餐声明

本子 agent 由父会话经 ZCode 客户端原生 Agent 工具创建，运行于父会话同一模型配置（客户端声明 GLM-5.3-Flash / Start Plan）；后端实际模型标识：未提供。

## 上下文披露

全新启动（不继承父会话对话），仅收到入口提示词与规则书所列输入（分片清单、标注提示词、check_answer_v3.py、normalize_label_alias.py）。四个输入文件 sha256 均与规则书期望一致，开工前逐一核验通过。分片内按清单顺序处理；后图上下文包含前图与前答，不是相互独立的 API 调用。客户端层自动注入内容无法从子 agent 内枚举，记未提供。每张恰好一次视觉读取（Read 原位置 PNG），未搬图、未拼图、未用其他模型或几何程序预读。同伴 STOPPED* 每张开工前用 `test -e` 查过存在性，未读取同伴目录任何内容。

## 逐张状态表

| sample_id | checker1 | normalizer | checker2 | status | pending_review | t0 | t1 | 耗时(ms) |
|---|---|---|---|---|---|---|---|---|
| wafer_00044180_010 | 0 | 0 | 0 | raw_pass | false | 2026-10-03 01:02:24 +0800 | 2026-10-03 01:03:07 +0800 | 42900 |
| wafer_00044187_005 | 0 | 0 | 0 | raw_pass | false | 2026-10-03 01:03:22 +0800 | 2026-10-03 01:03:57 +0800 | 34327 |
| wafer_00044190_015 | 0 | 0 | 0 | raw_pass | false | 2026-10-03 01:04:08 +0800 | 2026-10-03 01:04:25 +0800 | 17231 |
| wafer_00044193_023 | 0 | 0 | 0 | raw_pass | false | 2026-10-03 01:04:39 +0800 | 2026-10-03 01:04:58 +0800 | 19142 |
| wafer_00044298_014 | 0 | 0 | 0 | raw_pass | false | 2026-10-03 01:05:12 +0800 | 2026-10-03 01:05:35 +0800 | 23381 |
| wafer_00044444_010 | 0 | 0 | 0 | raw_pass | false | 2026-10-03 01:05:46 +0800 | 2026-10-03 01:06:16 +0800 | 30182 |

全部 6 张 PNG 的 sha256 与分片清单期望值一致；图片指纹无漂移。

## 计数

- 尝试：6
- 原始通过（raw_pass）：6
- 副本补救通过（raw_fail_canonical_pass）：0
- 隔离（raw_fail / isolated_bad_json）：0
- unknown：0
- 未执行（not_run）：0

别名规范化：6 张的 normalization.json 中 changes 均为空数组（alias_changes=0），canonical 与 raw 仅哈希不同因重序列化，无类别字符串改动。

## 待复核清单

无 pending_review 项：6 张的 content_review_warnings 均为空。

## 停止原因

无。未写任何 STOPPED_*.txt（同伴目录亦无 STOPPED 文件）。

## 声明

格式通过（checker1/checker2 退出码 0）不等于事实正确；本轮类别、形态与位置描述均为模型标注（GLM-5.3-Flash，客户端声明身份），未作人工复核，不构成人工 gold。答案中 extent_r 一律为 null，未写未经验证的数值尺寸/比例/覆盖率，未推测工艺根因。

## 产物清单

每张 5 件：`<sample_id>_raw.json`（原答，保存后未改写）、`<sample_id>_canonical.json`（规范化副本）、`<sample_id>_normalization.json`、`<sample_id>_check.log`（checker1/normalizer/checker2 退出码与原始终端输出）、以及本目录 `worker_log.jsonl`（6 行逐张流水）。
