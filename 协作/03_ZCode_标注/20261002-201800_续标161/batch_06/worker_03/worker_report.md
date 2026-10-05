# worker_report — batch_06 / worker_03（续标161 · 20261002-201800）

## 模型与套餐声明

本子 agent 由父会话经 ZCode 客户端原生 Agent 工具创建，运行于父会话同一模型配置（客户端声明 GLM-5.3-Flash / Start Plan）；后端实际模型标识：未提供。

## 上下文披露

- 全新启动，不继承父会话对话；仅收到入口提示词与本规则书所列输入（分片清单、标注提示词、原格式检查器、类别别名规范化器）。
- 分片内按 item_115→item_120 顺序处理；后图上下文含前图与前答，不是相互独立的 API 调用。
- 客户端层自动注入内容无法从子 agent 内枚举，记未提供。
- 四个固定输入开工前逐一核验 sha256，全部匹配（分片 a998ab56…89912、提示词 7df73031…954a、检查器 794a254e…ec55、规范化器 01e02d58…8f2c）；分片 6 行与清单一致。
- 同伴 worker_01/02/03 目录仅用 `test -e`/`ls` 检查 STOPPED* 存在性，全程未发现，未读取其内容。
- 每张恰好一次视觉读取（Read 原 PNG），未搬运、未拼接、未用其他模型或几何程序预读；未重试、未补写、未改答。
- `D:/python/python.exe` 存在，未发生 python 替换。

## 逐张状态表

| sample_id | checker1 | normalizer | checker2 | status | pending_review | t0 (t1) | 耗时 ms |
|---|---|---|---|---|---|---|---|
| wafer_00042359_002 | 0 | 0 | 0 | raw_pass | false | 00:48:16 (00:48:52 +0800) | 35833 |
| wafer_00042367_004 | 0 | 0 | 0 | raw_pass | false | 00:49:01 (00:49:33 +0800) | 32341 |
| wafer_00042397_008 | 0 | 0 | 0 | raw_pass | false | 00:49:42 (00:50:09 +0800) | 27389 |
| wafer_00042702_020 | 0 | 0 | 0 | raw_pass | false | 00:50:16 (00:50:44 +0800) | 27762 |
| wafer_00042724_003 | 8 | 0 | 8 | raw_fail（隔离） | false | 00:50:51 (00:51:35 +0800) | 43872 |
| wafer_00042865_024 | 8 | 0 | 8 | raw_fail（隔离） | false | 00:51:55 (00:52:44 +0800) | 48111 |

全部 6 张 PNG 指纹与分片逐一比对一致，无 STOPPED_FINGERPRINT。

## 计数

- 尝试：6
- 原始通过（raw_pass）：4
- 副本补救通过（raw_fail_canonical_pass）：0
- 隔离（raw_fail / isolated_bad_json）：2
- unknown 作答：0
- 未执行（not_run）：0
- 别名规范化改动（alias_changes 总和）：0

## 隔离项记录（保留原件与全部退出码，未改写、未重试）

两例均为检查器退出码 8：morphology/caption_zh 在剔除钟点表达式后仍含数字（源自我对钟点区间的写法"约7至11点钟方向"/"约5至7点钟方向"，其中"7点钟方向"被剥离后区间起点的阿拉伯数字仍残留）。原答与 canonical 副本均保留，check.log 三段齐全。

- wafer_00042724_003：`morphology 剔除钟点后仍含数字 ['7']`；`caption_zh 剔除钟点后仍含数字 ['7']`（clock_expressions_stripped: "11点钟方向"）
- wafer_00042865_024：`morphology 剔除钟点后仍含数字 ['5']`；`caption_zh 剔除钟点后仍含数字 ['5']`（clock_expressions_stripped: "7点钟方向"）

## 待复核清单

- pending_review 项：无（全部 6 张 normalization.json 的 content_review_warnings 均为空）。

## 停止原因与标记文件

- 无系统停止条件触发，未写任何 STOPPED_*.txt；分片 6 张全部按流程执行完毕。

## 声明

格式通过（checker 退出码 0）≠ 事实正确；本报告与全部 raw/canonical 答案未作人工复核。模型标注/模型复核不冒称人工 gold。
