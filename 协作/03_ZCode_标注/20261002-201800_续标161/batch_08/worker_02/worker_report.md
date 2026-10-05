# worker_report — batch_08 / worker_02（2026-10-03）

## 模型与套餐声明

本子 agent 由父会话经 ZCode 客户端原生 Agent 工具创建，运行于父会话同一模型配置（客户端声明 GLM-5.3-Flash / Start Plan）；后端实际模型标识：未提供。

## 上下文披露

- 全新启动（不继承父会话对话），仅收到入口提示词与 worker_instructions.md 所列输入。
- 分片内按顺序处理，后图上下文含前图与前答，不是相互独立的 API 调用。
- 客户端层自动注入内容无法从子 agent 内枚举，记：未提供。

## 输入核验

四个固定输入 sha256 均与规则书期望值一致（分片清单 / 标注提示词 / 原检查器 / 类别别名规范化器），全部通过后才开工。

## 逐张状态表

| sample_id | checker1 | normalizer | checker2 | status | pending_review | 起止时间 | 耗时 |
|---|---|---|---|---|---|---|---|
| wafer_00045623_009 | 0 | 0 | 0 | raw_pass | false | 01:09:53–01:10:53 +0800 | 59697 ms |
| wafer_00045625_016 | 0 | 0 | 0 | raw_pass | false | 01:11:12–01:11:46 +0800 | 34035 ms |
| wafer_00045634_017 | 0 | 0 | 0 | raw_pass | false | 01:12:08–01:12:36 +0800 | 28404 ms |
| wafer_00045635_023 | 0 | 0 | 0 | raw_pass | false | 01:12:45–01:13:11 +0800 | 25673 ms |
| wafer_00045639_020 | 0 | 0 | 0 | raw_pass | false | 01:13:23–01:13:53 +0800 | 30476 ms |
| wafer_00045687_006 | 0 | 0 | 0 | raw_pass | false | 01:14:08–01:15:12 +0800 | 63679 ms |
| wafer_00045738_008 | 8 | 0 | 8 | raw_fail | false | 01:15:20–01:15:47 +0800 | 27531 ms |

每张恰好一次视觉读取；每张的 raw.json 保存后未做任何改写；PNG 指纹 7/7 与分片一致；同伴 STOPPED* 检查每张开 工前执行，均不存在。

## 计数

- 尝试：7
- 原始通过（raw_pass）：6
- 副本补救通过（raw_fail_canonical_pass）：0
- 隔离（格式未过，保留原件与全部退出码）：1（wafer_00045738_008）
- unknown 类别作答：0
- 未执行：0

## 隔离与失败详情

- **wafer_00045738_008**：checker1 退出码 8，失败信息：`morphology 剔除钟点后仍含数字 ['6']`（原文"底部边缘约6至7点钟处"，检查器剥离"7点钟"钟点表达后残留数字"6"）。normalizer 退出码 0 并已生成 canonical.json（changes 为空），副本 checker2 同为 8。原件与副本均保留，未重试、未改写。类别 Random 与其余字段本身未被判错。
- 参考性信息（非失败）：wafer_00045639_020 的 checker detail 中出现 `clock_expressions_stripped: {"morphology": ["12点钟"]}`，退出码仍为 0。

## 待复核清单

- pending_review 项：无（7 张 normalization.json 的 content_review_warnings 均为空）。
- 别名规范化（alias_changes）：7 张均为 0 次，无 Near-full/Edge-Loc/Edge-Ring 别名改写。

## 停止原因与标记文件

- 无系统停止条件触发；未写任何 STOPPED_*.txt；分片 7/7 全部执行完毕。

## 声明

格式通过 ≠ 事实正确。本报告与全部 raw/canonical 产物均为模型标注，未作人工复核；类别与描述的事实一致性待后续独立核验。
