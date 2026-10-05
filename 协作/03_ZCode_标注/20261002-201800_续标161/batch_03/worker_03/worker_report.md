# worker_report — batch_03 / worker_03（续标161 · 20261002-201800）

## 模型与套餐声明

本子 agent 由父会话经 ZCode 客户端原生 Agent 工具创建，运行于父会话同一模型配置（客户端声明 GLM-5.3-Flash / Start Plan）；后端实际模型标识：未提供。

## 上下文披露

- 全新启动（不继承父会话对话），仅收到入口提示词与规则书所列四个固定输入；分片内按顺序逐张处理，后图上下文含前图与前答，不是相互独立的 API 调用；客户端层自动注入内容无法从子 agent 内枚举，记未提供。
- 输入核验：四个固定输入（分片清单 / 标注提示词 / check_answer_v3.py / normalize_label_alias.py）sha256 全部与规则书期望一致后才开工。
- 每张图恰一次视觉读取（Read 原位置 PNG），PNG sha256 与分片期望一致；T0 在视觉读取前记录。
- Python：使用 `D:/python/python.exe`（存在，未替换）。
- 输出目录内产物：每张 `*_raw.json` / `*_check.log` / `*_canonical.json` / `*_normalization.json`，外加 `worker_log.jsonl`。原答保存后未改写；canonical 由规范化器生成（alias_changes=0，副本与原答差异仅为 JSON 缩进格式）。

## 逐张状态表

| sample_id | checker1 | normalizer | checker2 | status | pending_review | t0 | t1 | 耗时(ms) |
|---|---|---|---|---|---|---|---|---|
| wafer_00014395_016 | 0 | 0 | 0 | raw_pass | false | 2026-10-02 20:35:23 +0800 | 2026-10-02 20:38:05 +0800 | 162208 |
| wafer_00015381_014 | 0 | 0 | 0 | raw_pass | false | 2026-10-02 20:38:22 +0800 | 2026-10-02 20:39:34 +0800 | 72148 |
| wafer_00015561_024 | 0 | 0 | 0 | raw_pass | false | 2026-10-02 20:39:48 +0800 | 2026-10-02 20:40:25 +0800 | 37174 |
| wafer_00015563_008 | 0 | 0 | 0 | raw_pass | false | 2026-10-02 20:40:37 +0800 | 2026-10-02 20:41:28 +0800 | 50611 |
| wafer_00015633_024 | 0 | 0 | 0 | raw_pass | false | 2026-10-02 20:41:34 +0800 | 2026-10-02 20:42:19 +0800 | 45085 |
| wafer_00015646_019 | 0 | 0 | 0 | raw_pass | false | 2026-10-02 20:42:36 +0800 | 2026-10-02 20:43:24 +0800 | 48616 |

累计净耗时约 415.8 s（duration_ms 合计 415842；含两次检查器、规范化器与哈希计算开销）。

## 计数

- 尝试：6
- 原始通过（raw_pass）：6
- 副本补救通过（raw_fail_canonical_pass）：0
- 隔离（raw_fail / isolated_bad_json）：0
- unknown 类别：0
- 未执行（not_run）：0

## 待复核清单

无（全部 6 张 pending_review=false，normalization 的 content_review_warnings 均为空）。

## 停止原因与标记文件

无。未发生任何系统停止条件；未写任何 STOPPED_* 文件。

## 声明

格式通过（检查器退出码 0）≠ 事实正确；本 worker 未作人工复核，全部答案为模型标注，类别与描述待人工/后续内容核验。
