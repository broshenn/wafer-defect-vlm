# worker_report — batch_02 / worker_01（续标161 · 20261002-201800）

## 模型与套餐声明

本子 agent 由父会话经 ZCode 客户端原生 Agent 工具创建，运行于父会话同一模型配置（客户端声明 GLM-5.3-Flash / Start Plan）；后端实际模型标识：未提供。

## 上下文披露

- 全新启动（不继承父会话对话），仅收到入口提示词与规则书所列输入（分片清单、标注提示词、原格式检查器、类别别名规范化器）。
- 分片内按清单顺序处理；后图上下文含前图与前答，不是相互独立的 API 调用。
- 客户端层自动注入内容无法从子 agent 内枚举，记未提供。
- 每张图恰好一次视觉读取（Read 工具读原位置 PNG），未搬图、未拼图、未用其他模型或几何程序预读。

## 输入核验

开工时四个固定输入 sha256 全部与规则书期望值一致：

| 输入 | 核验结果 |
|---|---|
| 分片清单 shards/batch_02__worker_01.jsonl | 匹配（44e91768…82f24），7 行 |
| prompt_image_only_v2_20261002.txt | 匹配（7df73031…1954a） |
| check_answer_v3.py | 匹配（794a254e…3ec55） |
| normalize_label_alias.py | 匹配（01e02d58…28f2c） |

python 使用 `D:/python/python.exe`（存在，未替换）。每张 PNG 指纹均与清单一致，无指纹漂移。

## 逐张状态表

| sample_id | checker1 | normalizer | checker2 | status | pending_review | alias_changes | t0 | t1 | 耗时(ms) |
|---|---|---|---|---|---|---|---|---|---|
| wafer_00000025_003 | 0 | 0 | 0 | raw_pass | false | 0 | 2026-10-02 20:25:01 +0800 | 2026-10-02 20:26:09 +0800 | 67935 |
| wafer_00000042_024 | 0 | 0 | 0 | raw_pass | false | 0 | 2026-10-02 20:26:21 +0800 | 2026-10-02 20:27:09 +0800 | 48283 |
| wafer_00000184_008 | 0 | 0 | 0 | raw_pass | false | 0 | 2026-10-02 20:27:19 +0800 | 2026-10-02 20:28:21 +0800 | 62353 |
| wafer_00000216_023 | 0 | 0 | 0 | raw_pass | false | 0 | 2026-10-02 20:28:27 +0800 | 2026-10-02 20:29:13 +0800 | 45386 |
| wafer_00001419_023 | 0 | 0 | 0 | raw_pass | false | 0 | 2026-10-02 20:29:28 +0800 | 2026-10-02 20:30:12 +0800 | 43956 |
| wafer_00001470_008 | 0 | 0 | 0 | raw_pass | false | 0 | 2026-10-02 20:30:22 +0800 | 2026-10-02 20:32:06 +0800 | 104065 |
| wafer_00002074_020 | 0 | 0 | 0 | raw_pass | false | 0 | 2026-10-02 20:32:14 +0800 | 2026-10-02 20:33:06 +0800 | 52609 |

备注：wafer_00000042_024 的 checker detail 中 `clock_expressions_stripped` 列出 morphology/caption_zh 里的「12点钟方向」字样（信息性 detail，非 failure，退出码 0）；该图为顶部边缘致密红区，clock_direction 字段按提示词填写为「12点钟」。

## 计数

- 尝试：7（分片 7 行全部执行）
- 原始通过（raw_pass）：7
- 副本补救通过（raw_fail_canonical_pass）：0
- 隔离（raw_fail / isolated_bad_json）：0
- unknown 类别作答：0
- 未执行（not_run）：0

## 待复核清单（pending_review）

无。所有 7 张的 `content_review_warnings` 均为空，无警告原文需记录。

## 停止原因与标记文件

无系统停止条件触发，未写任何 STOPPED_*.txt。同伴 worker_01..03 目录的 STOPPED* 存在性检查在每张处理前执行（仅 test -e，未读取内容），全程均不存在。

## 声明

- 全部 7 张为「格式通过 + 副本规范化通过」；**格式通过不等于事实正确**，类别、形态、位置描述均未经人工复核，属模型标注产物，不得作为人工 gold。
- 原答文件（*_raw.json）保存后未做任何改写；canonical.json 与 normalization.json 由规范化器生成，未手工修改。
- 逐张退出码、stdout、stderr 原文见各 `*_check.log`；逐张结构化记录见 `worker_log.jsonl`。
