# worker_report — batch_02 / worker_02（续标161 · 20261002-201800）

## 模型与套餐声明

本子 agent 由父会话经 ZCode 客户端原生 Agent 工具创建，运行于父会话同一模型配置（客户端声明 GLM-5.3-Flash / Start Plan）；后端实际模型标识：未提供。

## 上下文披露

- 全新启动，不继承父会话对话；仅收到入口提示词与本规则书所列输入。
- 分片内按顺序处理（item_028 → item_034），后图上下文含前图与前答，不是相互独立的 API 调用。
- 客户端层自动注入内容无法从子 agent 内枚举，记未提供。
- 每张恰好一次视觉读取（Read 原位置 PNG），未搬图、未拼图、未用其他模型或几何程序预读。

## 输入核验

| 输入 | 期望 sha256 | 实测 | 结果 |
|---|---|---|---|
| 分片清单 shards/batch_02__worker_02.jsonl | 33cf05e8…798d16 | 33cf05e8914cd03c150e0c86eaafb5e8cd2905cc9ab893b7e5627938ac798d16（7 行） | 一致 |
| prompt_image_only_v2_20261002.txt | 7df73031…11954a | 7df73031ee1a30dfb018dbb2417f505ea9ff139280307bce2934de604119954a | 一致 |
| check_answer_v3.py | 794a254e…03ec55 | 794a254e005eaa232362c539817abde98a1c2406ecb52574b9e9edb35303ec55 | 一致 |
| normalize_label_alias.py | 01e02d58…f828f2c | 01e02d587648d41618329908de8376851293551773d62140ecc610963f828f2c | 一致 |

- 7 张 PNG 指纹逐一与分片比对，全部一致。
- 开工时及每张处理前以 `test -e` 检查同伴 worker_01..03 目录 STOPPED*：均不存在（只查存在性，未读内容）。
- python 使用 `D:/python/python.exe`（存在，未替换）。

## 逐张状态表

| sample_id | item_id | checker1 | normalizer | checker2 | status | pending_review | t0 | t1 | 耗时(ms) | alias_changes |
|---|---|---|---|---|---|---|---|---|---|---|
| wafer_00002149_019 | item_028 | 0 | 0 | 0 | raw_pass | false | 2026-10-02 20:22:47 +0800 | 20:23:42 | 55041 | 0 |
| wafer_00002425_025 | item_029 | 0 | 0 | 0 | raw_pass | false | 2026-10-02 20:23:47 +0800 | 20:24:20 | 32722 | 0 |
| wafer_00002434_021 | item_030 | 0 | 0 | 0 | raw_pass | false | 2026-10-02 20:24:25 +0800 | 20:24:53 | 27721 | 0 |
| wafer_00002498_020 | item_031 | 0 | 0 | 0 | raw_pass | false | 2026-10-02 20:24:57 +0800 | 20:25:48 | 50656 | 0 |
| wafer_00002731_016 | item_032 | 0 | 0 | 0 | raw_pass | false | 2026-10-02 20:25:57 +0800 | 20:26:57 | 60137 | 0 |
| wafer_00003299_003 | item_033 | 0 | 0 | 0 | raw_pass | false | 2026-10-02 20:27:02 +0800 | 20:27:28 | 26563 | 0 |
| wafer_00004083_023 | item_034 | 0 | 0 | 0 | raw_pass | false | 2026-10-02 20:27:33 +0800 | 20:28:17 | 43971 | 0 |

备注：wafer_00002731_016 与 wafer_00004083_023 的 check.log 中 `clock_expressions_stripped` 非空（前者 caption_zh 中的「4-5点钟」被检查器剥离，后者为空记录无剥离项）；该项属检查器 detail 信息，非 failure，退出码均为 0，未改写任何原答。

## 计数

- 尝试：7
- 原始通过（raw_pass）：7
- 副本补救通过（raw_fail_canonical_pass）：0
- 隔离（raw_fail / isolated_bad_json）：0
- unknown 类别：0
- 未执行（not_run）：0

## 待复核清单

无（全部样本 content_review_warnings 为空，pending_review=false）。

## 停止原因与标记文件

无停止事件；本目录下无任何 STOPPED_* 文件。

## 声明

格式通过 ≠ 事实正确。以上全部为模型标注且仅通过格式/结构检查，未经人工复核；类别与描述的事实一致性（含 wafer_00002731_016 的混合模式判断等不确定项，见各 raw/canonical 文件 uncertainty 字段）留待人工审核。
