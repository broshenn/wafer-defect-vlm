# GLM 续标 161 张 · 父会话批次交付报告

- 任务书：Codex《给ZCode：继续原180候选清单，剩余最多161张》（2026-10-02 放行，用户转交本父会话）
- 执行：ZCode 父会话（调度/计数/阶段索引，未看任何图）+ 25 个全新 GLM 子 agent（pending_020 单 worker + 8 组 × 3 worker）
- 批次目录：`协作/03_ZCode_标注/20261002-201800_续标161/`（每组 `stage_index.json` + 各 worker 产物）
- 时间窗：2026-10-02 20:16 – 10-03 01:33 (+0800)，其中 10-02 晚按用户指示暂停一次（batch_05 启动前取消，零消耗零产物）
- **全部 161 份回答为 GLM-5.3-Flash 模型标注，事实正确性未审，未作人工审核，不自动宣布可进训练集。**
- 批次结果：**计划 161 · 尝试 161 · 原始通过 153 · 副本补救 0 · 隔离失败 8 · unknown 0 · 未执行 0**

## 一、核实了什么

1. **输入链完整**：索引 SHA256 `60762f2e…f55e0` ✓；9 组清单、提示词、原检查器、规范化器共 12 个文件哈希全部吻合 ✓；行键干净；161 ID 与 SHA256 全库唯一、跨组零重叠；**未回头标注 batch_01 已完成的 19 张**；pending_020 即 batch_01 未执行的 `wafer_00047003_005`（此前从无答案）。
2. **执行结果**：25 个 worker 全部完成各自分片，无任何系统停止（全批无 STOPPED_* 文件）；每张恰好一次视觉读取、原答逐字节保存。
3. **两轮格式检查**：原检查器 153/161 原始通过；8 件隔离（详见 `batch_log.json` 的 `failure_count`/`failure_details`）；别名规范化器全部成功运行但 **alias_changes 合计为 0**（无一是 Near-full/Edge-Loc/Edge-Ring 拼写），副本全部生成且副本检查与原答同判。
4. **父会话全量复跑**：161 份原答重跑原检查器，退出码与 worker 记录 **100% 一致**（153×0 + 8×8）；8 件隔离的 canonical 副本重跑均为 8（0 件副本补救）；覆盖率/查重/STOPPED 标记核对完毕（唯一重复原答对见 §四.3）。
5. **失败模式单一且已定性**：8 件隔离全部为退出码 8——模型写「约X至Y点钟方向」类区间钟点表述，检查器的钟点剥离规则只识别部分形式，剔除后残留数字。**按协议原件保留、未改写未重试**；是否调整剥离规则或提示词由 Codex 决定（见 `batch_log.json` `failure_pattern_note`）。
6. **吞吐分布**（如实记录，不解读）：Random 61、Center 21、Near_full 18、Edge_Ring 17、Edge_Loc 15、Loc 14、Scratch 12、Donut 2、none 1。

## 二、仍未知什么（未证实 / 未测量）

1. **161 份回答的事实正确性未审**——待 Codex 逐张内容验收；格式通过≠事实正确。
2. 后端实际模型标识、后端请求数、自动重试、逐张 token、套餐余额、finish_reason、生成参数：**未提供**。
3. `subagent_tokens`（客户端工具回报，全批约 4700 万）口径未知，与套餐扣量关系未提供，不推算单张成本。
4. 客户端层向子 agent 自动注入的内容无法从子 agent 内枚举：未提供。
5. 「约X至Y点钟」表述对应图是否真的多为跨区间缺陷：未测量（属内容验收范畴）。

## 三、改了什么

- **仅新建**：批次目录下的各 worker 产物（raw/canonical/normalization/check.log/worker_log/worker_report）、9 份 `stage_index.json`、派生分片 `shards/`、父会话 `worker_instructions.md`、`parent_stage_index.py`、`pause_state.json`（暂停留痕）、`batch_log.json` 与本报告。
- 未改输入文件、共享文档、历史产物；worker 互相只查 STOPPED 存在性；父会话未看任何 PNG。

## 四、执行偏离与披露（如实记录）

1. **用户暂停一次**：10-02 晚 batch_05 三个 worker 在启动前被取消（零产物零消耗），恢复后按原计划续跑，未重派任何已尝试图片（留痕 `pause_state.json`）。
2. **父会话沿用披露**：本父会话为此前试标会话；开工污染比对确认 161 张中仅 `wafer_00047178_011`（batch_09/worker_02）与本会话九张试标重叠——该图由全新 worker 标注，父会话知识未进入 worker 输入，`batch_09/stage_index.json` 保留披露标记，提请 Codex 验收时重点复核。
3. **唯一字节相同原答对**：`wafer_00011630_025` 与 `wafer_00011856_003`（batch_03/worker_01 自述两图同形态、作答文字一致；父会话查重确认 sha `eca0739f…` 相同）。raw 文件不含 sample_id，同形态图答案字节相同可能成立，提请验收时比对两图。
4. **worker 日志引号修正两起**（batch_02/worker_03、batch_03/worker_02）：`worker_log.jsonl` 首写因 shell 引号产生非法 JSON，按原值机械修正并自校验；父会话核验 raw 哈希与磁盘一致、答案文件未动。
5. **信息性 checker 记录**：少数图（如 batch_07/worker_02 第 1 张）检查器 detail 记录了被剥离的钟点表达（退出码 0，非失败非警告），已随 check.log 保留。

## 五、下一步（均未运行）

1. 用户将本目录原始结果与报告转回 Codex 逐张内容验收；
2. 8 件隔离件与全部 pending_review（本批为 0）的处理方式、检查器钟点剥离规则是否调整、`wafer_00047178_011` 与同字节原答对的复核结论，由 Codex 决定；
3. 原 180 清单至此 19+161=180 全部有交代（161 完成 + 8 隔离件仍在 161 内）；是否补跑、如何补跑由 Codex 新授权明确；ZCode 不自行扩批、不上传、不训练、不自动宣布可进训练集。

## 六、证据路径与实际消耗

- 批次目录：`协作/03_ZCode_标注/20261002-201800_续标161/`
  - 逐张：`<组>/<worker>/<sample_id>_raw.json`（逐字节原答）、`_canonical.json` + `_normalization.json`（别名副本与报告）、`_check.log`（两轮检查退出码+输出）
  - 逐 worker：`worker_log.jsonl`（指纹/时间/状态）、`worker_report.md`
  - 逐组：`stage_index.json`；调度产物：`shards/`（派生分片，哈希在 batch_log 与生成记录）；父会话：`batch_log.json`（含 `failure_count=8`、`failure_details` 8 条、复跑结果、工单级披露）
- 规则书：`worker_instructions.md`（全部 worker 同一规则，含停止条件与禁令）
- 输入：`协作/01_Codex_指挥/GLM续标_20261002_无答案输入/inputs.json` 及其列出的清单/提示词/检查器/规范化器（哈希见 `batch_log.json`）
- **实际消耗**：161 次标注尝试（GLM-5.3-Flash / Start Plan，客户端声明）；25 个子 agent 的客户端回报 subagent_tokens/用时逐个见 `batch_log.json` 的 `worker_runs`；额外付费 0；后端分账：未提供

**交付完。父会话就此停止，等待用户转交 Codex 验收。**
