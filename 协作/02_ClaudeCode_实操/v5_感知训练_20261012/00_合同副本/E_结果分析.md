# 任务卡 E：结果汇总（**08:45 前必须产出初版**）

产物：`v5_感知训练_20261012/E_结果/`。08:30 开始，哪怕夜跑没全完成也要出报告。

## 步骤
1. 从服务器拉回：`state.json`、`status.md`、每条线的 `v5_config.json`、`runtime_gate.json`、`gate_sft.json`、`winner.json`、`passk.json`、全部 `metrics.json` 与 `per_sample.jsonl`（核字节数）。
2. 对尚未打分的原答跑 `score_v5.py`；有时间再跑 `bootstrap_v5.py`（预注册比较对，合同 8.3）。
3. 写 `v5_初版报告.md`，结构固定：
   - **一句话结论**（RL 主结论是否达标；达不到如实写）。
   - **执行状态表**：每条线 = DONE / partial（实际步数）/ failed / not run，附原因。
   - **主表**：TEST200 上各模型 GeoScore、coverage / zones / angular 正确率、扇区正确率（附正样本数）、measure 误差、类别准确率与 Macro-F1、格式合法率（剥包装与原始两种口径）。
   - 旋转等变（ROT100）、去图（SHUF50）结果与多数类基线。
   - 预注册比较对的差值与区间（没算完就写 pending）。
   - 门槛证据：起点加载（args.json + 张量计数）、S2 visual 键、奖励审计里零方差组比例、是否启用 dynamic_sample。
   - DECISIONS.md 全部决策摘要。
   - 未完成与限制。
4. 08:45 前 git commit 初版；还有时间就补全量 TEST 与 bootstrap，另存 `v5_报告_v2.md`。

## 必须守住
- **not run 不是 0**：没跑的格子写 not run，partial 写实际步数，不用 0 填。
- 命名：公开标签 / 程序真值 / 模型输出；**不写"金标"**。
- 程序算出的几何不算模型能力；只报模型输出对程序真值的准确率。
- 区间不含 0 才写"有效"，否则写"本批未观察到差异"。
