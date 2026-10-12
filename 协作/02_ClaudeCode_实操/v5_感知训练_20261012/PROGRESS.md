# v5 总进度（A 阶段起）
合同 00_总纲与合同.md SHA256 = e98a2a1cde19e353b4afe2af3402801361136488c104070b56e2ab1308f8740b

## 清单（本文件即 TodoWrite 的持久化版；每步完成打勾）
- [x] 第 0 步 准备（目录、合同副本、prompt_v5.txt、atomic_templates.json）
- [x] 第 1 步 全量几何真值（step1_geometry_all.py）→ 5902 可用 + 2 不可用 = 5904，69s
- [x] 第 2 步 划分与隔离检查（step2_splits.py + isolation_check.py + test_isolation.py）
- [x] 第 3 步 旋转增广 + 等变性自检（step3_rotations.py）
- [x] 第 4 步 合成图 1500（step4_synthetic.py + 单测）
- [x] 第 5 步 六个训练文件（step5_build_sft.py）
- [x] 第 6 步 评测与 RL 文件（step6_build_eval.py）
- [x] 第 7 步 img896（step7_img896.py）
- [x] 第 8 步 叠加核验图（step8_overlays.py, P2 已做）
- [x] 第 9 步 打包与交付报告（img448.tgz 25.7MB / img896.tgz，交付报告已写）（A_数据/交付报告.md）
- [x] 任务卡 B 全部完成（02:30）：16 文件 + 10 测试文件全 PASS（common10/reward23/configs20/score11/sched18/rewrite5/screen5/bootstrap4/winner+plan13/gate6）；交付报告见 B_代码/交付报告.md
- [x] 任务卡 C 完成（02:48）：上传全量 SHA 校验✅、服务器 setup✅、冒烟全过✅（含 S2 visual 220 键、Rfull 探针 gate 全过）、night_plan 126 任务已生成
- [~] 任务卡 D：占位已释放✅ → 调度器已启动✅（pid 3696693）→ 巡检循环进行中（sft_S1 7.23s/it，ETA 45min/线）
- [ ] 任务卡 E：初版报告 + ALL_DONE.flag

## 服务器状态
- 02:07（服务器钟）：连通正常（hostname=GPU-5090）。GPU0 他人占用（26.3GB），**GPU1–7 已全部占位锁定**（每卡 24GB 占位进程，PID 记录在 /WS/v5/night/holds/，释放脚本 release_holds.sh，12600s 自动过期兜底）。
- **夜跑启动前必须先 release_holds.sh 释放全部占位**（调度器要求启动时卡上零进程）。
- 时钟复核：服务器与本机仅差约 6 秒（基本同步），08:30 硬截止可直接用服务器本地时间。（02:16 复核）

## 结果摘要
- 第 0 步（02:12 完成）：prompt_v5.txt SHA=b6aa1e28d3a690b040c9d4a543d08946f2a1a644389cdced23378c0edb0c441a（15 行）；
  atomic_templates.json SHA=94e9359e8a5fc21656ac1c0f57b5b8f4037ba616d863878a1ffd614bd468647d（T1..T6 六条）。
  本机解释器：D:\anaconda\python.exe（numpy 1.26.4 / PIL 12.3.0 / pandas 2.3.3）。

## 巡检记录（服务器时间）
- 02:57 巡检1：调度器活；DONE 6 / RUNNING 7 / PENDING 113。SFT：S1 85/375@7.03s/it、S2 76/375@7.51、S1p 83/375@6.85、S0 30/375@18.33（S0 全主任务长序列，速率属正常）。评测：base_select(g5)、base_test200(g7)、L_test200(g6) 并行中。GPU1-7 均在用。
- 03:06 巡检2：DONE 12 / RUNNING 6 / PENDING+SKIPPED 余下。**gate_S3 FAILED（rc=1）**：swift 数据集报 "Failed to retrieve the dataset"（896 图 + IMAGE_MAX_TOKEN_NUM=1024 令样本超 max_length=2048）；S3 是 P2 线 → 按 0.4 表标 failed，所有 S3 评测连锁 SKIPPED，不动合同数值（max_length 固定 2048）。sft_S4（P2）gate 已过、在 g6 训练中。四条主线 S1/S0/S2/S1p 进入主任务长序列段 ~18.5s/it，预计 04:20-04:45 完成。eval_base_select/base_test200/L_select/L_test200 已完成，base_rot100 运行中。
- 03:15 巡检3：SFT 四线 44-45%（~18.5s/it，ETA 04:20-04:45）；probe（P2）首跑 FAILED（视觉塔 grid_thw 在 CPU + 返回输出对象两处接口问题）→ 两处小修后已用空闲 g5 单跑，特征提取进行中（700/2000）。其余照常。
- 03:36-04:00 巡检4-7：SFT 四线稳定（18.3-18.9s/it）；probe 经 4 处接口小修后跑完（2000×19584 特征；coverage 0.3277/zones 0.3243/angular 0.1715，均低于多数类基线——按合同配方如实记录）；sft_S4（P2）g6 训练中。
- 04:23 巡检10：**S1/S2/S1p 完成 375 步**（1h30-1h35）；自动进入评测链（S1_select/S1_test200/S1p_test200/S2_select/S2_test200 并行）；S0 83%（ETA 04:45）；S4（P2）仍在训。cnn 修复后重跑完成：**acc=0.7827 macro_f1=0.7723**（ResNet18 从零训 20 轮）。
- 04:50 巡检13：**S0 完成 375 步（1h54m）——四条主 SFT 线全部完成**；评测链持续推进（S0_select/S0_test200/S1_shuf50/S1p_shuf50/S2_rot100/S4_select 并行）。CNN acc=0.7827。待 S0_select 与 S4_select 完成 → select_winner → 图池筛选 → RL。
- 05:09 巡检15：**select_winner 完成：赢家 = S0**（SELECT200 GeoScore 0.9273、类别 acc 0.875；S1 0.9093/S2 0.9115/S4 0.9069；S3 failed 不参选）。screen_0-3 已在 g1-g4 开跑；S0/S4 评测链与 S1 全量 TEST 继续。
- 05:55 巡检20：**RL 全面开跑**：rl_RSFT(g1, 55/160@7.4s/it)、rl_Rfull(g2)、rl_Rbinary(g3)、rl_Rfullp(g5)（GRPO ~31s/步）；rl_Rnoproc 等 g4（S0 全量 TEST 收尾）。passk：**320 候选全部奖励有方差（zero_variance=0）、入选 160 无需补足；pass@1=0.70、pass@4=0.9313**（S0 赢家模型）。screen 全部分片完成。
- 08:29（本机）：E 阶段完成——rot-pair 全模型补算、拉回 132 个文件、**E_结果/v5_初版报告.md 完成并提交、ALL_DONE.flag 已创建**。核心结果：SFT 赢家 S0（SELECT200 GeoScore 0.927）；RL 主结论未达标（反向）：Rfull−RSFT = −0.0399 [−0.0646,−0.0154]；但 Rfullp（同配置 seed 3408）= 0.943 > Rfull 0.878，种子方差被单独记录。服务器继续跑 P1 全量 TEST 与 backup（预计 ~10:00 全部结束）；之后出 v2 完整报告。
- 08:45（本机）：终态确认（126 任务全终态，08:21:15 调度器正常退出）；**v2 报告完成**（含全量 TEST 附表 5 模型、探针预注册解读规则：探测 0.274 < S1−0.05 → 判"视觉特征不足"，与 S2−S1 方向核对为"不一致（微弱，区间含 0）"）。backup 手动补跑中（调度器的 backup 曾被 gate_S3 连锁跳过，已修复 make_night_plan 并记录 D15）。
- 08:55（本机）：**全链路完成**。backup 6.38GB + 清单已就位并拉回 `E_结果/server_final/`；服务器我方进程零残留、GPU1-7 空闲、GPU0 他人未扰。交付物：`E_结果/v5_初版报告.md`（08:30 提交 + ALL_DONE.flag）、`E_结果/v5_报告_v2.md`（08:45 提交，含全量 TEST 附表与探针预注册解读）、`D_夜跑/启动回执.md`（含完成记录）。夜色未尽而活已干完——无仍在运行的任务；如需 S3 修复重跑/多种子等新实验，等用户指令。
