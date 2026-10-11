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
