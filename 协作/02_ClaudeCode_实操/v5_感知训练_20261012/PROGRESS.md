# v5 总进度（A 阶段起）
合同 00_总纲与合同.md SHA256 = e98a2a1cde19e353b4afe2af3402801361136488c104070b56e2ab1308f8740b

## 清单（本文件即 TodoWrite 的持久化版；每步完成打勾）
- [x] 第 0 步 准备（目录、合同副本、prompt_v5.txt、atomic_templates.json）
- [x] 第 1 步 全量几何真值（step1_geometry_all.py）→ 5902 可用 + 2 不可用 = 5904，69s
- [ ] 第 2 步 划分与隔离检查（step2_splits.py + isolation_check.py + test_isolation.py）
- [ ] 第 3 步 旋转增广 + 等变性自检（step3_rotations.py）
- [ ] 第 4 步 合成图 1500（step4_synthetic.py + 单测）
- [ ] 第 5 步 六个训练文件（step5_build_sft.py）
- [ ] 第 6 步 评测与 RL 文件（step6_build_eval.py）
- [ ] 第 7 步 img896（step7_img896.py）
- [ ] 第 8 步 叠加核验图（step8_overlays.py, P2）
- [ ] 第 9 步 打包与交付报告（A_数据/交付报告.md）
- [ ] 任务卡 B：common/reward/train/configs/gate/eval/score/bootstrap/select_winner/screen/rewrite/scheduler/make_night_plan/backup/probe/cnn
- [ ] 任务卡 C：连通✅→选卡✅→上传→冒烟→night_plan
- [ ] 任务卡 D：**先释放占位** → 启动夜跑 → 巡检循环
- [ ] 任务卡 E：初版报告 + ALL_DONE.flag

## 服务器状态
- 02:07（服务器钟）：连通正常（hostname=GPU-5090）。GPU0 他人占用（26.3GB），**GPU1–7 已全部占位锁定**（每卡 24GB 占位进程，PID 记录在 /WS/v5/night/holds/，释放脚本 release_holds.sh，12600s 自动过期兜底）。
- **夜跑启动前必须先 release_holds.sh 释放全部占位**（调度器要求启动时卡上零进程）。
- 注意服务器时钟比本机慢约 16–17 分钟：08:30 硬截止按本机时间执行，换算到服务器钟需再核。

## 结果摘要
- 第 0 步（02:12 完成）：prompt_v5.txt SHA=b6aa1e28d3a690b040c9d4a543d08946f2a1a644389cdced23378c0edb0c441a（15 行）；
  atomic_templates.json SHA=94e9359e8a5fc21656ac1c0f57b5b8f4037ba616d863878a1ffd614bd468647d（T1..T6 六条）。
  本机解释器：D:\anaconda\python.exe（numpy 1.26.4 / PIL 12.3.0 / pandas 2.3.3）。
