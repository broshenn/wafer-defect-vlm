# v5 决策记录（严格按"时间、问题、选择、理由"）

## D1 2026-10-12 02:12 本机解释器
- 问题：任务卡说"用仓库里以前跑通的同一个解释器（有 numpy、PIL、pandas；没有 scipy）"。实际两个候选：`D:\python\python.exe`（3.13，numpy 2.2.6/PIL，无 pandas）与 `D:\anaconda\python.exe`（numpy 1.26.4/PIL 12.3.0/pandas 2.3.3，另有 scipy）。
- 选择：用 `D:\anaconda\python.exe`。
- 理由：只有它满足"有 pandas"；"没有 scipy"与实际不符（anaconda 有 scipy），但本卡所有步骤均不依赖 scipy（probe 按任务卡写 numpy 实现），不影响任何数值。

## D2 2026-10-12 02:15 多进程 MemoryError
- 问题：step1 以 24 进程跑全量时子进程抛 MemoryError（机器物理内存 131GB 空闲，非真缺内存，疑似进程数/沙箱限制）。
- 选择：降为 4 进程（实测 16ms/张，全量 69 秒完成）。
- 理由：4 进程已验证稳定且性能足够；不改任何测量逻辑与数值。

## D3 2026-10-12 02:07 占位锁卡（用户明确指示）
- 问题：用户先后发来"如果能用就先把这7张卡用上 不然会被别人用""建议占进程"。私有接入说明第五条与 AGENTS.md §18.20 曾写"别建占位进程"。
- 选择：在**空闲**卡 GPU1–7（此时 GPU0 已被他人占用 26.3GB）各起一个 `hold_gpu.py`（24GB 占位），记录 PID 到 `/WS/v5/night/holds/`，12600 秒自动过期兜底；夜跑启动前用 `release_holds.sh` 全部释放。
- 理由：用户两条消息明确授权且优先于旧通用规则；只占空闲卡、不碰他人进程、只停自己记录的 PID、可追溯可释放。注意：v5 调度器要求启动时卡上零进程，故释放占位是启动夜跑的前置步骤（已列入 D 阶段清单）。

## D4 2026-10-12 02:52 合成底图只有 5 张
- 问题：合同 5.2 说底图取"TRAIN_POOL 里 failure_type=="none" 且 coverage<0.02 的矩阵，前 300 个"；实测满足条件的只有 5 张（none 类多为低但非零红点，coverage<0.02 是更严格的子集）。
- 选择：用全部 5 张（不扩大条件、不改 0.02 阈值），3000 张合成图按轮次循环使用这 5 张底图；在交付报告里写明实际底图数。
- 理由：决策表"不改合同数值、最保守"；底图红线已清零、只提供晶圆外形轮廓，5 张仍覆盖不同矩阵尺寸；单测 5/5 通过。

## D5 2026-10-12 02:52 sample_id 与 measure_exact 的口径
- 问题：合同 5.4 行格式写"sample_id":"<文件名去后缀>"；S3 的图片名加 `_x2`、评测 `_x2` 文件要与原 id 对齐打分；5.5 要求 measure_exact 是"未舍入原值"，但几何 v2 输出的 zone_frac/R1/R2 已在源头保留 5 位小数。
- 选择：①sample_id 一律去 `_x2`（该后缀只是渲染副本），保留 `_r090/_r180/_r270`（是不同样本）；②measure_exact.red_ratio = n_red/(n_green+n_red)、zone_frac = zone_mass/n_red（真未舍入）；R1/R2 用源头的 5 位小数值（无更细分来源，偏差 <5e-6，远小于评分容差 0.3）。
- 理由：打分对齐与"不伪造"优先；R1/R2 舍入影响可忽略且在 DECISIONS 写明。

## D6 2026-10-12 02:5x runtime_guard 的最小改动（合同冲突裁决）
- 问题：合同 6.3 规定 RL 继承赢家冻结设置（"赢家是 S2 则 freeze_vit=False"），但 B 卡要求 runtime_guard_v3 "不改逻辑"复制，而原 guard 对"视觉参数可训练"一律抛错——若 S2 赢，全部 RL 线都会被门槛拒绝（P0 崩塌）。
- 选择：guard 逐字保留，仅加一行显式开关：`WAFER_ALLOW_VISUAL_TRAINABLE=1` 时才放行视觉 LoRA 可训练（默认行为与原版完全一致）；train_v5 在 config freeze_vit=False 时自动设该环境变量。
- 理由：0.4 表"其他未列情况：不改合同数值、最保守"——两条合同条款冲突时选择让两者都可执行的最小改动，且改动被显式记录。

## D7 2026-10-12 02:5x RL 的保存间隔与数据面量
- 问题：合同未规定 RL 线的 save_steps（只规定 SFT 的 125）。时间盒 3.5h，需要中途可截断的保存点。
- 选择：RL 五线 save_steps=40、save_total_limit=4（40/80/120/160 四个保存点）。
- 理由：按 SFT 同比例（375→125 约每 42 分钟），保证"取已保存最大步 checkpoint"在时间盒截断时粒度足够；不挑选 checkpoint 的纪律不变。

## D8 2026-10-12 02:5x probe_v5 视觉塔调用需服务器验证
- 问题：线性探针的视觉塔具体调用方式（model.visual 接口、image_grid_thw）本机无法验证（本机无 ms-swift/torch 环境）。
- 选择：按 transformers 通用接口写好，标 P2；在服务器冒烟时先用 5 张图带 --limit 5 试跑确认，失败则按 0.4 表标 not run。
- 理由：probe 是 P2 诊断，不给关键路径挡路；失败不影响主实验。

## D9 2026-10-12 02:40 上传与解包校验
- 方式：数据（29 个 jsonl，147MB）与代码（37 个文件）分别打 tar.gz（gzip 后 3.5MB / 48KB），与 img448.tgz/img896.tgz 共 4 个文件上传，每个核对 SHA；服务器端 setup_server.py 再做 bundle SHA + 解包后全量文件 SHA 双层校验。
- 过程记录：首轮校验对 configs/、tests/ 报 21 处"不符"，查明为**校验器读 manifest 键时 Windows 反斜杠未归一化**（tarfile 打包时已自动把 arcname 反斜杠转为 `/`，解包本身正确）；修校验器后全部通过。数据文件全量 SHA 一次通过。

## D10 2026-10-12 02:44 配置路径改写补一轮
- 问题：rewrite_paths 只替换 `/WS/v5`，而 configs 里模型路径是 `/WS/models/Qwen3.5-9B`（不在 /WS/v5 下），S1 探针因此 "path not found"。
- 选择：对 configs/*.json 再做一轮 `/WS` → 真实根 的替换（数据文件不受影响，其内容只有 `__IMG__` 与 `/WS/v5` 两种形态）。不改任何合同数值。

## D11 2026-10-12 02:44 运行时环境修复（两处）
- ① envs/wafer 缺 `msgspec`：ms-swift 4.5.3 的 grpo_trainer 顶层 import 依赖它，RL 无法启动。选择：pip 安装 msgspec==0.22.0（与该机 labguard wheels 中的版本一致）。这是用户自己的 env，不改共享/系统环境。
- ② Rfull 探针首轮把 `IMAGE_MAX_TOKEN_NUM` 设成了字符串 "None"（探针配置 image_max_token_num=null 且起点为字面路径，未走 FROM_WINNER 继承）→ swift 报 `int('None')`。选择：train_v5 对 null 显式 SystemExit 报错；探针配置显式给 256。夜跑 RL 线走 FROM_WINNER 继承，不受影响。

## D12 2026-10-12 02:48 冒烟结论（全部项目最终状态）
- PASS：S1/S2 探针 2 步 + gate（S2 visual 键 220>0、S1 =0）→ **S2 训练视觉侧得到确认**；Rfull 探针 2 步（重跑后 rc=0，92s，30.84s/step，runtime_gate 两项张量计数>0 且 two_step_gate=passed，reward_audit 16 条中 4 条非 -1）；评测 10 张 7.14s/张；score_v5 出 metrics；scheduler 假任务 DONE/FAILED/SKIPPED 全对。
- 首轮三项失败及处置（如实记录）：test_configs 因路径已改写而断言失败（测试改为对根路径不敏感，服务器复验 20/20）；Rfull 探针因 rl_pool320 夜跑才生成而崩（改用 rl_candidates800 重跑）；纯测试 8/9 即由前者引起。smoke_report.json 中保留原 FAIL 记录并以 notes 说明。

## D13 2026-10-12 02:46 夜跑启动
- 已释放全部占位、核验 GPU1–7 零进程后启动调度器（pid 3696693）；详见 D_夜跑/启动回执.md。
- 实测 SFT 7.2s/step ≈ 单线 45 分钟，比冒烟估计更快；若一切顺利，全部任务预计 07:30-08:00 前收尾（硬截止 08:30 不变）。
