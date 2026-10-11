# 任务卡 D：启动夜跑（目标 04:45 启动）

产物：`v5_感知训练_20261012/D_夜跑/`。

## 步骤
1. 把 `night_plan.json` 上传到 `/WS/v5/night/`，核字节数。
2. 写 `start_night.sh`（上传后执行）：
   `cd /WS/v5/code && setsid nohup /WS/envs/wafer/bin/python scheduler.py --plan /WS/v5/night/night_plan.json --state /WS/v5/night/state.json < /dev/null > /WS/v5/night/scheduler.log 2>&1 & echo $! > /WS/v5/night/scheduler.pid`
3. 启动后 2 分钟、10 分钟、30 分钟各看一次 `status.md` 与 `nvidia-smi`：gate 任务应 DONE、SFT 任务应 RUNNING 且日志里 loss 在输出。失败的线按 0.4 处理（替补提上来：在服务器上写一个追加计划，调度器支持重读计划则重读，否则停掉调度器 → 改计划 → 重启，已 DONE 的任务不重跑）。
4. 把启动回执写进 `D_夜跑/启动回执.md`（调度器 PID、tmux/nohup 方式、日志路径、预计各阶段时间）。
5. 之后**每 30 分钟**看一次状态（会话还在就看；会话断了也没关系，调度器在服务器上照跑），直到 08:30。08:30 确认调度器已对训练任务发 SIGTERM；然后进入任务卡 E。

## 必须守住
- 调度器是服务器端进程，**不依赖本会话**；不要用前台 ssh 跑训练。
- 时间盒到点只取**已保存的最大步 checkpoint**，不挑中间最好的，不续训。
- 每个任务日志必须落盘到 `/WS/v5/night/logs/`；训练线必须有 `v5_config.json`（含训练器实际参数）；RL 线必须有 `runtime_gate.json` 与 `reward_audit.jsonl`。
- 数据顺序固定（`dataset_shuffle=False`、`train_dataloader_shuffle=False`），seed 按合同。
- 不碰他人进程；不占用计划外的卡。
