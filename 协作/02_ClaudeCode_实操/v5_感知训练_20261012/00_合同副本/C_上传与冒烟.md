# 任务卡 C：上传与冒烟（学校 5090，1 张卡；目标 04:00–04:45）

产物目录：`v5_感知训练_20261012/C_冒烟/`。不停下：问题按总纲 0.4 处理，写 `DECISIONS.md`。

## 步骤
1. **连通**：`cd D:/pycode/.ssh-tmp && bash hop89.sh -c 'hostname; nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader'`。跳板机不通就每 5 分钟重试，最多 1 小时；仍不通则写 DECISIONS.md 并结束（不能伪造夜跑）。
2. **选卡**：只选 `memory.used < 500 MiB` 且 `nvidia-smi --query-compute-apps=gpu_uuid,pid` 下没有任何进程的卡，最多 7 张，记 UUID 到 `gpus.json`。不足 7 张就按 0.3 砍法减任务。
3. **上传**：`img448.tgz`、`img896.tgz`（P2 用，时间紧可不传）、全部 jsonl、`B_代码/` 打包。用 `upload89.sh` 传到 `/WS/v5/` 下（真实根从仓库外私有文件读）。解包、`rewrite_paths.py` 改路径。
4. **冒烟（单卡，每项几步）**：
   - 跑 `tests/` 里全部纯 Python 测试（服务器环境再跑一遍）。
   - S1 配置 `--probe-steps 2` → `gate_sft.py` 通过；S2 同样（检查 visual 键 > 0）。记录 s/step、峰值显存。
   - 用 S1 的 2 步 checkpoint 当假起点，Rfull 配置 `--probe-steps 2`：`runtime_guard_v3` 写出 `runtime_gate.json`，`policy_equal_M0_tensor_count`、`reference_equal_M0_tensor_count` 必须 > 0 且等于可训练张量数；检查 `reward_audit.jsonl` 里奖励不是全 −1（全 −1 说明解析/空 think 有问题，必须先修再继续）。
   - `eval_v5.py` 跑 10 张 TEST200，记录 s/张；`score_v5.py` 能出 metrics。
   - `scheduler.py` 用 3 个 `sleep 5` 假任务（含一个故意失败、一个依赖它的）跑通：DONE / FAILED / SKIPPED 状态正确。
5. 写 `smoke_timing.json`，运行 `make_night_plan.py` 生成 `night_plan.json`，把预计时间表写进 `冒烟报告.md`。预计会超过 08:30 的 P1/P2 任务直接从计划里删掉并记录。
6. 自动门槛：上面全部通过 → 进入任务卡 D。某项失败 → 按 0.4 处理（例如 G=8 OOM 改 G=4），修好重测该项后继续。

## DeepSeek 最容易出错的地方（逐条照做）
- **`CUDA_VISIBLE_DEVICES` 必须在 Python 启动前由父 shell 设**（`CUDA_VISIBLE_DEVICES=<UUID> python ...`），进程内设无效；脚本里断言 `device_count()==1`。
- **上传会静默写 0 字节**：每个文件上传后比对本地与远端字节数和 SHA，不一致就重传。
- **`hop89.sh -c '...'` 会吃掉引号、`$`、中文**：多于一条的命令一律写进 `.sh` 文件，上传后 `bash 文件` 执行。
- **不抢卡、不杀他人进程**：只用空卡；只 kill 自己记录的 PID。
- **起点 adapter 必须真加载**：看训练器自存的 `args.json` 里 `adapters` 指向起点，不看自己打印的路径；`runtime_gate.json` 张量计数为 0 就是没加载。
- **空 `<think></think>` 前缀**：聊天模板会自己加，解析前必须先剥；奖励全 −1 就是这里错了。
- **超参只能按合同第 9 节 / 0.4 表改**，其他一律不改。
