# 任务卡 B：训练 / 奖励 / 评测 / 调度代码（本机 CPU）

> **01:50 覆盖修订（优先于本卡正文）**：①同一主会话在 A 之后执行，不并行；目标 **04:00 前完成**。②优先级：P0 必做 = 1 common、2 reward（测试必须全过）、3 train、4 configs、5 gate_sft、6 eval、7 score、9 select_winner、10 screen、13 rewrite、14 scheduler、15 make_night_plan、16 backup；P1 = 8 bootstrap、12 cnn；P2 = 11 probe（来不及就标 not run）。③步数与规模按总纲 0.3（SFT 375 步，RL 160 步）；make_night_plan 按总纲 0.3 的卡分配与 08:30 硬截止生成：调度器在 08:30 对所有训练任务发 SIGTERM（取已保存最大 checkpoint），之后只跑评测与 08:40 的报告任务。④遇到问题不停，按 0.4 决策表处理并写 `DECISIONS.md`。

> 先完整读 `00_总纲与合同.md` 第 3、6、7、8、9、10 节。**数值以合同为准。**
> 工作目录：`D:/pycode/晶圆图研究/协作/02_ClaudeCode_实操/v5_感知训练_20261012/B_代码/`
> 本机没有 GPU、没有 ms-swift：本会话里**能跑的只有纯 Python 测试**（奖励、解析、打分、统计、选择、调度器逻辑）；训练和推理脚本只做 `python -m py_compile` 和 `--dry-run`，真跑在会话 C。
> 会话 A 的数据可能还没出来：需要样例时，自己按合同 3.2 / 5.5 的格式手写 3–5 条假数据放 `B_代码/tests/fixtures/`。

## 工作方式（必须遵守）
- 每个文件一个职责，开头写中文注释：用途、输入、输出、对应合同哪一节。
- **能复用旧代码就复制过来再改**，不要从头重写。复制时在文件头注明来源路径和原 SHA。
- 每写完一个文件就写它的测试并跑通，然后在 `PROGRESS.md` 打勾并 git commit。
- 不确定 ms-swift 某个参数是否存在：**不要猜**，在代码里写成"运行时检查 `hasattr`/`dataclasses.fields`，不存在就按合同第 9 节处理并记录"，并在 `DECISIONS.md` 记一笔，留给会话 C 冒烟时确认。
- 上下文中断后先读 `PROGRESS.md` 接着做。

---

## 文件清单与规格

### 1. `common_v5.py` —— 常量与解析（合同 3.5）
- 从 `协作/02_ClaudeCode_实操/一天半优化_20261010/core_v3_复核修复.py` **复制** `EMPTY_THINK, FENCE, strip_envelope, _pairs, _constant, parse_raw_strict, schema_errors, CLASSES, COVERAGE, ANGULAR, ZONES, FIELDS` 到本文件（不 import 旧文件，避免路径依赖；文件头记来源 SHA）。
- 新增：
  - `MEASURE_RE`：合同 3.5 第 2 条的正则，`re.S`。
  - `parse_v5(text) -> dict`：返回 `{"ok": bool, "raw_ok": bool, "measure": {...} 或 None, "obj": {...} 或 None, "errors": [...]}`。`raw_ok` = 原始全文（不剥包装）`.lstrip().startswith("<measure>")` 且整体匹配正则。`ok` = 剥包装后匹配正则 + JSON 严格解析成功 + `schema_errors` 为空。measure 里的数值转成 float，zone_frac 为长度 3 的 list。
  - `COV_ORDER = ["zero","under_5pct","5_to_under_20pct","20_to_under_50pct","half_or_more"]`
  - `band_of_ratio(x)`：0 → zero；(0,0.05) → under_5pct；[0.05,0.20) → 5_to_under_20pct；[0.20,0.50) → 20_to_under_50pct；≥0.50 → half_or_more。
  - `zones_of_frac(fr)`：返回 `[z for z,f in zip(ZONES, fr) if f >= 0.1]`。
- 测试 `tests/test_common.py`（全部要 PASS）：
  1. 合同 3.2 的样例答案 → ok=True、raw_ok=True。
  2. 同样例前面加 `<think>\n\n</think>\n\n` → ok=True、raw_ok=False。
  3. 前面加非空 think `<think>abc</think>` → ok=False。
  4. 少了测量行 → ok=False。
  5. 测量行 red_ratio 写成一位小数 `0.3` → ok=False。
  6. JSON 有重复键 → ok=False。
  7. JSON 少一个字段 → ok=False。
  8. JSON 后面多一行废话 → ok=False。
  9. `band_of_ratio` 在 0、0.0499、0.05、0.1999、0.2、0.4999、0.5、1.0 上的输出全部正确。
  10. `zones_of_frac([0.0,0.1,0.9]) == ["middle","edge"]`。

### 2. `reward_v5.py` —— 奖励（合同第 7 节，逐条实现）
- `score_v5(text, ref, mode) -> dict`：返回 `{"reward", "ok", "raw_ok", "parts": {...}, "weights_used": {...}, "errors"}`。
- `class WaferV5Reward`：`__init__` 读环境变量 `WAFER_REWARD_MODE`，不在 {full, binary, noproc} 就 `raise`；`__call__(self, completions, reference_json=None, sample_id=None, **kwargs)` 与 v3 相同的检查（缺列或长度不齐就 raise）；每条追加审计到 `self.audit`。
- 测试 `tests/test_reward.py`（全部 PASS，每条打印名字和实际奖励值）：
  1. 完全正确答案，full → 1.0；binary → 1.0；noproc → 1.0。
  2. 格式不合法 → −1（三种模式都是）。
  3. 只有类别错、其余全对，full → (0.30×(−1)+0.70×1)/1.0 = 0.40。
  4. 类别答 unknown、其余全对，full → (0.30×0.2+0.70)/1.0 = 0.76。
  5. coverage 差一档，其余全对：full 中 coverage 分量 = 0.5；binary 中 = 0。
  6. zones 参考 [middle, edge]、答 [edge]：full 中 zones = 0.5；binary 中 = 0。
  7. 参考 single [3]、答 single [4]：full 中 angular = 0.6+0.4×0.5 = 0.8；binary 中 = 0。
  8. 参考 opposed [3,9]、答 opposed [4,10]：full 中 angular = 0.8。
  9. 参考 nondirectional []、答 nondirectional []：angular = 1.0；答 single [3]：angular = 0。
  10. 参考 angular_type 为 unknown：angular 分量不在 `weights_used` 里，分母相应减少。
  11. **全 unknown 刷分攻击**：类别 unknown、coverage unknown、zones unknown、angular unknown、扇区 []、测量全 0，参考全部已知 → full 奖励必须 < 0.25（打印实际值）。
  12. **自相矛盾**：red_ratio=0.03 但答 coverage `20_to_under_50pct` → consistency 第①项 = 0。
  13. noproc 模式下 measure、consistency 都不在 `weights_used` 里。
  14. 奖励类：缺 `reference_json` → raise；长度不齐 → raise；未设模式 → raise。
  15. 审计记录里有 `had_empty_think_prefix` 和 `raw_ok` 两个字段。

### 3. `train_v5.py` —— 训练（以 `server_v3_train.py` 为模板复制修改）
命令行：`python train_v5.py --config configs/<线名>.json --out <输出目录> [--probe-steps 2] [--dry-run]`
- **在 import torch / swift 之前**读取 config，设置 `os.environ["IMAGE_MAX_TOKEN_NUM"]`、`WAFER_REWARD_MODE`（GRPO 线）。
- 保留模板里的所有门槛：父进程必须设 `CUDA_VISIBLE_DEVICES`、`device_count()==1`、`world_size==1`。
- 三种模式（config 的 `mode` 字段）：
  - `sft_base`：不加载 adapter（`adapters` 不传），按合同 6.1。
  - `sft_cont`（R-SFT）：`adapters=[起点]`、`resume_only_model=True`，装载 `runtime_guard_v3`（直接复制 v3 的 `runtime_guard_v3.py` 到本目录，不改逻辑），`WAFER_FIX_MODE=sft`。
  - `rl`：同上 + `ref_adapters=[起点]`，`WAFER_FIX_MODE=rl`，`reward_funcs=[WaferV5Reward()]`，按合同 6.3。
- 起点来源：config 里 `start_adapter` 可以写字面路径，或写 `"FROM_WINNER"`——此时读 `/WS/v5/night/winner.json` 的 `checkpoint` 字段；并把 `freeze_vit`、`freeze_aligner`、`IMAGE_MAX_TOKEN_NUM` **从赢家的 config 继承**（合同 6.3）。`WAFER_EXPECT_M0_SHA` 自动设为起点 `adapter_model.safetensors` 的 SHA。
- `dynamic_sample`：用 `dataclasses.fields(RLHFArguments)` 检查是否存在；存在就按合同设置，不存在就不传，并在输出的 `v5_config.json` 里写 `"dynamic_sample": "unsupported"`。
- `--probe-steps N`：覆盖 max_steps=N、save_steps=N（用于门槛），输出目录必须与正式目录不同。
- `--dry-run`：不 import torch/swift，只打印解析后的完整参数字典并退出 0（本机可跑）。
- 结束时写 `v5_config.json`：同模板，另加 `line, mode, config_sha256, data_sha256, start_adapter_sha256, reward_mode, dynamic_sample, trainer_actual_args`；GRPO 线把 `reward.audit` 写 `reward_audit.jsonl`。
- 本机验收：对每个 config 跑 `--dry-run`，打印结果贴进交付报告。

### 4. `configs/` —— 11 个配置文件
`S0.json S1.json S2.json S3.json S4.json S1p.json RSFT.json Rfull.json Rbinary.json Rnoproc.json Rfullp.json`。字段：`line, mode, data(服务器绝对路径), model, model_type, template, lora_rank, lora_alpha, lora_dropout, target_modules, freeze_vit, freeze_aligner, image_max_token_num, lr, warmup_ratio, scheduler, per_device_train_batch_size, gradient_accumulation_steps, max_steps, save_steps, save_total_limit, max_length, seed, data_seed, start_adapter, reward_mode, num_generations, temperature, top_p, beta, epsilon, max_completion_length, time_box_min`。数值全部照抄合同 6.1 / 6.3。服务器路径用 `/WS/v5/...` 占位，会话 C 用 `rewrite_paths.py` 统一替换成真实根。
- 测试 `tests/test_configs.py`：逐项断言与合同一致（尤其 S2 的 freeze 为 False、S3 的 token 为 1024、S1p 的 seed 为 3408、RL 线 G=8、步数 375/160）。

### 5. `gate_sft.py` —— SFT 两步门槛检查
命令行：`python gate_sft.py --config configs/S2.json --ckpt <probe 输出里的 checkpoint-2>`
检查并输出 `gate_sft.json`：
- `adapter_model.safetensors` 的键里含 `visual` 的数量：S2 必须 > 0，其余必须 = 0。
- 所有 `lora_B` 张量中至少一半非全零（两步更新后应非零）。
- checkpoint 里的 `args.json`：`adapters` 为空或不存在；`freeze_vit` 与 config 一致。
- 任一不过 → 退出码 3。
- 本机测试：用 `safetensors.numpy` 写一个假的小 safetensors（两个键，一个含 visual）测检测逻辑；如果本机没装 safetensors，就把检测逻辑写成"输入键名列表与张量是否全零"的纯函数并对纯函数测试。

### 6. `eval_v5.py` —— 推理（以 `server_v3_eval.py` 为模板）
命令行：`python eval_v5.py --model <基座> [--adapter <dir> | --adapter-from-line <线名>] --tag <模型名> --dataset <jsonl> --out <dir> --max-tokens 384 --batch-size <N> --budget-s <秒>`
- 保留模板的：锁卡断言、台账、断点续跑、原答两列（raw / stripped）、单条失败不中断。
- 改动：图片路径直接用数据行里的 `images[0]`（已是服务器绝对路径）；`RequestConfig(max_tokens=384, temperature=0.0, seed=3407)`；一次把 `batch_size` 条请求作为列表传给 `eng.infer`（批推理），批内某条失败时退回逐条重试一次。
- `--adapter-from-line <线名>`：在 `/WS/v5/runs/<线名>/` 下找最大步数的 `checkpoint-*`（递归），把实际用到的路径和步数写进台账。S3 系列的评测必须用 `_x2` 数据（由调度计划保证，这里只要打印数据文件名）。
- L 模型：`--adapter` 读 `/WS/night/L-N3072-3407.ckpt` 文件内容得到路径，`--max-tokens 64`。
- 本机只做 `py_compile`。

### 7. `score_v5.py` —— 打分（合同 8.2）
命令行：`python score_v5.py --raw <原答.jsonl> --ref <评测数据.jsonl> --kind main|L --out <metrics.json> [--rot-pair <原图 TEST 原答.jsonl>]`
- `main`：对每条用 `parse_v5`，计算合同 8.2 全部指标；每条样本的逐项对错写进 `per_sample.jsonl`（bootstrap 要用，必须带 `sample_id` 和 `lot_name`）。
- `L`：从 L 的回答里抽 `defect_class`（用 v3 的 `parse`，失败记错），只算类别指标。
- `--rot-pair`：ROT200 的旋转等变指标（合同 8.2）：按 `src_id` 与原图预测配对。
- 缺失的原答（not run / 失败）计入"未完成"数量，不算对也不算错，在 metrics 里单列。
- 测试 `tests/test_score.py`：用 fixtures 里 4 条手写原答（全对、格式错、类别错、扇区差一）算出的各指标与手算一致。

### 8. `bootstrap_v5.py` —— 配对 lot bootstrap（合同 8.3）
命令行：`python bootstrap_v5.py --a <per_sample_A.jsonl> --b <per_sample_B.jsonl> --metric geoscore|class_macro_f1|class_acc --n 10000 --seed 3407`
- 只在两边都有的 sample_id 上比；按 `lot_name` 成组有放回抽样；GeoScore 与 Macro-F1 每次重算整体值再相减。输出点估计（原样本差）、bootstrap 均值、2.5% / 97.5% 分位。
- 测试：两份完全相同的输入 → 差值 0、区间 [0,0]；把 B 的所有 coverage 改错 → 差值为正且区间不含 0。

### 9. `select_winner.py` —— 赢家选择（合同 6.2，逐条照做）
- 输入：`/WS/v5/eval/<线>/select200/metrics.json`（S0–S4）与 L 的 select200 metrics。
- 输出 `/WS/v5/night/winner.json`：`{"winner", "checkpoint", "config", "scores": {...}, "guard": {...}, "reason"}`。
- 测试：构造 4 组假 metrics（正常、并列、都不过守门、某线缺失）验证规则。

### 10. `screen_rl_pool.py` —— RL 图池筛选（合同 6.5）
- `--shard i --nshards 6`：只处理候选里 `index % 6 == i` 的图；每图采样 4 次（`RequestConfig(max_tokens=384, temperature=1.0, seed=3407+k)`），用 `score_v5(..., "full")` 打分；写 `screen_shard_i.jsonl`。
- `--merge`：合并 6 个分片，按候选原顺序判"4 次奖励不全相同"，取前 160，不足按顺序补足；写 `rl_pool320.jsonl`（GRPO 行：只有 user 消息 + `reference_json` + `sample_id`）、`rsft_pool320.jsonl`（每图 4 行相同正确答案，从 `rsft_rows_template.jsonl` 取）、`passk.json`（pass@1、pass@4、入选数、补足数）。
- 测试：merge 逻辑用假分片测。

### 11. `probe_v5.py` —— 线性探针（合同 8.4）
- 用 transformers 直接加载基座（`AutoProcessor`、`AutoModelForImageTextToText` 或 ms-swift 的 `get_model_tokenizer`，以能跑通为准），对每张图走处理器得到 `pixel_values` 与 `image_grid_thw`，调用视觉塔得到图像 token 特征 → 平均池化 + 4×4 空间平均池化（按 `image_grid_thw` 还原成二维网格后池化）→ 存 `.npy`。
- 第二阶段（CPU）：PCA 到 256 维（用 numpy SVD 实现，不依赖 sklearn），逻辑回归优先用 sklearn；没有 sklearn 就用 numpy 实现的多类 softmax 回归（全批量梯度下降 500 步、L2=1e-3）。
- 输出 `probe.json`：三字段的 TEST 准确率、多数类基线、样本数、特征维度、所用实现。
- 视觉塔的具体调用方式本机无法验证：写好之后在 `DECISIONS.md` 注明"需会话 C 在服务器上用 5 张图试跑确认"。

### 12. `cnn_v5.py` —— CNN 类别基线（合同 8.4）
- torchvision ResNet18（`weights=None`，`num_classes=9`），不可用就用 4 层卷积网络；输入 224×224；20 轮；Adam 1e-3；batch 64；seed 3407；只用 TRAIN_POOL 原图训练；TEST 上输出类别准确率、Macro-F1、逐类召回，并写 `per_sample.jsonl`（与 score_v5 同格式，便于 bootstrap）。

### 13. `rewrite_paths.py` —— 路径改写
- 把数据 jsonl 里的 `__IMG__/` 换成给定前缀（img448 或 img896 的真实目录），把 config 里的 `/WS/v5` 换成真实根。输出新文件（不覆盖原件），并写 `rewrite_log.json`：每个文件改写前后 SHA、替换次数。
- 测试：用 fixtures 测替换次数与 SHA 记录。

### 14. `scheduler.py` —— 夜跑调度器（服务器端常驻）
命令行：`setsid nohup /WS/envs/wafer/bin/python scheduler.py --plan /WS/v5/night/night_plan.json --state /WS/v5/night/state.json < /dev/null > /WS/v5/night/scheduler.log 2>&1 &`
- `night_plan.json` 结构：
  ```json
  {"gpus": {"g1": "GPU-uuid...", "...": "..."},
   "jobs": [{"id": "sft_S1", "gpus": ["g2"], "cmd": ["..."], "after": ["gate_S1"],
             "after_mode": "all_ok", "timeout_min": 300, "timeout_ok": true,
             "retries": 0, "log": "/WS/v5/night/logs/sft_S1.log"}]}
  ```
  - `after_mode`：`all_ok`（依赖都 DONE，或 TIMEOUT 且该依赖 `timeout_ok`）或 `all_finished`（依赖都结束即可，不论成败；用于 select_winner 和最后的 backup）。
  - `gpus`：候选卡列表，按顺序取第一张空闲的；`"cpu"` 表示不需要卡。
- 主循环每 20 秒：
  1. 检查运行中的任务：进程结束 → 退出码 0 记 DONE，否则 FAILED（有剩余重试就重新排队）；超时 → `os.killpg(pgid, SIGTERM)`，60 秒后仍在就 `SIGKILL`，记 TIMEOUT。
  2. 找就绪任务：依赖满足；任一依赖 FAILED（或不允许的 TIMEOUT）且 `after_mode=all_ok` → 记 SKIPPED（连锁跳过）。
  3. 为就绪任务找卡：卡上没有本调度器的运行任务，且 `nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader` 里该 UUID 下没有**任何**进程（防止别人占用）；找到就启动：`subprocess.Popen(cmd, env=os.environ + {"CUDA_VISIBLE_DEVICES": uuid}, stdout=日志, stderr=STDOUT, stdin=DEVNULL, start_new_session=True)`。
  4. 原子写 `state.json`（先写临时文件再 `os.replace`）和人类可读的 `status.md`（每个任务的状态、卡、开始/结束时间、耗时、日志路径）。
  5. 所有任务都进入终态 → 写 `ALL_DONE` 标记文件并退出。
- 收到 SIGTERM：终止自己启动的全部进程组后退出（只杀自己记录的 PID）。
- 本机测试 `tests/test_scheduler_logic.py`：把"就绪判断、连锁跳过、选卡"写成纯函数并测试；真实进程测试留给会话 C 在服务器上用 `sleep` 假任务跑。

### 15. `make_night_plan.py` —— 生成夜跑计划
- 输入：会话 C 冒烟测得的速度（`smoke_timing.json`）、GPU UUID 列表（7 张）、真实根路径（从环境变量读，不写进仓库）。
- 按下面的依赖图生成 `night_plan.json`（卡号 g1–g7）：

| 阶段 | 任务 | 卡 | 依赖 |
|---|---|---|---|
| N0 | `gate_S0…gate_S4, gate_S1p`：`train_v5 --probe-steps 2` 后接 `gate_sft.py`（用 `bash -c` 串起来，任一失败整体失败） | g1–g6 各一 | 无 |
| N0 | `eval_base_select`、`eval_L_select` | g7 | 无 |
| N1 | `sft_S0…sft_S4, sft_S1p` | 与各自 gate 同卡 | 对应 gate |
| N1 | `eval_base_test/rot/shuf`、`eval_L_test`、`probe`、`cnn` | g7 | 无（排在 N0 的 g7 任务之后） |
| N2 | `eval_<S>_select`（S0–S4） | 优先原卡，可用 g1–g6 任一 | 对应 sft（timeout_ok） |
| N2 | `select_winner` | cpu | 5 个 select 评测 + `eval_L_select`，`all_finished` |
| N2 | `screen_0…screen_5` | g1–g6 | select_winner |
| N2 | `screen_merge` | cpu | 6 个 screen |
| N3 | `rl_RSFT, rl_Rfull, rl_Rbinary, rl_Rnoproc, rl_Rfullp` | g1–g5 | screen_merge |
| N3 | `eval_<S>_test/rot/shuf`（S0–S4、S1p） | g6、g7 | 对应 sft |
| N4 | `eval_<R>_test/rot/shuf` | g1–g7 任一 | 对应 rl（timeout_ok） |
| N5 | `backup` | cpu | 全部，`all_finished` |

- 每个评测任务后面接一个 `score_v5` 的 cpu 任务（依赖该评测）。
- 超时：gate 20 分钟；SFT 300；RL 210；评测按冒烟速度 ×1.5 估，最少 20 分钟；screen 60。
- 输出计划同时打印一张预计时间表（按冒烟速度估算每张卡的累计时长），交给会话 C 写进冒烟报告。
- 测试：生成的计划无环、每个依赖 id 都存在、每个 GPU 任务都有卡、S3 系列评测用的是 `_x2` 数据、RL 任务都依赖 screen_merge。

### 16. `backup.sh`
- 把每条线的最终（或时间盒下最大步）checkpoint、`v5_config.json`、`runtime_gate.json`、`reward_audit.jsonl`、全部评测原答与 metrics 打包到 `/WS/v5/backup/v5_night_<日期>.tgz`，生成 `backup_manifest.json`（每个文件 SHA）。不删除任何原文件。

---

## 交付报告（`交付报告.md`）
必须包含：
1. 文件清单与每个文件的 SHA、来源（复制自哪个旧文件）。
2. 所有测试的输出（PASS/FAIL 计数，失败为 0 才算交付）。
3. 每个 config 的 `--dry-run` 输出。
4. `DECISIONS.md` 全部内容（尤其是需要冒烟确认的 ms-swift 参数、视觉塔调用方式）。
5. 本会话没做到的事。
git commit。告诉用户：然后直接进入任务卡 C。

## 禁止
- 不改合同数值；不发明合同之外的奖励项、指标或比较对。
- 不修改任何旧文件（只复制）。
- 不在本机尝试安装 ms-swift / torch（会话 C 在服务器上用现成环境）。
- 不写任何凭据或真实服务器路径。
