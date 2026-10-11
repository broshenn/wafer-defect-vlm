# 任务卡 A：数据引擎（本机 CPU）

> **01:50 覆盖修订（优先于本卡正文）**：①遇到问题不停，按总纲 0.4 决策表处理并写 `DECISIONS.md`。②规模已压缩：SFT 文件每个 12000 行，合成图 1500 张；文件名里的 `800`、`320`、`200`、`100` 等数字保持原名不改，以总纲 0.3 的实际行数为准。③评测文件**额外**生成 `test200_main.jsonl`（TEST 分层 200，seed 3407）、`rot100_main.jsonl`（取 ROT200 前 100）、`shuf50_main.jsonl`（取 SHUF100 前 50）及 `_x2` 版；P0 只用这些。④第 8 步叠加核验图降为 P2：时间紧就跳过并在交付报告写 not run。⑤目标 **03:00 前完成本卡**，完成后直接进入任务卡 B。

> 先完整读 `00_总纲与合同.md` 第 2–5 节和第 10 节。本卡只是把合同拆成步骤；**数值以合同为准**。
> 工作目录：`D:/pycode/晶圆图研究/协作/02_ClaudeCode_实操/v5_感知训练_20261012/A_数据/`
> 本机 Python：用仓库里以前跑通的同一个解释器（有 numpy、PIL、pandas；**没有 scipy**）。先 `python -c "import numpy, PIL; print(numpy.__version__, PIL.__version__)"` 确认。
> 不需要 GPU、不需要联网、不调用任何模型。

## 工作方式（必须遵守）
- 每一步写成**一个独立的小脚本**（文件名见下），脚本开头有中文注释说明输入、输出、验收。
- 每个脚本跑完打印验收结果；**验收不过就停**，把现象写进 `DECISIONS.md`，不要改合同数值。
- 每完成一步：在 `PROGRESS.md` 打勾、写一行结果摘要，然后 `git add` 本目录并 `git commit -m "v5-A: 第N步 ..."`。
- 上下文快满或中断后：先读 `PROGRESS.md`，从第一个没打勾的步骤继续，**不要重做已打勾的步骤**。
- 所有随机数：`random.Random(3407)` 或 `numpy.random.default_rng(3407)`，每个脚本只建一次生成器。
- 所有输出 jsonl：UTF-8、`\n` 换行、`ensure_ascii=False`。每生成一个数据文件，就把它的行数和 SHA256 追加到 `数据清单.json`。

---

## 第 0 步：准备
- [ ] 建目录 `v5_感知训练_20261012/` 及子目录 `00_合同副本/ A_数据/ B_代码/ C_冒烟/ D_夜跑/ E_结果/`。
- [ ] 把 `D:/pycode/长鑫项目/学习计划_20261011/v5执行手册/` 整个复制到 `00_合同副本/`，算出 `00_总纲与合同.md` 的 SHA 写进 `A_数据/PROGRESS.md` 第一行。
- [ ] 把合同 3.1 节的题面逐字写入 `A_数据/prompt_v5.txt`（不含首尾空行），记 SHA。
- [ ] 把合同 5.3 节六个模板逐字写入 `A_数据/atomic_templates.json`（`{"T1": "...", ...}`），记 SHA。

## 第 1 步：`step1_geometry_all.py` —— 全量几何真值
输入：`data/manifest.jsonl`、`data/images/*.png`。
做法：
1. 用 `importlib` 按路径加载 `协作/01_Codex_指挥/无卡CPU整备_20261010/core_v3.py`（它内部会加载钉住 SHA 的几何 v2；SHA 不符会报错——报错就停）。
2. 用同样方式加载 `协作/02_ClaudeCode_实操/口径修正与准备_20261002-021207/wafer_png_v2.py`。
3. 对每个 manifest 行：`recover_matrix(png, hint_hw=tuple(row["matrix_shape"]))`（用几何 v2 模块里的 `recover_matrix` 包装函数），要求 `roundtrip_validated=True`；然后 `g = core_v3.measure_v3(matrix)`。
4. 判可用（合同 4.1）。可用的写入 `geom_all.jsonl`：`sample_id, split, lot_name, failure_type, png_sha256, matrix_hw, status, coverage, coverage_band(=COV_MAP[g["coverage_level"]]), red_mass_zones, angular_type(=ANG_MAP[g["direction_type"]]), clock_sectors, zone_frac[3], R1, R2, n_red`。不可用的写 `ineligible.jsonl`（含原因）。
5. 同时把矩阵存成 `matrices/<sample_id>.npy`（后面旋转和合成要用，避免重复反解）。
验收（打印）：
- 总数 = 5904；可用 + 不可用 = 5904。
- 各 split 可用数量；各 coverage_band、angular_type 的分布。
- 抽 5 张 `no_red` 图（若有）确认 zone_frac 全 0、R1=R2=0。
- 耗时。若超过 2 小时未完成，用 `multiprocessing.Pool(进程数=CPU核数-1)` 并行，结果按 sample_id 排序后再写。

## 第 2 步：`step2_splits.py` —— 划分与隔离检查
1. 按合同 4.2 读取 8 个受保护文件里的全部 `sample_id`。每个文件格式不同：先打印每个文件的前 300 个字符，看清结构再写读取函数；对每个文件打印"读到 N 个 ID"。**任何一个文件读到 0 个 → 按 0.4 决策表处理，写 DECISIONS.md。**
2. 按合同 4.3 生成 `TRAIN_POOL.txt`、`SELECT200.txt`、`TEST.txt`、`ROT200.txt`、`SHUF100.txt`（每行一个 sample_id，排序后写出）。
3. 写 `isolation_check.py`（独立脚本，可被第 7 步复用），函数 `check(train_ids, eval_ids, meta)` 检查 sample_id / lot_name / png_sha256 三层交集，交集非空或任一输入为空都 `raise`。
4. 反例测试 `test_isolation.py`：①正常输入通过；②往 train 副本里加 1 个 TEST 的 ID → 必须报错；③传空 train → 必须报错；④构造一个与 TEST 某图 SHA 相同的假条目 → 必须报错。四项都要打印 PASS。
验收：打印四个集合的大小、各类分布、HOLD56 与 CONF120 在 TEST 里的命中数、`isolation_check` 结果；`test_isolation.py` 4/4 PASS。

## 第 3 步：`step3_rotations.py` —— 旋转增广 + 程序等变性自检
1. 对 TRAIN_POOL 每张：读 `matrices/<id>.npy`，`np.rot90(m, k=-1/-2/-3)` 得三个矩阵；`matrix_to_image(m_rot, 448)` 存 `img448/<id>_r090.png` 等；`measure_v3(m_rot)` 重算几何；可用性同第 1 步（不可用的旋转版丢弃并计数）。几何写 `geom_rot.jsonl`（字段同第 1 步，另加 `src_id`、`rot_deg`）。
2. 原图也复制到 `img448/<id>.png`（TRAIN_POOL、SELECT200、TEST 全部复制，后面统一从 `img448/` 取）。
3. 等变性自检（合同 5.1）：在原图 angular_type 为 single 的样本上比较 r090 扇区与原扇区 +3；比较三个旋转版的 coverage_band 与 red_mass_zones 是否与原图完全相同。输出 `rotation_selfcheck.json`：总数、一致数、不一致清单（前 50 条）、因并列变 unknown 的数量。
验收：扇区一致率 ≥ 95%（排除变 unknown 的）；coverage_band 与 zones 不变率 ≥ 99%。不达标 → 按 0.4 决策表处理，写 DECISIONS.md（这说明几何定义对旋转不稳，必须先裁决）。

## 第 4 步：`step4_synthetic.py` —— 合成图
严格按合同 5.2。实现要点：
- 写一个函数 `uv_of(matrix)`：调用几何 v2 模块的 `_ellipse_of(valid_mask)` 与 `_normalized_uv(H, W, cy, cx, a, b)`，返回 `u, v, r, deg, valid`。
- 七个图案各写一个函数，输入 `(base_matrix, u, v, r, deg, valid, rng)`，返回新矩阵（只改 valid 内的格，值 2 = 红）。
- 生成 3000 张 → 用 `measure_v3` 重算 → 丢弃不可用 → 按 angular_type 挑 1500（合同 5.2）。
- 存 `img448/syn_xxxxxx.png`，几何写 `geom_syn.jsonl`（另加 `base_id, pattern, params`）。
- 单元测试 `test_synthetic.py`：①center_blob 半径 0.2 → 程序测得 zones 含 center 且不含 edge；②edge_arc 在 3 点方向 → 程序测得 angular_type=single 且扇区为 3（若刚好落在并列容差内则换 θ=95° 重试）；③ring r_in=0.8 → zones 只含 edge；④random_scatter p=0.2 → coverage_band 为 `20_to_under_50pct` 或 `5_to_under_20pct`；⑤所有红点都在 valid 内。5/5 PASS 才继续。
- 渲染 20 张合成图的联系表 `syn_contact_sheet.png`（4×5 网格，每格下方写图案名和程序测得的 coverage_band / zones / angular_type），供用户肉眼看。
验收：最终 1500 张；打印各 angular_type 实际数量、各图案数量；5/5 PASS。

## 第 5 步：`step5_build_sft.py` —— 六个训练文件
严格按合同 5.3–5.4。实现要点：
- 函数 `main_answer(geom_row, public_class)`：生成合同 3.2 的两行答案（测量行 + JSON）。JSON 的六字段用 `core_v3.reference(g, public_class)` 的同一映射；为了不重复反解，可以直接用第 1/3 步保存的字段，但**必须**写一个校验：随机抽 200 条，用 `core_v3.reference(measure_v3(np.load(...)), cls)` 重算，与生成的 JSON 完全一致。
- 函数 `atomic_answer(geom_row, template_id)`：按合同 5.3 表格。
- 图池抽取：主任务池 = TRAIN_POOL 原图 + 旋转版（S0/S1/S2/S3/S1p）；S4 只有原图。"无放回抽完一轮再下一轮"：每轮先把池打乱一次。
- 原子段的模板分配：先按比例算出每个模板的条数（四舍五入后调整到总数 4000），T6 只从 angular_type∈{single, opposed} 的图里抽；其余模板从全部图里抽。S1 原子段图源：真实/旋转 2500 + 合成 1500。S4 原子段：只用原图 4000。
- 写出 `sft_S0.jsonl … sft_S1p.jsonl`。S2、S1p 与 S1 字节相同（直接复制并核 SHA 相同）；S3 与 S1 相同但图片名换 `_x2.png`。
验收（打印并写 `sft_stats.json`）：
- 每个文件行数 = 12000；S1/S2/S1p SHA 相同。
- 每个文件：原子段/主任务段条数、各模板条数、图源（real/rot/syn）条数、主任务段里 angular_type 为 single/opposed 的条数。
- 随机抽 5 条主任务和 5 条原子问答，原样打印出来（用户要看）。
- 200 条重算校验 200/200 一致。

## 第 6 步：`step6_build_eval.py` —— 评测与 RL 文件
按合同 5.4 生成：`select200_main.jsonl`、`test_main.jsonl`、`rot200_main.jsonl`、`shuf100_main.jsonl` 及各自 `_x2` 版、`select200_L.jsonl`、`test_L.jsonl`、`rl_candidates800.jsonl`、`rsft_rows_template.jsonl`。
- ROT200 的图：TEST 原图顺时针 90°，存 `img448/<id>_r090.png`，几何用 `measure_v3` 重算；`reference_json` 用旋转后的几何；另存字段 `src_id`。
- SHUF100：对 448×448×3 的像素数组做**同一个**置换（`perm = rng.permutation(448*448)`，一次生成，全部图共用），存 `img448/<id>_shuf.png`；`reference_json` 仍用原图的几何（打分时看模型是否还能答对——不应该能）。
- `reference_json` 格式按合同 5.5（含 `measure` 舍入值与 `measure_exact` 原值）。
- L 题面：从 `协作/02_ClaudeCode_实操/夜间扩容_20261008_0244/数据/L_N3072.jsonl` 第一行取 user 内容，原样使用（它以 `<image>` 开头）；参考只要公开类别。
- `rl_candidates800`：TRAIN_POOL 原图分层抽 320（seed 3407）。`rsft_rows_template`：每张 4 行相同的正确主任务答案。
验收：每个文件行数正确（SELECT200=200、TEST=全部可用 test、ROT200=200、SHUF100=100、rl=320、rsft=1280）；每个文件跑一遍 `isolation_check`（评测文件 vs TRAIN_POOL 必须 0 交集；rl_candidates 必须全部属于 TRAIN_POOL）。

## 第 7 步：`step7_img896.py` —— S3 用的 896 图
- 找出 `sft_S3.jsonl` 与全部 `_x2` 评测文件里引用的图片，对 `img448/` 里的对应原图用 `Image.resize((896, 896), Image.NEAREST)` 存 `img896/<名>_x2.png`。
验收：引用的每一个 `_x2` 文件都存在；抽 3 张检查尺寸 896×896、颜色只有黑绿红三种。

## 第 8 步：`step8_overlays.py` —— 叠加核验图（给会话 F 和用户看）
- 从 TEST 分层抽 120 张（seed 3407）+ 合成图 20 张，画叠加图存 `overlays/<名>.png`：
  - 原图放大到 672×672（最近邻）；
  - 用几何的椭圆中心和半轴画外轮廓（白线）、r=1/3 与 2/3 两条区带分界（黄色虚线）；
  - 从中心画 12 条扇区分界线（灰色细线，分界在 12 点两侧 ±15°，与 v3 钟点契约一致），在外圈标 1–12；
  - 若 angular_type 为 single/opposed，把对应扇区涂半透明蓝色；
  - 图下方白底黑字写：`coverage=0.27 (20_to_under_50pct) zones=[middle,edge] angular=single sectors=[1]`。
- 写 `overlays/index.jsonl`：每张的文件名、sample_id、程序真值六字段（不含类别）。
验收：140 张都生成；打开 3 张自查文字和线条位置（DeepSeek 不能看图的话，就打印每张图的尺寸并把 3 张路径写进交付报告，请用户看）。

## 第 9 步：打包与交付报告
- 把 `img448/` 和 `img896/` 分别打成 `img448.tgz`、`img896.tgz`（用 Python `tarfile`），记 SHA 和文件数。
- 写 `交付报告.md`，必须包含：
  1. 合同 SHA；`prompt_v5.txt`、`atomic_templates.json` 的 SHA。
  2. 每一步的验收结果（直接贴数字）。
  3. `数据清单.json` 的内容（文件名、行数、SHA）。
  4. 四个集合的大小与类别分布；HOLD56、CONF120 命中数。
  5. 方向样本统计：TRAIN_POOL 原图、旋转、合成、各 SFT 文件主任务段中 single/opposed 的数量；TEST 中 single/opposed 的数量（这决定扇区指标有没有分母）。
  6. 随机抽的 10 条训练样本原文。
  7. 所有 DECISIONS.md 里未解决的问题。
  8. 本会话**没做**的事（如果有）。
- git commit。告诉用户：交付报告路径；然后直接进入任务卡 B。

## 禁止
- 不改合同里的任何数值、阈值、文件名、题面文字。
- 不修改 `core_v3.py`、几何 v2、`wafer_png_v2.py` 或任何旧文件（只读加载）。
- 不用任何模型生成标签；所有标签只来自公开 `failure_type` 和程序几何。
- 不把 SELECT200 / TEST / ROT200 / SHUF100 的任何图用于训练或增广来源。
