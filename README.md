# 晶圆图缺陷 VLM —— 本地工作副本

2026-09-19 整理。这里是**服务器上那个项目的一份离线快照**，加上我自己工作过程留下的补丁与检查脚本（都在 `archive/`，不参与阅读）。

- **项目是什么**：用 Qwen3.5-9B + ms-swift 对晶圆缺陷图做 LoRA/QLoRA SFT，再做 GRPO/GSPO 强化学习后训练，在自建 benchmark `wafer_bench_v1` 上评五类任务（分类 / 结构化字段 / Caption / 鲁棒性 / 检索），结论写进三份互相校验的文档。
- **服务器在哪里**：AutoDL 实例，项目根 `/root/autodl-fs/wafer-vlm`。**目前实例已关机**；数据在独立的网络盘上，重启不丢。权重（约 119 G）没有拉下来，本机也没有；要重跑训练才需要开机。
- **哪些是权威的**：`docs/` 和 `results/` 里的东西是从服务器原样搬过来的（只移动，未改写）。

---

## 从哪里开始读

| # | 读什么 | 为什么 |
|---|---|---|
| 1 | `docs/FINAL_REPORT.md`（212 行） | 全项目的结论摘要。先读这个 |
| 2 | `docs/comparison.md` | **12 个 run × 16 项指标的主表**，所有讨论的锚点 |
| 3 | `docs/LIMITATIONS.md`（1530 行） | 重点两节：**§5.2.x** 每条结论的限定语，**§8** 20 条失败分类（这个项目的灵魂） |
| 4 | `results/readouts/error_cases_sft.md` | 最自信的 20 个错误 + 最自信的 20 个正确，人可读 |
| 5 | `samples/*.png` | 9 个缺陷类别各一张代表图，直接看 |
| 6 | `results/reports/seed_variance.json`、`paired_significance.json` | 三次抽样的噪声地板、配对显著性检验 |
| 7 | `logs/land_finish_recheck.log` 尾部 | 文档自检体系的最后一段输出 |

想跑代码或看流水线设计，读 `docs/README-project.md`（服务器上那份 README，含从 WM811K 到训练评测的完整复现路径）和 `AGENTS.md`（项目宪章，约束全部工作）。

---

## 目录

| 目录 | 文件数 | 体积 | 是什么 |
|---|---|---|---|
| `docs/` | 6 | 0.2 MB | 三份文档 + 主表 + 项目 README + 参考文献精读 |
| `results/` | 103 | 20.5 MB | 报告 JSON（51）、原始模型输出（12 个 run + 5 个辅助）、检索排名（12）、损失曲线（11）、合并校验（7）、人可读摘要（5） |
| `benchmark/` | 18 | 5.2 MB | benchmark 冻结件：五类题面、`SHA256SUMS`、`review/`（两张审核表） |
| `data/` | 5909 | 26 MB | `manifest.jsonl`（5904 行，含 `lot_name` 与 `split`）+ 5904 张 448×448 预处理图 |
| `code/` | 202 | 1.4 MB | 服务器上的代码：`tools/`（检查器）+ `projects/wafer-defect-vlm/`（src / tests / scripts，逐字节照搬） |
| `logs/` | 116 | 10 MB | 队列、训练、评测、收尾的日志 |
| `samples/` | 10 | 0.03 MB | 9 个类别各一张代表图 |
| `papers/` | 10 | 9.1 MB | 参考论文 PDF 与其附带数据 |
| `archive/` | 314 | 16 MB | 历史与工作副本 —— **不参与阅读主线**，见下 |

根目录只留 4 个快照文件：本文件、`AGENTS.md`、`.env.example`、`.gitignore`。
（另有四份**我加的、不属于快照**的文件：`面试讲解.md` 是把项目讲给面试官听的版本；
`阅读指南.md` 是阅读路线图 —— 五条主线、三条路线、代号表、以及「哪些文件会把你带偏」；
`第一课_QLoRA与SFT.md` 讲训练方法本身（SFT / LoRA / QLoRA 分别解决什么）；
`面试学习路线与事实底稿.md` 是学习入口 —— **它把「能核对的事实」和「需要修正的解释」分开写**，
里面有些地方是在纠正早期口述里的说法，以它为准。
这四份的数字都引自 `docs/` 与 `results/reports/`，没有新算的。）

### `archive/` 里是什么

| 子目录 | 是什么 |
|---|---|
| `patches/` (135) | 逐个改服务器的补丁脚本 `patch*.py` |
| `working-copies/` (70) | 根目录散放的检查/诊断脚本 |
| `queue-scripts/` (21) | 一次性队列脚本 |
| `local-skeleton/` (74) | 本机 9 月 14/15 日的旧骨架（src/tests/scripts/tools…），已被 `code/` 取代 |
| `commit-messages/` (4) | commit message 草稿 |
| `superseded-2026-09-15/` (1) | 根目录那份**过期的** `LIMITATIONS.md`（31 KB；现行版见 `docs/LIMITATIONS.md`，125 KB） |
| `tarballs/` (4) | 当晚从服务器拉取的 4 个原始压缩包，解压即得未经整理的形态 |
| `tooling/` (4) | 本次整理的三个脚本 + `_verify_reorg.py`（内容完整性校验，可随时重跑） |
| `REORG_MAP.txt` | 本次整理做了什么、删了什么、怎么倒回去 |

---

## 这个项目说了什么、没说什么

**说了**：零样本分类准确率 0.2143 → SFT 0.6230；12 个 run 里最高 0.6429（`GRPO_lr1e5_s3`）；检索 mAP@10 从 0.3236 到最高的 0.4337（`GSPO_G32`）；`root-cause hallucination rate` 从 0.2183 降到 0.0000。

**主要结论是否证**：预注册的假设是"序列级归一化（GSPO）比 token 级（GRPO）更能容忍大学习率"。结果**不支持**——两种归一化在测过的组大小上不可分；而学习率本身的效应在 4 个可用格子里有 3 个显著。

**噪声地板被触发**：换种子规则（种子差 ≥ lr 效应的一半即判定不可分）在第三个抽样后触发，范围 0.0357 ≥ 阈值 0.0278，§5.2.2 已按此重写。

**没说的**（重要）：benchmark 仍是 `draft_pending_human_review`，`gold_withheld: true`——`benchmark/review/` 里两张 252 行的审核表**每一列标签都是空的**，人工双审这一步**没有做，也没有被伪造**。另外 `radial_zone` / `clock_direction` 两个字段被收集、被提问、被评测，但**没有任何东西消费它们**，这个问题还悬着（而模型恰好在这两个字段上最弱）。

---

## 注意

- 根目录曾经有一份 `LIMITATIONS.md`（9 月 15 日，31 KB）**是过期的**，已移到 `archive/superseded-2026-09-15/`。别读它——它把第 14 条之后的整个 §8 和 5.2.7/5.2.8 全缺了。
- `results/raw_outputs/*.jsonl` 每个文件 1 MB 且"一条记录一行"，**别用记事本打开**；用 `python -c` 或 `head -c` 读。同样的建议适用于 `data/manifest.jsonl`。
- 整理只移动文件，**没有改写任何内容**；完整性由 `archive/tooling/_verify_reorg.py` 校验（把 4 个 tarball 解出来，逐个文件按 md5 在新树里找）：6418 个文件里 6355 个按内容找到，63 个是故意删掉的构建缓存，**0 个丢失**（另有 14 个本机 `.pyc` 也被删，它们不在压缩包里，所以内容校验覆盖不到——账目和推断值都写在 `archive/REORG_MAP.txt` 里）。
