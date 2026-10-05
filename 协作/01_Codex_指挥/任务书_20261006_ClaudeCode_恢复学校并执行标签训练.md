# 给Claude Code：恢复学校接入，直接执行原标签训练

**签发：Codex，2026-10-06。用户要求继续执行。** 当前数据已准备，不重新选教师、不重新标注。目标：学校服务器完成20条/20步小测试，通过后直接完成180条/60步SFT-A。沿用原单卡、累计两GPU小时授权，不逐步等待批准。

## 1. 先恢复接入并核对已有任务

工作目录：`D:/pycode/晶圆图研究`。先读根AGENTS §18和协作约定；接入方式仅从仓库外 `D:/pycode/tailscale-code-browser/AGENT_USAGE.md`、当前用户家目录 `.codex/skills/tailscale-ssh-hop/SKILL.md`、`school-a100-server-usage/SKILL.md`取。

短检查两跳连通和SSH认证，复用合法现有认证，不猜密码、不输出/存凭据，不从网页Cookie取得执行权限，不循环长超时。连接成功先核对本人工作区、已有tmux/训练日志与checkpoint：原任务已在跑则接续/回收，不重复启动。远端账号、IP与真实私有路径不写公开仓库。

重新检查当时GPU状态。原已授权GPU0空闲且使用条件有效时直接使用，不再问同一许可；被他人占用不能抢占或擅自共用。其他卡只有明确获得分配才用，仍单卡。不终止他人进程，不改公共环境、容器或服务。

复用已经建立的个人环境、Qwen3.5-9B权重和公开图片；核实实际版本，不重装、不重下完整模型。历史环境为torch 2.6.0+cu124、transformers 5.16.1、ms-swift 4.5.3，须以当前现场为准。模型revision若只取得master，补实际文件SHA256清单，不冒称固定commit。

## 2. 数据包现成，传输后只重映射路径

数据包：`D:/pycode/晶圆图研究/协作/01_Codex_指挥/训练数据就绪_20261003/训练数据包_20261003.zip`

SHA256：`8c9c90c0ed084ef565a9de2739dd03fca28c5a92d8e20d802a3797ce8d7ca469`。

该包320128字节，含198张原PNG及datasets/下的训练文件。传到个人获准目录，核对整包hash，解压到新版本目录，检查成员路径不越界，不覆盖历史输入。已有图片匹配hash可复用，缺项补传。模型权重与凭据不在包内。

**本次仅使用：**

- `datasets/sft_a_smoke_20.jsonl`，SHA256 `3d2d133b48807ed4483a29cb7a3b6133ef18c53ab2c408cecd42706be4a6e05e`。
- `datasets/sft_a_train_180.jsonl`，SHA256 `7b306f00b9d7da970655099512a8e05a6df32d31c134a942c207d197b89be861`。
- `datasets/sft_a_dev_18.jsonl`，SHA256 `5a96fe15321865c005370610aebf66e1cc8e5282bf9881e5f70351f36c9fa19c`。

三份文件仍含Windows图片路径。服务器另存转换版，仅重映射images到实际PNG目录：Windows反斜杠路径用PureWindowsPath取文件名，不能用Linux Path直接取错。保留全部样本ID、messages、题面、原类别和顺序，验证每图存在且hash匹配。源JSON与路径转换JSON分别记录hash，转换版私有绝对路径只留服务器，不回传公开仓库。

隔离与原始元数据来源：`协作/02_ClaudeCode_实操/学校训练准备_20261002-025848/`的原train_180/dev_18/smoke_20/dataset_plan，以及 `data/manifest.jsonl`（SHA256 `c2f8a9afbf7ab3bc7bac92dc2f5896f8896bf379de2ac1c9d3667afcebcbaa19`）。必要时传这些公开元数据，独立核对训练/开发/测试/benchmark lot交集，避免从全是train的集合取空val/test；派生benchmark ID映射原晶圆。当前首批198图足够启动原训练；全量5904图上传另记完成状态，不冒称已全部传完。

**不使用caption_train_158、multitask_candidate_338或structured_candidates_90训练。** 它们仅是后续描述候选，不属于本次GPU授权。分类答案来自原公开标签，不使用GLM预测覆盖原类别。

## 3. 基座、模板、冻结检查通过后立即训练

在实际固定ms-swift版本下确认视觉模板、真实pixel_values/image_grid与回答loss mask，只对assistant回答监督。核实LoRA可训练对象，ViT/aligner冻结；参数名字存在不等于实际冻结。保存非敏感配置与可训练参数摘要。

核对旧基座dev18原答及实际题面/系统消息/模板/解码设置。完全同题才可复用；否则先重跑这18张基座，保存原答，不把题面改变算训练提升。

原冻结LoRA起点：rank16、alpha32、dropout0.05、LLM all-linear、freeze_vit/aligner、LR1e-4、BF16、max_length2048、image token256、gradient checkpointing、seed3407。具体参数按现场ms-swift核对，不套旧VisualGLM层名、padding或历史路径。

1. **小测试：**20条，max_steps20、batch1、GA1。保存有限loss、图像输入、回答mask、LoRA实际更新和checkpoint；加载Adapter做明确记录为train/dev的分类推理。进程启动/退出0不能代替这些通过条件。
2. **通过就继续：**180条，max_steps60、batch1、GA4，其他冻结配置相同。独立输出目录，固定选择最终第60步checkpoint，不挑最好。基座和Adapter在同dev18、同题面、同解码/解析规则下评测。
3. 分类输出只有defect_class一个字段，不能用GLM七字段检查器直接判它。保存原答，报告原始解析/类别结果；如使用别名规范化，Base与Adapter都同规则，分列原始与规范化结果，解析失败不静默剔除。

用tmux或实验室调度器保活，完整记录日志、实际步数、GPU型号/卡数、峰值显存、耗时、数据/代码/模型/环境版本。累计GPU推理+训练不超过原120分钟：先核原账（旧报告约10分钟已用），断连期间已有任务是否继续运行先查证，不从0重算。数据准备/传输耗时单列。

在原预算内诊断修复技术错误；非有限loss、泄漏、错误视觉模板、资源权限问题、合理缩减后仍OOM、预算到达或扰动他人任务时停止。未合并就写not run，不能把Adapter加载冒称合并验证。不开新参数网格、Caption训练或RL，不购云GPU。

## 4. 卡住也给出具体结果

连接/认证/GPU无法满足时，完成能做的路径转换、完整性和启动配置，报告卡在第几跳、具体错误与尚缺的实际条件；训练明确not run。不无限等待、高频轮询或自动抢卡。

新产物只写自己的新日期目录 `协作/02_ClaudeCode_实操/`，不改Codex/ZCode文件和历史快照。六项交付：已核实与出处、未知项、改动、下一步、证据路径、真实消耗。成功时交小测试和60步的日志/原答/Adapter位置别名；未运行/未完成逐项区分。

完成这两轮或到预算后停止，将结果由用户转回Codex验收。**不要再次停在“准备好了、要不要开始”：条件满足、小测试通过后，原60步已获授权，直接执行。**
