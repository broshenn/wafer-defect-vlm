# 晶圆图缺陷 Qwen3.5-9B 研究：全局开发约定

本文件作用于整个仓库。后续在服务器上进行数据处理、训练、评测和部署时，所有开发者与 AI Agent 都必须先阅读并遵守本文件。

## 1. 项目目标与技术选型

目标：建立一条可复现的“晶圆 BIN 图 → 高质量多模态数据 → Qwen3.5-9B 后训练 → Caption/分类/结构化理解 → 图像检索评估”全流程。

已经确定的主方案：

- 基础模型：`Qwen/Qwen3.5-9B`，它是带 Vision Encoder 的统一视觉语言模型，不使用纯文本模型替代。
- 后训练框架：阿里 ModelScope `ms-swift` 4.x。
- 训练方法：先做多模态 SFT/LoRA；只有 SFT 基线稳定后，才考虑 DPO、GSPO/GRPO 或其他强化后训练。
- 原 VisualGLM-6B、SwissArmyTransformer、自定义 LoRA 和旧 EVA-ViT 训练代码仅作为历史对照，不作为新主线。
- 所有模型、框架和依赖必须固定具体版本或 Git commit，禁止只记录 `latest`。

官方参考：

- Qwen3.5-9B：https://huggingface.co/Qwen/Qwen3.5-9B
- ms-swift：https://github.com/modelscope/ms-swift
- ms-swift 自定义数据：https://github.com/modelscope/ms-swift/blob/main/docs/source/Customization/Custom-dataset.md

如果 mentor 指定的“阿里后训练框架”不是 `ms-swift`，必须先确认框架名称，再修改训练方案，不得自行猜测。

## 2. 安全与数据合规

- 禁止在代码、文档、日志、数据样例或 Git 历史中写入 API Key、Token、密码、工号、内网地址、服务器凭证和客户信息。
- 所有密钥只允许通过服务器环境变量或未纳入版本控制的 `.env` 注入。
- 公司内部数据必须确认允许迁移后才能上传服务器；不能迁移时只使用公开 WM811K 或经过批准的脱敏产物。
- 日志中的路径、lot 名和样本内容也要按公司要求脱敏。
- 提交前必须扫描敏感信息；发现曾经泄露的凭证时，不仅删除文本，还应立即轮换凭证。

## 3. 新旧方案对应关系

| 原方案 | 新方案 |
|---|---|
| VisualGLM-6B | Qwen3.5-9B 原生多模态模型 |
| SAT/DeepSpeed 自定义训练 | ms-swift `swift sft` |
| 手写 LoRA | ms-swift/PEFT LoRA 或 QLoRA |
| Stage 1 自定义 CE+InfoNCE | 首先取消；以零样本基线和标准多模态 SFT 为主 |
| Stage 2 Caption 微调 | ms-swift 标准 `messages + images` JSONL |
| 只训练英文 Caption | 推荐中英文 Caption + 结构化识别多任务 |
| 单一 LLM Judge | 数学规则为主、人审为准、LLM Judge 为辅 |
| 单 query/类检索 | 多 query、跨 lot、包含 hard negatives 的固定 Benchmark |

不要直接把 VisualGLM 的 `<img> + 32 pad token`、QFormer、LoRA 层名或 SAT 参数迁移到 Qwen3.5。Qwen3.5 的图像模板、视觉 token 和可训练模块必须交给对应版本的 ms-swift 处理。

## 4. 推荐目录结构

```text
.
├── AGENTS.md
├── configs/
│   ├── data.yaml
│   ├── sft_lora.yaml
│   ├── sft_vision.yaml
│   └── benchmark.yaml
├── data/
│   ├── raw/                  # 原始 pkl，只读
│   ├── images/               # 标准化 PNG
│   ├── metadata/             # 清洗后的表和统计
│   ├── splits/               # train/val/test JSONL
│   └── quarantine/           # 无效、冲突、低置信样本
├── benchmarks/
│   └── wafer_bench_v1/       # 发布后只读
├── src/
│   ├── data/
│   ├── benchmark/
│   ├── train/
│   ├── eval/
│   └── retrieval/
├── tests/
├── outputs/
│   ├── baselines/
│   ├── checkpoints/
│   └── reports/
└── manifests/               # 版本、哈希、运行配置
```

原始数据目录必须只读。任何清洗和转换都写到新目录，不覆盖输入。

## 5. 数据主流程

### 5.1 统一样本清单

先生成唯一的 `manifest.parquet`，每行至少包含：

- `sample_id`：稳定且唯一，不依赖临时行号。
- `lot_name`、`wafer_index`。
- `failure_type`：原始标签。
- `predicted_type`：如果存在，必须单独保存，不能冒充 ground truth。
- `label_source`：`ground_truth`、`predicted`、`human` 或 `unknown`。
- `image_path`、原矩阵 shape、缺陷像素数、有效 die 数。
- 原始 Caption、二次 Caption、各自置信度与最终 Caption 来源。
- 数学特征、异常标志和数据版本。

`predictDefectType` 的生成方式目前未知。在来源查明前：

- 不得把它描述为人工真值。
- 分类评测只使用可信的 `failureType` 或人工标签。
- 使用预测标签训练时必须单独报告，不可与真值训练结果混合。

### 5.2 图片标准化

主数据版本统一使用：

- `0 → 黑色 [0,0,0]`
- `1 → 绿色 [0,255,0]`
- `2 → 红色 [255,0,0]`
- 最近邻插值。
- 默认 `448×448`。
- 保存无损 PNG，不使用 JPG，避免离散 BIN 图产生压缩伪影。
- 默认不添加坐标轴、文字、圆圈等人工提示。
- Caption 生成、训练、验证和测试必须使用同一渲染函数。

去噪版本和原始版本必须分别保存，禁止静默覆盖。若 Caption 是根据未去噪图片生成，训练时原则上使用同一版本图片。

### 5.3 Caption 筛选修正规则

修复旧 `tran2Json.py` 漏样本的问题，最终 Caption 选择规则统一为：

```text
若 primary_confidence >= 60：使用 primary_caption
否则若 secondary_confidence > 60：使用 secondary_caption
否则：进入 quarantine，不进入主训练集
```

要求：

- Primary 和 Secondary 应来自真正不同的模型或检查策略；同一模型重复调用不能宣称“双模型交叉标注”。
- 去除 `ERROR`、`无法描述`、空文本和无法解析的结果。
- 按 `sample_id` 去重。
- 保留原始中文，不强制只留英文。
- 英文翻译作为附加训练目标；翻译不得引入根因或新事实。
- `none` 样本应保留一部分，避免模型看到正常晶圆也强行生成缺陷，但必须控制比例。

### 5.4 教师模型选择与数据规模

禁止默认使用 GLM-4.5V 或旧 Qwen3-VL 作为主教师，也禁止让待训练的 Qwen3.5-9B 生成自己的主要训练答案。教师必须在晶圆领域小规模盲测后确定。

教师候选优先级：

1. `Qwen3.5-397B-A17B` 或当前可通过 API 获得的同级旗舰多模态模型。
2. `Qwen3.5-122B-A10B` 作为成本较低的候选。
3. 一个不同模型家族的当前旗舰视觉模型，用于交叉核验，具体型号按公司可用 API 决定。

当前已实际验证的 API 能力（2026-09-14）：

- 阿里云百炼 `deepseek-v4.1-flash`：文本和图片请求均成功。
- 阿里云百炼 `qwen3.5-397b-a17b`：文本和图片请求均成功。
- DeepSeek 官方 `deepseek-flash`：文本请求成功；该滚动别名当前指向 V4.1 Flash。

当前预算策略：优先消耗百炼 `deepseek-v4.1-flash` 免费额度，但免费模型不能未经验证直接成为训练真值。推荐执行：

1. DeepSeek V4.1 Flash 对候选数据做第一遍结构化 Caption。
2. Python 数学规则检查类别、方向、径向区域、尺寸和格式。
3. `qwen3.5-397b-a17b` 重标所有失败、边界和模型不确定样本。
4. Qwen3.5-397B 随机审计 DeepSeek 已通过样本的10%～20%。
5. 若审计错误率超过预设阈值，停止批量接受 DeepSeek 结果，扩大 Qwen 重标范围。
6. 只有规则通过且达到审计门槛的数据进入主训练集。

API Key 只能通过环境变量注入，严禁写入本文件或仓库。

先建立独立的 `teacher_selection` 集：

- 每类20～30张，覆盖典型、边界、噪声和多模式样本。
- 不与最终 Benchmark 重合。
- 所有候选使用相同图片、Prompt、解码参数和 rubric。
- 评分包含类别、形态、位置、尺寸、空回答率、幻觉率和人工盲评。
- 只有显著优于未微调 Qwen3.5-9B 的候选才能作为教师。

教师生成时可以同时输入图片和由图片确定性计算出的数学特征，以减少方位、尺寸和环形判断错误；学生训练时只输入图片和问题。

数据规模采用逐级扩展，而不是直接追求8万张：

- Pipeline smoke test：每类20张左右。
- 第一轮 SFT：约1,000～2,000张高质量唯一晶圆。
- 主训练集：约3,000～8,000张高质量唯一晶圆，尽量类别平衡。
- 每张图可派生2～3种合法任务，但必须保持在同一 split。
- 只有学习曲线仍持续改善且伪标签错误率稳定时，才扩展到10,000～20,000张。
- 不以样本数量作为质量指标；宁可保留5,000张可靠样本，也不训练80,000张弱教师噪声样本。

### 5.5 数据拆分

必须在生成多种 Prompt、增强图片或复制样本之前拆分数据。

- 默认随机种子：`3407`。
- 默认比例：train/validation/test = 80/10/10。
- 以 `lot_name` 为 group，任何一个 lot 只能出现在一个 split 中。
- 在 group 隔离的前提下尽量保持缺陷类别分布。
- 同一晶圆的原图、去噪图、旋转图、双语 Caption 和所有问答必须留在同一 split。
- Benchmark 样本一旦冻结，禁止回流训练集、Prompt 调试集或 Judge 调参集。

### 5.6 ms-swift 数据格式

使用 JSONL，每行一条多模态对话：

```json
{"messages":[{"role":"system","content":"你是半导体晶圆缺陷分析专家。"},{"role":"user","content":"<image>\n请描述晶圆图中的主要缺陷形态、位置和尺寸。"},{"role":"assistant","content":"一个致密的环形缺陷簇，位于晶圆边缘，宽度约为0.1R。"}],"images":["data/images/wafer_xxx.png"]}
```

训练目标建议包含三类，但同一图片的所有变体必须属于同一 split：

1. 自然语言描述：形状、位置、尺寸。
2. 结构化输出：缺陷类别、形态、径向区域、钟点方向、尺寸和 Caption。
3. 简短 VQA：类别、位置、形态、覆盖程度。

禁止在问题中透露答案，例如不要问“这是不是 Center 缺陷”。没有可靠根因标签时，不创建根因问答。

## 6. 数学特征使用原则

主 Judge 使用 `calculate_static_param` 类特征，但新实现要修正旧代码问题：

- 圆心和半径由有效晶圆轮廓拟合。
- 缺陷比例分母使用晶圆内有效 die 数，而不是整个矩阵面积。
- 径向直方图固定使用 `[0, wafer_radius]` 的10个 bin，不能使用“缺陷最大半径”作为终点。
- 角度以12点钟为0，顺时针划分12个扇区；坐标定义必须写入测试。
- 各向异性比若定义为 `lambda_min/lambda_max`，越接近0越线状，不能解释反了。
- 禁止继续使用硬编码 `52×52`、`R=26` 的旧 `calculate_geo_param`。
- 对无缺陷、滤波后无缺陷、单点、共线点和拟合失败必须有显式状态码。

数学特征用于：

- 数据质检与异常检测。
- 生成结构化 gold 字段。
- Benchmark 的确定性评分。
- LLM Judge 的事实依据。

数学特征不是人工真值的完全替代，边界样本必须人审。

## 7. Benchmark 构造规范

Benchmark 必须先于训练冻结，建议命名为 `wafer_bench_v1`，并保存版本清单和 SHA256。

### 7.1 子任务

至少包含以下五部分：

1. **九类识别**：Center、Donut、Edge-Loc、Edge-Ring、Loc、Near-full、Random、Scratch、none。
2. **结构化视觉理解**：形态、中心/中间/边缘径向区域、钟点方向、尺寸比例、覆盖率。
3. **Caption 质量**：形状、位置、尺寸事实一致性，以及幻觉率。
4. **鲁棒性**：不同矩阵尺寸、噪点密度、稀疏缺陷、边界案例和渲染扰动。
5. **图像检索**：多 query、跨 lot、hard negatives、粗粒度和细粒度相关性。

### 7.2 样本选择

- Core Benchmark 每类至少30条；不足时使用全部可信样本并明确标注。
- 每类同时覆盖典型样本和困难样本，不能只挑“最好看”的图。
- 每类至少10个 query 用于检索，禁止沿用“一类只有一个 query”的旧设计。
- Query 与相关 Gallery 不能来自同一 lot。
- Gallery 应混入外观相似但类别不同、类别相同但位置不同的 hard negatives。
- `none` 类单独报告，不能依靠数量优势抬高总体准确率。
- 多缺陷模式、标签冲突和低可信预测标签进入单独 challenge 集，不混入干净主榜。

### 7.3 Gold 构造

Gold 来源优先级：

```text
双人复核人工标注 > 原始可信标签 + 数学特征 > 单人标注 > Teacher VLM输出
```

- Teacher 生成的 Caption 不能直接作为唯一测试真值。
- 数学特征自动生成位置、径向和尺寸候选，然后由人工确认边界样本。
- 每条 Benchmark 样本记录 `annotator_count`、`agreement`、`gold_source`。
- 无法达成一致的样本放入 challenge，不进入主分数。

### 7.4 检索相关性

同时报告两套定义：

- 粗粒度相关：同一缺陷类别为相关。
- 细粒度相关：类别相同，并且形态、径向区域或主方向也一致/接近。

建议保存 graded relevance：

- 2：类别和关键空间形态均一致。
- 1：类别一致但空间形态不同。
- 0：类别不同。

AP/Recall 使用明确的二值规则；graded relevance 另外计算 nDCG。

### 7.5 评估指标

- 分类：Accuracy、Macro-F1、每类 Precision/Recall/F1、混淆矩阵。
- 结构化理解：字段准确率、Macro-F1、钟点方向误差、径向区域准确率、尺寸绝对误差。
- Caption：shape/position/size rubric 分数、must-hit、must-avoid、空回答率和幻觉率。
- 检索：mAP@1/5/10/25/50、Recall@N、Recall@1.5N、nDCG@10/50。
- 所有指标先按类别分别计算，再做 macro average。
- 必须报告置信区间或 bootstrap 波动，不只给单个数字。

基线至少包括：

1. 随机检索/多数类分类。
2. 未微调 Qwen3.5-9B 零样本结果。
3. 冻结视觉编码器的 LoRA SFT。
4. 可选的视觉侧适配模型。

CSV 原始顺序不能称为“随机 baseline”。随机基线必须使用固定种子重复多次并报告均值与标准差。

### 7.6 鲁棒性注意

- 颜色或分辨率扰动不能改变 gold 标签。
- 旋转会改变钟点方向。测试类别不变性时可旋转，但位置 gold 必须同步旋转，不能沿用原方向标签。
- 所有自动扰动必须记录参数，并能由 seed 完全重现。

## 8. Qwen3.5-9B 训练计划

### 8.1 实验0：零样本基线

训练前先用未微调的 `Qwen/Qwen3.5-9B` 跑完整 Benchmark，保存原始输出。这是判断微调是否真正有效的必要基线。

Caption 任务关闭 thinking，避免短描述生成冗长推理。框架具体参数以当前固定版本的 Qwen3.5 模板为准。

### 8.2 实验1：冻结视觉侧的 LoRA SFT

先做风险最低的基线：

- `swift sft`
- `tuner_type=lora`
- 冻结 Vision Encoder 和视觉 Aligner/Merger。
- LLM 侧 `target_modules=all-linear`。
- 建议起点：LoRA rank 16、alpha 32、dropout 0～0.05。
- BF16 优先；24GB 显存不足时使用 QLoRA、batch size 1、梯度累积和 gradient checkpointing。
- 建议起始学习率 `1e-4`，warmup ratio `0.03～0.05`，训练1～2 epoch。
- `max_length` 从2048起步；晶圆 Caption 不应使用超长上下文。
- `IMAGE_MAX_TOKEN_NUM` 从256起步，并对256/512做显存与精度对比。
- 有效 batch 建议达到32左右，但需按实际 GPU 调整。

这些是实验起点，不是最终结论。任何修改必须写入运行配置和实验记录。

命令结构参考，不得未检查当前 ms-swift 版本就直接执行：

```bash
IMAGE_MAX_TOKEN_NUM=256 \
swift sft \
  --model Qwen/Qwen3.5-9B \
  --dataset data/splits/train.jsonl \
  --val_dataset data/splits/val.jsonl \
  --tuner_type lora \
  --target_modules all-linear \
  --freeze_vit true \
  --freeze_aligner true \
  --torch_dtype bfloat16 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --learning_rate 1e-4 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 32 \
  --max_length 2048 \
  --warmup_ratio 0.05 \
  --gradient_checkpointing true \
  --output_dir outputs/checkpoints/qwen35_9b_lora
```

### 8.3 实验2：视觉侧适配

只有当实验1表现出“语言格式学会了，但图像类别/位置仍识别错误”时，才训练视觉侧：

- 优先使用 ms-swift 支持的 `lora_llm` 或等价模式。
- LLM 使用 LoRA，Vision Encoder/Aligner 使用更小学习率。
- 建议起点：LLM `1e-4`，ViT/Aligner `1e-5`。
- 先只训练1 epoch并监测过拟合。
- 不直接迁移旧 VisualGLM 的 CE+InfoNCE，除非已经完成独立消融并实现 Qwen3.5 兼容的自定义 Trainer。

视觉侧训练比冻结视觉侧更耗显存。资源不足时，优先完成可靠的实验1，不要为了复刻旧 Stage 1 牺牲可复现性。

### 8.4 可选的强化后训练

SFT 基线稳定后，若 mentor 明确要求，可使用 ms-swift 的 DPO/GSPO/GRPO：

- 奖励必须基于确定性的 shape/position/size rubric 和 must-avoid 幻觉惩罚。
- 不得只依赖另一个 LLM 的主观分数。
- 强化训练必须与 SFT 使用独立验证集。
- 未证明 SFT 瓶颈前，不启动强化学习。

## 9. 检索方案

Qwen3.5 的生成式 SFT不会自动保证视觉向量适合检索。必须区分两类目标：

1. **生成能力**：看图生成类别、形态、位置和尺寸。
2. **检索能力**：让相似缺陷图在向量空间中接近。

第一阶段可以提取 Qwen3.5 Vision Encoder/视觉融合层的 pooled feature，比较微调前后检索指标，但具体层和 pooling 必须根据固定版本模型接口验证并记录。

如果生成提高但检索没有提高，不应判定训练失败；应另设检索训练：

- 方案A：根据 Qwen3.5 生成的结构化描述进行文本检索。
- 方案B：为视觉特征增加对比学习/Embedding 目标。
- 方案C：使用专用多模态 Embedding 模型作为检索基线。

禁止在没有对比损失的情况下宣称 SFT 已直接优化图像检索。

## 10. 评测与实验记录

每次实验必须保存：

- Git commit。
- ms-swift、transformers、torch、CUDA 版本。
- 模型 revision。
- 完整训练参数和环境变量中非敏感部分。
- 数据 manifest 版本与 SHA256。
- 随机种子。
- GPU 型号和数量。
- loss 曲线、验证指标、最终 Benchmark 报告。
- Base、Adapter、Merged 三种模型的对比结果。

禁止只汇报最好 checkpoint。需要记录选择 checkpoint 的预先规则。

## 11. 开发与验证规范

- 搜索文件和文本优先使用 `rg`、`rg --files`。
- 文件修改使用补丁方式，保留用户已有修改；禁止随意覆盖原始数据和实验输出。
- 所有数据转换脚本必须支持 `--dry-run`、固定 seed、断点续传和幂等执行。
- 所有核心步骤都要有小型合成测试，不依赖完整数据或 GPU。
- 数学特征、坐标方向、ID转换、split隔离和指标公式必须写单元测试。
- 在大规模运行前先做20条 smoke test，再做每类小样本测试，最后才跑全量。
- 训练前必须打印 train/val/test 的 lot 交集；任何非空交集都应立即失败。
- 训练前必须抽样解码 ms-swift 模板，确认 `<image>`、labels mask 和回答部分正确。
- 发现旧代码 bug 时，同时记录 `legacy behavior` 与 `fixed behavior`，不能静默修改后宣称是原始复现。

## 12. 阶段验收门槛

### 数据阶段

- manifest 行数、唯一 ID 数和图片数一致。
- 无 train/val/test lot 泄漏。
- 最终 Caption 来源和置信度可追溯。
- 每类数量、丢弃原因和异常比例有报告。

### Benchmark 阶段

- Benchmark 已冻结并生成哈希。
- 每类、每难度层都有覆盖。
- 人工抽检完成，争议样本已隔离。
- 未被训练或 Prompt 调试使用。

### 训练阶段

- 先完成零样本基线。
- 20条数据可以完成端到端 SFT 和推理。
- loss 正常下降，验证集没有明显退化。
- Adapter 推理和合并后推理结果基本一致。

### 最终评估

- 同时报告生成、分类、结构化理解、鲁棒性和检索指标。
- 按类别报告并使用 macro average。
- 展示典型成功、失败和高置信错误样本。
- 与未微调 Qwen3.5-9B 进行同一 Benchmark 的公平对比。

## 13. 开工前仍需确认

服务器开发开始时，先确认以下信息并记录到 `manifests/project_decisions.md`：

1. mentor 所指框架是否确定为 ms-swift。
2. GPU 型号、数量和显存。
3. 可迁移的数据文件、字段和标签来源。
4. `predictDefectType` 的真实来源。
5. 最终主要输出语言：中文、英文或双语。
6. 主任务优先级：Caption、分类、结构化输出还是检索。
7. 是否允许使用 QLoRA，以及是否要求合并模型权重。
8. Benchmark 是否需要独立人工标注。

在以上信息未完全明确时，可以进行只读数据审计、格式转换代码和小型测试，但不得启动昂贵的全量训练。
