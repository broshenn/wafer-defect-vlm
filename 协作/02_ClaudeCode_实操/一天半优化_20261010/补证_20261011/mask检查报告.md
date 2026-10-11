# 真实 collator 的监督掩码检查（CPU，2026-10-11）

> 任务书第 2 项。**用框架自己的 collator，不手造 mask、不加载 9B 权重、不做前向。**
> 脚本：`server_mask_check.py`（服务器上跑，产物 `mask_check.json`）；本地副本见 `mask_check.json`。

## 一、先纠正我自己上一轮的一个说法

我上轮写过"`encode()` 不返回 labels，所以 CPU 这一层无法验证掩码" —— **这个结论是错的**。
Codex 的批评（"`encode()` 没有 labels 不等于 collator 不能查"）成立。真相是：

- swift 的 `Template._encode()` 结尾有一段：
  `if not self.is_training: encoded[k] = None`（对 `labels` / `loss_scale`）；
- 而 `is_training` 是**只读属性**，由 `mode` 推导：
  `is_training = mode not in {'transformers','vllm','lmdeploy','sglang'}`。

**我第一版没设 mode，模板停在推理模式，labels 被抹掉**，于是我看到"batch 里没有 labels"——
那是**检查工具的开关问题，不是训练没有监督**。设对 mode 后 collator 正常产出 labels。

**顺带查清一个我原先猜错的事实**：swift 的
`mode_mapping = {'kto':'kto', 'gkd':'train', 'ppo':'transformers', 'grpo':'train'}` ——
**GRPO 与 SFT 用的是同一个模板 mode（`train`）**，不是 `rlhf`。
两者在本轮的真实差别**只在 `loss_scale` 的 base strategy（`default` vs `last_round`）**。

## 二、怎么做的（可复跑）

1. `get_model_processor(model_dir, model_type='qwen3_5', load_model=False)` 取 processor（CPU）；
2. `get_template(proc, template_type='qwen3_5', loss_scale=...)` + `set_mode('train')`；
3. 对真实训练样本先 `encode()` 再 **`data_collator([...])`** —— 这是训练时真正走的路径；
4. 逐 token 记录文本 / `labels` / 是否被监督。

**只加了一个进程内适配（不改框架源码）**：collator 为 `position_ids` 会内部建 dummy model，
它调用 `get_model_processor(model_dir)` **不带 model_type**，而本目录能匹配 4 个候选类型 → 报错。
我在本进程内把该函数包一层，缺 `model_type` 时补 `qwen3_5`。

输入 SHA：`adapt512_sft.jsonl` `417b3d7f…`、`group256_grpo.jsonl` `8a62af8a…`（见 `mask_check.json`）。

## 三、结果

### 3.1 合成短对话（先核对问答边界）

| 口径 | 总 token | 被监督 | 区间 |
|---|---:|---:|---|
| `loss_scale=default` | 17 | 8 | [9, 16] |
| `loss_scale=last_round` | 17 | 8 | [9, 16] |

答案只有 `PONG`（2 token），被监督 8 个 = `<|im_start|>assistant\n<think>\n\n</think>\n\nPONG<|im_end|>`
—— **监督从 assistant 轮开始（含模板的空 think 前缀）到结束符**；问题段（0–8）**不监督** ✓。

### 3.2 真实 SFT 样本（adapt512，含最短/最长答案与方向形式）

| 样本 | 总 token | 被监督 | 区间 | 答案本身 token 数 | 结论 |
|---|---:|---:|---|---:|---|
| `wafer_00007864_013`（最短） | 630 | **43** | [587, 629] | 37 | 尾部监督 ✓ |
| `wafer_00015851_009`（最长） | 647 | **60** | [587, 646] | 54 | 尾部监督 ✓ |
| `wafer_00006467_002`（含 single） | 632 | **45** | [587, 631] | 39 | 尾部监督 ✓ |

**逐 token 实证（`wafer_00007864_013`）**：

- 被监督段解码 = `'<think>\n\n</think>\n\n{"defect_class":"Loc","coverage_band":"under_5pct",…'`
  → **模板的空 think 前缀在监督范围内**（模型被训练着输出它）；
- 前 9 个 token = `'<|im_start|>user\n<|vision_start|><|image_pad|>…'` → **问题与图片不监督** ✓；
- 最后 5 个 token：`ertainty` / `":""` / `}` / `<|im_end|>` / `\n` → **全部监督**（含结束符）✓；
- 被监督数比"答案本身 token 数"多 6 个 = assistant 前缀（`<|im_start|>assistant\n<think>\n\n</think>\n\n`）
  + `<|im_end|>` + 换行的 token 数。
- **总长 630–647，远小于 `max_length=2048` → 无截断** ✓

### 3.3 GRPO 单问样本（group256_grpo）

被监督 **0** 个 token（参考"答案本应监督 384"是脚本里对整行文本的粗略估计，不适用）——
GRPO 数据只有一行 user（**生成发生在 rollout 时**），**prompt 本来就不该被监督** ✓。

### 3.4 SFT 与 GRPO 的 `loss_scale` 差别**在本轮数据上没有体现**

- `default` 与 `last_round` 两种口径下，**监督位逐 token 完全一致**（三组真实样本 + 合成样本全部一致）；
- **collator 输出的 batch 键里没有 `loss_scale`**
  （键为 `attention_mask / cu_seq_lens_k / cu_seq_lens_q / image_grid_thw / input_ids / labels /
  max_length_k / max_length_q / mm_token_type_ids / pixel_values / position_ids / text_position_ids`），
  → 说明**损失权重不是在 collator 里加进去的**，而在 trainer 侧使用。

**结论（对"两条线是否公平"的直接回答）**：
**在本轮单问单答的数据形态下，`default` 与 `last_round` 产生了相同的位置掩码，
因此 `loss_scale` 的差异不构成 SFT 与 GRPO 表现差异的解释。**
它可能影响取值权重（未在本层观察），但**不影响哪些 token 参与监督**。

## 四、边界（不能过度声明）

- 这是**当前实现的独立检查**，**不回填成历史训练当时已做过的证据**。
- 我只检查了 **3 个真实样本 + 1 个合成样本**，**不是全量 1536 行**；结论按这 4 条说。
- **没有**验证 `loss_scale` 在 trainer 里如何进入损失；只说"collator 层没有它"。
- 未加载 9B 权重、未做前向，**不能用 loss 下降代替掩码正确性**。
