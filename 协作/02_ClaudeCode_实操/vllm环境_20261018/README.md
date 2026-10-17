# vLLM 环境（2026-10-18）

用户要求：在学校服务器上新开一个环境装 vLLM，并且要能跑 Qwen3.5-9B。

## 结论（先说结果）

**环境建好了、vLLM 装上了、模型架构也认了，但在这台机器上跑不起来。**
根因不是配置，是**驱动版本**：机器驱动是 570.133.07（CUDA 12.8），而所有可用的
vLLM 预编译 wheel 都是 CUDA 12.9 / 13.0 构建，它们的 kernel 里带的 PTX 版本
570 驱动的 JIT 不支持，一跑就报：

```
torch.AcceleratorError: CUDA error: the provided PTX was compiled with an unsupported toolchain.
cudaErrorUnsupportedPtxVersion
```

**唯一的解法是把 NVIDIA 驱动升到 580 或更高**（需要 root，会影响整台共享机器）。
升完之后 `envs/vllm312`（vllm 0.31.0）可以直接用，不用再装。

## 硬事实（逐条可核对）

| 项 | 值 |
|---|---|
| 服务器 | 114.214.255.89（8× RTX 5090，32 GB/卡） |
| 系统 | Ubuntu 20.04.6，kernel 5.15.0-67 |
| 驱动 | **570.133.07**，`nvidia-smi` 报 **CUDA Version: 12.8** |
| 账号下原有 vLLM | **没有**。5 个 Python 环境（wafer + 4 个 conda）全部 `ModuleNotFoundError`；全盘无 `vllm` 可执行文件 |
| 唯一的离线 wheel | `/data/renjinyao/labguard-gpu/wheels/vllm-0.31.0-...whl`（320 MB，CUDA 13 构建） |
| 根盘 `/` | **99% 满，只剩 11 G**（所以环境建在 `/data`） |
| 模型 | `/data/renjinyao/models/Qwen3.5-9B`，架构 `Qwen3_5ForConditionalGeneration` |

## 建了两个环境（都在 /data，wafer 训练环境一点没碰）

| 路径 | 内容 | 结果 |
|---|---|---|
| `/data/renjinyao/envs/vllm312` | Python 3.12 + vllm **0.31.0** + torch **2.13.0+cu130** | **CUDA 初始化直接失败**：`The NVIDIA driver on your system is too old (found version 12080)` |
| `/data/renjinyao/envs/vllm026` | Python 3.12 + vllm **0.26.0+cu129** + torch **2.11.0+cu129** + torchcodec **0.15.0** | torch 能跑，**vLLM kernel 跑不了**（PTX 版本） |

## 实际验证到哪一步

**能跑通的**（`envs/vllm026`，GPU4 空卡）：

```
torch 2.11.0+cu129  cuda 12.9  avail True
matmul OK on NVIDIA GeForce RTX 5090
import vllm                  OK（vllm 0.26.0+cu129）
Resolved architecture: Qwen3_5ForConditionalGeneration   ← 模型架构识别正确
```

**跑不通的**，三个位置，同一个错：

1. 默认后端 → 视觉编码器 `_vllm_fa2_C.varlen_fwd`
2. `mm_encoder_attn_backend=TORCH_SDPA` 后 → 视觉这关过了，卡在 FlashInfer 采样器
   （`FlashInfer requires GPUs with sm75 or higher`，对 sm_120 误判）
3. 再加 `VLLM_USE_FLASHINFER_SAMPLER=0` 后 → 卡在 **LLM 主注意力**
   `vllm/v1/attention/backends/flash_attn.py` 的 `_vllm_fa2_C.varlen_fwd`，还是 PTX 错误

三次尝试的日志：`/data/renjinyao/vllm_test/{result_cu129,result_sdpa,result_sdpa2,result_sdpa3}.json`

## 为什么绕不过去

- vllm 0.27–0.31 全钉 `torch==2.13.0`，torch 2.13 **只有 cu129/cu130**，没有 cu128。
- vllm 0.26.0 钉 `torch==2.11.0`（有 cu128），但 **PyPI 上的 vllm 0.26 wheel 本身是 CUDA 13**。
- vLLM 官方 wheel 索引 `https://wheels.vllm.ai/<版本>/` 只提供 **cu129 / cu130**，从来没有 cu128。
- CUDA 12.9 的**次版本兼容**只到「torch 的 kernel 能用」这一步（torch 带 sm_120 的 SASS），
  但 vLLM 自带的 FlashAttention kernel 没有 sm_120 的 SASS，只能靠 ptX JIT，
  而 12.9 的 PTX 又超出 570 驱动能 JIT 的范围。

## 该怎么办

1. **找管机器的人把驱动升到 580+**（推荐）。升完 `envs/vllm312` 直接可用。
2. 或者换一台驱动更新的机器（比如云端实例）。
3. 在那之前，项目的评测继续用现有的 transformers 路径（`envs/wafer`，ms-swift 4.5.3），
   这条一直是好用的，只是慢（200 张图约 20–30 分钟）。

## 服务器上留下的东西

| 路径 | 是什么 |
|---|---|
| `/data/renjinyao/envs/vllm312` | Python 3.12 + vllm 0.31.0（等驱动升级后可用），约 8.5 G |
| `/data/renjinyao/envs/vllm026` | Python 3.12 + vllm 0.26.0+cu129 + torch 2.11.0+cu129，约 11 G |
| `/data/renjinyao/vllm_test/` | 测试脚本、题面、全部运行日志与结果 JSON |
| `/data/renjinyao/tmp/vllm026/` | vllm 0.26 wheel（可删） |

## 没做的事

- 没有动 `wafer` 训练环境（torch 2.8.0+cu128 / ms-swift 4.5.3 原样）。
- 没有 pkill 任何别人的进程；GPU0（tpy 的任务）全程未碰；实测只用过空闲的 GPU4。
- 没有把 vLLM 接进项目评测流程。
- 没有升级驱动（需要 root，且会影响整台共享机器，必须由人决定）。
