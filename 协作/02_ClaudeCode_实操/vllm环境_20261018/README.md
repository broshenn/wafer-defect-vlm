# vLLM 环境（2026-10-18）

用户要求：在学校服务器上新开一个环境装 vLLM，并且要能跑 Qwen3.5-9B。

## 结论：**能跑通了**

模型能加载、纯文本能生成、带晶圆图也能生成。但**不能走 vLLM 的默认后端**——必须换成
运行时 JIT 的注意力实现，原因见下。

实际跑出来的数字（GPU4，RTX 5090）：

| 项 | 值 |
|---|---|
| vLLM | `0.26.0+cu129` |
| torch | `2.11.0+cu129`（`cuda_available: true`，`device: NVIDIA GeForce RTX 5090`） |
| 架构识别 | `Qwen3_5ForConditionalGeneration` → `qwen3_5_registered: true` |
| 模型加载 | 48.2 s |
| 纯文本生成 | 12.7 s（256 max_tokens） |
| **带图生成** | **24.1 s**（448×448 PNG，256 max_tokens） |
| `mm_ok` / `text_ok` | **都是 true** |

## 为什么默认后端不行

机器驱动是 **570.133.07（CUDA 12.8）**，而 vLLM 的预编译 wheel 全是 **CUDA 12.9 / 13.0** 构建。
它们自带的 FlashAttention kernel 没有 sm_120 的 SASS，只能靠 PTX JIT，而 12.9 的 PTX
超出 570 驱动能 JIT 的范围 → `cudaErrorUnsupportedPtxVersion`。

绕开的办法是**完全不用那些预编译 kernel**：

| 位置 | 默认 | 本机必须改成 | 为什么 |
|---|---|---|---|
| 视觉编码器注意力 | FLASH_ATTN | `mm_encoder_attn_backend="TORCH_SDPA"` | 纯 PyTorch SDPA，无预编译 kernel |
| LLM 注意力 | FLASH_ATTN | `attention_backend="TRITON_ATTN"` | Triton 运行时 JIT，按驱动支持的 PTX 生成 |
| 采样器 | FlashInfer | `VLLM_USE_FLASHINFER_SAMPLER=0` | FlashInfer 对 sm_120 误判为 `sm<75` |
| CUDA graph | 开启 | `enforce_eager=True` | 跳过 graph 捕获，少一条 kernel 路径 |

注意：`TORCH_SDPA` 在 vLLM 0.26 里**只对 ViT 有效**（源码注释写着 `this tag is only used for ViT`），
LLM 必须用 `TRITON_ATTN`。

## 一条命令复现

```bash
/data/renjinyao/vllm_test/run_test.sh 4
```

（参数是 GPU 序号；脚本里已经带好上面四个环境变量/参数。）

## 走不通、已经排除的路

| 路 | 结果 |
|---|---|
| `envs/vllm312`：vllm 0.31.0 + torch 2.13.0+cu130 | CUDA 13 要 580+ 驱动 → `driver too old` |
| PyPI 的 vllm 0.26.0（cu13 构建）+ torch 2.11+cu128 | `ImportError: libcudart.so.13` |
| cuda-compat 用户态驱动（`cuda-compat-12-9`，575 用户态配 570 内核模块） | `Error 804: forward compatibility was attempted on non supported HW` —— **NVIDIA 的 forward-compat 不支持 GeForce**，只对数据中心卡开放 |
| Docker | 容器共用宿主机内核模块，`nvidia-smi` 在容器里仍是 570，**改不了这个问题** |
| 降级到有 cu128 构建的 vllm | vllm 0.27–0.31 全钉 torch 2.13；vllm 0.26 的 PyPI wheel 本身是 cu13；官方 wheel 索引只有 cu129/cu130 |

## 代价：速度

`TRITON_ATTN + TORCH_SDPA + enforce_eager` 是**慢路径**。实测带图生成 24 s/条（256 tokens），
和现在 transformers 那条路（200 张约 20–30 分钟）**同一量级，没有明显变快**。
如果目标是提速，真正的解法还是把驱动升到 580+，恢复 FA2 与 CUDA graph。

## 服务器上有什么

| 路径 | 是什么 |
|---|---|
| `/data/renjinyao/envs/vllm312` | Python 3.12 + vllm 0.31.0（**等驱动升级后可用**），约 8.5 G |
| `/data/renjinyao/envs/vllm026` | Python 3.12 + vllm 0.26.0+cu129 + torch 2.11.0+cu129 + torchcodec 0.15.0，约 11 G |
| `/data/renjinyao/vllm_test/` | 测试脚本、题面、`run_test.sh`、全部运行日志与结果 JSON |
| `/data/renjinyao/cuda_compat_12_9/` | 试过但失败的 cuda-compat 用户态库（可删） |
| `/data/renjinyao/tmp/vllm026/` | vllm 0.26 wheel（可删） |

## 环境本身的两个坑（都已处理，换环境时会再遇到）

1. **torchcodec 版本必须跟 torch 对齐**：torch 从 2.13 降到 2.11 后，torchcodec 0.17 的
   `.so` 加载失败，换成 **0.15.0** 才能 `import vllm`。
2. **测试脚本必须有 `if __name__ == "__main__"` 保护**：vLLM 用 `spawn` 起引擎子进程会重新
   导入主模块，否则报 `An attempt has been made to start a new process before ...`。

## 没做的事

- 没有动 `wafer` 训练环境（torch 2.8.0+cu128 / ms-swift 4.5.3 原样）。
- 没有 pkill 任何别人的进程；GPU0（tpy 的任务）全程未碰；实测只用过空闲的 GPU4，已释放。
- 没有把 vLLM 接进项目评测流程，也没有和 transformers 路径做完整吞吐对比。
- 没有升级驱动（需要 root，且影响整台共享机器，必须由人决定）。
- 模型这次是**原生基座**、没挂 adapter，输出的是思考文本而不是 v5 的 `<measure>+JSON` 格式；
  这一轮只验证「环境能跑」，不验证答题质量。
