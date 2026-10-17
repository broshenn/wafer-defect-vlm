# vLLM 环境（2026-10-18）

用户要求：在学校服务器上新开一个环境装 vLLM，并且要能跑 Qwen3.5-9B。

## 结论：**能跑通了**

模型能加载、纯文本能生成、带晶圆图也能生成。但**不能走 vLLM 的默认后端**——必须换成
运行时 JIT 的注意力实现，原因见下。

实际跑出来的数字（GPU4，RTX 5090）：

| 项 | `enforce_eager=True` | `enforce_eager=False`（开 CUDA graph） |
|---|---|---|
| 模型加载 | 48.2 s | 184.4 s |
| 纯文本生成（256 tok） | 12.7 s | **3.9 s** |
| 带图生成（448×448 PNG，256 tok） | 24.1 s | **19.8 s** |
| `mm_ok` / `text_ok` | true / true | true / true |

**开 CUDA graph 是值得的**：启动多花两分钟，但纯文本快 3.2 倍、带图快约 18%。
`run_test.sh` 默认用 `VLLM_TEST_EAGER=0`。

| 环境事实 | 值 |
|---|---|
| vLLM | `0.26.0+cu129` |
| torch | `2.11.0+cu129`（`cuda_available: true`，`device: NVIDIA GeForce RTX 5090`） |
| 架构识别 | `Qwen3_5ForConditionalGeneration` → `qwen3_5_registered: true` |

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
| Docker | **实测过，不行，而且比宿主机方案更差**（见下） |
| 降级到有 cu128 构建的 vllm | vllm 0.27–0.31 全钉 torch 2.13；vllm 0.26 的 PyPI wheel 本身是 cu13；官方 wheel 索引只有 cu129/cu130 |

## 降级 vLLM 为什么也不行（版本谱系查过了）

| vLLM 版本 | 官方 wheel 有的 CUDA 构建 |
|---|---|
| 0.10.0 | cu118 / cu126 |
| 0.11.0 | cu129 |
| 0.12 – 0.26 | 只有 cu129 / cu130 |

- **0.12 到 0.26 之间从来没出过 cu128 构建**，"降到一个有 cu128 的版本"这个选项不存在。
- 再往前退到 0.10（cu126）：cu126 **根本不支持 sm_120**（Blackwell / RTX 5090），
  而且 0.10 早于 Qwen3.5，不认 `Qwen3_5ForConditionalGeneration`。
- 结论：降级是死路，不是没查。

## 真正能恢复快路径的办法：把驱动升到 575

Ubuntu 20.04 官方源最高只到 `nvidia-driver-570`；但 **NVIDIA 自己的 CUDA 源
（`ubuntu2004/x86_64`）里有 `nvidia-driver-575-open_575.57.08`**，而这台机器用的正是
open kernel module。

关键：**CUDA 12.9 的最低驱动要求就是 575.51**。升到 575 之后，已经装好的
`envs/vllm026`（vllm 0.26.0+cu129）可以直接走**默认 FA2 后端**，不需要任何环境变量，
CUDA graph 和 FlashInfer 采样器全部恢复——这才是真正的快路径。

代价：需要 root，要卸载/重载内核模块（等于**重启**），会中断机器上其他所有人的任务。
这是共享机器，必须由人协调，我不能单方面做。

想要 580+（从而用 `envs/vllm312` 的 vllm 0.31）的话，ubuntu2004 源里没有，
得混用 22.04 的包，风险更高。**575 是这台机器上性价比最高的一跳。**

## Docker 为什么救不了（已实测，不是推断）

服务器上 Docker 是装好的（`docker 28.1.1`，`nvidia-container-cli 1.18.0`，用户在 docker 组），
本地还有现成镜像 `heteropool/vllm-mooncake:0.26.0-cu129-mc0.3.12.post1`。实测结果：

```
$ docker run --rm --gpus all --entrypoint nvidia-smi <镜像>
| NVIDIA-SMI 570.133.07   Driver Version: 570.133.07   CUDA Version: 12.9 |

$ docker run --rm --gpus all --entrypoint python3 <镜像> -c "import torch; ..."
torch 2.11.0+cu129 | cuda 12.9 | avail False
RuntimeError: ... Error 804: forward compatibility was attempted on non supported HW
```

**容器里 `torch.cuda.is_available()` 直接是 False**，比宿主机方案更早失败。

原因在 NVIDIA 的容器 GPU 模型：

1. `nvidia-container-toolkit` 把**宿主机的驱动用户态库**（`libcuda.so`、`libnvidia-ptxjitcompiler.so` 等）
   挂进容器，并暴露 `/dev/nvidia*`；
2. **内核模块 `nvidia.ko` 永远在宿主机上**，容器不能替换它，也不需要 —— 这就是"容器共享驱动"的含义；
3. 镜像里的 CUDA toolkit 只提供用户态 runtime/编译库，**PTX JIT 能力来自驱动**；
4. 镜像自带的更新版 `libcuda`（forward-compat）想接管，NVIDIA 在 GeForce 上直接拒绝 → `Error 804`。

所以 Docker 能换的是**用户态**，换不了**驱动的 PTX 能力**。而这个问题的根因恰好在后者。
反过来说：宿主机那套 `TRITON_ATTN` 方案反而是**唯一能跑通的**，因为它用的是宿主机原生 libcuda 570，
只把 vLLM 的 kernel 换成运行时 JIT 的。

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
