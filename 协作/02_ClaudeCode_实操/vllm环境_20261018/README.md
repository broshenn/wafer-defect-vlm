# vLLM 环境（2026-10-18）

用户要求：在学校服务器上新开一个环境装 vLLM，并且要能跑 Qwen3.5-9B。

## 一句话状态

**环境已建、vLLM 已装，但撞上驱动/CUDA 版本不匹配；已把「换 cu129 再试」的活做成服务器上的
detached 任务继续跑，本地电脑关机不影响。** 结果看 `/data/renjinyao/vllm_test/overnight_*.log`。

## 硬事实（可核对）

| 项 | 值 |
|---|---|
| 服务器 | 114.214.255.89（8× RTX 5090，32 GB/卡） |
| 系统 | Ubuntu 20.04.6，kernel 5.15.0-67 |
| 驱动 | **570.133.07**，`nvidia-smi` 报 **CUDA Version: 12.8** |
| 原有 vLLM | **没有**。账号下 5 个 Python 环境（wafer 与 4 个 conda）全部 `ModuleNotFoundError`，全盘无 `vllm` 可执行文件 |
| 唯一的离线 wheel | `/data/renjinyao/labguard-gpu/wheels/vllm-0.31.0-...whl`（320 MB，CUDA 13 构建） |
| 根盘 `/` | **99% 满，只剩 11 G**；`/data` 剩 2.4 T |
| 模型 | `/data/renjinyao/models/Qwen3.5-9B`，架构 `Qwen3_5ForConditionalGeneration` |

## 建了两个环境（都在 /data，没碰 wafer 训练环境）

| 路径 | 内容 | 状态 |
|---|---|---|
| `/data/renjinyao/envs/vllm312` | Python 3.12 + vllm **0.31.0** + torch **2.13.0+cu130** | **跑不了**：CUDA 13 需要 580+ 驱动，本机 570 → `RuntimeError: driver too old` |
| `/data/renjinyao/envs/vllm026` | Python 3.12 + vllm **0.26.0** + torch **2.11.0** | 正在试 cu129（见下） |

## 为什么绕了一圈

- vllm 0.31/0.30/0.29/0.28/0.27 都钉 `torch==2.13.0`，而 torch 2.13 只有 cu129/cu130 构建，**没有 cu128**。
- vllm 0.26.0 钉 `torch==2.11.0`，torch 2.11 **有 cu128 构建**；且 vllm 0.26 的 registry 里
  **确实有 `Qwen3_5ForConditionalGeneration`**。
- 但 PyPI 上的 vllm 0.26.0 wheel 本身是 **CUDA 13 构建** → `ImportError: libcudart.so.13`。
- vLLM 官方 wheel 索引 `https://wheels.vllm.ai/<版本>/` 只提供 **cu129 / cu130**，没有 cu128。
- 所以最后的希望是 **cu129**：CUDA 12.9 有次版本兼容，12.x 驱动通常能跑 12.9 程序。

## 正在跑的 detached 任务

`/data/renjinyao/vllm_test/overnight.sh`（`setsid nohup` 启动，本地关机不影响）：

1. 等 torch 2.11.0+cu129 装完；
2. 在 **GPU4**（当时唯一的空卡）上做 CUDA 自检 + 2048×2048 矩阵乘；
3. 装 `vllm-0.26.0+cu129` 的 x86_64 wheel（直接按 URL 装，覆盖 PyPI 的 cu13 版）；
4. `import vllm` 自检；
5. 真实加载 Qwen3.5-9B：纯文本生成一次 + 带一张晶圆 PNG 生成一次，把耗时和原答写进 `result_cu129.json`。

看进度：

```bash
ls -t /data/renjinyao/vllm_test/overnight_*.log | head -1 | xargs tail -40
```

## 如果 cu129 也不行

那就只剩**升级 NVIDIA 驱动到 580 及以上**（需要 root、会影响整台共享机器），
升完直接能用已经装好的 `/data/renjinyao/envs/vllm312`（vllm 0.31.0）。

## 已上传的脚本

- `/data/renjinyao/vllm_test/test_qwen35.py`（= 本目录 `test_qwen35.py`）
- `/data/renjinyao/vllm_test/prompt_v5.txt`
- `/data/renjinyao/vllm_test/overnight.sh`（= 本目录 `overnight.sh`）

## 没做的事

- 没有动 `wafer` 训练环境（torch 2.8.0+cu128 / ms-swift 4.5.3 原样）。
- 没有 pkill 任何别人的进程；GPU0（tpy 的任务）全程未碰。
- 没有把 vLLM 接进项目评测流程——目前只是「装好并验证能跑」。
