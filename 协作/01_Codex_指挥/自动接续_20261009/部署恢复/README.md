# 描述服务 D 的恢复启动包

这是已经在租用 RTX5090 上跑过真实 HTTP 的原型代码，附固定36图工程回放样本。当前服务已停止，本包不是在线服务。标准输入为448×448、黑/绿/红三色无损PNG，不支持任意产线图像。

## 所需资产

- 本 ZIP 解压到任意目录，里面包含服务、页面、冻结题面、冻结检查器、CPU测试和36图回放样本；没有业务API依赖。
- 自行准备公开 Qwen/Qwen3.5-9B 基座。本包不附19GB基座。
- D-N3072-3407 的两个 adapter 文件在项目仓库外已备份。其权重 SHA256 为 `babf37ecced78d37f56cc6cdea4d4dd839bce9b8324ef8caf4c7bb7b45fb018b`。原配置含历史本机路径，须保私有；实际基座由启动参数指定。
- 已验证运行环境：Python3.12.3、torch2.8.0+cu128、transformers5.16.1、peft0.20.0、ms-swift4.5.3；还需Pillow与numpy。历史报告没有单列后二者精确版本，不伪称完整环境锁文件。5090需支持sm_120的CUDA/PyTorch构建。
- 获准使用的单张GPU与它的GPU UUID。不能仅看空闲就绕过学校或平台的使用规则。

## 只检查文件，不加载模型

```bash
python verify_bundle.py --files-only
export WAFER_CHECKER="$PWD/schema_check.py"
python test_cpu_contract.py
```

CPU检查不等于GPU、模型能力或完整新机器复现。

## 真正启动（使用现有隔离环境）

```bash
bash run_d.sh /your/base/Qwen3.5-9B /your/adapters/D-N3072-3407 GPU-your-authorized-uuid /your/new-runtime-records
```

启动脚本先核包内文件、核心版本、D权重指纹及6项CPU契约，再启动服务。访问本机 `http://127.0.0.1:7860/`，或通过本人SSH端口转发访问。不要无认证地公网发布。

按 Ctrl-C 结束前台服务，45分钟硬停止覆盖模型加载与空闲时间。程序退出不等于租用平台关机或停止计费，必须单独确认平台状态。

已启动服务时，可运行一次工程回放（指定全新输出目录）：

```bash
python replay_selftest.py --samples 样本清单.json --images 图 --out /your/new-replay-records
```

本次自动交付没有重跑GPU回放。此前真实记录：D在36图上HTTP成功36/36、类别25/36，needs_review19/36；客户端端到端P50=4.177秒、P95=5.376秒，峰值已分配显存17.88GiB。无并发吞吐或生产稳定性证据。needs_review不是经过校准的正确率保证；内容仍需复核。

类别模型L也已备份，但L的HTTP服务性能没有新测，不能把D时延/描述能力归给L。36图是开发样本，其中一图PNG与训练图逐字节相同，附加35图敏感性另见最终报告。
