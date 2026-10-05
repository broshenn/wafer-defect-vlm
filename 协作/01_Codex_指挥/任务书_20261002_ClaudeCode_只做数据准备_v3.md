# 给Claude Code：只做GLM标注的数据准备

**Codex，2026-10-02。最新独立入口，替代此前混合准备/标注说明。** 当前阶段C/D的数据准备。你负责清单与校验；看图作答由ZCode完成。用户转交成果，不自动创建或通知其他会话。

本次新增模型请求0、付费0、GPU0、上传0。产物只写 `D:/pycode/晶圆图研究/协作/02_ClaudeCode_实操/GLM训练标注准备_20261002/`；若已存在不同产物，另建版本并报告实际路径，不能覆盖旧文件。

## 1. 固定来源

来源目录：`D:/pycode/晶圆图研究/协作/02_ClaudeCode_实操/学校训练准备_20261002-025848/`。

| 输入 | SHA256 |
|---|---|
| train_180.jsonl | 04febd99674f70bf4619c3b529fe4c1313a5b8d739e89c69c390fe686324db47 |
| smoke_20.jsonl | f59ae6ba5ba917fae410a9a82f6747c886abde20df189878093467242dd98eec |
| dev_18.jsonl | 08b267ea73def5b3119dbf6ca73d27630ef565bb4011f6e53b404d753f7972ac |

沿用既有180个训练ID，不重新选样。首批20条保持smoke_20原顺序；剩余160保持train_180原顺序并剔除首批ID。最终9批、每批20，不按教师表现调整名单。dev_18只做隔离核对，不给ZCode标注。核对manifest/图片对应、ground_truth来源及train/dev/test/benchmark的lot隔离；漂移即停，不静默修原数据。

## 2. 你要产出的文件

1. `blind_inputs_all_180.jsonl`及`batch_01_blind.jsonl`至`batch_09_blind.jsonl`。每行**仅item_id/sample_id/image_path/image_sha256**，全局ID唯一。
2. 首批子清单：`batch_01_worker_01.jsonl`至`batch_01_worker_04.jsonl`。按首批顺序分别取1–5、6–10、11–15、16–20条；各5条、互不重叠、并集精确等于首批20条。
3. `audit_selected.jsonl`保留标签、来源、lot、split、manifest行，只给Codex/Claude；ZCode禁止读取。另存`prepare_report.md`和含审计信息的`plan.json`，不交标注会话。
4. 准备器支持dry-run、固定seed3407、resume和拒绝覆盖；独立检查器不import准备器，验证ID/分片并集/图片指纹/无答案字段/隔离。无需改已有格式检查器或重复教师测试。
5. **唯一给ZCode的输入索引 `zcode_batch_01_inputs.json`**，不含标签、类别数量、lot、split、几何或评价，字段如下：

```json
{"batch_id":"batch_01","total_limit":20,"max_concurrency":4,
 "batch_manifest":{"path":"实际首批清单绝对路径","sha256":"实测hash"},
 "prompt":{"path":"下述固定提示词路径","sha256":"下述固定hash"},
 "checker":{"path":"下述固定检查器路径","sha256":"下述固定hash"},
 "workers":[{"worker_id":"worker_01","input_path":"实际子清单绝对路径","input_sha256":"实测hash","count":5}]}
```

workers必须有四条完整记录。示例中的占位文字必须替换为真实路径/hash，不得猜值。其余批可预备相同子清单，但本次只交首批执行索引。

固定提示词：`D:/pycode/晶圆图研究/协作/01_Codex_指挥/prompt_image_only_v2_20261002.txt`，SHA256 `7df73031ee1a30dfb018dbb2417f505ea9ff139280307bce2934de604119954a`。

固定检查器：`D:/pycode/晶圆图研究/协作/02_ClaudeCode_实操/口径修正与准备_20261002-021207/check_answer_v3.py`，SHA256 `794a254e005eaa232362c539817abde98a1c2406ecb52574b9e9edb35303ec55`。

**不改提示词、图片、检查器、原标签和已有清单。不生成GLM答案，不启动标注子agent。**

## 3. 交回什么

只给Codex交 `给Codex的回复.md`：输入hash与独立检查结果、首批主/四子清单的路径/hash、zcode_batch_01_inputs.json的路径/hash、未知项和实际消耗；说明本轮准备完成、真实标注未执行。

Codex核对后，用户把**另一份ZCode任务书**转给ZCode，不把本任务、audit、plan或准备报告一起转过去。你不需要再编写ZCode的工作步骤。

学校训练仍由Claude负责，沿用已有学校任务及20步/60步授权；本任务先完成本地公开图片清单准备，训练另按学校连接与资源条件推进，不在本地运行GPU训练。无需为此重新安装环境或下载基座。
