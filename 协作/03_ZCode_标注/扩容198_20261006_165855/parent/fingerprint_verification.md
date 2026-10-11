# 指纹核验记录（父会话，2026-10-06 16:58）

执行依据：`协作/01_Codex_指挥/任务书_20261006_ZCode_198张扩容描述_已放行.md`
运行目录：`协作/03_ZCode_标注/扩容198_20261006_165855/`

## 一、任务书锁定指纹逐项核对（sha256sum，2026-10-06 16:58）

| 文件 | 任务书锁定值 | 实测 | 结果 |
|---|---|---|---|
| zcode_input_index.json | c2405a22ca2a3a8a71eef59773b13340e78514c5b83c67044adf277118a1db55 | 同 | 一致 |
| zcode_blind_new.jsonl | 587c30a1be9b620059f29b1ff0614c27f8ba7662d5002b45d42c01b5e22827a0 | 同 | 一致 |
| zcode_batch_01_首批20.jsonl | dc0341948dd2ce4a032b11a5d7bda4901971a8c832a69e4f5c4f21eb08af4b7d | 同 | 一致 |
| prompt_image_only_v2_20261002.txt | 7df73031ee1a30dfb018dbb2417f505ea9ff139280307bce2934de604119954a | 同 | 一致 |
| check_answer_v4.py | 6a2db3c69685c622fdb18c7407505902f26a18ae9d787c0e4ddec9063677fdf2 | 同 | 一致 |
| check_answer_v3.py（依赖） | 794a254e005eaa232362c539817abde98a1c2406ecb52574b9e9edb35303ec55 | 同 | 一致 |

## 二、分片与结构核验（脚本 parent/verify_structure.py，输出 parent/verify_report.txt）

- 27个分片文件 sha256 全部与索引 `batches[].sha256` 一致；条数 20 + 7×25 + 3 = 198，与索引 `n` 一致。
- 每行恰好四个字段：`item_id`、`sample_id`、`image_path`、`image_sha256`（无重复键、无多余/缺失字段）。
- item_id 无重复，序列恰为 zc_001–zc_198。
- 分片并集 = blind_new = 索引 items：键集合相同且四字段值逐条相同（mismatches=0）。
- 198 张 PNG 全部存在于原位置且 sha256 与索引一致（png_bad=0）。
- 索引自述"只含任务输入路径与指纹，不含类别、lot、几何、描述或筛选原因"，与实际字段相符。

VERDICT=PASS（退出码0）

## 三、禁读边界执行情况

输入目录中 `共同图片清单.jsonl`、`标注分账.json`、`校验结果.json`、`给ZCode的执行提示词.md` 未读取（仅见文件名）。父会话未读取任何标签、lot、几何、历史raw、audit/manifest 文件，未按 sample_id 检索任何答案。允许读取范围：锁定索引、27个分片、blind_new、提示词、检查器v4及依赖v3（只读）。

## 四、提示词与检查器要点（供本运行执行；原文以锁定文件为准）

- 提示词要求：只输出一个七字段 JSON 对象（defect_class/morphology/radial_zone/clock_direction/extent_r/caption_zh/uncertainty）；extent_r 必须 null；描述不得含未验证数值（钟点表达除外）；类别无法可靠判断填 unknown。
- 检查器 v4 退出码：0 通过；2 非法JSON；3 顶层非对象；4 字段集合不符；5 类型错误；6 枚举未知；7 extent_r 非 null；8 描述含数值（钟点已剔除，v4 放行"至"区间）。v4 在导入时自校验 v3 依赖 hash。
- raw 文件约定：`<item_id>_raw.json` 内容即模型作答原文（七字段 JSON 对象文本本身），不解析重写、不修补、不重试。
