# 内容复核30 · Worker 共用指令（存档）

适用于本轮全部内容复核 worker（GLM-5.3-Flash 会话，新上下文，互不继承答案）。
派发时每个 worker 只拿到自己分片的条目清单；本文件为记录副本，实际指令内嵌于派发 prompt。

## 每图流程（严格按序，本图视觉读取只做一次）

1. 核SHA：真实程序现算 PNG SHA256，与派发清单中的期望值比对。一致 → 写 `<review_id>.started.json`；
   **不一致 → 停止本 worker 全部新派发**，写 worker_status 后结束（输入完整性问题）。
2. 只看图一次：Read 该 PNG（唯一一次视觉读取）。**此刻之前与之后都不得打开旧 raw，直到第4步。**
3. 独立观察：按下方七字段题面作答，把模型原答逐字写入 `<review_id>.observation_raw.json`；
   随即现算其 SHA256 与 UTC 时间，写 `<review_id>.observation_meta.json`。
   观察落盘后**不得改写、不得重试**；它只是本轮模型候选，不是 gold。
4. 打开旧答：Read `协作/04_WorkBuddy_复核/方向形态84_v3_20261010_200950/<item_id>.raw.txt`，现算其 SHA256。
5. 对照判读：写 `<review_id>.review.json`（格式见下）。
6. 闭合：写 `<review_id>.done.json`。

## 七字段题面（与上轮84图完全一致，逐字）

只根据当前这一张公开晶圆 BIN 图实际可见的内容作答。黑色是片外，绿色是合格格点，红色是失效格点。
你的任务是描述主要形态、主要结构位置与可靠的方向；不是猜公开类别，也不是反推工艺原因。
图片上方是12点钟、右方3点钟，顺时针。区分"结构位于哪个钟点"与"线条沿哪个轴延伸"：偏离中心的团块可能有位置钟点，但没有线条轴；中心长线可以有延伸轴，但没有偏离中心的位置钟点。环状或全局散布不要强填唯一钟点。
只输出一个 JSON 对象，严格包含以下七个字段：
{"morphology_zh":"主要可见形态","main_location_zh":"主要结构位置","location_clock":[],"orientation_clock":[],"secondary_observations":[],"caption_zh":"一句话概述主要形态与位置","uncertainty":""}
morphology_zh：仅描述真实可见形态，例如点状、团块、环、弧、细长带；看不清时如实说明。
main_location_zh：描述主要结构的中心、中部、边缘、偏离方向或全局分布，不把全部红色散点的分布冒充主要结构位置。
location_clock：偏离中心的主要结构方向，可靠时填恰好一个1到12的整数，不适用/不确定填[]。
orientation_clock：真实细长结构的延伸轴，可靠时填恰好两个相对钟点整数（相差6），不适用/不确定填[]。两个相对团块并不足以判定中间有连续线，不要仅凭两端红点猜穿中心划痕。
secondary_observations：重要的次要可见结构或散点，每项短句；没有则[]。
caption_zh：中文一句话，不编造未见图案、精确数量、比例、格数、半径、尺寸或工艺根因。少数散点不等于无红点。
uncertainty：有歧义时说明，无则空字符串。不要为了凑字段猜测。
不写 Markdown、推理、前后缀或多对象。一次看图、一次原答，首次答案保留，不进行自我改写。

## review.json 格式

必含字段：review_id、item_id、sample_id、image_sha256（现算64位）、original_raw_sha256（现算）、observation_raw_sha256、model_display_name（"GLM-5.3-Flash"）、checks、overall、suggested_revision、uncertainty。

checks 必须逐一覆盖七项：morphology_zh、main_location_zh、location_clock、orientation_clock、secondary_observations、caption_zh、uncertainty（对**被审的旧原答**逐项判）。每项：
`{"verdict":"supported|contradicted|unverifiable|not_applicable","evidence":"具体指出图上位置/结构和被审句子，不写空泛同意"}`

- supported：图上可见证据支持被审句子；contradicted：图上可见证据与之明确矛盾；unverifiable：图上无法核实；not_applicable：该字段对本图无实体可核（如原答字段为[]且图上也确无对应结构时，判 supported 更贴切时用 supported，仅当该字段完全无对象时才用 not_applicable）。
- 被审原答与你的独立观察不一致时：如实写矛盾或分歧，在 uncertainty 中保存分歧，**不按多数票强定真值**。
- overall：usable / requires_review / reject。reject 仅用于核心形态或主要位置与图明显矛盾；个别字段 unverifiable 或非核心矛盾 → requires_review。
- suggested_revision：仅建议文字，不能覆盖旧raw、不能自动进入训练。
- 图上判断纪律：主要结构位置≠全部红点的平均方向；两团块≠一条连续线；有中心分支≠纯竖直十字；none类别未提供，不猜公开类；"无明显模式"≠无红点；钟点数字本身不是尺寸幻觉；不从图中推工艺根因、精确覆盖比例、格数或物理尺寸。无法判定写 unverifiable，不强猜钟点。

## 纪律红线

- 只允许读：派发清单指定的 PNG、旧 raw（第4步之后）、以及写自己分片的输出文件。禁止读盲清单其他行、其他图、旧check/meta/done、任务书、验收文档、manifest、任何含公开类别/lot/几何特征的数据。
- 单图失败：写 `<review_id>.failed.json`（stage+reason），隔离该图，继续其余图。
- PNG SHA 不匹配 / 视觉读取异常 / 写盘异常 → 停止新派发并如实报告。
- 不重试、不改写已落盘内容；原84答案不修改不覆盖。
- model_display_name 如实写 "GLM-5.3-Flash"；真实后端型号与采样参数未提供未核实，不得宣称后端独立已证。
- 结束时写 `worker_status_<分片名>.json`：planned/observed/reviewed/failed 列表/stopped_early/finished_at_utc。
