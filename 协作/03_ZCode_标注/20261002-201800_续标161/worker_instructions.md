# GLM 标注 worker 通用规则书（续标161 · 20261002-201800）

本文件由父会话一次性写入，对所有 worker 相同。你的入口提示词给出：你的身份（组/worker号）、你的分片清单路径与期望 sha256、你的输出目录、你的组目录。本文件给出全部工作规则。两者冲突时以本文件为准；本文件与父会话入口提示词都没有的授权一律视为没有。

## 固定输入（开工先逐一核验 sha256，任一不符→在自己目录写 STOPPED_INPUT_HASH.txt 后结束，不处理任何图）

1. 你的分片清单：见入口提示词（每行 JSONL：item_id/sample_id/image_path/image_sha256）
2. 标注提示词：`D:\pycode\晶圆图研究\协作\01_Codex_指挥\prompt_image_only_v2_20261002.txt`（期望 sha256：`7df73031ee1a30dfb018dbb2417f505ea9ff139280307bce2934de604119954a`）
3. 原格式检查器：`D:\pycode\晶圆图研究\协作\02_ClaudeCode_实操\口径修正与准备_20261002-021207\check_answer_v3.py`（期望 sha256：`794a254e005eaa232362c539817abde98a1c2406ecb52574b9e9edb35303ec55`）
4. 类别别名规范化器：`D:\pycode\晶圆图研究\协作\01_Codex_指挥\验收_20261002_GLM首批20\normalize_label_alias.py`（期望 sha256：`01e02d587648d41618329908de8376851293551773d62140ecc610963f828f2c`；只规范化 Near-full/Edge-Loc/Edge-Ring 三个精确别名，不读标签、不改其他字段）

## 你的输出目录（只准写这里）

入口提示词给出的 `<你的目录>`。同伴目录 = 你的组目录下全部 worker_XX 子目录（含你自己的）——**只准用 Bash `test -e` 检查其中是否存在 STOPPED\* 文件，严禁读取其任何内容**。

## 逐张流程（按分片顺序；单张失败只隔离该张并继续下一张，不重试）

a. Bash `test -e` 检查同伴目录 STOPPED\*（只查存在性）；任一存在→写自己 `STOPPED_PEER.txt`，本张起不再处理，结束。
b. Bash 记 T0（`date '+%Y-%m-%d %H:%M:%S %z'` 与 `date +%s%3N`）。
c. 计算 PNG sha256 与分片比对；不符→写 `STOPPED_FINGERPRINT.txt`（sample_id、期望/实测），结束，不再处理后续任何图。
d. 用 Read 工具读取原位置 PNG（清单里的 image_path）。这是唯一视觉输入：每张恰好一次视觉读取；不搬图、不拼图、不只发路径、不用其他模型/几何程序预读。无法获得图像内容→写 `STOPPED_NO_VISION.txt`，结束。
e. 严格按提示词作答：仅输出一个七字段 JSON 对象（defect_class/morphology/radial_zone/clock_direction/extent_r/caption_zh/uncertainty）；extent_r 必须为 null；不写未经验证的数值尺寸/比例/覆盖率、不推测工艺根因；无法可靠分类填 unknown（合法，不是错误）；clock_direction 仅在可靠识别局部缺陷方向时写简短钟点方向，否则 null。
f. 未经改写原样 Write 到 `<你的目录>\<sample_id>_raw.json`。保存后一律不改写、不补字段、不删改、不重新生成。
g. 原检查器（第一轮）：
   `"D:/python/python.exe" -X utf8 -B "D:/pycode/晶圆图研究/协作/02_ClaudeCode_实操/口径修正与准备_20261002-021207/check_answer_v3.py" "<你的目录>/<sample_id>_raw.json" --json`
   把进程退出码、stdout、stderr 原样写入 `<你的目录>/<sample_id>_check.log`（三段都标明）。若 `D:/python/python.exe` 不存在，改用 PATH 中的 python 并在日志注明替换。
h. 规范化器（原答不动）：
   `"D:/python/python.exe" -X utf8 -B "D:/pycode/晶圆图研究/协作/01_Codex_指挥/验收_20261002_GLM首批20/normalize_label_alias.py" --raw "<你的目录>/<sample_id>_raw.json" --out "<你的目录>/<sample_id>_canonical.json" --report "<你的目录>/<sample_id>_normalization.json"`
   退出码与 stdout 记入同一 check.log。退出码 0 → 生成 canonical.json 与 normalization.json；非 0（坏 JSON、重复键等）不生成副本。该工具拒绝覆盖既有文件：每张只用自己 sample_id 的路径，绝不复用、绝不删除重跑。
i. 若 canonical.json 已生成 → 对副本再跑原检查器（第二轮，命令同 g，路径换成 canonical），退出码与输出追加到 check.log。
j. Bash 记 T1；向 `<你的目录>/worker_log.jsonl` 追加一行 JSON：
   `sample_id, image_sha256, raw_sha256, canonical_sha256(无则 null), checker1_exit, normalizer_exit, checker2_exit(未跑则 null), alias_changes(normalization.json 的 changes 数组长度), pending_review(=content_review_warnings 非空；true 时把警告原文记入 worker_report.md), t0, t1, duration_ms, status`
   status 取值：`raw_pass`（checker1=0）/ `raw_fail_canonical_pass`（checker1≠0 但副本 checker2=0）/ `raw_fail`（checker1≠0 且无副本或副本也未过）/ `isolated_bad_json`（normalizer 因 JSON 解析失败）/ `not_run`。
k. 坏 JSON、未知类别、字段/格式错误（checker1 或 checker2 非 0）→ 只隔离该张（保留原件与全部退出码），**继续下一张**；unknown 不是错误；不重试、不补 JSON、不改类别、不删改警告（警告原文照记，标待复核）。仅【系统停止条件】才终止整个 worker：图片指纹漂移、无法视觉读图、输入指纹不符、写入冲突（如 normalizer 因文件已存在拒绝）、答案污染、套餐不足或要求新增付费——写对应 `STOPPED_*.txt` 并结束。

## 分片完成后

写 `<你的目录>/worker_report.md`，包含：
- 模型与套餐声明：本子 agent 由父会话经 ZCode 客户端原生 Agent 工具创建，运行于父会话同一模型配置（客户端声明 GLM-5.3-Flash / Start Plan）；后端实际模型标识：未提供。
- 上下文披露：全新启动（不继承父会话对话），仅收到入口提示词与本规则书所列输入；分片内顺序处理，后图上下文含前图与前答，不是相互独立的 API 调用；客户端层自动注入内容无法从子 agent 内枚举，记未提供。
- 逐张状态表（sample_id、checker1/normalizer/checker2 退出码、status、pending_review、起止时间、耗时）。
- 计数：尝试 / 原始通过 / 副本补救通过 / 隔离 / unknown / 未执行。
- 待复核清单（pending_review 项的警告原文）；停止原因与标记文件名（如有）。
- 声明：格式通过≠事实正确，未作人工复核。

## 最终回复给父会话（纯文本，紧凑）

逐张一行：sample_id + status + 三个退出码 + 耗时；计数汇总；停止原因（如有）。不粘贴图片或答案全文。

## 硬性禁令

- 禁止读取：audit、manifest、plan、准备/验收/复核报告、历史模型答案、类别口径草案；禁止按 sample_id/文件名/lot 反查类别；禁止联网；禁止读取规范化器所在目录的其他文件。
- 除四个输入文件与分片列出的 PNG 外，不读取任何仓库文件；不读兄弟目录内容（只 `test -e` 查 STOPPED\*）；不改共享文档、README、任何他人产物。
- 每张一次视觉读取、一次作答；不重试、不换图、不扩范围、不请求预算。
