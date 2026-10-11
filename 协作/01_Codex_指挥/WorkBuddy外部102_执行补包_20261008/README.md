# WorkBuddy外部102执行补包

1200主标原答完成并冻结后，另开独立新会话。使用GLM-5.3-Flash现有客户端套餐，新增付费0；不延续训练标注会话，不读取训练答案。这里不含真实图片类别、gold、学生输出或成绩。

允许读取的输入：

- 本补包的 `prompt_7f.txt`、`schema_check.py` 与本说明。
- 原盲包 `D:\pycode\晶圆图研究\协作\02_ClaudeCode_实操\夜间扩容_20261008_0244\客户端盲包\WorkBuddy_外部评测102\` 内的README、总索引与15个分片。
- 索引指向的公开PNG；只用于当次视觉输入与图片hash检查。

总索引为 `WB_eval102_索引.jsonl`，SHA256：`41d6e79c7ae6026fae34a609f39af4ced07aa080b163c18e813d1205c9dcd0c1`。每行仅item_id、sample_id、image_path、image_sha256。共102图、15分片，每片最多7图。题面文件是UTF-8、无BOM、无末尾换行，共591字符，SHA256：`8a8315f785a2398ba5a823754c40b8a8658f89b79aad1a0dbd68058acb43cfda`。

最多两路新GLM worker，每个worker最多7图，不继承其他worker答案。每图核图片hash后一次视觉输入、一次作答，不重试、不纠正已经冻结的原答。全部102图按分片自动续，不限首3图，不需每片审批。系统输入、视觉链路、写盘、套餐或新增付费异常时停止新派发；单图失败保留并继续。

每次逐字发送 `prompt_7f.txt` 全文，不能改写；保存可获取的实际提交文本并现算hash。拿不到客户端后台记录、精确后端标识或用量时写未知，不能用固定填写的hash冒称实际请求已核验。输出只写本人新目录 `协作/04_WorkBuddy_复核/外部评测102_20261008_<实际时间>/`，保留原答、图片/题面/原答hash、显示模型、时间、已尝试/失败/未执行及检查结果。

校验命令（原答文件为直接UTF-8文本或JSON对象）：

```powershell
D:/python/python.exe -X utf8 -B schema_check.py --input '<单原答文件>' --out '<另存校验文件.json>'
```

若输入为包含原答的JSON记录，显式加 `--raw-key raw_answer`（或 `--raw-key raw`）。若传入的是dict，校验会标明不能恢复逐字原文，不伪造原文hash。校验输出不覆盖输入；目标校验文件已存在时拒绝覆盖。

口径分列：`strict_json`拒绝重复键、非有限常量、围栏、多个对象和尾随文本；只允许原协议中的空think前缀。`schema_ok`要求恰好七字段、固定枚举和原类型约定；`extent_r`必须真正JSON null。`clock_direction`允许字符串或JSON null，字符串`"null"`仍是类型合法，但另列 `contract_warnings`，不得将其悄悄改成null或回写原答。原格式不合规保留失败，不能转换为主口径成功。结构合格不等于视觉事实正确，不生成或读取答案表。

离线回归：`schema_check.py --self-test`，13例全部为合成文本，不是图片答案。补包指纹见 `SHA256.json`。
