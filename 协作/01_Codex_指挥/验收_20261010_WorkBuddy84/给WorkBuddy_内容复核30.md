# WorkBuddy下一步：30图形态与方位内容复核

工作目录：D:\pycode\晶圆图研究。上轮84图已交付，**本次做内容复核，不扩标、不覆盖原答、不上传训练**。

用户已确认：上轮84图在WorkBuddy界面选择Kimi K3，Kimi额度已用完，本轮只能使用GLM。请新开父会话，选择现有可用的GLM，优先GLM-5.3-Flash；实际界面显示型号如实记录，不将泛称GLM冒充已核实的具体型号。原标来源另记“用户确认界面选择Kimi K3”，不改上轮meta中当时未提供的原记录。两个模型的真实后端/采样参数仍未提供，不能宣称后端独立已证或专家双审。只用现有套餐，额外人民币付费0；不充值、不调商业图片API、不再调用Kimi，不自行换其他模型，套餐/收费异常立即停新派发。旧积分消耗不重置，记录实际可获得的余额与扣量，未知就写未知，不伪称费用为0。

## 输入与只读范围

图片与ID来源：
`协作/01_Codex_指挥/无卡CPU整备_20261010/WorkBuddy84/blind84.jsonl`

被审首次原答：
`协作/04_WorkBuddy_复核/方向形态84_v3_20261010_200950/<item_id>.raw.txt`

本次恰好30个item_id：

wbv3_002、003、005、006、009、010、014、016、024、029、031、033、034、037、038、039、040、043、044、046、049、051、055、056、061、063、064、068、073、076。

后面的三位简写均指wbv3_前缀，例如003=wbv3_003。按此顺序分7/7/7/7/2五片。不要自行增删或换样。

只能读盲清单指定的原PNG、上轮固定标注题面、上面的被审原答以及本次任务。不要读取公开类别、lot、程序几何、来源审计、Codex初步判读、旧实验成绩或相邻父目录。父会话的验收理由不得传给worker。

## 执行步骤

1. **先补机械记录，不用重新看84图**：用真实程序现算84输入PNG的SHA，复跑既有workbuddy_schema_check.py，生成本次时间的独立核验表。wbv3_056旧started中的hash被截为61字符，请追加更正说明、旧值与现算新值；旧started/raw/check/meta/done原件保持不动。不声称补测能重建当时的读图或发送证据，不把新测量回填为旧时间。
2. 新输出只写`协作/04_WorkBuddy_复核/内容复核30_v3_实际开始时间/`。先做首7图走通输入、视觉、写盘链路，正常后自动续其余23。最多2个原生worker并行，每worker至多7图，新上下文、不继承其他答案。
3. 每图先核真实PNG SHA并记录started；**先不打开旧raw**，只看该图一次，按上轮七字段题面独立观察，原文先写`<review_id>.observation_raw.json`并计算SHA/时间。该观察只是本轮模型候选，不是gold。
4. 独立观察写盘之后，再打开对应旧raw，对照图与两份描述写`<review_id>.review.json`。本图视觉读取只做一次；观察与对照判读是两次不同阶段作答，如实分开记录，不冒充后台请求数。若客户端不能隔离这两个阶段，停止并说明，不假装独立。
5. 原84每图首次答案不改、不覆盖、不重试。单图无法判定就写unverifiable，不强猜钟点；单图失败隔离，不影响其余。图片hash不匹配、视觉/写入/套餐系统异常停新派发。

## 对照判读格式

每个review.json包含review_id、item_id、sample_id、image_sha256、original_raw_sha256、observation_raw_sha256、model_display_name、checks、overall、suggested_revision、uncertainty。

checks逐一覆盖morphology_zh、main_location_zh、location_clock、orientation_clock、secondary_observations、caption_zh、uncertainty。每项写：

`{"verdict":"supported/contradicted/unverifiable/not_applicable","evidence":"具体指出图上位置/结构和被审句子，不写空泛同意"}`

overall取usable/requires_review/reject；suggested_revision仅给建议，不能覆盖旧raw或自动进入训练。格式以外的内容问题必须如实留档。若两模型意见不一致，保存分歧，不按多数票强定真值。

注意：主要结构的位置不等于全部红点的平均方向；两团块不等于一条连续线；有中心分支不一定是纯竖直十字；none类别没有提供，不猜公开类，也不把“无明显模式”理解为无红点；钟点数字本身不是尺寸幻觉。不要从图中推工艺根因、精确覆盖比例、格数或物理尺寸。

## 最终交付

30计划/实际读图/独立观察/对照判读/失败/未执行分别计数；独立观察落盘先于打开旧答案的客户端记录、所有raw/review/meta/sha清单、84机械复核表及056追加更正齐全。仍属AI复核，不是人审/专家gold，不将定向抽查30图的通过率推广到全84。

完成即停，交Codex review；Claude Code随后负责合格数据副本与训练执行，不自行扩批或启动GPU。
