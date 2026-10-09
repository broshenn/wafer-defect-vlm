from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
SRC=ROOT/'协作/02_ClaudeCode_实操/56图候选补测与最终结项_20261009'
html=(SRC/'项目经历页_56图版.html').read_text(encoding='utf-8')
html=html.replace('准确率 <strong>0.2679 → 0.8214</strong>', '全schema门控类别准确率 <strong>0.2679 → 0.8214</strong>')
html=html.replace('拖成「未解析」','使全schema门控拒收').replace('严格口径 0.8214、宽松口径 0.8393','全schema门控 0.8214、类别字段独立口径 0.8393')
html=html.replace('（5）56 张里<strong>至少 4 张公开标签与图上明显不符</strong>（例：标 Center 的图除最外圈外几乎全红），<strong>「答错」里含标签噪声</strong>。',
                  '（5）记录<strong>AI与公开标签的疑似判读分歧</strong>，原标签错误尚未证实，保留原标签计分。')
html=html.replace('基座原生输出带 Markdown 围栏，严格 JSON 判分几乎全失败：<strong>必须把「格式失败」与「内容错误」分开计</strong>。',
                  '本56图三模型严格JSON均56/56可解析；<strong>JSON解析、完整schema与类别内容分别统计</strong>。')
html=html.replace('峰值显存 <strong>17.88 GiB</strong>','峰值已分配显存 <strong>17.88 GiB</strong>')
# 屏幕页宽210mm不能覆盖打印可用190mm；在打印媒体中明确覆盖高优先级的页面模式宽度。
html=html.replace('</head>', '<style>@media print { body {font-size:10pt !important;} body[data-page-mode="continuous"] main.sheet, body[data-page-mode="paged"] main.sheet { width:190mm !important; max-width:190mm !important; } main.sheet[data-fit-scale] > * { width:100% !important; zoom:1 !important; } .company-name {min-width:0;flex:1;overflow-wrap:anywhere;} .company-meta {flex-shrink:0;padding-left:8px;} }</style></head>')
# 全部细节在验收报告留存，项目经历页缩短重复的负结果与边界，保持正常字体与单页。
html=re.sub(r'<li class="nested"><strong>负结果（同批同图实测）：</strong>.*?</li>',
            '<li class="nested"><strong>负结果与口径：</strong>36图的AI描述复核不支持本项目超过最强外部模型；56图的L与D配对区间跨0。L的51/56条clock字符串null是约定警告，完整schema不合法仅1条。该条类别答对，故类别字段独立口径47/56、全schema门控46/56。公开标签与AI的疑似分歧留档，未认定标签错误。</li>',html,count=1,flags=re.S)
html=re.sub(r'<li><strong>个人边界：</strong>数据全部来自<strong>公开数据集</strong>.*?</li>',
            '<li><strong>个人边界：</strong>公开数据研究，未使用内部数据、未做人审或产线部署。36图是开发集，含1张与训练PNG相同的内容（另列35图诊断）；56图是七类留出候选，完整历史接触未核，缺Near_full/none。外部模型零样本，训练预算不同，保留比较局限。</li>',html,count=1,flags=re.S)
html=re.sub(r'<li><strong>背景与目标：</strong>晶圆 BIN 图.*?</li>', '<li><strong>目标：</strong>在公开BIN图上分别评价类别与描述，为质检沟通提供可核对的模型结果。</li>',html,count=1,flags=re.S)
html=re.sub(r'<li><strong>背景与目标：</strong>工业质检.*?</li>', '<li><strong>目标：</strong>让推理结果可审计、可消费，保留原答及不确定/人工复核提示。</li>',html,count=1,flags=re.S)
html=re.sub(r'<li class="nested"><strong>不达标项（照实写）：</strong>.*?</li>', '<li class="nested"><strong>限度：</strong>HTTP成功不等于内容正确；时延含本客户端流程，12次连接失败未重发，缺生产吞吐和稳定性证据。</li>',html,count=1,flags=re.S)
(HERE/'项目经历页_56图_核验版.html').write_text(html,encoding='utf-8')
md=(SRC/'简历_两版精简段落与90秒.md').read_text(encoding='utf-8')
md=md.replace('`FastAPI`','`Python标准库HTTP`').replace('`LoRA 热挂载`','`启动时加载LoRA`')
md=md.replace('严格 Acc','全schema门控 Acc').replace('严格口径 0.8214，宽松口径 0.8393','全schema门控0.8214，类别字段独立口径0.8393')
md=md.replace('拖成"未解析"','使全schema门控拒收').replace('拖成“未解析”','使全schema门控拒收')
md=md.replace('服务启动时**自校验三个 SHA**（题面 / 校验器 / adapter），不一致直接拒绝启动。',
              '服务校验题面与检查器SHA，逐请求记录adapter指纹；另存恢复启动包在加载前核D权重SHA。')
md=md.replace('- **负结果照写**：56 张里**至少 4 张公开标签与图上明显不符**（例：标 `Center` 的图除最外圈外几乎全红）。\n  判分仍按公开真值，但说明**"答错"里含标签噪声**。',
              '- **争议样本留档**：记录AI与公开标签的疑似判读分歧，尚未证实原标签错误，判分仍按公开标签。')
md=md.replace('峰值显存 **17.88 GiB**','峰值已分配显存 **17.88 GiB**')
md=md.replace('同图请求 **哈希一致、`request_id` 不同**','每次请求保存图片/原答哈希与唯一`request_id`')
md=md.replace('而且我自己测出来**至少 4 张图的公开标签和图上明显不符**，', '另外记录了AI与公开标签的疑似分歧，尚未证实标签错误，')
md=md.replace('**真人复核本项目已决定不做**，报告里真人栏是**永久留空**的。', '**本轮未进行真人复核**，报告里真人栏为空。')
# 其余位置若仍谈“已证实标签噪声”，在交付中明确给出本轮核验更正，而不改变真实指标。
md += '\n\n## Codex最终核验补记\n\n本报告中的“严格”主成绩以完整schema门控为条件；三个模型56份JSON均可解析。H052使L从类别字段正确47/56变成门控46/56。公开标签噪声未坐实；H003的渲染红占比约72.7%，不足以证明Center标错。H049的0.98R是未核测量依据，不以点少直接认定跨度不可能。旧主成绩与AI原判断保持留档。\n'
(HERE/'简历项目稿_56图_核验版.md').write_text(md,encoding='utf-8')
print('Saved corrected HTML and project text copies; original Claude files unchanged.')
