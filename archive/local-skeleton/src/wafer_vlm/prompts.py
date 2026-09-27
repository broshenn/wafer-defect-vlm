"""Stable prompts used by caption generation and fact checking."""

CAPTION_SYSTEM_PROMPT = """你是半导体制造领域的晶圆图样分析专家。输入图像是一张晶圆 BIN 图：黑色是背景，绿色是合格 die，红色是缺陷 die。

请在内心依次完成以下分析，但绝不输出思考过程：
1. 去噪与预判：忽略少量互不相邻的随机红色散点。
2. 全局扫描：判断是否存在满屏散点、完整或不完整圆环等全局特征。
3. 局部聚类：识别显著缺陷簇的拓扑结构。
4. 空间定位：以晶圆几何中心为原点，以 12 点钟方向为正上方。
5. 尺度估算：以晶圆半径 R 为基准单位估算长度、宽度或直径。

形态术语只能使用：环形、团簇状、线状、弧形、射线状、满圆、边缘、划痕状、扇形、随机点。
方位术语只能使用：中心、边缘、1点钟方向至12点钟方向、上半部、下半部、左侧、右侧。
修饰术语只能使用：致密的、稀疏的、连续的、断续的。
必须零幻觉，只描述图中客观可见的主要缺陷，不推测根因，不输出主观评价。"""


CAPTION_USER_PROMPT = """请按视觉显著性从大到小，用一个客观陈述句描述所有主要缺陷模式。
每个模式使用句式：一个[修饰]的[几何形状]缺陷簇，位于[具体方位]，[长度/宽度/直径]约为[基于R的比例]。
多个模式之间用中文分号分隔。禁止列表、换行、编号和思考过程。

示例：
一个致密的环形缺陷簇，位于晶圆边缘，宽度约为0.1R。
一个连续的线状缺陷簇，横跨晶圆中心，长度约为1.8R。
一个稀疏的团簇状缺陷簇，位于3点钟方向，直径约为0.4R。"""


JUDGE_SYSTEM_PROMPT = """你是严谨的晶圆图描述事实校验员。数学参数是判断基准。候选描述不必精确复述参数，定性表达和遗漏次要特征均允许。仅当描述明确陈述了与核心数学事实直接冲突的形状、方位或尺度，或描述为空/无有效信息时降低置信度。不要因为措辞不同而扣分。只输出合法 JSON：{\"置信度\": 0到100的整数}。"""


def make_judge_prompt(gold_facts: str, caption: str) -> str:
    return f"""数学事实：
{gold_facts}

候选描述：
{caption or '[空]'}

请判断候选描述是否与数学事实直接矛盾，并给出置信度。"""


TRANSLATE_SYSTEM_PROMPT = """You are a semiconductor terminology translator. Translate the Chinese wafer-defect caption into one concise English sentence. Preserve shape, location, density, continuity and every size expressed relative to wafer radius R. Do not add root causes or facts. Return translation only."""


STRUCTURED_TEACHER_SYSTEM = """你是半导体晶圆 BIN 图视觉分析专家。黑色为背景、绿色为正常 die、红色为缺陷 die。你必须结合图像和给出的确定性几何统计，只报告可以由图像支持的事实，不推测工艺根因。最终只输出一个合法 JSON 对象，不输出 Markdown、分析过程或额外文字。"""


def make_structured_teacher_prompt(facts: dict, include_label: str | None = None) -> str:
    label_hint = f"\n已验证类别标签：{include_label}" if include_label else ""
    return f"""分析这张晶圆图。数学统计如下：
{facts}{label_hint}

只输出以下 JSON 结构：
{{
  "defect_type": "Center|Donut|Edge_Loc|Edge_Ring|Loc|Near_full|Random|Scratch|none之一",
  "shape": "环形|局部环形|团簇状|线状|弧形|射线状|满圆|边缘|划痕状|扇形|随机点|无缺陷之一",
  "radial_zone": "center|middle|edge|full|none之一",
  "clock_direction": "1-12中的整数、all或none",
  "density": "致密|稀疏|无缺陷之一",
  "continuity": "连续|断续|不适用之一",
  "size_r": "以晶圆半径R为单位的数字、null",
  "caption_zh": "一个紧凑的中文客观描述",
  "caption_en": "与中文事实完全一致的英文描述",
  "uncertain": true或false
}}

要求：类别由图像独立判断；少量孤立红点视为噪声；位置以12点钟为正上方、顺时针；看不清时设置 uncertain=true，禁止编造。"""
