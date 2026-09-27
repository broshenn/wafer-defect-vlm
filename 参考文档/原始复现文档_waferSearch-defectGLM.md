# 原始复现详细文档：waferSearch-feature-defectGLM

> **来源**：mentor 提供的原始完整文档，基于 `waferSearch-feature-defectGLM-c6ef014ba1c34c9eb7d937befca92144f91bfebe` 源码逐文件走读整理。文档生成时间 2026-09-14。
> **保存时间**：2026-09-27。
> **本文件的性质**：这是**最早那一版方案（VisualGLM-6B / SwissArmyTransformer / SAT）的复现记录**，即 `AGENTS.md` §3「新旧方案对应关系」里说的**原方案**。当前主线（Qwen3.5-9B + ms-swift）是从它改过来的，两者不是同一个东西，读的时候不要混。
>
> ⚠️ **本文件已脱敏**：原文 §12.1 含 API Key、工号与内网端点。按 `AGENTS.md` §2 / §17 的红线，这些一律不得写入仓库，已替换为 `[已脱敏：…]` 占位符。其余内容**一字未改**（含原文的错别字如 `trianTestLabel`、`failue_ratio`，以及已知 bug 的描述）。如需真实凭证，只从服务器 `secrets/api.env` 读，不要回填到本文件。

---

## 1. 原始数据：字段、类别、获取方式

### 1.1 WBM180（180 条，阶段1探索用）

文件：`wbm_caption_code/WBM180.pkl`

来源：从 Kaggle 的 WM811K / LSWMD 数据集中，每类缺陷抽 20 条，共 9 类 × 20 = 180 条

格式：pandas DataFrame，180 行 × 6 列

| 列名 | 类型 | 说明 |
|---|---|---|
| waferMap | np.ndarray (uint8) | 二维矩阵，0=背景, 1=合格die, 2=缺陷die，尺寸不一（约 20-60 行 × 20-60 列） |
| dieSize | - | die 尺寸信息 |
| lotName | str | 如 "lot1992" |
| waferIndex | float | wafer 在 lot 中的序号 |
| trianTestLabel | 嵌套列表 | `[["Training"]]` 或 `[["Test"]]`（注意原始拼写是 trianTestLabel） |
| failureType | 嵌套列表/np.ndarray | 如 `[['Donut']]`、`[['none']]`，需要 `get_failure_type_label()` 展平 |

实际验证（`wbm_caption_code/show_all_types.py` 运行结果）：

9 类各 20 条：Donut, Near-full, Loc, Edge-Ring, Scratch, none, Random, Edge-Loc, Center

标签格式是 `[['Donut']]`（双层嵌套列表）

### 1.2 WBM180_converted（160 条，带预测标签）

文件：`/ossfs/workspace/wbm_caption_code/WBM180_converted.pkl`（服务器路径）

说明：在 WBM180 基础上增加 `predictDefectType` 列，并筛选掉 `predictDefectType == 'None'` 的行

实际：180 条中 160 条有缺陷标签，20 条 None 被排除

代码引用：`WBM_caption_qwen3v.py:196` — `df_defect = df[df['predictDefectType'] != 'None']`

### 1.3 LSWMD 全量（约 8 万+缺陷晶圆）

来源：Kaggle WM811K Wafer Map

原始文件：`LSWMD.pkl`（约 200MB），约 81 万行

转换为：`LSWMD_wLabel_converted.pkl` — 在原始数据上增加 `predictDefectType` 列（**来源不详，代码中未找到生成该列的逻辑**）

筛选条件：`predictDefectType != 'None'`，得到约 8 万+条缺陷晶圆

### 1.4 缺陷类型（9 类）

| 类型 | 中文 | 说明 |
|---|---|---|
| Center | 中心缺陷 | 中心区域聚集 |
| Donut | 环形缺陷 | 中心正常，周围环形缺陷 |
| Edge-Loc | 边缘局部 | 边缘局部聚集 |
| Edge-Ring | 边缘环状 | 边缘完整环形 |
| Loc | 局部聚集 | 某区域聚集 |
| Near-full | 近满片 | 几乎全片缺陷 |
| Random | 随机 | 散布随机点 |
| Scratch | 划痕 | 线状缺陷 |
| none | 无缺陷 | 正常分布 |

标准化处理：`get_failure_type_label()`（`utils.py:35-63`），将 `-` 替换为 `_`（如 Edge-Loc → Edge_Loc），处理嵌套列表/ndarray/字符串等多种格式

---

## 2. 去噪与矩阵转图片

### 2.1 去噪

项目有两套去噪方法：

**方法 A：孤立点过滤（简单版）**

函数：`apply_filter(matrix)`（`WBM_caption/utils.py:512-538`）

逻辑：遍历矩阵中每个值=2 的点，如果其 8 邻域内没有其他值=2 的点，改为 1

用途：`calculate_static_param()` 内部调用（`utils.py:372`），算金标时先滤波

阶段1 caption 生成时不调用（`WBM_caption_qwen3v.py:77` 中 `add_filter=False`）

**方法 B：多尺度 C-Mean 过滤（进阶版）**

函数：`multi_scale_cmean_filtering(wafer_map, scale_thresholds=[1.1,1.2,1.3], scales=[3,5,7], voting_method='majority', min_valid_neighbors=1)`（`WBM_caption/utils.py:197-250`）

逻辑：

- 对每个缺陷点，在 3×3、5×5、7×7 三个尺度上分别计算邻域均值（排除中心点）
- 每个尺度判断 `avg_value < threshold`（是否为误判）
- 多数投票：≥2 个尺度判定为误判 → 改为 1

用途：`matrix2image()` 中 `AddFilter=True` 时调用（`utils.py:150`）

探索阶段用过（`wbm_caption_code/show_180.ipynb` 中调用了 `multi_scale_cmean_filtering`），但 caption 生成管线中 `add_filter=False`

### 2.2 矩阵转图片

有两个版本，caption 管线用的是 `_cached` 版本：

**版本 A：`matrix2image()`（`WBM_caption/utils.py:135-171`）— 探索期用**

```python
def matrix2image(wafer_matrix, resolution, color_stype='default', AddRef=False, AddFilter=False, pad_s=0)
```

支持多种配色方案（黑绿红/黑灰白/蓝绿红/白绿红/黑绿黄等 8 种）

用 `scipy.ndimage.zoom` 最近邻插值放大

可选添加参考线（`addXYCircle`）

注意：这个版本要求 `color_stype` 在预定义 `color_maps` 字典中，`'default'` 不是有效值

**版本 B：`matrix2image_cached()`（`WBM_caption_qwen3v.py:35-70`）— 管线实际用**

```python
def matrix2image_cached(wafer_matrix, resolution, color_map='黑绿红', add_ref=False, add_filter=False) -> PIL.Image
```

准确步骤和参数（阶段1，`WBM_caption_qwen3v.py`）：

- 如果 `add_filter=True`，先调 `apply_filter()`
- 创建 RGB 数组，颜色映射：`{0:[0,0,0], 1:[0,255,0], 2:[255,0,0]}`（黑绿红）
- 用 `scipy.ndimage.zoom` 放大到 `resolution`（管线中用 448）
- 如果 `add_ref=True`：用 `find_contours` 找轮廓 → `fit_circle_least_squares` 拟合圆心 → `addXYCircle` 画坐标轴和参考圆

阶段1实际参数：`resolution=448, color_map='黑绿红', add_ref=False, add_filter=False`（`WBM_caption_qwen3v.py:250-251`）

准确步骤和参数（阶段3，`WBM_caption_glm45v.py`）：

- 先做 `np.pad(waferMap, ((2,2),(2,2)))` — 边缘填充 2 圈（**阶段1没有这步**）
- 其余同阶段1

阶段3实际参数：`resolution=448, color_map='黑绿红', add_ref=False, add_filter=False`（`WBM_caption_glm45v.py:265-266`）

准确步骤和参数（阶段6，`tran2Json.py`）：

- 不做边缘填充

阶段6实际参数：`resolution=364, color_map='黑绿红', add_ref=False, add_filter=False`（`tran2Json.py:199`）

保存为 `.jpg`

参考：`addXYCircle()`（`WBM_caption/utils.py:81-132`）— **有已知 bug**

- 用 `find_contours` + `fit_circle_least_squares` 找晶圆中心和半径
- 用 `cv2.arrowedLine` 画 X/Y 坐标轴（青色）
- 用 `cv2.circle` 画晶圆边界圆（青色）
- ⚠️ Bug：第 120 行 `cv2.addWeighted(rgb_image, 1.0, axis_layer, axis_alpha, 0)` 中 `axis_alpha` 未定义，会 NameError
- `wbm_caption_code/show_180.ipynb` 中用户手动重写了这个函数，用了固定值替代
- 实际管线中 `add_ref=False`，所以这个 bug 不会被触发

---

## 3. 数学特征：公式、坐标系、阈值、异常处理

### 3.1 `calculate_static_param(wafer_matrix)` — 数学金标（judge 阶段用）

文件：`WBM_caption/utils.py:357-381`

用途：阶段2/4 事实校验时计算金标参数

计算步骤：

1. `find_contours(wafer_matrix > 0, level=0.5)` → 取最长轮廓
2. `fit_circle_least_squares` 拟合晶圆边界圆 → `(center_x, center_y, radius)`
3. `failue_ratio = mean(wafer==2) / mean(wafer>0)` — 缺陷占比
4. 异常处理：如果 `failue_ratio == 0`，返回 `-1, None`
5. 调 `compute_radial_tangential_stats(points, center, radius)` 算详细统计
6. 再对滤波后的矩阵算一次（`apply_filter` 后），如果滤波后 `failue_ratio == 0`（全是随机噪声），返回 `-2, {'ratio': failue_ratio}`

**`compute_radial_tangential_stats(points, center, radi)` — `utils.py:252-315`**

坐标系：以晶圆中心为原点，像素坐标系（x 向右，y 向下）

计算的特征：

| 特征 | 公式 | 说明 |
|---|---|---|
| L1_center | `mean(points - center) / radi` | 质心坐标，归一化到单位圆（[-1,1]范围） |
| L2_center | `sqrt(mean((points-center)^2)) / radi` | L2 质心 |
| mean_r | `mean(||points - center||)` | 平均径向距离 |
| std_r | `std(||points - center||)` | 径向距离标准差 |
| radial.hist | `histogram(rs, bins=10)` 然后除以圆环面积 | 10 个圆环的密度分布 |
| mean_theta | `90 + arctan2(sin_mean, cos_mean)` (度) | 平均角度（0点钟=90度） |
| R (resultant_length) | `sqrt(cos_mean^2 + sin_mean^2)` | 合成矢量长度 [0,1]，越高越集中 |
| angle_entropy | `entropy(angle_probs, base=2)` | 角度分布熵，越高越均匀 |
| angular.hist | `histogram(thetas, bins=12, range=(-π,π))`，然后偏移 3 个 bin | 12 扇区分布，偏移对齐 12 点钟 |
| covariance.matrix | `np.cov(rel.T)` | 2×2 协方差矩阵 |
| eigenvalues | `np.linalg.eigh(cov_matrix)` | 特征值 |
| anisotropy_ratio | `eigvals[0] / (eigvals[1] + 1e-12)` | 各向异性比，越高越线状 |

径向直方图细节：

- `np.histogram(rs, bins=10, range=(0, max(rs)))` — 10 个等距圆环
- 每个圆环的计数除以该圆环面积 `π(R_out² - R_in²)` 得到密度

角度直方图细节：

- `np.histogram(thetas, bins=12, range=(-π,π))` — 12 个 30° 扇区
- 偏移 3 个 bin：`np.concatenate([angle_hist[3:], angle_hist[:3]])` — 让 0 点钟方向（-π/2 附近）落在第一个 bin

### 3.2 `calculate_geo_param(wafer_matrix)` — 7 个核心指标

文件：`WBM_caption/utils.py:383-509`

⚠️ **硬编码 52×52**：`R=26.0`（第 430 行）、`C_cover = len(defect_points) / (52*52)`（第 470 行）、Moran's I 循环 `range(52)`（第 486-492 行）

用途：探索阶段用，**管线代码中没有调用**

| 指标 | 公式 | 说明 |
|---|---|---|
| C_s (空间集中度) | `1 - (std_x * std_y) / (range_x * range_y)`，clamp [0,1] | 越高越集中 |
| R_e (边缘距离比) | `\|\|mean(defect_coords)\|\| / 26.0` | 缺陷质心到中心的归一化距离 |
| C_shape (形态一致性) | `P / sqrt(4π * N)`，P=轮廓像素数，N=缺陷像素数 | 衡量形状规则度 |
| R²_scr (线性拟合) | 对缺陷坐标做 LinearRegression 的 R²，clamp [0,1] | 越高越像 Scratch |
| R_ring (环形特征) | `mean(radii)` | 缺陷点到中心的平均距离 |
| C_cover (覆盖度) | `N / (52*52)` | 硬编码 52×52 |
| I_ac (Moran's I) | `(n * numerator) / (S0 * denominator)` | 空间自相关，>0 聚集，<0 分散。遍历 52×52 |

### 3.3 `fit_circle_least_squares(x, y)` — `utils.py:66-79`

最小二乘法拟合圆：解 `A[a,b,c]^T = C`，其中 `A=[[x,y,1]]`，`C=x²+y²`

返回 `(center_x, center_y, radius)`，**均为 int（四舍五入）**

### 3.4 `extract_final_output(text, strict=False)` — `utils.py:340-355`

优先匹配 `<|begin_of_box|>...<|end_of_box|>`

其次匹配最后一个 `</think>` 之后的内容

`strict=True` 时找不到返回 `"无法描述"`

`strict=False` 时返回原文

### 3.5 异常处理总结

| 场景 | 处理 | 代码位置 |
|---|---|---|
| `failue_ratio == 0`（无缺陷） | `calculate_static_param` 返回 `-1, None` | `utils.py:364-365` |
| 滤波后无缺陷（全是随机噪声） | 返回 `-2, {'ratio': 0}` | `utils.py:375-376` |
| `calculate_static_param` 异常 | `safe_extract_params` 返回默认值 | `WBM_judge_qwen3vl.py:97-111` |
| `find_contours` 无轮廓 | `addXYCircle` 用图像中心 + `min(高,宽)/4` | `utils.py:92-94` |
| VLM API 失败 | 返回 `"ERROR"` 字符串，写入 CSV | `WBM_caption_qwen3v.py:151-159` |
| judge 解析 JSON 失败 | `confidence = -1` | `WBM_judge_qwen3vl.py:199-201` |

---

## 4. Caption 和 Judge 的 Prompt、模型参数、置信度及筛选规则

### 4.1 阶段1：Caption 生成

文件：`WBM_caption/WBM_caption_qwen3v.py:81-120`

原始模型：`Qwen3-VL-235B-A22B-Thinking-FP8`

服务器实际：通过 sed 替换为 `GLM-4.5V-fp8`（但代码中原始值未改）

API 参数：`max_tokens=8192, stream=False`（无 temperature 设定）

System Prompt（完整，`WBM_caption_qwen3v.py:84-104`）：见 §11.1

User Prompt（`WBM_caption_qwen3v.py:107-120`）：见 §11.2

输出提取：`extract_final_output(content, True)` — `strict=True`，提取 `<|begin_of_box|>...<|end_of_box|>` 后内容，找不到返回 `"无法描述"`

### 4.2 阶段2：事实校验

文件：`WBM_caption/WBM_judge_qwen3vl.py:34-95`

模型：原始 `Qwen3-VL-235B-A22B-Thinking-FP8`（服务器改为 GLM-4.5V）

API 参数：`max_tokens=7192, temperature=0`

**注意：judge 阶段只发文字，不发图片**（`create_assessment_prompt` 返回的 messages 只有 text 类型）

Prompt 设计要点（`WBM_judge_qwen3vl.py:51-87`）：

- 身份：资深半导体晶圆缺陷分析专家，扮演"事实校验员"（不是打分员）
- 给出：金标参数（数学事实）+ 候选描述
- 按缺陷类型给不同评估重点（9 种类型各有 guidance）
- 触发"不准确"的条件：仅当明确陈述与核心数学事实直接冲突
- 允许的宽容：定性表达、遗漏次要特征、区域描述方向大体一致
- 如果候选描述无有效信息 → 置信度 0
- 输出格式：`{"置信度": 0-100的整数}`

金标参数描述格式（`generate_param_description()`，`WBM_judge_qwen3vl.py:113-155`）：

```text
缺陷芯片占比：15.3%
分布形状：环状分布
质心坐标（单位圆坐标系）：(0.021, -0.003)
径向分布（10个圆环，从中心到边缘）：[0.001, 0.002, ..., 0.087]
角度分布（12个扇形，从0点方向顺时针）：[0.08, 0.07, ..., 0.05]
```

置信度解析（`WBM_judge_qwen3vl.py:191-197`）：

- `extract_final_output(response_text, True)` 提取
- `re.search(r'\{.*\}', box_text)` 匹配 JSON
- `json.loads` 解析，取 `result_dict.get("置信度", -1)`
- 失败时 `confidence = -1`

### 4.3 阶段3：GLM 交叉标注

文件：`WBM_caption/WBM_caption_glm45v.py:79-118`

模型：原始 `Qwen3-VL-235B-A22B-Thinking-FP8`（服务器改为 GLM-4.5V）

API 参数：`max_tokens=6192, stream=False`

Prompt：与阶段1**完全相同**（system + user 一字不差）

区别：对 waferMap 做了 `np.pad(waferMap, ((2,2),(2,2)))` 边缘填充

### 4.4 阶段4：GLM caption 校验

文件：`WBM_caption/WBM_judge_glm45v.py:30-91`

模型：原始 `Qwen3-VL-235B-A22B-Thinking-FP8`（服务器改为 GLM-4.5V）

API 参数：`max_tokens=8196, temperature=0`

Prompt：与阶段2**完全相同**

区别：评估的是 `description['glm45vl_result']`（阶段3的输出），不是 `qwen3vl_result`

### 4.5 筛选规则

| 阶段 | 筛选条件 | 代码位置 |
|---|---|---|
| 阶段1→2 | 全部数据进入校验，无筛选 | - |
| 阶段2→3 | `qwen3vl_confidence < 60` → 进入阶段3 | `WBM_caption_glm45v.py:251` |
| 阶段2高置信 | `qwen3vl_confidence >= 60` → 直接写入 csv，`glm45vl_result="None"` | `WBM_caption_glm45v.py:253-257` |
| 阶段3→4 | `glm45vl_result` 不是 ERROR 才校验 | `WBM_judge_glm45v.py:282` |
| 阶段4→6 | `glm45vl_confidence > 60` → 进入阶段6 | `tran2Json.py:245` |
| 阶段4低置信 | `glm45vl_confidence <= 60` 或 ERROR → 丢弃 | `tran2Json.py:282-288` |

⚠️ **已知问题**：阶段2 的 67 条高置信度数据（`qwen3vl_confidence >= 60`）没有被 `tran2Json.py` 处理，只有阶段4 的 14 条进入了 COCO 输出。`tran2Json.py:245` 只检查 `glm45vl_confidence > 60`。

---

## 5. COCO 数据如何产生

### 5.1 阶段6：`tran2Json.py`

输入：`judge_glm45v_checkpoints/latest.csv`（阶段4 输出）

筛选：`glm45vl_confidence > 60`（`tran2Json.py:245`）

### 5.2 处理流程

**加载已处理 ID**（`tran2Json.py:149-174`）：读 JSONL 里的 `image_id`，跳过已处理

**翻译**（`tran2Json.py:63-101`）：

- System: `"你是一名专业的半导体晶圆制造工程师"`
- 强制模板：`[密度] [形态] defect pattern [位置], with a [尺寸参数] of approximately [数值].`
- 示例：`"一个致密的团簇状缺陷簇，位于中心，直径约为0.4R。"` → `"A dense clustered defect pattern located at the wafer center, with a diameter of approximately 0.4 times the wafer radius."`
- 模型：原始 `Qwen3-VL-235B-A22B-Thinking-FP8`（服务器改为 GLM-4.5V）
- `max_tokens=3096`

**生成图片**（`tran2Json.py:199`）：

`matrix2image_cached(whole_df.loc[idx, 'waferMap'], 364, '黑绿红', False, False)`

分辨率 364，保存为 `.jpg`

**image_id 格式化**（`tran2Json.py:53-61`）：

`lot17115_16` → `wafer_00017115_016`（lot 号补零到 5 位，wafer 序号补零到 3 位）

**写入 COCO JSON**（`tran2Json.py:203-208`）：

```json
{
  "caption": "A dense clustered defect pattern...",
  "image": "wbm20251204/wafer_00001_016.jpg",
  "image_id": "wafer_00001_016",
  "defect_class": "Center"
}
```

### 5.3 批量写入优化

先写 JSONL，每 1000 条批量写一次（`tran2Json.py:106-130`）

`atexit.register(flush_remaining_cache)` — 程序退出时兜底写入剩余缓存（`tran2Json.py:145`）

最后合并 JSONL → JSON 数组（`tran2Json.py:280-311`）

### 5.4 产出文件

```text
output/coco/annotations/wbm20251204.json   # COCO JSON 数组
output/coco/wbm20251204/*.jpg              # 晶圆图图片
```

---

## 6. 两阶段训练：模型、冻结层、Loss、LoRA、训练参数

### 6.1 模型架构

基础模型：**VisualGLM-6B**

- `model/visualglm.py:30` — `VisualGLMModel(ChatGLMModel)`，继承 ChatGLM
- `model/visualglm.py:11-28` — `ImageMixin`：在 ChatGLM 中插入 BLIP2 图像编码器
- `model/blip2.py` — BLIP2 架构：EVA-ViT + Qformer
- 图像 token 长度：`image_length=32`（`visualglm.py:39`）
- 图像嵌入插入位置：`<img>` token 之后，覆盖 32 个 pad token（`visualglm.py:25`）

微调模型：`FineTuneVisualGLMModel(VisualGLMModel)`（`finetune_visualglm.py:20-61`）

### 6.2 Stage 1：ViT 适配

文件：`finetune_visualglm.py:141-168`（`forward_step_adaptation`）

数据集：`AdaptationDataset`（`finetune_visualglm.py:225-249`）

输入：`{"img": 图片路径, "class": 类别标签}`（JSON 格式）

不需要文本 caption，只需要图片 + 9 类标签

冻结策略：

- 调用 `model.get_mixin("eva").model.vit.start_adaptation(args.eva_args)`（`finetune_visualglm.py:297`）
- 只训练 ViT，冻结 LLM

数据增强：

- `transforms.RandomRotation(degrees=(-90, 90))`（`finetune_visualglm.py:148`）
- 对同一张图做旋转，得到 `aug_images`

特征提取：

- `model.get_mixin("eva").model.vit(images)[0]` → `(logits, representation)`
- `representation` 是 ViT 的输出特征

Loss（`finetune_visualglm.py:161-163`）：

```python
loss_infonce = InfoNCELoss()
loss_ce = CrossEntropyLoss()
loss = loss_ce(logits, labels) + loss_infonce(sim_score)
```

InfoNCE（`contrastive_learning.py:5-25`）：

- `temperature=0.1`
- `similarity_scores /= temperature`
- 双向损失：`(loss_vt + loss_tv) / 2`
- 正样本：原图和旋转图的特征对（对角线）
- 负样本：batch 内其他图片

`cosine_sim(x, y)`（`contrastive_learning.py:28-35`）：标准余弦相似度矩阵

CrossEntropy：9 类分类损失

**总 loss = CE + InfoNCE**

图像预处理：`BlipImageEvalProcessor(224)`（`finetune_visualglm.py:254`）— 224×224

### 6.3 Stage 2：LoRA 微调

文件：`finetune_visualglm.py:117-138`（`forward_step`）

数据集：`FewShotDataset`（`finetune_visualglm.py:176-222`）

输入 JSON 格式：`{"img": 图片路径, "prompt": 问题, "label": caption}`

构造序列：`<img>{32个pad}</img>问：{prompt}\n答：{label}`

标签：输入部分 mask 为 -100，只对答案部分计算 loss

`max_source_length=128, max_target_length=128`（`finetune_visualglm.sh:11-12`）

冻结策略（`finetune_visualglm.py:46-59`，`disable_untrainable_params`）：

- 只训练 `matrix_A` 和 `matrix_B`（LoRA 参数）
- 其余所有参数 `requires_grad_(False)`

Loss（`finetune_visualglm.py:129-134`）：

```python
shift_logits = lm_logits[..., :-1, :].contiguous()
shift_labels = labels[..., 1:].contiguous()
loss_fct = CrossEntropyLoss(ignore_index=-100)
loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
```

标准 causal LM next-token prediction

`ignore_index=-100`：pad 和输入部分不计算 loss

调用：`model.get_mixin("eva").model.vit.stop_adaptation(args.eva_args)` — 停止 ViT 适配模式（`finetune_visualglm.py:313`）

### 6.4 LoRA 配置

LoRA 实现：`lora_mixin.py`

**自定义实现，不使用 PEFT 库**

`LoraLinear`（`lora_mixin.py:71-130`）：

- `matrix_A`: shape `(r, in_dim)`，Kaiming 均匀初始化
- `matrix_B`: shape `(out_dim//partition, r)`，零初始化
- `scaling = lora_alpha / r`
- forward: `original(x) + (dropout(x) @ A^T @ B^T) * scaling`

`LoraMixin.reinit()`（`lora_mixin.py:191-226`）：

- 替换每层的 `attention.dense`（partition=1）和 `attention.query_key_value`（partition=3, head_first=True）
- 如果 `cross_attention=True` 且是 decoder 层，还替换 `cross_attention.dense/query/key_value`

`LoraMixin.merge_lora()`（`lora_mixin.py:228-237`）：

- `new_weight = original_weight + (A^T @ B^T) * scaling`

Stage 2 LoRA 配置（`finetune_visualglm.py:26-28`，`--use_lora`）：

```python
# LLM 侧 LoRA
self.add_mixin("lora", LoraMixin(args.num_layers, args.chatglm_lora_rank, layer_range=args.chatglm_layer_range), reinit=True)

# ViT 侧 glm_proj 层 LoRA
self.get_mixin("eva").model.glm_proj = replace_linear_with_lora(self.get_mixin("eva").model.glm_proj, 1, args.chatglm_lora_rank)

# ViT 内部 LoRA
self.get_mixin("eva").model.vit.add_mixin("lora", LoraMixin(args.num_layers, args.vit_lora_rank, layer_range=args.vit_layer_range), reinit=True)
```

| 参数 | 值 | 来源 |
|---|---|---|
| vit_lora_rank | 16 | `finetune_visualglm.sh:12` |
| vit_layer_range | 0 13 26 38 | `finetune_visualglm.sh:13` |
| chatglm_lora_rank | 16 | `finetune_visualglm.sh:14` |
| chatglm_layer_range | 0 19 27 | `finetune_visualglm.sh:15` |
| lora_alpha | **默认 1**（`lora_mixin.py:167`） | — |
| lora_dropout | 默认 0（`lora_mixin.py:168`） | — |
| scaling | `alpha / r = 1/16 = 0.0625` | — |

QLoRA 配置（`finetune_visualglm_qlora.sh`，`--use_qlora`）：

| 参数 | 值 |
|---|---|
| lora_rank | 10 |
| layer_range | 0 14 |
| lr | 0.0001 |
| batch_size | 1 |
| gradient_accumulation_steps | 4 |
| train_iters | 300 |

### 6.5 训练参数（`finetune_visualglm.sh`）

| 参数 | 值 | 说明 |
|---|---|---|
| train_iters | 4500 | 总训练步数 |
| lr | 0.001 | 学习率 |
| lr_decay_style | cosine | 余弦退火 |
| warmup | 0.02 | 预热比例 |
| batch_size | 16 | 批大小 |
| eval_batch_size | 10 | 评估批大小 |
| eval_interval | 100 | 每 100 步评估 |
| save_interval | 4500 | 最后保存 |
| fp16 | True | 半精度 |
| zero_stage | 1 | DeepSpeed ZeRO-1 |
| max_source_length | 128 | 输入最大长度 |
| max_target_length | 128 | 输出最大长度 |
| pre_seq_len | 4 | PTuningV2 序列长度（但 `--use_ptuning` 未开启） |
| num_gpus | 8 | 8 GPU |
| train_data | `finetune_data/dataset_trn.json` | 训练数据 |
| eval_data | `finetune_data/dataset_tst.json` | 评估数据 |
| image_size | 224 | `BlipImageEvalProcessor(224)` |

### 6.6 训练数据格式

Stage 1 适配数据（`AdaptationDataset`，`finetune_visualglm.py:225-249`）：

```json
[{"img": "path/to/wafer.jpg", "class": 0}]
```

`class` 是整数标签（0-8，对应 9 类缺陷）

Stage 2 微调数据（`FewShotDataset`，`finetune_visualglm.py:176-222`）：

```json
[{"img": "path/to/wafer.jpg", "prompt": "What is the defect pattern?", "label": "A dense clustered..."}]
```

`prompt` 是问题文本

`label` 是英文 caption

---

## 7. 检索评估：特征取自哪层、AP/Recall 怎么算

### 7.1 文件：`inference_online.py`

### 7.2 特征提取

**BLIP2 特征提取器**（`inference_online.py:107-167`，`BLIP2FeatureExtractor`）：

```python
# 特征提取流程 (inference_online.py:161-167)
vit_features = self.model.get_mixin("eva").model.vit(images)[0]      # ViT 输出
query_feature = self.model.get_mixin("eva").model.qformer(vit_features)[0]  # QFormer 输出
image_features = query_feature.mean(dim=1, keepdim=True)  # mean pooling
image_features = image_features / (image_features.norm(p=2, dim=-1, keepdim=True) + 1e-8)  # L2 归一化
```

取自哪层：ViT → QFormer → mean pooling（对 query token 维度取均值）→ L2 归一化

输出：`(B, D)` numpy 数组

图像预处理：`BlipImageEvalProcessor(224)`（`inference_online.py:581`）— 224×224

**DINOv3 特征提取器**（`inference_online.py:169-213`，`DINOv3FeatureExtractor`）：

取 MLP 前的特征，L2 归一化

`feature_dim=384`，`pooling_strategy="gap+cls"`

**DINOv3 Baseline**（`inference_online.py:216-244`）：

取 `pooler_output`，L2 归一化

实际使用：`main()` 中默认用 `BLIP2FeatureExtractor`（`inference_online.py:577-580`），DINOv3 代码被注释

### 7.3 检索与评估

检索（`retrieve_topk`，`inference_online.py:297-306`）：

```python
sim = cosine_similarity(query_feature.reshape(1, -1), gallery_features.squeeze())[0]
topk_idx = np.argsort(-sim)[:topk]
```

AP@K（`compute_ap_at_k`，`inference_online.py:309-323`）：见 §10 与 §10.19

K 值：`[1, 5, 10, 25, 50]`（`inference_online.py:429`）

`labels_in_topk`：top-K 结果中每个是否相关（1/0）

`total_relevant`：gallery 中总相关数

Recall@K（`compute_recall_at_k_single`，`inference_online.py:326-330`）：

```python
def compute_recall_at_k_single(labels_in_topk, total_relevant, k):
    return sum(labels_in_topk[:k_use]) / total_relevant
```

K 值：`k1 = total_relevant`（相关样本数 N），`k2 = min(int(N * 1.5), max_possible)`（`inference_online.py:484-485`）

Baseline 对比：

`baseline = CSV 原始顺序的 AP/Recall`（`inference_online.py:430`）

用于对比模型检索 vs 随机顺序

### 7.4 测试集结构（`inference_online.py:399-414`）

```text
TestDataset/
├── {category}/              # gallery 目录（该类的参考图片）
├── {category}_top500.csv    # 标注 CSV: Wafer_id, Label(1=相关, 0=不相关)
└── {category}_query.png     # 查询图片
```

每类一个 gallery 文件夹 + 一个 query 图 + 一个 CSV

CSV 中 `Label=1` 表示与 query 同类

### 7.5 评估流程（`process_category`，`inference_online.py:399-530`）

1. 读 CSV → `label_map = {filename: label}`
2. `total_relevant = sum(Label)` → 相关样本数
3. 提取 gallery 所有图片特征
4. 提取 query 图片特征
5. `retrieve_topk` 取 top-K
6. 对照 `label_map` 给检索结果打标签
7. 计算 AP@K 和 Recall@K
8. 输出 `retrieval_results.json` + 可视化 `retrieval.jpg`

### 7.6 汇总（`main`，`inference_online.py:537-686`）

遍历所有类别，汇总 per-category 指标

计算 overall Mean AP@50、Mean AP@5、Mean Recall

输出 `summary.json`

---

## 8. 源码与 REPRODUCTION_GUIDE.md 的不一致或缺失

### 8.1 数据相关

| 项 | REPRODUCTION_GUIDE.md 描述 | 源码实际情况 |
|---|---|---|
| 数据列名 | failureType 是嵌套列表 `[["Donut"]]` | ✅ 一致，实际验证确认 |
| predictDefectType 来源 | 未说明如何生成 | ❌ 源码中没有生成该列的逻辑，`WBM180_converted.pkl` 和 `LSWMD_wLabel_converted.pkl` 都是预先准备好的，来源不详 |
| dieSize 列 | 未提及 | ✅ WBM180.pkl 中存在此列 |
| trianTestLabel 拼写 | 未提及 | ✅ 原始数据确实是 `trianTestLabel`（拼写错误），不是 `trainTestLabel` |

### 8.2 矩阵转图片

| 项 | REPRODUCTION_GUIDE.md 描述 | 源码实际情况 |
|---|---|---|
| 分辨率 | "224 (分类用) 或 448 (caption 用)" | ✅ caption 阶段1/3 用 448，阶段6 用 364（指南未提及 364） |
| 阶段3 边缘填充 | 未提及 | ❌ `WBM_caption_glm45v.py:74` 做了 `np.pad(waferMap, ((2,2),(2,2)))`，阶段1 没有这步，指南未记录 |
| 颜色映射 | 0→黑色, 1→绿色, 2→红色 | ✅ `matrix2image_cached` 中硬编码 `{0:[0,0,0], 1:[0,255,0], 2:[255,0,0]}` |
| addXYCircle bug | 未提及 | ❌ `utils.py:120` 引用未定义的 `axis_alpha`，管线中 `add_ref=False` 所以不触发 |

### 8.3 Caption/Judge

| 项 | REPRODUCTION_GUIDE.md 描述 | 源码实际情况 |
|---|---|---|
| judge 不发图片 | "先算数学金标, 再让 LLM 判断" | ✅ 确认：`create_assessment_prompt` 返回的 messages 只有 text，没有 image_url |
| judge 模型 | "纯文本 LLM (Qwen3-235B)" | ❌ 源码中 `WBM_judge_qwen3vl.py:169` 的 payload model 是 Qwen3-VL-235B-A22B-Thinking-FP8（是 VLM 模型），只是没发图片 |
| 阶段1 max_tokens | 6192 | ❌ 实际是 8192（`WBM_caption_qwen3v.py:126`） |
| 阶段3 max_tokens | 未提及 | 实际是 6192（`WBM_caption_glm45v.py:123`） |
| judge max_tokens | 未提及 | 阶段2: 7192，阶段4: 8196（不一致） |
| 置信度阈值 | ≥60 → 高质量 | ✅ 阶段2→3 用 `< 60`，阶段4→6 用 `> 60`（注意一个是 `<`，一个是 `>`，等于 60 的处理不同） |

### 8.4 COCO 输出

| 项 | REPRODUCTION_GUIDE.md 描述 | 源码实际情况 |
|---|---|---|
| image_id 格式 | `lot17115_16` → `wafer_00017115_016` | ✅ 一致（`tran2Json.py:53-61`） |
| 翻译模型 | 未指定 | 源码中 `tran2Json.py:83` model 仍是 Qwen3-VL-235B-A22B-Thinking-FP8（服务器 sed 改为 GLM-4.5V） |
| COCO 只含14条 | 未提及 | ❌ `tran2Json.py:245` 只筛选 `glm45vl_confidence > 60`，漏掉了阶段2的67条高置信度数据。指南未指出这个bug |
| 图片分辨率 | "364×364" | ✅ `tran2Json.py:199` 确认 |

### 8.5 训练

| 项 | REPRODUCTION_GUIDE.md 描述 | 源码实际情况 |
|---|---|---|
| Stage 1 Loss | CrossEntropy + InfoNCE | ✅ 确认（`finetune_visualglm.py:163`） |
| InfoNCE temperature | "0.1" | ✅ 确认（`contrastive_learning.py:7`） |
| InfoNCE 正样本 | "原图和旋转图" | ✅ 确认，`RandomRotation(-90, 90)`（`finetune_visualglm.py:148`） |
| Stage 2 LoRA rank | "16" | ✅ 确认（`finetune_visualglm.sh:12-14`） |
| LoRA alpha | "16" | ❌ 源码中 `lora_mixin.py:167` 默认 `lora_alpha=1`，不是 16。`finetune_visualglm.sh` 中也没有设 `--lora_alpha` |
| scaling | 未提及 | 实际 = alpha/r = 1/16 = 0.0625 |
| ViT LoRA 层范围 | 未提及 | `vit_layer_range = 0 13 26 38`（`finetune_visualglm.sh:13`）— 指定第 0/13/26/38 层 |
| ChatGLM LoRA 层范围 | 未提及 | `chatglm_layer_range = 0 19 27`（`finetune_visualglm.sh:15`） |
| lr | "0.001(SAT) 或 2e-4(PEFT)" | ✅ SAT 脚本是 0.001，QLoRA 脚本是 0.0001 |
| Stage 2 数据格式 | `{"image": "...", "conversations": [...]}` | ❌ 源码中实际格式是 `{"img": "...", "prompt": "...", "label": "..."}`（`finetune_visualglm.py:186-191`），不是 conversations 格式 |
| 序列构造 | `<img>...image_tokens...</img>问：...答：...` | ✅ 确认（`finetune_visualglm.py:188-190`），但实际是 `<img>` + 32个pad_token + `</img>`问：{prompt}\n答：{label} |
| 训练数据路径 | 未提及 | `finetune_data/dataset_trn.json`（Stage 2），`./fewshot-data/dataset.json`（QLoRA） |
| 模型路径 | 未提及 | `/ossfs/workspace/Defect-GLM-master/THUDM/visualglm-6b`（`finetune_visualglm.py:278`） |

### 8.6 检索评估

| 项 | REPRODUCTION_GUIDE.md 描述 | 源码实际情况 |
|---|---|---|
| 特征提取流程 | "ViT → QFormer/投影 → mean pooling → L2 归一化" | ✅ 确认（`inference_online.py:161-167`） |
| AP@K | "K=1,5,10,25,50" | ✅ 确认（`inference_online.py:429`） |
| Recall@K | "K=total_relevant (N), K=1.5×N" | ✅ 确认（`inference_online.py:484-485`） |
| 特征提取器 | "微调后的 Defect-GLM/Qwen3-VL (主方案)" | ✅ `main()` 中默认用 `BLIP2FeatureExtractor`（`inference_online.py:577`），DINOv3 被注释 |
| 输入尺寸 | "256×256" | ❌ BLIP2 用 `BlipImageEvalProcessor(224)`（`inference_online.py:581`），DINOv3 路径用 256（`inference_online.py:289`）但被注释 |
| from_pretrained | 未提及 | 默认 `/THUDM/visualglm-6b`（`inference_online.py:54`） |

### 8.7 指南完全缺失的内容

- 阶段3 的 `np.pad` 边缘填充（`WBM_caption_glm45v.py:74`）— 阶段1 没有这步
- `calculate_static_param` 的返回值 `-1` 和 `-2` 的特殊含义（无缺陷 / 滤波后无缺陷）
- 角度直方图的 bin 偏移（偏移 3 个 bin 对齐 12 点钟）
- `extract_final_output` 的优先级：先匹配 `<|begin_of_box|>`，再匹配 `</think>`
- Stage 1 `AdaptationDataset` 的数据格式（`{"img": path, "class": int}`）
- `LoraMixin` 替换的具体层：`attention.dense`（partition=1）、`attention.query_key_value`（partition=3, head_first）、`cross_attention.dense/query/key_value`
- ViT 侧 LoRA：除了 `glm_proj` 层，还在 ViT 内部加了 `LoraMixin`（`finetune_visualglm.py:28`）
- `lora_alpha` 默认值是 **1 不是 16**，导致 `scaling=0.0625`
- judge prompt 按缺陷类型有不同 guidance（9 种类型各有评估重点，`WBM_judge_qwen3vl.py:35-45`）
- 阶段6 翻译的 `max_tokens=3096` 和强制模板格式
- `data_pipeline.py` 和 `data_pipeline_2model.py` 的存在（早期单文件版本，指南未提及）
- `custom_unpickler` 处理 numpy 2.x → 1.x 兼容性的具体模块映射（`utils.py:540-599`）

### 8.8 指南中存在但源码无法确认的内容

| 项 | 指南描述 | 源码状态 |
|---|---|---|
| predictDefectType 的生成方式 | 未描述 | 源码中没有生成逻辑，pkl 文件是预先准备好的 |
| Stage 1 eva_args 的内容 | 未描述 | `finetune_visualglm.py:297` 调用 `start_adaptation(args.eva_args)`，但 `eva_args` 来自 SAT 框架的模型配置，源码中未显式定义 |
| load_save_WM811K.ipynb 的内容 | 未描述 | 项目中存在此文件但未走读 |
| WBM180.pkl 的构建脚本 | "每类抽20条" | 源码中没有构建脚本，pkl 文件是预先准备好的 |

---

## 9. 依赖包清单

### 9.1 数据管线依赖（`WBM_caption/requirements.txt`）

```text
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
Pillow>=9.0.0
opencv-python-headless>=4.10.0
matplotlib>=3.7.0
scikit-image>=0.21.0
scikit-learn>=1.3.0
aiofiles
aiohttp
aiocsv
requests
openai
uvloop
```

### 9.2 训练依赖（`requirements.txt`）

```text
SwissArmyTransformer==0.4.4
torch==2.5.1
torchvision==0.20.1
transformers==4.30.0
mdtex2html
gradio
wandb
accelerate
scikit-learn
deepspeed==0.16.0
```

### 9.3 不含 DeepSpeed 的训练依赖（`requirements_wo_ds.txt`）

```text
torch>1.10.0
torchvision
transformers>=4.27.1
mdtex2html
gradio
sentencepiece
tensorboardX
datasets
cpm_kernels
einops
```

### 9.4 版本注意

- `SwissArmyTransformer` 必须 `>=0.3.6`，推荐 `0.4.4`
- `transformers==4.30.0` 是精确版本，不要升级
- `torch==2.5.1` + `torchvision==0.20.1` 是配对版本
- `deepspeed==0.16.0` 精确版本
- Windows 下 `uvloop` 不可用，需删除或替换为 asyncio 默认事件循环
- `aiocsv` 是异步 CSV 写入库，`WBM_caption_glm45v.py` 中用到

---

## 10. 关键源码完整片段

以下为复现所需的核心源码，按数据流顺序排列。标注了文件名和行号。

### 10.1 `matrix2image_cached()` — 矩阵转 RGB 图片

文件：`WBM_caption/WBM_caption_qwen3v.py:35-70`

```python
def matrix2image_cached(wafer_matrix: np.ndarray, resolution: int, 
                       color_map: str = '黑绿红', add_ref: bool = False, 
                       add_filter: bool = False) -> Image.Image:
    pad_sp = 2
    wafer_matrix = np.pad(wafer_matrix, ((pad_sp, pad_sp), (pad_sp, pad_sp)))
    if add_filter:
        wafer_matrix = apply_filter(wafer_matrix)
    
    H, W = wafer_matrix.shape
    output_image = np.zeros((H, W, 3), dtype=np.uint8)
    color_map_dict = {0: [0, 0, 0], 1: [0, 255, 0], 2: [255, 0, 0]}
    
    for value, color in color_map_dict.items():
        mask = wafer_matrix == value
        if np.any(mask):
            output_image[mask] = color
    
    if resolution != H or resolution != W:
        scale_h = resolution / H
        scale_w = resolution / W
        output_image = zoom(output_image, (scale_h, scale_w, 1), order=0)
    
    if add_ref:
        try:
            scaled_matrix = zoom(wafer_matrix, (resolution/H, resolution/W), order=0) > 0
            contours = measure.find_contours(scaled_matrix, level=0.5)
            if contours:
                contour = max(contours, key=len)
                x_center, y_center, radius = fit_circle_least_squares(
                    contour[:, 1], contour[:, 0]
                )
                output_image, _ = addXYCircle(output_image, [x_center, y_center, radius])
        except Exception as e:
            print(f"添加参考线时出错: {e}")
    
    return Image.fromarray(output_image)
```

注意：

- `WBM_caption_qwen3v.py` 版本有 `pad_sp = 2`（边缘填充 2 像素）
- `WBM_caption_glm45v.py:34-67` 版本没有 `pad_sp`，但 `process_wafer()` 中在外部做了 `np.pad(waferMap, ((2,2),(2,2)))`（`WBM_caption_glm45v.py:74`）
- `tran2Json.py` 中 `pad_sp = 2`（从 `WBM_caption_qwen3v.py` import 的）

### 10.2 `apply_filter()` — 孤立点去噪

文件：`WBM_caption/utils.py:512-538`

```python
def apply_filter(matrix):
    result_matrix = np.copy(matrix)
    rows, cols = matrix.shape
    
    def has_adjacent_two(r, c):
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if matrix[nr, nc] == 2:
                        return True
        return False
    
    for r in range(rows):
        for c in range(cols):
            if matrix[r, c] == 2 and not has_adjacent_two(r, c):
                result_matrix[r, c] = 1
                
    return result_matrix
```

### 10.3 `multi_scale_cmean_filtering()` — 多尺度 C-Mean 去噪

文件：`WBM_caption/utils.py:197-250`

```python
def multi_scale_cmean_filtering(wafer_map: np.ndarray,
                                scale_thresholds: List[float] = [1.1, 1.2, 1.3],
                                scales: List[int] = [3, 5, 7],
                                voting_method: str = 'majority',
                                min_valid_neighbors: int = 1) -> np.ndarray:
    filtered_map = np.copy(wafer_map)
    defect_mask = (wafer_map == 2)

    if not np.any(defect_mask) or len(scale_thresholds) != len(scales):
        return filtered_map

    defect_coords = np.argwhere(defect_mask)
    
    min_valid_ratios = {
        scale: min_valid_neighbors / (scale * scale - 1) 
        for scale in scales
    }

    for i, j in defect_coords:
        scale_decisions = []

        for scale, threshold in zip(scales, scale_thresholds):
            neighborhood = get_neighborhood(wafer_map, i, j, scale)
            avg_value = calculate_mean_excluding_center(neighborhood)

            valid_ratio = np.sum(neighborhood != 0) / neighborhood.size
            
            if valid_ratio < min_valid_ratios.get(scale, 0):
                scale_decisions.append(False)
            else:
                scale_decisions.append(avg_value < threshold)

        if voting_method == 'majority':
            should_change = sum(scale_decisions) > len(scale_decisions) // 2
        elif voting_method == 'all':
            should_change = all(scale_decisions)
        elif voting_method == 'any':
            should_change = any(scale_decisions)
        else:
            raise ValueError("voting_method must be 'majority', 'all', or 'any'")

        if should_change:
            filtered_map[i, j] = 1

    return filtered_map
```

### 10.4 `calculate_static_param()` — 数学金标计算

文件：`WBM_caption/utils.py:357-381`

```python
def calculate_static_param(wafer_matrix):
    contours = measure.find_contours(wafer_matrix > 0, level=0.5)
    contour = max(contours, key=len)
    center_x, center_y, radius = fit_circle_least_squares(contour[:, 1], contour[:, 0])

    failue_ratio = np.mean(wafer_matrix == 2) / np.mean(wafer_matrix > 0)

    if failue_ratio == 0:
        return -1, None
    center = np.array([center_x, center_y])
    rows_loc, cols_loc = np.where(wafer_matrix == 2)
    points = np.column_stack((cols_loc, rows_loc))
    stats = compute_radial_tangential_stats(points, center, radius)
    stats['ratio'] = failue_ratio

    waferMatrixFilter = apply_filter(wafer_matrix)
    rows_loc, cols_loc = np.where(waferMatrixFilter == 2)
    failue_ratio = np.mean(waferMatrixFilter == 2) / np.mean(waferMatrixFilter > 0)
    if failue_ratio == 0:
        return -2, {'ratio': failue_ratio}
    points = np.column_stack((cols_loc, rows_loc))
    statsFilter = compute_radial_tangential_stats(points, center, radius)
    statsFilter['ratio'] = failue_ratio

    return stats
```

返回值说明：

- 正常：返回 `stats` 字典（含 `ratio, L1_center, L2_center, radial, angular, covariance`）
- 无缺陷（`failue_ratio == 0`）：返回 `(-1, None)`
- 滤波后无缺陷（全是随机噪声）：返回 `(-2, {'ratio': 0})`

注意：`statsFilter` 计算了但未返回，是代码遗留问题

### 10.5 `compute_radial_tangential_stats()` — 径向/角度统计

文件：`WBM_caption/utils.py:252-315`

```python
def compute_radial_tangential_stats(points, center, radi):
    rel = points - np.array(center)  # shape (N, 2)

    mass_center = np.mean(rel, axis=0)
    mass_center_l2 = np.sqrt(np.mean(rel**2, axis=0))

    rs = np.linalg.norm(rel, axis=1)
    thetas = np.arctan2(rel[:, 1], rel[:, 0])  # [-π, π]

    # === 径向统计 ===
    mean_r = np.mean(rs)
    std_r = np.std(rs)
    
    radial_hist, radial_bins = np.histogram(rs, bins=10, range=(0, np.max(rs)), density=False)
    bin_areas = np.pi * (radial_bins[1:]**2 - radial_bins[:-1]**2)
    radial_density = np.divide(radial_hist, bin_areas, out=np.zeros_like(radial_hist, dtype=float), where=bin_areas!=0)

    # === 角度统计 ===
    cos_mean = np.mean(np.cos(thetas))
    sin_mean = np.mean(np.sin(thetas))
    R = np.sqrt(cos_mean**2 + sin_mean**2)  # [0,1]
    mean_theta = 90 + np.rad2deg(np.arctan2(sin_mean, cos_mean))

    n_angle_bins = 12
    angle_hist, _ = np.histogram(thetas, bins=n_angle_bins, range=(-np.pi, np.pi), density=False)
    angle_hist = np.concatenate([angle_hist[3:], angle_hist[:3]], axis=0) / np.sum(angle_hist)
    angle_probs = angle_hist / np.sum(angle_hist + 1e-12)
    angle_entropy = entropy(angle_probs + 1e-12, base=2)

    # === 协方差分析 ===
    cov_matrix = np.cov(rel.T)
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    anisotropy_ratio = eigvals[0] / (eigvals[1] + 1e-12) if eigvals[1] > 0 else np.inf

    stats = {
        'L1_center': mass_center / radi,
        'L2_center': mass_center_l2 / radi,
        'radial': {
            'mean': mean_r,
            'std': std_r,
            'hist': (radial_density, radial_bins)
        },
        'angular': {
            'mean_direction': mean_theta,
            'resultant_length': R,
            'entropy': angle_entropy,
            'hist': (angle_hist, np.linspace(-np.pi, np.pi, n_angle_bins+1))
        },
        'covariance': {
            'matrix': cov_matrix,
            'eigenvalues': eigvals,
            'anisotropy_ratio': anisotropy_ratio
        }
    }
    return stats
```

### 10.6 `generate_param_description()` — 金标参数格式化

文件：`WBM_caption/WBM_judge_qwen3vl.py:113-155`

```python
def generate_param_description(defect_type: str, params: Dict) -> str:
    shape_descriptions = {
        'Loc': "团簇状分布",
        'Center': "团簇状分布（中心区域）",
        'Edge_Loc': "团簇状分布（边缘局部区域）",
        'Donut': "环状分布",
        'Edge_Ring': "环状分布（边缘环状）",
        'Scratch': "线形分布",
        'Random': "随机分布",
        'Near-full': "近全片分布",
        'none': "无明显缺陷模式"
    }
    
    shape = shape_descriptions.get(defect_type, f"{defect_type}分布")
    
    ratio_percent = f"{params['ratio']*100:.1f}%" if isinstance(params['ratio'], (int, float)) else str(params['ratio'])
    
    if isinstance(params['L1_center'], (tuple, list)) and len(params['L1_center']) >= 2:
        center_x, center_y = params['L1_center'][:2]
        center_desc = f"({center_x:.3f}, {center_y:.3f})"
    else:
        center_desc = str(params['L1_center'])
    
    radial_hist = params['radial']['hist'][0] if 'hist' in params['radial'] and params['radial']['hist'] else [0]*10
    radial_desc = "[" + ", ".join([f"{x:.3f}" for x in radial_hist]) + "]"
    
    angular_hist = params['angular']['hist'][0] if 'hist' in params['angular'] and params['angular']['hist'] else [0]*12
    angular_desc = "[" + ", ".join([f"{x:.3f}" for x in angular_hist]) + "]"
    
    return (
        f"缺陷芯片占比：{ratio_percent}\n"
        f"分布形状：{shape}\n"
        f"质心坐标（单位圆坐标系）：{center_desc}\n"
        f"径向分布（10个圆环，从中心到边缘）：{radial_desc}\n"
        f"角度分布（12个扇形，从0点方向顺时针）：{angular_desc}"
    )
```

### 10.7 `get_failure_type_label()` — 缺陷标签标准化

文件：`WBM_caption/utils.py:35-63`

```python
def get_failure_type_label(failure_type):
    if failure_type is None:
        return 'unknown'
    
    try:
        if isinstance(failure_type, np.ndarray):
            flattened = failure_type.flatten()
            valid = [x for x in flattened if not pd.isna(x)]
            if not valid:
                return 'unknown'
            return str(valid[0])
        elif isinstance(failure_type, list):
            non_empty_list = [item for item in failure_type if item is not None and str(item).strip()]
            if not non_empty_list:
                return 'unknown'
            label = str(non_empty_list[0])
        elif isinstance(failure_type, str):
            stripped_label = failure_type.strip()
            label = stripped_label if stripped_label else 'unknown'
        else:
            label = str(failure_type).strip()
            if not label:
                return 'unknown'
        normalized_label = label.replace('-', '_').strip()
        return normalized_label if normalized_label else 'unknown'
    except (ValueError, TypeError, IndexError, AttributeError) as e:
        return 'unknown'
```

### 10.8 `fit_circle_least_squares()` — 最小二乘拟合圆

文件：`WBM_caption/utils.py:66-79`

```python
def fit_circle_least_squares(x, y):
    A = np.column_stack([x, y, np.ones(len(x))])
    C = x**2 + y**2

    abc, _, _, _ = np.linalg.lstsq(A, C, rcond=None)
    
    a, b, c = abc
    center_x = a / 2
    center_y = b / 2
    radius = np.sqrt(c + center_x**2 + center_y**2)
    
    return int(np.around(center_x)), int(np.around(center_y)), int(np.around(radius))
```

### 10.9 `extract_final_output()` — VLM 输出提取

文件：`WBM_caption/utils.py:340-355`

```python
def extract_final_output(text: str, strict=False) -> str:
    box_match = re.search(r'<\|begin_of_box\|>\s*(.*?)\s*<\|end_of_box\|>', text, re.DOTALL)
    if box_match:
        return box_match.group(1).strip()

    marker = "</think>"
    if marker in text:
        return text.split(marker, 1)[1].strip()

    if strict:
        return "无法描述"
    return text.strip()
```

### 10.10 `image_to_base64()` — 图片转 base64

文件：`WBM_caption/utils.py:24-32`

```python
def image_to_base64(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    with io.BytesIO() as output:
        image.save(output, format='PNG')
        image_bytes = output.getvalue()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        return image_base64
```

### 10.11 `contrastive_learning.py` — 完整（36行）

文件：`contrastive_learning.py:1-36`

```python
import torch
import torch.nn as nn


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.loss = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, similarity_scores):
        batch_size = similarity_scores.size(0)

        # Apply temperature scaling
        similarity_scores /= self.temperature

        # Split similarity scores into video-to-text and text-to-video scores
        similarity_vt = similarity_scores
        similarity_tv = similarity_scores.t()

        # Construct labels and calculate loss
        labels = torch.arange(batch_size).to(similarity_scores.device)
        loss_vt = self.loss(similarity_vt, labels)
        loss_tv = self.loss(similarity_tv, labels)
        return (loss_vt + loss_tv)/2


def cosine_sim(x, y):
    inner_prod = x.mm(y.t())
    im_norm = torch.sqrt((x ** 2).sum(1).view(-1, 1) + 1e-18)
    s_norm = torch.sqrt((y ** 2).sum(1).view(1, -1) + 1e-18)
    sim = inner_prod / (im_norm * s_norm)
    return sim
```

### 10.12 BLIP2 架构 — 完整

文件：`model/blip2.py:1-126`

```python
import torch
import torch.nn as nn

from sat.model import ViTModel, BaseModel
from sat.model import BaseMixin
from sat import AutoModel
from copy import deepcopy
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from sat.model.finetune.lora2 import LoraMixin

class LNFinalyMixin(BaseMixin):
    def __init__(self, hidden_size):
        super().__init__()
        self.ln_vision = nn.LayerNorm(hidden_size)

    def final_forward(self, logits, **kw_args):
        return self.ln_vision(logits)

class ClsMixin(BaseMixin):
    def __init__(self, hidden_size, num_classes):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, num_classes)

    def final_forward(self, logits, **kw_args):
        representation = logits[:, 0]
        logits = self.classifier(representation)
        return logits, representation

class EVAViT(ViTModel):
    def __init__(self, args, transformer=None, parallel_output=True, **kwargs):
        super().__init__(args, transformer=transformer, parallel_output=parallel_output, **kwargs)
        self.del_mixin("cls")
        self.add_mixin("cls", LNFinalyMixin(args.hidden_size))
        
    def forward(self, image):
        batch_size = image.size(0)
        input_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=image.device)
        attention_mask = torch.tensor([[1.]], dtype=image.dtype, device=image.device)
        return super().forward(input_ids=input_ids, position_ids=None, attention_mask=attention_mask, image=image)
    
    def start_adaptation(self, args):
        self.del_mixin("cls")
        self.add_mixin("cls", ClsMixin(args['hidden_size'], 36))
    
    def stop_adaptation(self, args):
        self.del_mixin("cls")
        self.add_mixin("cls", LNFinalyMixin(args['hidden_size']))

class QFormer(BaseModel):
    def __init__(self, args, transformer=None, parallel_output=True, **kwargs):
        super().__init__(args, transformer=transformer, parallel_output=parallel_output, activation_func=nn.functional.gelu, **kwargs)
        self.transformer.position_embeddings = None
    
    def final_forward(self, logits, **kw_args):
        return logits

    def position_embedding_forward(self, position_ids, **kw_args):
        return None
    
    def forward(self, encoder_outputs):
        batch_size = encoder_outputs.size(0)
        input_ids = torch.arange(32, dtype=torch.long, device=encoder_outputs.device).unsqueeze(0).expand(batch_size, -1)
        attention_mask = torch.tensor([[1.]], dtype=encoder_outputs.dtype, device=encoder_outputs.device)
        cross_attention_mask = torch.tensor([[1.]], dtype=encoder_outputs.dtype, device=encoder_outputs.device)
        return super().forward(input_ids=input_ids, position_ids=None, attention_mask=attention_mask, encoder_outputs=encoder_outputs, cross_attention_mask=cross_attention_mask)


class BLIP2(torch.nn.Module):
    def __init__(self, eva_args, qformer_args, vit=None, qformer=None, **kwargs):
        super().__init__()
        if vit is not None:
            self.vit = vit
        else:
            self.vit = EVAViT(EVAViT.get_args(**eva_args))
        if qformer is not None:
            self.qformer = qformer
        else:
            self.qformer = QFormer(QFormer.get_args(**qformer_args))
        
        self.glm_proj = nn.Linear(768, 4096).to(self.qformer.parameters().__next__().device).to(self.qformer.parameters().__next__().dtype)

    def forward(self, image, **kwargs):
        enc = self.vit(image)[0]
        out = self.qformer(enc)[0]
        return self.glm_proj(out)


class BlipImageBaseProcessor():
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)
        self.normalize = transforms.Normalize(mean, std)

class BlipImageEvalProcessor(BlipImageBaseProcessor):
    def __init__(self, image_size=384, mean=None, std=None):
        super().__init__(mean=mean, std=std)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            self.normalize,
        ])
        self.augmentation_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            self.normalize,
            transforms.RandomRotation(degrees=(-90, 90)),
        ])

    def __call__(self, item, augmentation=False):
        if augmentation:
            return self.augmentation_transform(item)
        return self.transform(item)
```

关键架构细节：

- `EVAViT.start_adaptation()`: 替换 `LNFinalyMixin` 为 `ClsMixin(hidden_size, 36)` — **36 个类别（不是 9 类，可能包含细分子类）**
- `ClsMixin.final_forward()`: 返回 `(logits, representation)`，`representation = logits[:, 0]`（取 CLS token）
- `QFormer`: 32 个 query token（`input_ids = torch.arange(32)`），无位置嵌入
- `BLIP2.forward()`: `vit(image) → qformer(enc) → glm_proj(out)`
- `glm_proj`: `nn.Linear(768, 4096)` — 将 QFormer 输出投影到 ChatGLM 嵌入空间
- `BlipImageEvalProcessor`:
  - 默认 `image_size=384`，但训练代码中用 `BlipImageEvalProcessor(224)`
  - 归一化均值: `(0.48145466, 0.4578275, 0.40821073)`
  - 归一化标准差: `(0.26862954, 0.26130258, 0.27577711)`
  - 插值方式: BICUBIC
  - `augmentation=True` 时额外加 `RandomRotation(-90, 90)`

### 10.13 VisualGLMModel — 完整

文件：`model/visualglm.py:1-43`

```python
import torch
from sat.model.official import ChatGLMModel
from sat.model.base_model import BaseMixin
from copy import deepcopy
import json
from .blip2 import BLIP2

from sat.resources.urls import MODEL_URLS
MODEL_URLS['visualglm-6b'] = 'r2://visualglm-6b.zip'

class ImageMixin(BaseMixin):
    def __init__(self, args):
        super().__init__()
        self.args = deepcopy(args)
        if hasattr(args, 'model_parallel_size'):
            args.eva_args['model_parallel_size'] = args.model_parallel_size
            args.qformer_args['model_parallel_size'] = args.model_parallel_size
        self.model = BLIP2(args.eva_args, args.qformer_args)

    def word_embedding_forward(self, input_ids, output_cross_layer, **kw_args):
        if kw_args["pre_image"] > input_ids.shape[1] or kw_args.get("image", None) is None:
            return self.transformer.word_embeddings(input_ids)
        image_emb = self.model(**kw_args)
        pre_id, pads, post_id = torch.tensor_split(input_ids, [kw_args["pre_image"], kw_args["pre_image"]+self.args.image_length], dim=1)
        pre_txt_emb = self.transformer.word_embeddings(pre_id)
        post_txt_emb = self.transformer.word_embeddings(post_id)
        return torch.cat([pre_txt_emb, image_emb, post_txt_emb], dim=1)

class VisualGLMModel(ChatGLMModel):
    def __init__(self, args, transformer=None, **kwargs):
        super().__init__(args, transformer=transformer, **kwargs)
        self.image_length = args.image_length
        self.add_mixin("eva", ImageMixin(args))

    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('VisualGLM', 'VisualGLM Configurations')
        group.add_argument('--image_length', type=int, default=32)
        group.add_argument('--eva_args', type=json.loads, default={})
        group.add_argument('--qformer_args', type=json.loads, default={})
        return super().add_model_specific_args(parser)
```

关键细节：

- `image_length=32`：图像嵌入占 32 个 token 位置
- 图像嵌入插入方式：`<img>` token 之后，覆盖 32 个 pad token
- `ImageMixin.word_embedding_forward()`: 将 pad token 替换为图像嵌入
- `ChatGLMModel` 是 SAT 框架的官方实现

### 10.14 Stage 1 适配训练 forward

文件：`finetune_visualglm.py:141-168`

```python
def forward_step_adaptation(data_iterator, model, args, timers):
    timers('batch generator').start()
    labels, images = get_batch_adaptation(data_iterator, args, timers)
    augmentation = transforms.RandomRotation(degrees=(-90, 90))
    aug_images = augmentation(images)
    timers('batch generator').stop()
    
    output = model.get_mixin("eva").model.vit(images)[0]
    logits, representation = output[0], output[1]
    aug_output = model.get_mixin("eva").model.vit(aug_images)[0]
    aug_representation = aug_output[1]
    sim_score = cosine_sim(representation, aug_representation)
    dtype = logits.dtype
    lm_logits = logits.to(torch.float32)
    
    loss_infonce = InfoNCELoss() 
    loss_ce = CrossEntropyLoss()
    loss = loss_ce(logits, labels) + loss_infonce(sim_score)
    lm_logits = lm_logits.to(dtype)
    loss = loss.to(dtype)
    return loss, {'loss': loss}
```

注意：

- `vit(images)[0]` 返回 `(logits, representation)` — 来自 `ClsMixin.final_forward()`
- `representation` 是 CLS token 特征（`logits[:, 0]`）
- `InfoNCELoss()` 用默认 `temperature=0.1`
- `loss = CrossEntropy(logits, labels) + InfoNCE(sim_score)`
- `sim_score = cosine_sim(representation, aug_representation)` — 原图和旋转图的余弦相似度

### 10.15 Stage 2 微调 forward

文件：`finetune_visualglm.py:117-138`

```python
def forward_step(data_iterator, model, args, timers):
    timers('batch generator').start()
    tokens, labels, image, pre_image = get_batch(data_iterator, args, timers)
    timers('batch generator').stop()
    logits = model(input_ids=tokens, image=image, pre_image=pre_image)[0]
    dtype = logits.dtype
    lm_logits = logits.to(torch.float32)

    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    lm_logits = lm_logits.to(dtype)
    loss = loss.to(dtype)
    return loss, {'loss': loss}
```

### 10.16 FewShotDataset — Stage 2 数据加载

文件：`finetune_visualglm.py:176-222`

```python
class FewShotDataset(Dataset):
    def __init__(self, path, processor, tokenizer, args):
        max_seq_length = args.max_source_length + args.max_target_length
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.images = []
        self.input_ids = []
        self.labels = []
        self.processor = processor
        for item in data:
            image = Image.open(item['img']).convert('RGB')
            input0 = tokenizer.encode("<img>", add_special_tokens=False)
            input1 = [tokenizer.pad_token_id] * args.image_length
            input2 = tokenizer.encode("</img>问：" + item['prompt'] + "\n答：", add_special_tokens=False)
            a_ids = sum([input0, input1, input2], [])
            b_ids = tokenizer.encode(text=item['label'], add_special_tokens=False)
            if len(a_ids) > args.max_source_length - 1:
                a_ids = a_ids[: args.max_source_length - 1]
            if len(b_ids) > args.max_target_length - 2:
                b_ids = b_ids[: args.max_target_length - 2]
            pre_image = len(input0)
            input_ids = tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)

            context_length = input_ids.index(tokenizer.bos_token_id)
            mask_position = context_length - 1
            labels = [-100] * context_length + input_ids[mask_position + 1:]

            pad_len = max_seq_length - len(input_ids)
            input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
            labels = labels + [tokenizer.pad_token_id] * pad_len
            if args.ignore_pad_token_for_loss:
                labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]
            self.images.append(image)
            self.input_ids.append(input_ids)
            self.labels.append(labels)
        self.pre_image = pre_image

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {
            "image": self.processor(self.images[idx]),
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx],
            "pre_image": self.pre_image
        }
```

Stage 2 训练数据 JSON 格式：

```json
[{"img": "path/to/wafer.jpg", "prompt": "What is the defect pattern?", "label": "A dense clustered..."}]
```

序列构造：

`<img>` + 32个pad_token + `</img>`问：{prompt}\n答：{label}

`pre_image = len("<img>")` — 图像嵌入插入位置

labels: 输入部分为 -100，只对答案部分计算 loss

### 10.17 AdaptationDataset — Stage 1 数据加载

文件：`finetune_visualglm.py:225-249`

```python
class AdaptationDataset(Dataset):
    def __init__(self, path, processor):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.images = []
        self.labels = []
        self.processor = processor
        for item in data:
            image = Image.open(item['img']).convert('RGB')
            label = item['class']
            self.images.append(image)
            self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {
            "image": self.processor(self.images[idx]),
            "labels": self.labels[idx],
        }
```

Stage 1 训练数据 JSON 格式：

```json
[{"img": "path/to/wafer.jpg", "class": 0}]
```

`class` 是整数标签

### 10.18 BLIP2FeatureExtractor — 检索特征提取

文件：`inference_online.py:107-167`

```python
class BLIP2FeatureExtractor(BaseFeatureExtractor):
    def __init__(self, model_name="Salesforce/blip2-opt-2.7b", device="cuda"):
        super().__init__(device)
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.load_model()
        self.dtype = torch.float16

    def load_model(self):
        model_config = BLIP2WaferModelConfig()
        base_args = model_config.get_args()
        base_args_dict = vars(base_args)
        new_params = {
            "fp16": True,
            "skip_init": True,
            "use_gpu_initialization": True if (torch.cuda.is_available() and base_args.quant is None) else False,
            "device": 'cuda' if (torch.cuda.is_available() and base_args.quant is None) else 'cpu',
        }
        merged_args_dict = {**base_args_dict, **new_params}
        merged_args = argparse.Namespace(**merged_args_dict) 
        model, model_args = AutoModel.from_pretrained(
            self.model_name,
            args=merged_args)
        model = model.eval()
        self.model = model

    @torch.no_grad()
    def __call__(self, images: torch.Tensor) -> np.ndarray:
        images = images.to(self.device).to(self.dtype)
        if hasattr(self.model, 'vit') and hasattr(self.model.vit, 'parameters'):
            model_dtype = next(self.model.vit.parameters()).dtype
            images = images.to(model_dtype)
        
        vit_features = self.model.get_mixin("eva").model.vit(images)[0]
        query_feature = self.model.get_mixin("eva").model.qformer(vit_features)[0]
        image_features = query_feature.mean(dim=1, keepdim=True)
        image_features = image_features / (image_features.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        
        return image_features.cpu().numpy()
```

特征提取路径：ViT → QFormer → `mean(dim=1)` → L2 normalize

`query_feature` shape: `(B, 32, D)` — 32 个 query token

`mean(dim=1)` → `(B, 1, D)` → squeeze → `(B, D)`

### 10.19 AP@K 和 Recall@K 计算

文件：`inference_online.py:309-330`

```python
def compute_ap_at_k(labels_in_topk, total_relevant, k=50):
    if total_relevant == 0:
        return 0.0
    k_use = min(k, len(labels_in_topk))
    if k_use == 0:
        return 0.0
    ap_sum = 0.0
    num_relevant = 0
    for i in range(k_use):
        if labels_in_topk[i] == 1:
            num_relevant += 1
            precision_at_i = num_relevant / (i + 1)
            ap_sum += precision_at_i
    normalization = min(total_relevant, k_use)
    return ap_sum / normalization


def compute_recall_at_k_single(labels_in_topk, total_relevant, k):
    if total_relevant == 0:
        return 0.0
    k_use = min(k, len(labels_in_topk))
    return sum(labels_in_topk[:k_use]) / total_relevant
```

### 10.20 阶段5 LLM-as-judge 三维打分

文件：`WBM_caption/judge_score.py:50-129`

Prompt：

```text
您是一位专家评估员，专门评估晶圆BIN图中缺陷分布描述的一致性。
由缺陷芯片组成的几何对象位于以晶圆原点为中心、半径为 R 的圆形晶圆内。
比较以下两个文本描述，它们指的是晶圆BIN图上*同一个*缺陷分布对象：

金标描述: "{golden}"
候选描述: "{candidate}"

请从以下三个方面进行评估：
1. 形状（例如，满圆分布 vs 缺陷大面积覆盖晶圆 → 完全一致; 圆形 vs 圆弧形 → 完全一致；正方形 vs 矩形 → 接近一致; 线性缺陷簇 vs 划痕 → 完全一致；团簇状缺陷分布）
2. 大小（例如，长轴约R vs 大小约R → 完全一致；0.5R vs 0.6R → 接近一致）
3. 位置（例如，左上角 vs 10点钟方向 → 完全一致；靠近边缘 vs 紧贴边缘 → 接近一致）

对于每个方面，请按以下标准评分：
- 如果完全相同或明显等价，则得 2 分；
- 如果语义接近但不完全相同，则得 1 分；
- 如果完全不相关或相互矛盾，则得 0 分。

仅返回一个 JSON 对象，该对象包含key: "shape" "size" "position"，每个的值可以是 0、1 或 2。
示例输出：
{"shape": 2, "size": 1, "position": 0}
```

模型：`test-Qwen3-235B-GPTQ-Int8`（**纯文本模型**）

输入：`golden.txt`（金标描述 CSV，注意字段名有空格 `描述`）+ 预测结果 CSV

输出：`evaluation_{pred_name}.csv`，增加 `shape_score, size_score, position_score, total_score` 列

---

## 11. 完整 Prompt 原文

### 11.1 Caption 生成 System Prompt（阶段1和阶段3共用）

文件：`WBM_caption_qwen3v.py:84-104` / `WBM_caption_glm45v.py:82-102`

```text
你是半导体制造领域的晶圆图样分析专家（Wafer Map Pattern Recognition Expert）。你的任务是将输入的晶圆BIN图（Wafer Bin Map）转化为结构化、标准化的自然语言描述，用于多模态模型的对比学习（Contrastive Learning）

### 图像特征定义
- 背景：{color_type[0]}色背景。
- 合格芯片（Good Die）：{color_type[1]}色像素点，通常占据大部分区域。
- 缺陷芯片（Defect Die）：{color_type[2]}色像素点，是我们关注的重点。
- 晶圆区域：由{color_type[1]}色（合格）和{color_type[2]}色（缺陷）像素组成的圆形区域，背景通常为{color_type[0]}色。请以该圆形区域的几何中心作为坐标原点。

### 核心分析逻辑（Chain of Thought）
在生成最终描述前，请在内心执行以下步骤：
1.  去噪与预判：除出现大量随机分布的"{color_type[2]}点"以外，忽略少量的"{color_type[2]}点"（可忽略的随机缺陷），仅关注具有聚集性或几何规律的轮廓和图案（系统性缺陷）。如果图像模糊不清，输出"无法识别"。
2.  全局扫描：寻找覆盖整个晶圆的特征（如：满屏散点、同心圆环）。
3.  局部聚类：识别孤立的缺陷簇。分析其拓扑结构（是实心的团块，还是细长的线条？）。
4.  空间定位：基于有效芯片（{color_type[1]}色）和缺陷芯片（{color_type[2]}色）构成的整体圆形轮廓，推断晶圆的几何中心。默认图像正上方为12点钟方向，建立虚拟坐标系，以此确定缺陷簇的质心位置（如：第一象限、3点钟方向）及径向分布（中心区域、中间环带、边缘区域）。
5.  尺度估算：以晶圆半径（R）为基准单位，估算缺陷簇的长轴、直径或厚度。

### 严格术语库（Controlled Vocabulary）
请仅使用以下几何与方位术语，禁止创造新词：
- 形状：环形 (Ring), 局部环形 (Partial Ring), 团簇状 (Cluster), 线状 (Line), 弧形 (Arc), 射线状 (Radial), 满圆 (Full Circle), 边缘 (Edge), 划痕状 (Scratch), 扇形（Circular Sector），随机点 (Random Dots)。
- 方位：中心 (Center), 边缘 (Edge), [1-12]点钟方向 ([1-12] o'clock), 上半部/下半部 (Upper/Lower half), 左侧/右侧 (Left/Right side)。
- 修饰：致密的 (Dense), 稀疏的 (Sparse), 连续的 (Continuous), 断续的 (Intermittent)。

### 输出约束
- 零幻觉：如果看不清或不确定，不要强行描述。
- 格式化：输出必须是客观的陈述句，不包含"我认为"、"看起来像"等主观词汇。
```

`{color_type}` 传入的是中文字符串如 `'黑绿红'`，取字符 `[0]='黑'`, `[1]='绿'`, `[2]='红'`

### 11.2 Caption 生成 User Prompt（阶段1和阶段3共用）

文件：`WBM_caption_qwen3v.py:107-120` / `WBM_caption_glm45v.py:106-118`

```text
请分析图中{color_type[2]}色缺陷芯片的分布模式。忽略随机散落少量的噪点，仅描述显著的系统性缺陷。
请生成一段紧凑的文本描述，严格遵守以下规则：
1.  排序：按缺陷区域的视觉显著性（面积大小或密度）从大到小排序。
2.  句式模板：
    "一个[修饰]的[几何形状]缺陷簇，位于[具体方位]，尺寸约为[基于晶圆半径R的比例]"
3.  连接符：不同模式之间用分号"；"分隔，最后以句号结尾。
4.  禁止事项：不要使用列表、换行符或编号。不要输出"分析步骤"或"思考过程"，直接输出最终标签。

参考示例（Positive Examples）：
- 示例1：一个致密的环形缺陷簇，位于晶圆边缘，宽度约为0.1R；一个线状缺陷簇，横跨晶圆中心，长度约为1.8R。
- 示例2：一个团簇状缺陷簇，位于3点钟方向边缘，直径约为0.3R。
- 示例3：满圆分布的稀疏点状缺陷，覆盖整个晶圆区域。

请开始描述：
```

### 11.3 Judge 事实校验完整 Prompt

文件：`WBM_judge_qwen3vl.py:34-87` / `WBM_judge_glm45v.py:30-83`

```text
您是一位资深半导体晶圆缺陷分析专家。您的任务不是评判文本描述的完美程度，而是扮演一个"事实校验员"。
请严格依据数学统计计算得出的金标参数，判断候选的文本描述中是否存在可以被证明是错误或与关键数学事实直接矛盾的陈述。只要文本描述没有出现可被证伪的错误，即可认为其基本准确。

## 背景信息
- 缺陷类型：{defect_type}
- 评估重点：{type_guidance}
- 晶圆坐标系：以晶圆中心为原点，半径为1的单位圆

## 待评估的描述
1. **金标参数（数学事实，判断基准）**：
{param_desc}
2. **候选描述（待评估的文本表达）**：
{predict_description}

## 评估规则与逻辑
1.核心原则：候选描述不必完全、精确地复述所有数学参数。它可能包含模糊、简化或定性的表达，这都是允许的。如果候选描述无有实质意义的信息，或者候选描述为空，则置信度为0。
2.触发"不准确"的条件：仅当候选描述中的明确陈述与金标参数中的核心数学事实发生直接冲突时，才判定其存在错误。
示例冲突：
金标：质心位于 (0.2, 0.3) → 文本："缺陷集中分布在左上象限"（质心在第一象限，与左上象限矛盾）。
金标：空间模式：无显著环状特征 → 文本："缺陷呈现明显的环形分布"（直接矛盾）。
金标：边缘密度 > 中心密度 → 文本："从中心到边缘缺陷逐渐减少"（梯度方向描述相反）。
3.允许的宽容情况：
文本未提及某些次要特征。
文本使用了"较为集中"、"相对稀疏"等定性词汇，即使没有精确的数值支撑。
文本对区域（如"右上部分"、"左侧"）的描述与精确坐标在空间指向上大体一致，即使边界不精确。

## 输出要求
请严格按照以下JSON格式输出，不要包含任何其他文字：
{"置信度": 0-100的整数}

## 注意事项
1. 金标描述中的数值是精确计算值，候选描述可能是定性描述
2. 允许合理的语言表达差异，但核心事实必须一致
3. 如果候选描述模糊但方向正确，可适当扣分
4. 如果候选描述包含金标未提及但合理的内容，不扣分
5. 如果候选描述无有效内容，给零分
```

`{type_guidance}` 按缺陷类型从以下字典取：

```text
Loc: "重点关注团簇的集中程度、位置分布（中心/边缘/局部区域）是否匹配"

Center: "重点关注中心区域的聚集特征、对称性、扩散范围是否匹配"

Edge_Loc: "重点关注边缘局部区域的聚集特征、与晶圆边缘的距离关系是否匹配"

Donut: "重点关注环状结构的完整性、宽度、内外径比、环中心位置是否匹配"

Edge_Ring: "重点关注边缘环状的连续性、宽度、与晶圆边缘的距离是否匹配"

Scratch: "重点关注线形的方向性、长度、宽度、弯曲程度、位置是否匹配"

Near-full: "重点关注覆盖范围、均匀性、缺失区域特征是否匹配"

Random: "重点关注随机分布的程度、密度均匀性是否匹配"

none: "重点关注是否有明显缺陷模式或是否为正常分布"
```

### 11.4 翻译 Prompt（阶段6）

文件：`tran2Json.py:64-80`

```text
你是一名专业的半导体晶圆制造工程师，精通Wafer BIN Map缺陷分析。你的任务是将中文的缺陷形态描述专业、准确、一致地翻译成英文，用于技术报告或国际交流。
请遵循以下要求：
使用标准的半导体工程术语；
保持描述的客观性与技术准确性；
如遇中文特有表述或模糊描述，根据行业常识进行合理意译。
R表示wafer半径，在翻译时进行补充。
句子结构强制模板：
标准模板： [密度] [形态] defect pattern [位置], with a [尺寸参数] of approximately [数值].
示例： "一个致密的团簇状缺陷簇，位于中心，直径约为0.4R。" -> A dense clustered defect pattern located at the wafer center, with a diameter of approximately 0.4 times the wafer radius.
输出格式指令：
严格按输入句子的顺序和数量输出。
每个句子的翻译单独成行，用";"分隔。
禁止添加任何注释。
现在，请翻译以下中文缺陷描述：{chinCaption}
```

---

## 12. API 配置和服务器路径

> ⚠️ **本节已脱敏**。原文含真实 API Key、工号与内网端点。按 `AGENTS.md` §2 / §17，这些不得写入仓库、日志或任何提交。真实凭证只从服务器 `secrets/api.env`（权限 600）读取。**不要回填到本文件。**

### 12.1 API 配置

| 配置项 | 原始值（源码中） | 服务器 sed 替换后 |
|---|---|---|
| GLM-4.5V endpoint | `[已脱敏：内网端点]` | 不变 |
| GLM-4.5V model name | GLM-4.5V-fp8 | 不变 |
| GLM-4.5V api_key | `[已脱敏：API Key]` | 不变 |
| Qwen3-VL endpoint | `[已脱敏：内网端点]` | `[已脱敏：内网端点]` |
| Qwen3-VL model name | Qwen3-VL-235B-A22B-Thinking-FP8 | GLM-4.5V-fp8 |
| Qwen3-VL api_key | `[已脱敏：API Key]` | `[已脱敏：API Key]` |
| Qwen3-VL header Bearer | `[已脱敏]` | 不变（未替换） |

**API Key 格式**：`模型名_RT` + `base64(工号&模型名&有效期)`

（原文档在此处还写出了两个 key 解出的具体工号与有效期。**已一并删除**，理由同上：工号属于禁止写入的凭证信息。）

### 12.2 各阶段 API 调用参数

| 阶段 | 文件 | model (原始) | max_tokens | temperature | stream |
|---|---|---|---|---|---|
| 1 caption | `WBM_caption_qwen3v.py:123` | Qwen3-VL-235B-A22B-Thinking-FP8 | 8192 | 未设 | False |
| 2 judge | `WBM_judge_qwen3vl.py:168-172` | Qwen3-VL-235B-A22B-Thinking-FP8 | 7192 | 0 | - |
| 3 caption | `WBM_caption_glm45v.py:121-124` | Qwen3-VL-235B-A22B-Thinking-FP8 | 6192 | 未设 | False |
| 4 judge | `WBM_judge_glm45v.py:164-168` | Qwen3-VL-235B-A22B-Thinking-FP8 | 8196 | 0 | - |
| 6 翻译 | `tran2Json.py:82-86` | Qwen3-VL-235B-A22B-Thinking-FP8 | 3096 | 未设 | False |
| 5 打分 | `judge_score.py:79-83` | Qwen3-VL-235B-A22B-Thinking-FP8 | 8196 | 未设 | False |

### 12.3 服务器关键路径

```text
# 项目根目录
/ossfs/workspace/waferSearch-feature-defectGLM-c6ef014ba1c34c9eb7d937befca92144f91bfebe/waferSearch-feature-defectGLM-c6ef014ba1c34c9eb7d937befca92144f91bfebe/

# 数据文件
/ossfs/workspace/wbm_caption_code/WBM180.pkl
/ossfs/workspace/wbm_caption_code/WBM180_converted.pkl
/ossfs/workspace/archive/LSWMD_wLabel_converted.pkl

# Checkpoint 输出（源码原始路径，服务器 sed 替换后）
# 原始: /ossfs/workspace/Defect-GLM-master/WBM_caption/qwen3vl_checkpoints/
# 替换: .../waferSearch-.../checkpoints/qwen3vl_checkpoints/

# COCO 输出（源码原始路径，服务器 sed 替换后）
# 原始: /ossfs/workspace/LAVIS-main/coco/wbm20251204/
# 替换: .../waferSearch-.../output/coco/wbm20251204/

# 模型权重
/ossfs/workspace/Defect-GLM-master/THUDM/visualglm-6b
```

### 12.4 并发架构模式

所有阶段共用同一套异步架构：

- `asyncio.Queue(maxsize=concurrency * 2)` — 限制队列大小
- N 个 worker 协程（`asyncio.create_task`）
- 每条结果即时追加写入 CSV（`aiofiles.open(mode='a')`）
- 启动时读取已有 CSV 的 `idx` 列，跳过已处理（断点续传）
- 每次启动先备份当前 `latest.csv` 为 `checkpoint_{timestamp}.csv`
- 失败时写 ERROR 行，不阻塞后续

各阶段并发数：

阶段1：默认 64（`WBM_caption_qwen3v.py:283`）

阶段2：默认 64（`WBM_judge_qwen3vl.py:314`）

阶段3：默认 64（`WBM_caption_glm45v.py:296`）

阶段4：默认 64（`WBM_judge_glm45v.py:320`）

阶段6：默认 64（`tran2Json.py:318`）

---

## 13. 数据实际验证结果

### 13.1 WBM180.pkl 实际验证

运行 `wbm_caption_code/show_all_types.py` 的结果：

```text
列名: ['waferMap', 'dieSize', 'lotName', 'waferIndex', 'trianTestLabel', 'failureType']
行数: 180
failureType 分布（每类 20 条）:
  [['Donut']]        20
  [['Near-full']]    20
  [['Loc']]          20
  [['Edge-Ring']]    20
  [['Scratch']]      20
  [['none']]         20
  [['Random']]       20
  [['Edge-Loc']]     20
  [['Center']]       20
```

标签格式确认是双层嵌套列表 `[['Donut']]`

`trianTestLabel` 拼写确认（不是 `trainTestLabel`）

每类正好 20 条，共 9 类 180 条

`waferMap` 尺寸不固定（约 20-60 行 × 20-60 列）

### 13.2 waferMap 矩阵值确认

```python
# 矩阵值只有 0, 1, 2
# 0 = 背景（晶圆外区域，黑色 [0,0,0]）
# 1 = 合格 die（绿色 [0,255,0]）
# 2 = 缺陷 die（红色 [255,0,0]）
```

### 13.3 各阶段产出 CSV 列名

| 阶段 | 文件 | 列名 |
|---|---|---|
| 1 | `qwen3vl_checkpoints/latest.csv` | idx, defectType, predictDefectType, qwen3vl_result, qwen3vl_think |
| 2 | `judge_qwen3vl_checkpoints/latest.csv` | +qwen3vl_confidence |
| 3 | `glm45v_checkpoints/latest.csv` | +glm45vl_result, glm45vl_think |
| 4 | `judge_glm45v_checkpoints/latest.csv` | +glm45vl_confidence |
| 6 | `wbm20251204.json` | caption, image, image_id, defect_class |

### 13.4 180 条数据的实际产出

```text
阶段1: 160 条缺陷晶圆 → caption
阶段2: 160 条 → 校验 → 67 条高置信度(≥60) + 93 条低置信度(<60)
阶段3: 93 条低置信度 → GLM 重新生成 caption
阶段4: 93 条 → 校验 → 14 条高置信度(>60)
阶段6: 14 条 → 翻译 + COCO JSON + jpg

总计 COCO 输出: 14 条（漏掉了阶段2的 67 条高置信度数据，是 tran2Json.py 的 bug）
```

---

## 14. 已知 Bug 和代码问题汇总

| # | 文件 | 行号 | 问题 | 严重程度 | 影响 |
|---|---|---|---|---|---|
| 1 | `utils.py` | 120 | `addXYCircle` 引用未定义的 `axis_alpha`，`cv2.addWeighted` 会 NameError | 高 | 仅 `add_ref=True` 时触发，管线中 `add_ref=False` |
| 2 | `utils.py` | 430 | `calculate_geo_param` 硬编码 `R=26.0`（假设 52×52） | 高 | WBM180 尺寸不固定，会算错；但管线未调用此函数 |
| 3 | `utils.py` | 470 | `C_cover` 硬编码 `52*52` | 高 | 同上 |
| 4 | `utils.py` | 486-492 | Moran's I 遍历 `range(52)` | 高 | 同上 |
| 5 | `tran2Json.py` | 245 | 只筛选 `glm45vl_confidence > 60`，漏掉阶段2的 67 条高置信度数据 | 高 | COCO 输出只有 14 条而非 81 条 |
| 6 | `utils.py` | 378-381 | `calculate_static_param` 算了 `statsFilter` 但没返回 | 低 | 浪费计算，不影响结果 |
| 7 | 多文件 | - | API header 仍写着旧服务的标识，但 model 名已 sed 替换 | 低 | 服务器上不影响功能 |
| 8 | `WBM_caption_glm45v.py` | 121 | payload model 仍为 Qwen3-VL-235B-A22B-Thinking-FP8 | 低 | 服务器上已 sed 替换 |
| 9 | `WBM_judge_qwen3vl.py` | 169 | judge 的 model 名是 VLM 模型，但实际只发纯文本 | 低 | 不影响功能，VLM 也能处理纯文本 |
| 10 | `tran2Json.py` | 17 | `client_qwen` 用的是 Qwen3 专用的内网 endpoint（该服务已 502） | 高 | 服务器上已 sed 替换为 GLM endpoint |
| 11 | `calculate_static_param` | - | 返回值 `-1`/`-2` 不是字典，下游 `generate_param_description` 会 KeyError | 中 | `safe_extract_params` 有 try-except 兜底 |
| 12 | 阶段2/4 | - | 置信度阈值不一致：阶段2用 `< 60`，阶段6用 `> 60`（等于 60 的处理不同） | 低 | 边界数据流向不同 |

---

## 附录：关键路径速查

```text
数据管线:
  WBM_caption_qwen3v.py    → 阶段1 caption 生成 (448×448, 黑绿红, 64并发)
  WBM_judge_qwen3vl.py     → 阶段2 事实校验 (纯文本, confidence 0-100)
  WBM_caption_glm45v.py    → 阶段3 GLM交叉标注 (448×448, np.pad 2)
  WBM_judge_glm45v.py      → 阶段4 GLM caption校验
  tran2Json.py             → 阶段6 翻译+COCO输出 (364×364 jpg)

训练:
  finetune_visualglm.py    → Stage1 ViT适配(InfoNCE+CE) + Stage2 LoRA微调(causalLM)
  contrastive_learning.py  → InfoNCELoss(temperature=0.1) + cosine_sim
  lora_mixin.py            → 自定义LoRA (A:Kaiming初始化, B:零初始化, scaling=alpha/r)
  finetune/finetune_visualglm.sh → LoRA训练参数 (rank=16, lr=0.001, 4500 iters)
  finetune/finetune_visualglm_qlora.sh → QLoRA训练参数 (rank=10, lr=0.0001, 300 iters)

评估:
  inference_online.py      → BLIP2特征提取(ViT→QFormer→mean pool→L2) + AP@K + Recall@K
  extract_param.py         → 从summary.json提取指标到CSV

工具:
  utils.py (769行)         → 核心工具库
  model/blip2.py           → BLIP2架构(EVA-ViT + Qformer)
  model/visualglm.py       → VisualGLM(ChatGLM + ImageMixin)
```
