# 结构化字段的平凡基线 / Trivial baselines, structured fields

## defect_type

- 取值分布 {'Edge_Loc': 30, 'none': 26, 'Scratch': 30, 'Near_full': 16, 'Donut': 30, 'Edge_Ring': 30, 'Loc': 30, 'Center': 30, 'Random': 30}
- 恒答多数类的准确率 **0.1190**（答案恒为 `Edge_Loc`）
- 均匀随机准确率 0.1111（k=9）

## radial_zone

- 取值分布 {'center': 210, 'middle': 36, 'edge': 5, 'none': 1}
- 恒答多数类的准确率 **0.8333**（答案恒为 `center`）
- 均匀随机准确率 0.2500（k=4）

## clock_sector

- 恒答 `5` 的 MAE **2.6932**（这是「忽略图像」的下限）

## size_r

- 恒答 `2.0313417723916634` 的 MAE **0.1101**（这是「忽略图像」的下限）

