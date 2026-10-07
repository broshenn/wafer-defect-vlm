# WorkBuddy 输入：训练缺口 1200 图（七字段描述）

用途：**主标注训练缺口的描述字段**。旧数据已有 shape/caption_zh，但 radial_zone/clock_direction 因与几何特征 100% 重合而不可用，此批用于补齐。

- 题面（七字段描述协议）纯文本 SHA256：`8a8315f785a2398ba5a823754c40b8a8658f89b79aad1a0dbd68058acb43cfda`
- 条目 1200 条；分片 172 个，每片至多 7 图
- 每行**仅** `item_id / sample_id / image_path / image_sha256`；**不含** gold、lot、旧答案或几何答案
- `image_path` 为本地公开 PNG，须核对 `image_sha256` 与学校版本一致
- 每图一次尝试、原答不可改写；输出目录见各自任务书
- **模型标注/复核，不是人工 gold**
