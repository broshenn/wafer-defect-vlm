# WorkBuddy 输入：90 图独立视觉复核

用途：**模型复核**训练候选的形态/位置问题；不是人工审核，不决定测试标签。

- 题面（七字段描述协议）纯文本 SHA256：`8a8315f785a2398ba5a823754c40b8a8658f89b79aad1a0dbd68058acb43cfda`
- 条目 90 条；分片 13 个，每片至多 7 图
- 每行仅含 `item_id / sample_id / image_path / image_sha256`；**不含** gold、lot、旧答案或几何答案
- `image_path` 为本地公开 PNG，客户端须核对 `image_sha256` 与学校版本一致
- 每图一次尝试、原答不可改写；输出目录见各自任务书
