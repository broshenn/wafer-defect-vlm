"""Record a defect class the provenance audit is structurally blind to.

Every earlier item in section 8 is a number that cannot be traced to a record,
or a result that was silently never produced. This is neither. The number is in
the record, byte for byte, and the sentence around it says the opposite of what
the number shows -- or compares two runs that differ in more than the variable
being discussed. audit_report_numbers.py asks only whether a number exists in a
record, so it passes this cleanly and always will.
"""
from pathlib import Path
import sys

p = Path("/root/autodl-fs/wafer-vlm/LIMITATIONS.md")
s = p.read_text(encoding="utf-8")

ITEM9 = '''9. **数字完全正确，但被用反了方向 —— 溯源核对结构上看不见这一类。**
   5.2.2 初稿有一句：「同一个 5e-5 在 G=8 上把 KL 抬到 1.02–1.12」。
   1.1201 与 1.0217 两个数字都真实存在于记录中，逐字节核对通过；
   但 1.1201 属于 **lr1e-5** 那次，1.0217 属于 **lr5e-5** 那次 ——
   这个区间横跨了两个学习率，恰好把「学习率升高时 KL 是**下降**的」掩盖了。
   五份记录逐行重算（150 步 × 5 run）确认 `mean_kl` 与日志逐值相等，
   错误不在测量，在这句话。

   更麻烦的是：把方向反过来写**也是错的**。`kl` 是逐 token 均值
   （证据：在 `frac_reward_zero_std == 1` 的步上优势恒为 0，于是
   `loss == beta * kl` 严格成立 —— 五个 run 的全部 271 个空转步均满足，
   最大偏差 2e-05，即日志自身的舍入）。但 5e-5 那两次的补全明显更短
   （GRPO 115.5 → 97.9 token，−15.2%；GSPO 114.4 → 109.2，−4.6%），
   均值取在不同的 token 跨度上，两次的 `kl` 不是同一个量。
   **因此该句被删除，而不是改写成相反方向。**

   同一段里的空转步论据犯的是第二种毛病：「把空转步从 36.00% 推到 38.00%」
   拿 GRPO(G=4, lr1e-5) 比 GSPO(G=8, lr5e-5) —— 算法与组大小同时变了。
   两个**同算法同组大小**的对照方向相反：
   GRPO(G=4) 36.00% → 32.67%（`idle_delta` −0.0333），
   GSPO(G=8) 15.33% → 38.00%（`idle_delta` +0.2267）。
   即「学习率把空转步推高」在 G=8 上成立、在 G=4 上不成立。

   **这一类的教训与前面八条不同：**前八条防的是「数字来路不明」，
   这一条防的是「数字来路清楚、但话说的不是它」。
   前者可以自动核对，后者目前**只能靠重算该比较本身、并检查符号** ——
   `tools/kl_length_confound.py` 就是为此写的，它把两个方向的证据
   （恒等式、补全长度）与结论一起存档，任何人可复跑。
   全部数值见 `outputs/reports/kl_length_confound.json`。

'''

ITEM10 = '''10. **本轮新写的两个核对工具，各自失败过一次。**
    记在这里，是因为工具是用来防前面九条的，工具自己出错却没人核对，
    就等于把防线建在了没验过的地基上。

    - `tools/audit_log_provenance.py` 第一版只搜命令行形式
      （`--importance_sampling_level token`），而 ms-swift 的 args dump 用的是
      `importance_sampling_level=token`。于是「找不到该字段」被当成「一致」，
      `grpo_train_result.json` 里为空的 `is_level` 被静默放过。
      **这正是该工具要防的那件事**：把没有证据当成一致。
      已改为：日志未声明某字段时报告为 UNVERIFIED，而不是通过。
    - 第二版补上了 args dump 形式，但**没加左边界**：`seed=` 同样能匹配
      `data_seed=42`，`beta=` 同样能匹配 `sdar_gate_beta=5.0`（二者都在同一份
      dump 里）。它当时仍给出正确答案，纯粹因为 args dump 按字母序打印，
      `seed=3407` 恰好排在 `data_seed=42` 之后、最后由「取最后一个匹配」胜出。
      **依赖字母序不是核对。** 已加 `(?<![A-Za-z0-9_])`。
    - `tools/kl_length_confound.py` 用同样的无边界写法读 `beta`，取到
      `sdar_gate_beta=5.0`，于是 `loss == beta * kl` 恒等式在全部 150 步上失败。
      这一次是**响亮失败**（打印 `0/54`）而不是静默通过，因为它断言的是一个
      **可否证的等式并打印通过步数** ——
      同一个错误，在断言恒等式的工具里表现为 `0/54`，在只报告「一致」的
      工具里则表现为静默通过。**这是本轮最有用的一条经验。**
    - 一处**尚未发作**的隐患：`scripts/29_grpo_train.sh` 的
      `RUN_TAG` 默认值是裸字符串 `grpo`，因此 GSPO-v1 那次覆盖了更早那次
      GRPO run 的 `29_grpo_train.log`。今天没有任何记录引用该文件，
      所以盘上没有错；但若有记录引用它，那个记录会指向一份**完整、真实、
      却属于另一次 run** 的日志 —— 正是本节一直在记的那类失败。
      今天两个队列都显式传了 `RUN_TAG`，不会复发；
      默认值本身的修复留到收盘后（见第 8 条：不得在 run 正在读脚本时改它）。
      已加 `tools/audit_log_provenance.py` 常态化核对「记录引用的日志是不是它自己的」。

'''

anchor = "这几处的共同点："
if s.count(anchor) != 1:
    sys.exit(f"FAILED: anchor appears {s.count(anchor)} times")
s = s.replace(anchor, ITEM9 + ITEM10 + anchor)
p.write_text(s, encoding="utf-8")
print("section 8: items 9 and 10 added")
