"""Record the effective-batch confound in 5.2, and the record defect in section 8.

Two things were found by reading the live argv of the running job:

  * the training records' grad_accum is a hardcoded 4, so three of five records
    misstate what ran (G=8 ran 8, G=32 ran 32);
  * accumulation was set equal to group size in every RL run, so effective batch
    grew as G^2 -- 16, 64, 1024 -- meaning group size was never varied alone.

The second one reaches into 5.2.5's verdict, which reads a G=8 -> G=32 contrast
as being about group size. It is not; the effective batch changed 16x alongside.

The lr contrasts are untouched: they hold G fixed, so accumulation and effective
batch are fixed within each. The patch says so explicitly, so that the correction
does not quietly weaken results that are actually fine.
"""
import sys
from pathlib import Path

D = Path("/root/autodl-fs/wafer-vlm/LIMITATIONS.md")
s = D.read_text(encoding="utf-8")

OLD1 = """以及补齐网格第四角用的
G=32 + lr 1e-5 —— 最后这一条是本设计里**唯一能把「组大小」与「IS 层级」分开**的对照
（同算法、同学习率、只改组大小），当时仍在队列 41 里跑，本节其余部分未使用它的结果。
**其中「G=32」这一条，是我先断言不可行、后被实测推翻的。**
"""

NEW1 = """以及补齐网格第四角用的
G=32 + lr 1e-5 —— 最后这一条给出的是**同一组大小、同一算法、只改学习率**的对照
（G=8 那一对也是同一种对照），当时仍在队列 41 里跑，本节其余部分未使用它的结果。
**初稿在这里写过它「唯一能把「组大小」与「IS 层级」分开」—— 那句是错的**：
两条都是 sequence，改的是组大小；而紧接的这一段会说明，本设计连「只改组大小」
都没有做到（组大小一变，有效批次跟着按平方变）。它能分开的是学习率。
**其中「G=32」这一条，是我先断言不可行、后被实测推翻的。**

**本节有一个贯穿全部「组大小」读数的混淆，在此单独说明。**

训练记录里的 `grad_accum` 字段是**写死的常量 4**，不是实测值。因此它既显示不出、
也掩盖不了这个事实：**每个 run 的梯度累积都设成了与组大小相等**（4/4、8/8、32/32）：

| 组大小 | 真实累积步 | 有效批次（= 组大小 × 微批次 1 × 累积步）|
| --- | --- | --- |
| G=4 | 4 | 16 |
| G=8 | 8 | 64 |
| G=32 | 32 | **1024** |

**有效批次按组大小的平方增长，所以本设计里从来没有单独改过组大小**：
G=8 → G=32 是组大小 4 倍、有效批次 **16 倍**。5.2.5 的「加组大小」一行、
以及 `group_size_paired` 的全部成对对照，比较的都是同时变了两个量的两个 run，
**因此它们不能把差异归给组大小**。

这是与 5.2.2 的 IS 层级混淆**相互独立**的第二个混淆（token 永远是 G=4、
sequence 永远是 G=8 或 G=32）。两个合起来：本设计**既没有单独变过 IS 层级，
也没有单独变过组大小** —— 论文的两个关键量与别的东西绑在了一起。

**学习率的对照不受影响**，这一点必须说清楚，以免这条更正顺手削弱了好结果：
它们在同一个组大小内比较，累积步与有效批次都固定（G=4 两个学习率、
G=8 两个学习率、G=32 两个学习率）。**本节关于学习率的结论照旧成立。**

逐 run 的真实配置由训练日志里的 swift 参数还原、与记录逐条对照，见
`outputs/reports/effective_batch_correction.json`；写死字段这一点列入第 8 节第 11 条。
"""

OLD2 = """即**在 lr 5e-5 这一档上，算法与组大小两两都分不开，唯一分得开的是学习率** ——
这比「组大小有效但不足」更强，也更贴合数据。
"""

NEW2 = """即**在 lr 5e-5 这一档上，算法与组大小两两都分不开，唯一分得开的是学习率** ——
这比「组大小有效但不足」更强，也更贴合数据。

**但上面「加组大小」这一行还要再降一级。** 该对照把有效批次从 64 改成 1024
（组大小 4 倍、有效批次 16 倍，见本节开头的混淆说明），所以它连
「加组大小无效」都只能写成「**加组大小或加有效批次无效**」——
本设计分不开这两个。方向与显著性都成立，被削掉的只是归因。
"""

OLD3 = """      已加 `tools/audit_log_provenance.py` 常态化核对「记录引用的日志是不是它自己的」。

这几处的共同点："""

NEW3 = """      已加 `tools/audit_log_provenance.py` 常态化核对「记录引用的日志是不是它自己的」。

11. **训练记录里的一个字段是写死的常量，因此三个记录写的不是实际发生的事。**
    `scripts/29_grpo_train.sh` 写结果的 heredoc 里，`"grad_accum": 4` 是**字面量**，
    不读任何环境变量。所以它不可能与实际不符 —— 它根本不知道实际是什么。
    训练日志里的 swift 参数才是实测值：

    | run | 实际累积步 | 记录写的 |
    | --- | --- | --- |
    | GSPO G=8 lr1e-5 | 8 | 4 |
    | GSPO G=8 lr5e-5 | 8 | 4 |
    | GSPO G=32 lr5e-5 | 32 | 4 |
    | GRPO G=4 两个 | 4 | 4（对，但只是常量碰巧等于实测值）|

    **本条暴露的不是配置错，而是「记录」与「事实」之间没有校验**：
    前三条记录给出的有效批次全部错了同一个倍数，而本项目此前所有的
    「组大小」分析都建立在那些记录上。发现方式与前述九条不同 ——
    这次是在准备让 GPU 并发跑第二个任务、去读**正在运行的进程**的 argv 时
    顺手读到了真相。**核对一个数字时顺手看一眼它的来源，比专门去查它更容易查出来。**

    连带发现的第二个问题（组大小与有效批次混淆）记在 5.2 节开头。
    修复同样留到收盘后：该 heredoc 在 `29_grpo_train.sh` 里，而队列 41 正在执行该文件，
    改动运行中的 `.sh` 会移动 bash 的续读偏移 —— 这正是第 8 条已经造成过两次事故的做法。
    修法：改成 `int(os.environ.get("GRAD_ACCUM", 4))`。

这几处的共同点："""

for old, new, tag in ((OLD1, NEW1, "5.2 intro"),
                      (OLD2, NEW2, "5.2.5 verdict"),
                      (OLD3, NEW3, "section 8 item 11")):
    n = s.count(old)
    if n != 1:
        sys.exit(f"FAILED: {tag} anchor appears {n} times, expected 1")
    s = s.replace(old, new)

if s.count("**") % 2:
    sys.exit("bold markup unbalanced after patch; refusing to write")

D.write_text(s, encoding="utf-8")
print("LIMITATIONS.md: 5.2 confound block added, 5.2.5 verdict downgraded, "
      "section 8 item 11 recorded")
print(f"lines: {s.count(chr(10)) + 1}, ** = {s.count('**')} (even)")
