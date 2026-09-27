"""Teach the report generator about the effective-batch confound.

FINAL_REPORT.md is generated, and queue 41 regenerates it when it finishes, so a
correction that lives only in LIMITATIONS.md would be silently absent from the
report a reader sees first. The caveat is emitted from the record rather than
written as prose in the generator, so the numbers in it cannot drift.

Values are read from outputs/reports/effective_batch_correction.json; if that
record is missing the bullet is omitted rather than printed with blanks.
"""
import ast
import sys
from pathlib import Path

P = Path("/root/autodl-fs/wafer-vlm/tools/final_report.py")
s = P.read_text(encoding="utf-8")

OLD = """        _gs = ps.get("group_size_paired")
        if _gs:
            out.append(f"  - 组大小（GSPO lr 5e-5，G=8 → G=32）：Δ准确率 "
                       f"{_gs['accuracy_delta_g32_minus_g8']:+.4f}，"
                       f"p = {_gs['mcnemar']['p_exact_two_sided']:.6g} —— "
                       f"点估计为正但**未达显著**。即在 lr 5e-5 上，算法与组大小两两都"
                       f"分不开，**唯一分得开的是学习率**。")
"""

NEW = """        _gs = ps.get("group_size_paired")
        if _gs:
            out.append(f"  - 组大小（GSPO lr 5e-5，G=8 → G=32）：Δ准确率 "
                       f"{_gs['accuracy_delta_g32_minus_g8']:+.4f}，"
                       f"p = {_gs['mcnemar']['p_exact_two_sided']:.6g} —— "
                       f"点估计为正但**未达显著**。即在 lr 5e-5 上，算法与组大小两两都"
                       f"分不开，**唯一分得开的是学习率**。")
        # The group-size contrast above is confounded with effective batch size,
        # which the training records cannot show: their grad_accum field is a
        # hardcoded constant. Read the real figures from the correction record.
        _ebf = reports / "effective_batch_correction.json"
        _eb = (json.loads(_ebf.read_text(encoding="utf-8"))
               if _ebf.is_file() else None)
        if _eb:
            _b = (_eb.get("design_confound", {})
                  .get("effective_batch_by_G", {}))
            out.append(
                f"  - **但「组大小」这一行被有效批次混淆，读法要再退一步**："
                f"训练时梯度累积被设成等于组大小（4/4、8/8、32/32），"
                f"故有效批次 = 组大小² ——G=4 → {_b.get('4')}、G=8 → {_b.get('8')}、"
                f"G=32 → **{_b.get('32')}**。该对照因此同时改了组大小 4 倍与有效批次 16 倍，"
                f"**只能读作「加组大小或加有效批次无效」**，分不开这两个。"
                f"同一个混淆也约束论文的中心主张（token 永远是 G=4、"
                f"sequence 永远是 G=8 或 G=32，两个量从未单独变过）。"
                f"**学习率的对照不受影响**：同一组大小内累积步与有效批次都固定。"
                f"另有一个记录缺陷 —— 训练记录里的 `grad_accum` 字段是写死的常量 4，"
                f"三条记录（两个 G=8、一个 G=32 lr5e-5）因此写错了实际值，"
                f"见 LIMITATIONS 第 8 节第 11 条。")
"""

if s.count(OLD) != 1:
    sys.exit(f"anchor appears {s.count(OLD)} times, expected 1")
s = s.replace(OLD, NEW)

ast.parse(s)
P.write_text(s, encoding="utf-8")
print("final_report.py: effective-batch confound bullet added to the "
      "paired-significance block")
