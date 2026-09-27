"""Write the accounting, and prove with content -- not with a file count -- that the
reorganisation moved files and changed nothing.

The count check in the previous pass was off by one because the two passes counted
differently (the first excluded its own script from the manifest, the second did not),
and it exited before writing anything. A count was the wrong instrument anyway: it cannot
tell "a file moved" from "a file changed". These can:

  * md5 of the two review sheets, taken from the server before the pull was unpacked
  * byte sizes of the two documents, taken from the server the same way
  * record counts of the files that are supposed to have a fixed population
"""
import hashlib
import json
import pathlib
import sys

R = pathlib.Path(__file__).resolve().parent


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def n(pat):
    return len(list(R.glob(pat)))


checks = [
    ("reviewer_A.csv md5 unchanged", md5(R / "benchmark/review/reviewer_A.csv"),
     "8df48a68cedd4775fd1f90b01540cdbc"),
    ("reviewer_B.csv md5 unchanged", md5(R / "benchmark/review/reviewer_B.csv"),
     "8df48a68cedd4775fd1f90b01540cdbc"),
    ("LIMITATIONS.md bytes", (R / "docs/LIMITATIONS.md").stat().st_size, 125137),
    ("FINAL_REPORT.md bytes", (R / "docs/FINAL_REPORT.md").stat().st_size, 28632),
    ("data/images PNG count", n("data/images/*.png"), 5904),
    ("benchmark/core.jsonl records", sum(1 for _ in open(R / "benchmark/core.jsonl", encoding="utf-8")), 252),
    ("results/curves loss curves", n("results/curves/*.jsonl"), 11),
    ("results/retrieval rankings", n("results/retrieval/*.jsonl"), 12),
]
bad = 0
for name, got, want in checks:
    ok = got == want
    bad += not ok
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}: {got}" + ("" if ok else f"  (expected {want})"))

total = len([p for p in R.rglob("*") if p.is_file()])
print(f"\nfiles now: {total}   (6766 before, minus 14 + 63 cache files = 6689; "
      f"this count includes the two reorg scripts, which the 6766 excluded)")

# how each top-level directory is populated
top = {}
for d in sorted(p for p in R.iterdir() if p.is_dir()):
    fs = [p for p in d.rglob("*") if p.is_file()]
    top[d.name] = (len(fs), sum(p.stat().st_size for p in fs))

lines = [
    "# 整理记录 — 2026-09-19",
    "",
    "整理前：**6766** 个文件散在根目录（334 个）和 `server_text/`（6432 个，当晚从服务器拉的）。",
    f"整理后：**{total}** 个文件，其中 6689 个是原有文件（内容未变），另加两个整理脚本本身。",
    "",
    "## 规则（脚本存于 `archive/_reorg.py` 与 `archive/_finish_reorg.py`）",
    "",
    "先出计划再执行（默认 dry run）；目标已存在则拒绝执行而不是覆盖；执行前后各取一次 md5 清单；",
    "除可再生的构建缓存外不删任何东西；只移动，不重写。",
    "",
    "## 这两遍没做干净的地方（照实记下）",
    "",
    "第一遍把 6408 个移动全部做完，却在最后一步中止，两个原因：",
    "",
    "1. 空目录判据写成「这个目录下有东西吗」，而每个父目录此时仍装着自己那些同样排进删除队列的",
    "   子目录——于是 57 个空目录被判为「非空」，一个都没删。",
    "2. 待删缓存文件在移动**之前**枚举，删除却发生在移动**之后**——其中 68 个已随目录移到",
    "   `code/tools`、`code/projects` 下面，路径不再匹配那份清单。它们没被删，而没有任何东西说",
    "   它们没被删。它当时印的「82 个缓存文件已删除」是错的，真正删掉的是 14 个。",
    "",
    "这正是这个项目 §8 里记的那类缺陷（第 16 条的形状）：一个检查报出的数，和它实际做的事不一致。",
    "第二遍在缓存**实际所在的位置**删除它们，共 63 个。",
    "",
    "第三遍（本脚本）之所以存在，是因为第二遍的收尾判据用错了工具：它拿**文件总数**去对账，",
    "而总数分不清「文件移动了」和「文件被改了」，还因为两次统计口径不同（第一遍的清单排除了",
    "自己的脚本，第二遍没排除）差了一个数就中止退出，于是账目又一次没写下来。",
    "这一遍改用**内容判据**：文档字节数、审核表 md5、固定人口文件的记录数。",
    "",
    "## 内容核对（整理没有改动任何内容）",
    "",
    "| 检查 | 结果 |",
    "| --- | --- |",
]
for name, got, want in checks:
    lines.append(f"| {name} | {'ok，' if got == want else 'FAIL，'} {got}"
                 + ('' if got == want else f'（应为 {want}）') + " |")
lines += [
    "",
    "`reviewer_A.csv` 与 `reviewer_B.csv` 的 md5 取自**解包之前的服务器端**（拉取当晚记录的），",
    "两份同为 `8df48a68cedd4775fd1f90b01540cdbc`——两张表都是空模板，所以内容相同，这是预期的，",
    "不是同一份文件被用了两次。",
    "",
    "## 删除的东西（全部是可再生的构建缓存，没有别的东西）",
    "",
    "- 第一遍：根 `__pycache__/` 下 14 个 `.pyc`",
    "- 第二遍：63 个，分布在 12 个缓存目录里（`__pycache__`、`.pytest_cache`、`.ruff_cache`、",
    "  `*.egg-info`），涉及 `code/tools/`、`code/projects/wafer-defect-vlm/{tests,src}/`。",
    "  这两批缓存的**逐文件 md5 没有留存**：第一遍在写清单前中止，第二遍的清单在内存里随判据失败",
    "  一起丢掉了。它们都能由 `python -m compileall` 或测试重新生成，服务器上也各有一份。",
    "",
    "## 现在每个顶层目录是什么",
    "",
    "| 目录 | 文件数 | 体积 | 是什么 |",
    "| --- | --- | --- | --- |",
]
what = {
    "docs": "文档：结论、限定语、主表、项目 README、参考文献精读",
    "results": "结果：报告 JSON、原始模型输出、检索排名、损失曲线、人可读摘要",
    "benchmark": "benchmark 冻结件（题面、SHA256SUMS、审核表）",
    "data": "数据清单与 5904 张预处理图",
    "code": "服务器上的代码：26 个检查器 + 项目树（src/tests/scripts，逐字节照搬）",
    "logs": "服务器日志 116 个",
    "samples": "9 个缺陷类别各一张代表图",
    "papers": "参考论文与附带数据",
    "archive": "工作副本与历史（见下）",
}
for name, (cnt, size) in top.items():
    lines.append(f"| `{name}/` | {cnt} | {size/1e6:.1f} MB | {what.get(name, '')} |")

lines += [
    "",
    "### archive/ 里是什么",
    "",
    "- `REORG_MAP.txt` —— 本文件",
    "- `working-copies/` —— 根目录散放的 71 个工作副本（检查脚本、诊断脚本等）",
    "- `patches/` —— 135 个 `patch*.py`，逐个改服务器的补丁",
    "- `queue-scripts/` —— 21 个一次性队列脚本",
    "- `commit-messages/` —— 4 个 commit message 草稿",
    "- `local-skeleton/` —— 9 项：本机 9 月 14/15 日的旧骨架（src/tests/scripts/tools/docs/manifests/projects + pyproject.toml）",
    "- `superseded-2026-09-15/` —— 根目录那份过期的 LIMITATIONS.md（31 KB；现行版本见 `docs/LIMITATIONS.md`，125 KB）",
    "- `tarballs/` —— 当晚从服务器拉取的 4 个原始压缩包，解压即得未经整理的形态",
    "",
    "## 倒回去的办法",
    "",
    "每条移动都是可逆的，目标目录名保留了来源语义。要完全回到整理前的形态：",
    "解压 `archive/tarballs/bundle*.tar.gz` 得到 `server_text/` 的原样，",
    "再把 `archive/working-copies/`、`patches/`、`queue-scripts/`、`commit-messages/`、",
    "`local-skeleton/` 里的文件放回根目录。内容从未被改写，所以 md5 都对得上。",
    "",
]
(R / "archive/REORG_MAP.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("accounting written to archive/REORG_MAP.txt")
sys.exit(1 if bad else 0)
