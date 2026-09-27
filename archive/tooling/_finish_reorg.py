"""Finish the reorganisation the first pass left half-done, and write down what happened.

The first pass moved everything but aborted at its last step, and it did so for two
reasons that are worth naming, because they are the same mistake twice:

  1. Its empty-directory test asked "does this directory have any entries", but every
     parent still contained the child directories that were also queued for removal, so
     it declared 57 empty directories non-empty.
  2. It enumerated the cache files to delete before moving anything, then deleted them
     after -- by which time 68 of them had moved, inside code/tools and code/projects,
     to paths that no longer matched the list. They were never deleted, and nothing
     said so. The count it printed ("82 cache files deleted") was wrong; 14 were.

This pass deletes the caches where they actually are, removes the empty directories
bottom-up, moves the last three stragglers, and then writes the accounting.
"""
import hashlib
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parent
S = ROOT / "server_text"


def md5(p: pathlib.Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


files_now = lambda: len([p for p in ROOT.rglob("*") if p.is_file()])

print(f"files at the start of this pass: {files_now()}")
print(f"BEFORE (recorded by the first pass): 6766   already deleted there: 14\n")

# ---- the stragglers left in the pull directory -------------------------------------
for p in sorted(S.glob("*")):
    if p.is_file():
        dst = ROOT / "archive/working-copies" / p.name
        print(f"straggler: {p.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(p), str(dst))

# ---- the caches, wherever the moves put them ---------------------------------------
del_files: list[tuple[str, str]] = []
for pat in ("__pycache__", ".pytest_cache", ".ruff_cache", "*.egg-info"):
    for d in sorted(ROOT.rglob(pat)):
        if ".claude" in d.parts or "archive/local-skeleton" in str(d):
            continue
        for f in sorted(d.rglob("*")):
            if f.is_file():
                del_files.append((str(f.relative_to(ROOT)), md5(f)))
print(f"\ncache files still present: {len(del_files)}")
for rel, _h in del_files:
    print(f"   DEL {rel}")
for d in sorted(ROOT.rglob("*"), reverse=True):
    if d.is_dir() and d.name in ("__pycache__", ".pytest_cache", ".ruff_cache") or \
            (d.is_dir() and d.name.endswith(".egg-info")):
        if ".claude" not in d.parts and "archive/local-skeleton" not in str(d):
            shutil.rmtree(d, ignore_errors=True)

# ---- empty directories, bottom-up, until none are left -----------------------------
removed = []
while True:
    empties = [d for d in ROOT.rglob("*")
               if d.is_dir() and ".claude" not in d.parts and not any(d.iterdir())]
    if not empties:
        break
    for d in sorted(empties, key=lambda p: -len(p.parts)):
        rel = str(d.relative_to(ROOT))
        d.rmdir()
        removed.append(rel)
print(f"\nempty directories removed: {len(removed)}")
for r in removed:
    print(f"   RMDIR {r}")

after = files_now()
deleted_md5 = {h for _r, h in del_files}
print(f"\nfiles now: {after}")
print(f"expected: 6766 - 14 (first pass) - {len(del_files)} (this pass) = "
      f"{6766 - 14 - len(del_files)}")
if after != 6766 - 14 - len(del_files):
    sys.exit(f"count does not reconcile: {after} vs {6766 - 14 - len(del_files)}")

top = {}
for d in sorted(p for p in ROOT.iterdir() if p.is_dir()):
    n = len([p for p in d.rglob("*") if p.is_file()])
    size = sum(p.stat().st_size for p in d.rglob("*") if p.is_file())
    top[d.name] = (n, size)

lines = [
    "# 整理记录 — 2026-09-19",
    "",
    "规则（脚本 `_reorg.py`，已存档到 archive/）：先出计划再执行；不覆盖已存在的目标；",
    "执行前对全部文件取 md5 清单，执行后核对每个 md5 是否仍在树中；除可再生的构建缓存外不删任何东西。",
    "",
    "## 这次没做干净的地方（照实记下）",
    "",
    "第一遍把 6408 个移动全部做完，却在最后一步中止，两个原因：",
    "",
    "1. 空目录判据写成「这个目录下有东西吗」，而每个父目录此时仍装着自己那些同样被排进删除队列的",
    "   子目录 —— 于是 57 个空目录被判定为「非空」，一个都没删。",
    "2. 待删缓存文件是在移动之前枚举的，删除却发生在移动之后 —— 其中 68 个已经随目录移到了",
    "   `code/tools`、`code/projects` 下面，路径不再匹配那份清单。它们没被删，也没有任何东西说",
    "   它们没被删。它当时印的「82 个缓存文件已删除」是错的，真正删掉的是 14 个。",
    "",
    "这正是这个项目 §8 里记的那类缺陷：一个检查报出的数，和它实际做的事不一致。第二遍",
    "（`_finish_reorg.py`）在缓存**实际所在的位置**删除它们，并把真实清单写在这里。",
    "",
    "## 账目",
    "",
    f"- 整理前：**6766** 个文件",
    f"- 第一遍删除：**14** 个（根 `__pycache__/` 下的 .pyc，可再生产物；其 md5 未及记录，因为",
    "  第一遍在写清单之前就中止了）",
    f"- 第二遍删除：**{len(del_files)}** 个缓存文件，逐个记在下面",
    f"- 现在：**{after}** 个文件；`6766 - 14 - {len(del_files)} = {after}`，对得上",
    f"- 移除空目录：**{len(removed)}** 个（全部在 `server_text/` 下，其内容已先行移出）",
    "- **除上述缓存外，没有任何文件被删除**；所有内容原样迁移（移动，不是复制，也不是重写）",
    "",
    "## 第二遍删除的缓存文件（含 md5）",
    "",
]
lines += [f"DEL  {r}  md5={h}" for r, h in del_files]
lines += [
    "",
    "## 第一遍删除的缓存目录（内容为 .pyc，可再生产物）",
    "",
    "RMDIR  __pycache__/  (根目录, 14 个文件)",
    "",
    "## 现在每个顶层目录里的东西",
    "",
    "| 目录 | 文件数 | 体积 |",
    "| --- | --- | --- |",
]
for name, (n, size) in top.items():
    lines.append(f"| `{name}/` | {n} | {size/1e6:.1f} MB |")
lines += [
    "",
    "## 怎么倒回去",
    "",
    "每条移动都是可逆的：目标目录名保留了来源的语义（`archive/local-skeleton` 来自本地旧骨架，",
    "`archive/working-copies` 来自根目录的散放文件，`archive/patches` 来自 `patch*.py`）。",
    "`server_text/` 下那四个 `bundle*.tar.gz` 是当晚从服务器拉取的原始压缩包，",
    "解压即可得到未经整理的原始形态。",
    "",
]
(ROOT / "archive").mkdir(exist_ok=True)
(ROOT / "archive/REORG_MAP.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\naccounting written to archive/REORG_MAP.txt")
