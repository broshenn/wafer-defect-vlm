"""Reorganise this directory into one readable project tree.

Rules this script follows, because the alternative is losing something quietly:

  * dry run by default -- it prints every move and writes nothing until --apply
  * it never overwrites: a destination that already exists is a hard error, not a merge
  * it records a manifest of every file (path + md5) before it starts, and after it
    finishes it checks that every one of those md5s is still present somewhere in the
    tree. Deletions are limited to build caches and provable duplicates, and the md5 of
    every file inside a deleted directory is written into the map first.
  * every action goes into archive/REORG_MAP.txt, so the whole thing is reversible.

The truth here is server_text/ (pulled from /root/autodl-fs/wafer-vlm on 2026-09-17). The
loose files at the root and the src/tests/scripts/tools trees on this machine are working
copies and an earlier skeleton; they move to archive/ rather than being deleted.

Three phases, because order matters: the old local directories have to leave before the
new ones are created, the copies read from the pull before it is broken up, and
everything else follows.
"""
import hashlib
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parent
APPLY = "--apply" in sys.argv
KEEP_AT_ROOT = {"README.md", "AGENTS.md", ".env.example", ".gitignore", ".claude"}

PH_EARLY, PH_COPY, PH_LATE = 0, 1, 2
actions: list[tuple[int, str, pathlib.Path, pathlib.Path, str]] = []
deletes: list[tuple[pathlib.Path, str]] = []        # cache dirs: their files are enumerated
empty_dirs: list[tuple[pathlib.Path, str]] = []     # must be empty when execution reaches them
planned: set[pathlib.Path] = set()


def md5(p: pathlib.Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest(root: pathlib.Path) -> dict[str, str]:
    return {str(p.relative_to(root)).replace("\\", "/"): md5(p)
            for p in sorted(root.rglob("*"))
            if p.is_file() and "_reorg.py" not in p.name}


def move(src, dst, why, phase=PH_LATE):
    src, dst = pathlib.Path(src), pathlib.Path(dst)
    if src.exists() and src not in planned:
        actions.append((phase, "MOVE", src, dst, why))
        planned.add(src)


def copy(src, dst, why, phase=PH_COPY):
    src, dst = pathlib.Path(src), pathlib.Path(dst)
    if src.exists():
        actions.append((phase, "COPY", src, dst, why))


def rm(p, why):
    if pathlib.Path(p).exists():
        deletes.append((pathlib.Path(p), why))


def rmdir(p, why):
    if pathlib.Path(p).is_dir():
        empty_dirs.append((pathlib.Path(p), why))


# ---------------------------------------------------------------- phase 1: caches
for pat in ("__pycache__", ".pytest_cache", ".ruff_cache", "*.egg-info"):
    for p in sorted(ROOT.rglob(pat)):
        if ".claude" not in p.parts:
            rm(p, "build/test cache, regenerable")

# ------------------------------------------------- phase 2: the earlier local skeleton
for name in ("src", "tests", "scripts", "tools", "docs", "manifests", "projects",
             "pyproject.toml", "uv.lock", "0.6197"):
    move(ROOT / name, ROOT / "archive/local-skeleton" / name,
         "earlier local copy (superseded by code/)", PH_EARLY)
move(ROOT / "tmp", ROOT / "papers", "papers and their data", PH_EARLY)
move(ROOT / "2604.27629v4.pdf", ROOT / "papers/2604.27629v4.pdf", "the reference paper", PH_EARLY)

# ------------------------------------------------- phase 3: loose working copies
for p in sorted(ROOT.glob("*")):
    if p.is_dir() or p.name in KEEP_AT_ROOT:
        continue
    if p.name in ("README.md", "LIMITATIONS.md"):
        move(p, ROOT / "archive/superseded-2026-09-15" / p.name, "superseded by docs/")
    elif p.name.startswith("patch") and p.suffix == ".py":
        move(p, ROOT / "archive/patches" / p.name, "one-off patch script")
    elif p.suffix == ".sh":
        move(p, ROOT / "archive/queue-scripts" / p.name, "one-off queue script")
    elif p.name.startswith("commit_msg"):
        move(p, ROOT / "archive/commit-messages" / p.name, "commit message draft")
    else:
        move(p, ROOT / "archive/working-copies" / p.name, "working copy")

# ------------------------------------------------- phase 4: build the real tree
S = ROOT / "server_text"
PR = S / "projects/wafer-defect-vlm"
B = S / "benchmarks/wafer_bench_v1"

move(S / "FINAL_REPORT.md", ROOT / "docs/FINAL_REPORT.md", "")
move(S / "LIMITATIONS.md", ROOT / "docs/LIMITATIONS.md", "")
copy(PR / "README.md", ROOT / "docs/README-project.md", "copied: code/ mirrors the server")
copy(PR / "docs/WaferSAGE_中文精读与训练方案.md", ROOT / "docs/WaferSAGE_中文精读与训练方案.md", "")
copy(PR / "manifests/project_decisions.md", ROOT / "docs/project_decisions.md", "")
copy(S / "outputs/reports/comparison.md", ROOT / "docs/comparison.md", "")

for p in sorted((S / "outputs/reports").glob("*.json")):
    move(p, ROOT / "results/reports" / p.name, "")
for p in sorted((S / "outputs/reports").glob("*.md")):
    move(p, ROOT / "results/readouts" / p.name, "")
for p in sorted((S / "outputs/baselines").glob("*")):
    move(p, ROOT / "results/raw_outputs" / p.name, "")
for p in sorted((S / "outputs/retrieval").glob("*")):
    move(p, ROOT / "results/retrieval" / p.name, "")
for p in sorted((S / "outputs/merge_check").glob("*")):
    move(p, ROOT / "results/merge_check" / p.name, "")
for p in sorted((S / "outputs/checkpoints").rglob("logging.jsonl")):
    move(p, ROOT / "results/curves" / f"{p.parent.parent.name}.jsonl", "")

for p in sorted(B.glob("*.jsonl")) + sorted(B.glob("*.json")):
    move(p, ROOT / "benchmark" / p.name, "")
move(B / "SHA256SUMS", ROOT / "benchmark/SHA256SUMS", "")
move(S / "core.jsonl", ROOT / "benchmark/core.jsonl", "")
move(S / "inference_requests.jsonl", ROOT / "benchmark/inference_requests.jsonl", "")
move(S / "robustness_inference_requests.jsonl",
     ROOT / "benchmark/robustness_inference_requests.jsonl", "")
# Two review/ directories hold the same five files (identical md5s): keep the copy inside
# benchmark/wafer_bench_v1, record the other as a duplicate deletion.
for p in sorted((S / "review").glob("*")):
    other = B / "review" / p.name
    if other.exists() and md5(p) == md5(other):
        rm(p, "duplicate of benchmark/review/ (identical md5)")
    else:
        move(p, ROOT / "benchmark/review" / p.name, "")
for p in sorted((B / "review").glob("*")):
    move(p, ROOT / "benchmark/review" / p.name, "")

for p in sorted((S / "data/prepared_v1/images").glob("*")):
    move(p, ROOT / "data/images" / p.name, "")
for p in sorted((S / "data/prepared_v1").glob("*")):
    if p.is_file():
        move(p, ROOT / "data" / p.name, "")
rmdir(S / "data/prepared_v1/images", "empty once its files moved")
for p in sorted((S / "samples").glob("*")):
    move(p, ROOT / "samples" / p.name, "")
for p in sorted((S / "logs").glob("*")):
    move(p, ROOT / "logs" / p.name, "")

move(S / "tools", ROOT / "code/tools", "the server's checkers, verbatim")
move(S / "projects", ROOT / "code/projects", "the server's project tree, verbatim")
for p in sorted(S.glob("bundle*.tar.gz")):
    move(p, ROOT / "archive/tarballs" / p.name, "the pull itself; contents now extracted")
for p in sorted(S.glob("*")):
    if p.is_file():
        move(p, ROOT / "archive/working-copies" / p.name, "left over in the pull directory")
for d in sorted(S.rglob("*"), reverse=True):
    if d.is_dir():
        rmdir(d, "empty pull directory")
rmdir(S, "empty pull directory")

# ---------------------------------------------------------------- plan
before = manifest(ROOT)

# Enumerate the cache files now, while they still exist: a directory deletion has to
# carry the md5s of what was inside it, or the check below cannot tell a deleted cache
# file from a lost one. (The same shape as the project's own rule that a check must be
# able to account for what it removed.)
del_files: dict[pathlib.Path, tuple[str, str]] = {}
for p, w in deletes:
    if p.is_dir():
        for f in sorted(p.rglob("*")):
            if f.is_file():
                del_files[f] = (md5(f), w)
    elif p.is_file():
        del_files[p] = (md5(p), w)

actions.sort(key=lambda a: a[0])
print(f"files before: {len(before)}")
print(f"\nplanned: {len(actions)} actions, {len(del_files)} cache files deleted in "
      f"{len(deletes)} director(ies), {len(empty_dirs)} directories emptied")
by_why: dict[str, int] = {}
for _ph, kind, _s, _d, w in actions:
    by_why[f"{kind}  {w or '(structural)'}"] = by_why.get(f"{kind}  {w or '(structural)'}", 0) + 1
for w, n in sorted(by_why.items(), key=lambda kv: -kv[1]):
    print(f"   {n:5d}  {w}")
print("\nthe first 10 actions:")
for _ph, kind, s, d, _w in actions[:10]:
    print(f"   {kind}  {s.relative_to(ROOT)}  ->  {d.relative_to(ROOT)}")
print(f"\nthe {len(deletes)} cache deletions and {len(empty_dirs)} emptied directories:")
for p, w in deletes:
    print(f"   {p.relative_to(ROOT)}   [{w}]")
print(f"   ... plus {len(empty_dirs)} emptied directories")

if not APPLY:
    print("\nDRY RUN -- nothing written. Re-run with --apply to execute.")
    raise SystemExit(0)

# Conflicts are checked one phase at a time, because an earlier phase is what clears
# the way: the new docs/ can only be written once the old docs/ has moved, and checking
# the whole plan up front reported that as a collision.
for ph in (PH_EARLY, PH_COPY, PH_LATE):
    todo = [a for a in actions if a[0] == ph]
    taken = [str(d) for _p, _k, _s, d, _w in todo if d.exists()]
    if taken:
        raise SystemExit(f"refusing to run phase {ph}: destination(s) already exist:\n  "
                         + "\n  ".join(taken))
    for _ph, kind, s, d, _w in todo:
        d.parent.mkdir(parents=True, exist_ok=True)
        if kind == "COPY":
            shutil.copy2(str(s), str(d))
        else:
            shutil.move(str(s), str(d))

lines = ["# every action this reorganisation took (2026-09-19)", ""]
for _ph, kind, s, d, w in actions:
    lines.append(f"{kind}  {s.relative_to(ROOT)}  ->  {d.relative_to(ROOT)}  [{w or 'structural'}]")
for f, (h, w) in sorted(del_files.items()):
    lines.append(f"DEL   {f.relative_to(ROOT)}  md5={h}  [{w}]")

for p, _w in deletes:
    if p.exists():
        shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink()
# A directory is removed only once it is verifiably empty: if a move failed, this refuses
# instead of taking the contents with it.
stubborn = [str(p.relative_to(ROOT)) for p, _w in empty_dirs if p.exists() and any(p.iterdir())]
if stubborn:
    raise SystemExit("refusing to remove non-empty directories:\n  " + "\n  ".join(stubborn))
for p, w in sorted(empty_dirs, key=lambda pw: -len(pw[0].parts)):
    if p.exists():
        lines.append(f"RMDIR {p.relative_to(ROOT)}  [{w}]")
        p.rmdir()
(ROOT / "archive").mkdir(exist_ok=True)
(ROOT / "archive/REORG_MAP.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

# ---------------------------------------------------------------- verify
after = manifest(ROOT)
after_md5 = set(after.values())
gone = {m for m in before.values() if m not in after_md5}
deleted_md5 = {h for h, _w in del_files.values()}
lost = gone - deleted_md5
print(f"\nfiles after: {len(after)}")
print(f"before-manifest md5s no longer present anywhere: {len(gone)}")
print(f"  accounted for as recorded deletions: {len(gone & deleted_md5)}")
print(f"  UNACCOUNTED: {len(lost)}")
if lost:
    print("  !! investigate:", sorted(lost)[:5])
    raise SystemExit(1)
print("map written to archive/REORG_MAP.txt")
