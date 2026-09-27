"""Is the training venv on the persistent disk or on the ephemeral one?

The constraint for this project is that all data, models, environments and
outputs live under /root/autodl-fs/wafer-vlm. The running swift processes were
launched from /root/autodl-tmp/wafer-vlm/venvs/wafer/bin/python, and /root/autodl-tmp
is the 50G system disk. Either that path is a link into autodl-fs (harmless) or
the environment actually in use lives on the ephemeral disk (not harmless).
Read-only; nothing here changes state while a run is training.
"""
import os
import subprocess
from pathlib import Path

A = Path("/root/autodl-tmp/wafer-vlm")
B = Path("/root/autodl-fs/wafer-vlm")

print("=== path identity ===")
for p in (Path("/root/autodl-tmp"), A, B):
    print(f"  {p}: exists={p.exists()} symlink={p.is_symlink()}"
          + (f" -> {os.readlink(p)}" if p.is_symlink() else ""))
print(f"  same inode (A vs B): {A.exists() and B.exists() and A.stat().st_ino == B.stat().st_ino}")

print("\n=== the two venv interpreters ===")
for p in (A / "venvs/wafer/bin/python", B / "venvs/wafer/bin/python"):
    if p.exists():
        st = p.stat()
        print(f"  {p}: ino={st.st_ino} size={st.st_size} mtime={st.st_mtime:.0f}")
    else:
        print(f"  {p}: MISSING")

print("\n=== where does each python say site-packages is ===")
for p in (A / "venvs/wafer/bin/python", B / "venvs/wafer/bin/python"):
    if not p.exists():
        continue
    out = subprocess.run([str(p), "-c",
                          "import sys,swift,os;print(os.path.dirname(swift.__file__))"],
                         capture_output=True, text=True)
    print(f"  {p}:\n    {out.stdout.strip() or out.stderr.strip()[:200]}")

print("\n=== top-level of /root/autodl-tmp/wafer-vlm ===")
if A.exists():
    for e in sorted(A.iterdir()):
        if e.is_symlink():
            print(f"  {e.name} -> {os.readlink(e)}")
        elif e.is_dir():
            n = sum(1 for _ in e.rglob("*")) if e.name not in ("venvs",) else "many"
            print(f"  {e.name}/ (dir)")
        else:
            print(f"  {e.name} ({e.stat().st_size/2**20:.1f}M)")

print("\n=== which python does each queue script invoke ===")
for sh in sorted((B / "projects/wafer-defect-vlm/scripts").glob("*queue*.sh"))[-5:]:
    txt = sh.read_text(encoding="utf-8", errors="replace")
    pys = sorted({t for t in txt.split() if "bin/python" in t})
    print(f"  {sh.name}: {pys if pys else '(no explicit python)'}")

print("\n=== du of the two trees (top level only) ===")
for mnt in ("/root/autodl-tmp/wafer-vlm", "/root/autodl-fs/wafer-vlm"):
    out = subprocess.run(["du", "-sh", "--one-file-system", mnt],
                         capture_output=True, text=True)
    print(f"  {out.stdout.strip()}")
