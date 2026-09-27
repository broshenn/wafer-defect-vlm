"""Is /root/autodl-tmp/wafer-vlm a bind mount, a duplicate, or a hardlink farm?

The venv interpreter has the SAME inode under both roots, but both report swift
from the autodl-tmp path. Same inode means same physical file; a duplicate 29G
tree would have different inodes. So the answer decides whether the ephemeral
disk holds unique bytes (a real loss risk) or a view of the persistent one.
"""
import os
import subprocess
from pathlib import Path

print("=== mounts touching the two roots ===")
mounts = Path("/proc/mounts").read_text()
for ln in mounts.split("\n"):
    if "autodl" in ln:
        print(f"  {ln}")

A = Path("/root/autodl-tmp/wafer-vlm")
B = Path("/root/autodl-fs/wafer-vlm")

print("\n=== per-subdir: inode of the dir itself under both roots ===")
print(f"  {'subdir':<14} {'tmp ino':>14} {'fs ino':>14}  verdict")
for name in ("venvs", "src", "models", "data", "benchmarks", "outputs", "logs",
             "projects", "secrets", "cache"):
    a, b = A / name, B / name
    ai = a.stat().st_ino if a.exists() else None
    bi = b.stat().st_ino if b.exists() else None
    if ai is None:
        verdict = "tmp-only"
    elif bi is None:
        verdict = "fs-only"
    elif ai == bi:
        verdict = "SAME FILE (bind/hardlink)"
    else:
        verdict = "distinct copies"
    print(f"  {name:<14} {str(ai):>14} {str(bi):>14}  {verdict}")

print("\n=== sizes, per root, no cross-filesystem descent ===")
for root in (A, B):
    print(f"  -- {root}")
    for name in ("venvs", "src", "models", "data", "benchmarks"):
        p = root / name
        if not p.exists():
            continue
        out = subprocess.run(["du", "-sh", "--one-file-system", str(p)],
                             capture_output=True, text=True).stdout.strip()
        print(f"     {out}")

print("\n=== how swift is wired into the venv ===")
sp = B / "venvs/wafer/lib/python3.12/site-packages"
if not sp.exists():
    cands = sorted((B / "venvs/wafer/lib").glob("python3.*/site-packages"))
    sp = cands[0] if cands else sp
print(f"  site-packages: {sp}")
for pat in ("*.pth", "__editable__*", "swift.egg-link", "*ms_swift*"):
    for f in sorted(sp.glob(pat)):
        try:
            body = f.read_text(errors="replace").strip()[:300]
        except Exception:
            body = "(unreadable)"
        print(f"    {f.name}: {body!r}")

print("\n=== free space ===")
for m in ("/root/autodl-tmp", "/root/autodl-fs"):
    st = os.statvfs(m)
    print(f"  {m}: {st.f_bavail*st.f_frsize/2**30:.1f}G free")
