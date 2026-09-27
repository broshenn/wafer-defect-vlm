"""The one remaining stale scope-word, and what venvs actually resolves to."""
import os
from pathlib import Path

R = Path("/root/autodl-fs/wafer-vlm")
lines = (R / "LIMITATIONS.md").read_text(encoding="utf-8").split("\n")

print("=== line 409 context (the remaining 四个 run) ===")
for i in range(404, 414):
    mark = ">>" if i + 1 == 409 else "  "
    print(f"{mark} {i+1}: {lines[i]}")

print("\n=== venvs resolution ===")
v = R / "venvs"
print(f"  is_symlink: {v.is_symlink()}")
if v.is_symlink():
    print(f"  -> {os.readlink(v)}  (resolves to {v.resolve()})")
print(f"  models is_symlink: {(R / 'models').is_symlink()}"
      f"  -> {os.readlink(R / 'models') if (R / 'models').is_symlink() else '-'}")
print(f"  outputs is_symlink: {(R / 'outputs').is_symlink()}"
      f"  -> {os.readlink(R / 'outputs') if (R / 'outputs').is_symlink() else '-'}")
print(f"  src    is_symlink: {(R / 'src').is_symlink()}"
      f"  -> {os.readlink(R / 'src') if (R / 'src').is_symlink() else '-'}")

print("\n=== disk ===")
import shutil
for mnt in ("/root/autodl-fs", "/root/autodl-tmp", "/"):
    t, u, f = shutil.disk_usage(mnt)
    print(f"  {mnt}: {u/2**30:.1f}G used of {t/2**30:.1f}G, {f/2**30:.1f}G free")
