"""Resolve the 4.5K-vs-readable contradiction, and locate the third path.

/root/autodl-fs/wafer-vlm/venvs is 4.5K by du yet its site-packages was read
successfully, so something inside must be a link. And an editable install points
at /autodl-fs/data/wafer-vlm/... , a path that is not the project root at all.
Both need naming before anything is moved.
"""
import os
import subprocess
from pathlib import Path

for root in ("/root/autodl-fs/wafer-vlm", "/root/autodl-tmp/wafer-vlm"):
    print(f"=== {root} : links at depth 1-2 ===")
    base = Path(root)
    for e in sorted(base.iterdir()):
        if e.is_symlink():
            tgt = os.readlink(e)
            alive = Path(tgt).exists() if not tgt.startswith("/autodl-fs") else "?"
            print(f"  {e.name} -> {tgt}   (target exists: {alive})")
        elif e.is_dir():
            subs = [f"{x.name} -> {os.readlink(x)}" for x in sorted(e.iterdir())
                    if x.is_symlink()]
            real = [x.name for x in sorted(e.iterdir()) if not x.is_symlink()][:6]
            print(f"  {e.name}/  links: {subs if subs else 'none'}"
                  f"   real entries: {real}")
    print()

print("=== does the editable target exist? ===")
for p in ("/autodl-fs/data/wafer-vlm/projects/wafer-defect-vlm/src",
          "/root/autodl-fs/wafer-vlm/projects/wafer-defect-vlm/src",
          "/root/autodl-tmp/wafer-vlm/projects/wafer-defect-vlm/src"):
    q = Path(p)
    print(f"  {p}: exists={q.exists()}"
          + (f" same-as-fs-root-src="
             f"{q.stat().st_ino == Path('/root/autodl-fs/wafer-vlm/projects/wafer-defect-vlm/src').stat().st_ino}"
             if q.exists() else ""))

print("\n=== what the editable finder maps swift to ===")
sp = Path("/root/autodl-fs/wafer-vlm/venvs/wafer/lib/python3.12/site-packages")
f = sp / "__editable___ms_swift_4_6_0_dev0_finder.py"
if f.exists():
    txt = f.read_text(errors="replace")
    i = txt.find("MAPPING")
    print("  " + txt[i:i + 400].replace("\n", "\n  ") if i >= 0 else "  no MAPPING")

print("\n=== site-packages of the venv actually used by the running job ===")
out = subprocess.run(["/root/autodl-tmp/wafer-vlm/venvs/wafer/bin/python", "-c",
                      "import sys;print(sys.prefix);print(sys.executable);"
                      "print([p for p in sys.path if 'site-packages' in p])"],
                     capture_output=True, text=True)
print(f"  {out.stdout.strip()}{out.stderr.strip()[:300]}")
