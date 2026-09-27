"""Prove nothing was lost: extract the four pull tarballs and look for every file they
contain, by content, in the reorganised tree.

A file count cannot do this -- it cannot tell "moved" from "changed", and it is exactly
the instrument that produced three wrong sentences earlier tonight. Content can: the
tarballs are the pull exactly as it arrived, so every file in them must have a byte-for
-byte counterpart somewhere in the new tree, unless it is a cache that was deliberately
deleted (those are checked against the cache patterns and reported separately).
"""
import hashlib
import pathlib
import shutil
import tarfile

R = pathlib.Path(__file__).resolve().parent
# The script lives in archive/tooling/, so the project root is whatever ancestor holds
# archive/tarballs -- do not assume a fixed depth.
while not (R / "archive/tarballs").is_dir() and R.parent != R:
    R = R.parent
TMP = R / "_verify"
CACHE = ("__pycache__", ".pytest_cache", ".ruff_cache", ".egg-info")


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


# ---- what the pull contained --------------------------------------------------------
if TMP.exists():
    shutil.rmtree(TMP)
TMP.mkdir()
pulled = {}
for t in sorted((R / "archive/tarballs").glob("bundle*.tar.gz")):
    with tarfile.open(t) as tf:
        tf.extractall(TMP)
for p in sorted(TMP.rglob("*")):
    if p.is_file():
        pulled[str(p.relative_to(TMP)).replace("\\", "/")] = md5(p)
print(f"files inside the four tarballs: {len(pulled)}")

# ---- what the tree holds now --------------------------------------------------------
current = {}
for p in sorted(R.rglob("*")):
    # Skip only this tooling directory. An underscore-prefix test looks like it does that
    # and instead also skips __init__.py, which is how the first run of this check
    # reported a missing file that was present all along.
    if p.is_file() and "_verify" not in p.parts and "archive/tooling" not in str(p).replace("\\", "/"):
        current.setdefault(md5(p), []).append(str(p.relative_to(R)).replace("\\", "/"))
print(f"files in the reorganised tree (excluding these scripts): "
      f"{sum(len(v) for v in current.values())}")

missing, deleted_caches = [], []
for rel, h in pulled.items():
    if h in current:
        continue
    if any(c in rel for c in CACHE):
        deleted_caches.append(rel)
    else:
        missing.append(rel)

print(f"\nfrom the pull, still present by content: {len(pulled) - len(missing) - len(deleted_caches)}")
print(f"absent, and a build cache (deleted on purpose): {len(deleted_caches)}")
for rel in deleted_caches:
    print(f"   {rel}")
print(f"absent and NOT a cache: {len(missing)}")
for rel in missing:
    print(f"   !! {rel}")

# ---- the earlier local files, which the tarballs do not contain ---------------------
arch = {}
for d in sorted((R / "archive").iterdir()):
    if d.is_dir():
        arch[d.name] = len([p for p in d.rglob("*") if p.is_file()])
print("\narchive/ contents:")
for k, v in arch.items():
    print(f"   {v:4d}  {k}")

shutil.rmtree(TMP)
print("\nVERDICT:", "nothing lost" if not missing else f"{len(missing)} file(s) unaccounted")
raise SystemExit(1 if missing else 0)
