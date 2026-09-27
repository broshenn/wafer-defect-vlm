"""Verify a prepared wafer dataset: image geometry, exact palette, features, leakage.

Palette check packs RGB into a uint32 before np.unique; np.unique(axis=0) on
604k rows is ~50x slower and dominated the runtime on the network mount.
"""
import argparse, json, sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ALLOWED_U32 = {0x000000, 0x00FF00, 0xFF0000}
REQUIRED_FEATURES = {
    "status", "valid_count", "raw_defect_count", "defect_count", "filtered_points_removed",
    "defect_ratio", "wafer_center_xy", "wafer_radius", "radial_zone", "radial_density",
    "angular_probability", "resultant_length", "angle_entropy_bits",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root")
    ap.add_argument("--resolution", type=int, default=448)
    ap.add_argument("--limit", type=int, default=0, help="check only the first N images")
    args = ap.parse_args()

    root = Path(args.root)
    records = [json.loads(l) for l in (root / "manifest.jsonl").open(encoding="utf-8") if l.strip()]
    print("records:", len(records))
    problems = []

    ids = [r["sample_id"] for r in records]
    if len(ids) != len(set(ids)):
        problems.append("duplicate sample_id")

    splits = {n: {r["lot_name"] for r in records if r["split"] == n} for n in ("train", "val", "test")}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        inter = splits[a] & splits[b]
        if inter:
            problems.append("lot leak %s/%s: %s" % (a, b, sorted(inter)[:5]))
    print("split counts:", dict(Counter(r["split"] for r in records)))
    print("class counts:", dict(sorted(Counter(r["failure_type"] for r in records).items())))

    missing_feat, bad_palette, bad_size, missing_img = Counter(), [], [], []
    statuses = Counter()
    defect_range = [10 ** 9, -1]
    checked = 0
    sample = None
    for r in records:
        feats = r.get("features", {})
        missing_feat.update(REQUIRED_FEATURES - set(feats))
        statuses[feats.get("status")] += 1
        dc = feats.get("defect_count")
        if isinstance(dc, int):
            defect_range[0] = min(defect_range[0], dc)
            defect_range[1] = max(defect_range[1], dc)
        if args.limit and checked >= args.limit:
            continue
        p = Path(r["image_path"])
        if not p.exists():
            missing_img.append(r["sample_id"])
            continue
        checked += 1
        with Image.open(p) as im:
            if im.size != (args.resolution, args.resolution):
                bad_size.append((r["sample_id"], im.size))
            a = np.asarray(im.convert("RGB"), dtype=np.uint32)
            packed = (a[..., 0] << 16) | (a[..., 1] << 8) | a[..., 2]
            extra = set(np.unique(packed).tolist()) - ALLOWED_U32
            if extra:
                bad_palette.append((r["sample_id"], ["#%06x" % c for c in sorted(extra)[:3]]))
        if sample is None and feats.get("status") == "ok":
            sample = r

    if missing_feat:
        problems.append("missing feature keys: %s" % dict(missing_feat))
    if missing_img:
        problems.append("missing images: %d e.g. %s" % (len(missing_img), missing_img[:3]))
    if bad_size:
        problems.append("wrong size: %d e.g. %s" % (len(bad_size), bad_size[:3]))
    if bad_palette:
        problems.append("palette outside {black,green,red}: %d e.g. %s" % (len(bad_palette), bad_palette[:3]))

    print("images checked:", checked)
    print("features.status:", dict(statuses))
    print("defect_count range:", defect_range[0], "..", defect_range[1])
    if sample:
        f = sample["features"]
        print("sample:", json.dumps({
            "sample_id": sample["sample_id"], "failure_type": sample["failure_type"],
            "split": sample["split"], "matrix_shape": sample["matrix_shape"],
            "radial_zone": f.get("radial_zone"), "clock_sector": f.get("clock_sector"),
            "centroid_radius_r": round(float(f.get("centroid_radius_r") or 0), 4),
            "wafer_radius": round(float(f.get("wafer_radius") or 0), 4),
            "defect_ratio": round(float(f.get("defect_ratio") or 0), 6),
            "radial_bins": len(f.get("radial_density") or []),
            "angular_bins": len(f.get("angular_probability") or []),
        }, ensure_ascii=False))

    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print("  -", p)
        return 1
    print("\nOK: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
