"""Add the two newly-recorded keys to radial_zone_audit.json, and nothing else.

The audit record is consumed by FINAL_REPORT section 6c. Re-running the tool
writes the whole file, so a mistake in the run arguments would silently change
the published re-grading table. This script only ever adds keys: it refuses to
write unless the re-run reproduced the existing gold_distributions and runs
exactly, so a mismatch stops the merge instead of overwriting evidence.
"""
import json
import sys
from pathlib import Path

RECORD = Path("/root/autodl-fs/wafer-vlm/outputs/reports/radial_zone_audit.json")
FRESH = Path("/tmp/audit_new.json")
NEW_KEYS = ("centroid_radius_by_class", "center_share")

old = json.loads(RECORD.read_text(encoding="utf-8"))
new = json.loads(FRESH.read_text(encoding="utf-8"))

for key in ("gold_distributions", "runs"):
    if old.get(key) != new.get(key):
        print(f"REFUSING TO WRITE: {key} differs between the record and the re-run")
        print(f"  existing: {json.dumps(old.get(key), sort_keys=True)[:300]}")
        print(f"  re-run  : {json.dumps(new.get(key), sort_keys=True)[:300]}")
        sys.exit(1)
    print(f"  {key}: identical")

missing = [k for k in NEW_KEYS if k not in new]
if missing:
    sys.exit(f"REFUSING TO WRITE: the re-run is missing {missing}")

for key in NEW_KEYS:
    if key in old:
        print(f"  {key}: already present, replacing")
    old[key] = new[key]

RECORD.write_text(json.dumps(old, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nwrote {RECORD}")
print("  center_share:", json.dumps(old["center_share"], ensure_ascii=False))
print("  median centroid radius by class:")
for cls, entry in old["centroid_radius_by_class"].items():
    print(f"    {cls:12s} n={entry['n']:3d} median={entry['median']:.4f}")
