"""Re-grade `radial_zone` against location-based definitions of the same field.

`calculate_static_features` sets `radial_zone = _zone(centroid_r)` -- the zone of
the defect's *centre of mass*. That is not where the defect is. For any pattern
symmetric about the wafer centre the centre of mass sits at the centre, so the
label collapses to "center" by arithmetic rather than by observation: in the
frozen benchmark `Edge_Ring` has a median centroid radius of 0.063 and 83.7% of
samples land in the "center" bucket. A model that answers "edge" for an edge ring
is scored wrong for being right about the wafer.

So before calling the model's 0.264 a failure, re-grade the identical predictions
against definitions that measure where the defect actually sits:

  centroid      the published rule, reproduced for reference
  mean_radius   mean radius of the defect pixels
  outer_extent  mean radius plus half the radial span -- how far it reaches
  curate_rule   the project's own teacher-acceptance rule (curate.py), which
                accepts "full" for Near_full; the fairest test of whether the
                model learned what it was actually asked for

Features are read from the prepared manifest rather than recomputed from the
PNGs: the published gold is byte-identical to the manifest (verified below, delta
exactly 0), whereas re-deriving geometry from the images is lossy because wafer
maps are not square and `matrix_to_image` resizes them to 448x448, distorting
radial ratios. An earlier version of this script did recompute from the PNGs and
failed its own reproduction check (0/251 exact, max delta 0.585) -- the check is
kept so that failure cannot pass silently again.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "projects" / "wafer-defect-vlm" / "src"))

from wafer_vlm.evaluate import align_predictions, response_text  # noqa: E402
from wafer_vlm.utils import extract_json_object, read_jsonl  # noqa: E402

ZONE_EDGES = (0.35, 0.72)


def zone(radius_ratio: float | None) -> str | None:
    """The same thresholds `_zone` uses, applied to any radial measure."""
    if radius_ratio is None:
        return "none"
    if radius_ratio < ZONE_EDGES[0]:
        return "center"
    if radius_ratio < ZONE_EDGES[1]:
        return "middle"
    return "edge"


def definitions(features: dict, failure_type: str) -> dict[str, str | None]:
    """Four readings of "where is this defect", including the project's own rule."""
    span = features.get("radial_span_r")
    mean_r = features.get("mean_radius_r")
    outer = None if (mean_r is None or span is None) else mean_r + span / 2
    published = features.get("radial_zone")
    return {
        "centroid": published if published is not None else zone(features.get("centroid_radius_r")),
        "mean_radius": zone(mean_r),
        "outer_extent": zone(outer),
        # curate.py accepts "full" for Near_full and "none" for the no-defect
        # class; anything else must match the published zone exactly.
        "curate_rule": published,
        "_special": "full" if failure_type == "Near_full" else (
            "none" if failure_type == "none" else None),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--run", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    gold_rows = read_jsonl(root / "benchmarks" / "wafer_bench_v1" / "structured.jsonl")
    requests = read_jsonl(root / "benchmarks" / "wafer_bench_v1" / "all_requests.jsonl")
    manifest = {row["sample_id"]: row for row in
                read_jsonl(root / "data" / "prepared_v1" / "manifest.jsonl")}

    print("=" * 78)
    print("STEP 1 -- the manifest must reproduce the published gold exactly")
    print("=" * 78)
    deltas = []
    for row in gold_rows:
        published = row["gold"].get("centroid_radius_r")
        if published is None:
            continue
        deltas.append(abs(manifest[row["sample_id"]]["features"]["centroid_radius_r"] - published))
    print(f"  samples {len(deltas)}  max |delta| {max(deltas):.3e}  "
          f"exact {sum(1 for d in deltas if d == 0)}/{len(deltas)}")
    if max(deltas) > 1e-9:
        print("  >>> manifest does not reproduce the gold; the re-grade below is NOT comparable")
        return 1

    golds, special = {}, {}
    radii_by_class: dict[str, list[float]] = collections.defaultdict(list)
    for row in gold_rows:
        sid = row["sample_id"]
        record = manifest[sid]
        golds[sid] = definitions(record["features"], record["failure_type"])
        special[sid] = golds[sid].pop("_special")
        # The report argues that the published radial_zone label collapses to
        # "center" by arithmetic, citing the median centroid radius of each
        # pattern as the evidence. Those medians used to be literals typed into
        # prose in two different files; recording them here makes the argument
        # checkable and stops the two copies from drifting apart.
        radius = record["features"].get("centroid_radius_r")
        if radius is not None:
            radii_by_class[str(record["failure_type"])].append(radius)

    print()
    for name in ("centroid", "mean_radius", "outer_extent"):
        counts = collections.Counter(v[name] for v in golds.values())
        top = max(counts.values()) / len(golds)
        print(f"  gold[{name:13s}] {dict(counts)}   majority floor {top:.4f}")

    # How often the published rule answers "center", by both denominators.
    # The prose and the record previously each used a different one -- 210/252
    # is 83.33% but 210/251, dropping the single unlabelled row, is 83.67% --
    # so both are written down under explicit names rather than left to
    # whichever number happened to be typed.
    centroid_zones = collections.Counter(v["centroid"] for v in golds.values())
    center_hits = centroid_zones.get("center", 0)
    n_all = len(golds)
    n_scored = n_all - centroid_zones.get("none", 0)

    result: dict[str, object] = {
        "manifest_reproduces_gold": True,
        "gold_distributions": {
            name: {"counts": dict(collections.Counter(v[name] for v in golds.values())),
                   "majority_floor": max(collections.Counter(
                       v[name] for v in golds.values()).values()) / len(golds)}
            for name in ("centroid", "mean_radius", "outer_extent")},
        "centroid_radius_by_class": {
            cls: {"n": len(vals), "median": statistics.median(vals)}
            for cls, vals in sorted(radii_by_class.items())},
        "center_share": {
            "definition": ("gold rows whose published radial_zone is 'center' "
                           "under the _zone(centroid_r) rule"),
            "count": center_hits,
            "n_all": n_all,
            "n_scored": n_scored,
            "of_all": center_hits / n_all,
            "of_scored": center_hits / n_scored,
        },
        "runs": {},
    }

    for spec in args.run:
        name, _, path = spec.partition("=")
        aligned = align_predictions(requests, read_jsonl(root / path))
        predicted: dict[str, str] = {}
        for row in aligned:
            if row.get("task") != "structured":
                continue
            try:
                js = extract_json_object(response_text(row))
            except ValueError:
                continue
            value = js.get("radial_zone")
            if value is not None:
                predicted[row["sample_id"]] = str(value).strip().lower()

        print()
        print("=" * 78)
        print(f"RUN {name}   parsed radial_zone predictions {len(predicted)}")
        print("=" * 78)
        print(f"  predicted  {dict(collections.Counter(predicted.values()))}")
        per_run = {}
        for definition in ("centroid", "mean_radius", "outer_extent"):
            hits = sum(1 for sid, p in predicted.items()
                       if sid in golds and golds[sid][definition] == p)
            total = sum(1 for sid in predicted if sid in golds)
            floor = result["gold_distributions"][definition]["majority_floor"]
            acc = hits / max(total, 1)
            print(f"  vs gold[{definition:13s}] {acc:.4f}  (floor {floor:.4f})  "
                  f"{'beats floor' if acc > floor else 'BELOW floor'}")
            per_run[definition] = {"accuracy": acc, "floor": floor, "hits": hits, "n": total}

        # The project's own acceptance rule: an answer of "full" counts as
        # correct for Near_full, "none" for the no-defect class.
        hits = sum(1 for sid, p in predicted.items() if sid in golds and (
            p == golds[sid]["curate_rule"] or (special.get(sid) and p == special[sid])))
        total = sum(1 for sid in predicted if sid in golds)
        floor_acc = sum(1 for sid in golds if golds[sid]["curate_rule"] == "center") / len(golds)
        print(f"  vs gold[curate_rule  ] {hits/max(total,1):.4f}  (always-'center' {floor_acc:.4f})"
              f"  {'beats' if hits/max(total,1) > floor_acc else 'BELOW'}")
        per_run["curate_rule"] = {"accuracy": hits / max(total, 1),
                                  "floor": floor_acc, "hits": hits, "n": total}
        result["runs"][name] = per_run

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
