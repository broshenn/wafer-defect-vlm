"""Assemble the Base / SFT / RL comparison report from the per-run result files.

A metric that was never measured is printed as ``not run`` rather than as a
zero: a zero and a missing measurement look identical in a table, and the
difference matters when the whole point of the table is to say what the
fine-tuning bought. Missing values are also listed separately so a reader can
see at a glance which rows of the comparison are incomplete.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FIELDS = [
    ("classification.accuracy", "classification accuracy"),
    ("classification.macro_f1", "classification macro-F1"),
    ("classification.macro_f1_95ci", "macro-F1 95% CI"),
    ("classification.invalid_predictions", "off-vocabulary answers"),
    ("structured.field_accuracy.defect_type", "structured defect_type acc"),
    ("structured.field_accuracy.radial_zone", "structured radial_zone acc"),
    ("structured.clock_circular_mae_sectors", "clock circular MAE (sectors)"),
    ("structured.size_mae_r", "size MAE (R)"),
    ("caption.must_hit_rate", "caption must-hit rate"),
    ("caption.empty_rate", "caption empty rate"),
    ("caption.root_cause_hallucination_rate", "root-cause hallucination rate"),
    ("robustness.accuracy", "robustness accuracy"),
    ("robustness.flip_rate_vs_clean", "flip rate vs clean"),
    ("retrieval.mAP@10", "retrieval mAP@10"),
    ("retrieval.nDCG@10", "retrieval nDCG@10"),
    ("retrieval.Recall@10", "retrieval Recall@10"),
]


def dig(payload: dict, dotted: str):
    node = payload
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def fmt(value) -> str:
    if value is None:
        return "not run"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (list, tuple)):
        if len(value) == 2 and all(isinstance(v, (int, float)) for v in value):
            return f"[{value[0]:.4f}, {value[1]:.4f}]"
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True,
                        help="name=path to a run report JSON; repeat for each run")
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown", required=True)
    parser.add_argument("--provenance", action="append", default=[],
                        help="path to a JSON file whose contents are appended verbatim")
    parser.add_argument("--note", action="append", default=[])
    args = parser.parse_args()

    runs: dict[str, dict] = {}
    for spec in args.run:
        name, _, path = spec.partition("=")
        report_path = Path(path)
        runs[name] = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
        runs[name]["_source"] = str(report_path)
        runs[name]["_present"] = report_path.is_file()

    names = list(runs)
    missing: dict[str, list[str]] = {name: [] for name in names}
    rows = []
    for dotted, label in FIELDS:
        values = {}
        for name in names:
            value = dig(runs[name], dotted)
            values[name] = value
            if value is None:
                missing[name].append(label)
        rows.append((label, values))

    report = {
        "runs": {name: {"source": runs[name]["_source"], "present": runs[name]["_present"],
                        "requests_scored_from": runs[name].get("requests_scored_from"),
                        "requests_expected": runs[name].get("requests_expected")}
                 for name in names},
        "metrics": {label: {name: values[name] for name in names} for label, values in rows},
        "not_measured": missing,
        "notes": args.note,
    }
    for path in args.provenance:
        source = Path(path)
        if source.is_file():
            report.setdefault("provenance", {})[source.name] = json.loads(source.read_text(encoding="utf-8"))

    header = "| metric | " + " | ".join(names) + " |"
    divider = "| --- | " + " | ".join("---" for _ in names) + " |"
    lines = [header, divider]
    for label, values in rows:
        lines.append("| " + label + " | " + " | ".join(fmt(values[name]) for name in names) + " |")

    markdown = ["# Base / SFT comparison", "", "```", *lines, "```", ""]
    markdown.append("Metrics reported as `not run` were never measured; they are not zeros.")
    for name in names:
        if not runs[name]["_present"]:
            markdown.append(f"- **{name}**: no result file at `{runs[name]['_source']}`")
        elif missing[name]:
            markdown.append(f"- **{name}** missing: {', '.join(missing[name])}")
    for note in args.note:
        markdown.append(f"- {note}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.markdown).write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
