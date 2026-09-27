"""Compare two ms-swift result files row by row.

Used to check whether the merged model reproduces the adapter's generations.
Rows are matched by response position, which is sound because both runs consume
the same request file in the same order.

Two statistics are reported because exact string equality is the wrong sole
criterion here. The answers are free-form captions and JSON, so a single token
flipped by a rounding difference cascades through the rest of the generation:
the same adapter rendered in 4-bit and in bf16 agrees with itself on only ~43%
of rows. ``exact_agreement`` therefore measures numeric reproducibility, while
``functional_agreement`` — the defect_type the answer asserts, or the class
label for a classification reply — measures whether the two models actually say
the same thing. The noise floor for exact agreement should be established by
comparing one model against itself at a different precision, not assumed to be
1.0.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

LABELS = ("Edge_Loc", "Edge_Ring", "Center", "Donut", "Loc", "Near_full",
          "Random", "Scratch", "none")

JSON_FIELD = re.compile(r'"defect_type"\s*:\s*"([^"]+)"')


def responses(path: str | Path) -> list[str]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return [str(row.get("response", "")) for row in rows]


def answer_of(response: str) -> str:
    """The part of the generation after the thinking block, if there is one."""
    return response.split("</think>")[-1].strip()


def functional_signature(response: str) -> str:
    """What the answer asserts, independent of wording or tokenisation.

    A structured reply carries its class in a JSON field; a classification
    reply is the label itself. Captions have no single field, so they fall back
    to their text, where agreement is necessarily weaker.
    """
    answer = answer_of(response)
    match = JSON_FIELD.search(answer)
    if match:
        return match.group(1)
    for label in LABELS:
        if label in answer:
            return label
    return answer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    a, b = responses(args.a), responses(args.b)
    if len(a) != len(b):
        report = {
            "comparable": False,
            "reason": f"row count differs: {len(a)} vs {len(b)}",
            "rows_a": len(a), "rows_b": len(b),
        }
    else:
        exact = [x.strip() == y.strip() for x, y in zip(a, b)]
        functional = [functional_signature(x) == functional_signature(y) for x, y in zip(a, b)]
        report = {
            "comparable": True,
            "rows": len(a),
            "exact_matches": sum(exact),
            "agreement": sum(exact) / len(a) if a else None,
            "functional_matches": sum(functional),
            "functional_agreement": sum(functional) / len(a) if a else None,
            "mismatch_indices": [i for i, same in enumerate(exact) if not same][:20],
            "criterion_note": (
                "exact_agreement is a numeric-reproducibility measure, not a "
                "correctness one: the same model at a different precision does "
                "not reach 1.0 on free-form answers. Compare functional_agreement "
                "against the same-model precision floor before concluding the "
                "merge changed behaviour."
            ),
        }
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("agreement") == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
