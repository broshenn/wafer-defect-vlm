"""Map every report file to the config recorded inside its own run, so no
learning-rate or group-size label is taken from a filename or from the document.

A mislabelled column would make every comparison in the affected tables wrong in a
way that no internal arithmetic check could catch, so the label gets its own check
here: the value in the doc's table must be traceable to a record whose own config
says what the column header claims.
"""
import json
from pathlib import Path

R = Path("/root/autodl-fs/wafer-vlm")
REP = R / "outputs/reports"

reports = sorted(REP.glob("*__report.json"))
print(f"{len(reports)} report files\n")

KEYS = ["learning_rate", "num_generations", "importance_sampling_level",
        "grad_accum", "grad_accum_from_log", "per_device_batch", "seed",
        "beta", "temperature", "max_steps"]

for f in reports:
    tag = f.name[:-len("__report.json")]
    d = json.loads(f.read_text(encoding="utf-8"))
    cls = d.get("classification") or {}
    tr = REP / f"{tag}_train_result.json"
    cfg = {}
    if tr.is_file():
        cfg = (json.loads(tr.read_text(encoding="utf-8")) or {}).get("config") or {}
    else:
        cfg = d.get("config") or {}
    print(f"{tag}")
    print(f"    acc={cls.get('accuracy')}  macro_f1={cls.get('macro_f1')}")
    print(f"    cfg: " + "  ".join(f"{k}={cfg.get(k)}" for k in KEYS
                                  if cfg.get(k) is not None))
    print(f"    train_result present: {tr.is_file()}")
    print()
