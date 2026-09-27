"""Record the real effective batch per run, and the confound it exposes.

Found by reading the live argv of the running G=32 job, then the training logs.
Two separate problems, and only one of them is about configuration:

1. THE RECORD IS WRONG. 29_grpo_train.sh's result-writing heredoc contains
   "grad_accum": 4 as a literal. It is a constant, not a measurement, so it
   cannot disclose a mismatch -- which is why this survived five runs. The
   training logs carry the real swift argv:

       GSPO G=8  lr1e-5   ran accum 8   record says 4
       GSPO G=8  lr5e-5   ran accum 8   record says 4
       GSPO G=32 lr5e-5   ran accum 32  record says 4

   Every effective-batch figure derived from those records is wrong by the same
   factor. The GRPO G=4 records are right, but only because the constant
   happens to equal their real value -- they are not evidence either.

2. THE DESIGN CONFOUNDS GROUP SIZE WITH BATCH SIZE. Accumulation was set equal
   to the group size in every RL run (4/4, 8/8, 32/32), so

       effective batch = num_generations x per_device_batch(1) x accum = G^2

       G=4  -> 16      G=8 -> 64      G=32 -> 1024

   Group size was therefore never varied alone: a 4x change in G is a 16x change
   in effective batch, and 8x in G is 64x. Every group-size claim in section
   5.2.5, and the G=8 -> G=32 paired contrasts, compare runs that differ in both.
   This is a SECOND confound, independent of the IS-level one already recorded
   (token-level is always G=4, sequence-level always G=8 or G=32).

   It does NOT invalidate the learning-rate contrasts, which hold G and hence
   accumulation fixed: GRPO G=4 at both lrs, GSPO G=8 at both lrs, and GSPO G=32
   at both lrs once queue 41 lands (it runs accum 32, matching the completed
   G=32 run). The lr results stand.

Also recorded: what the fix is, and why it is deferred. The heredoc lives in
29_grpo_train.sh, which queue 41 is executing right now. Editing a .sh while
bash is reading it shifts the resume offset -- this project has already lost two
runs to exactly that. So the fix waits until every run has stopped.
"""
import json
import re
import time
from pathlib import Path

R = Path("/root/autodl-fs/wafer-vlm")
LOGS, REPORTS = R / "logs", R / "outputs/reports"
OUT = REPORTS / "effective_batch_correction.json"

ARGS = ("gradient_accumulation_steps", "num_generations",
        "per_device_train_batch_size", "learning_rate",
        "importance_sampling_level")

RUNS = [("GRPO G=4 lr1e-5", "grpo_train_result.json", None),
        ("GRPO G=4 lr5e-5", "qwen35_9b_grpo_lr5e5_train_result.json",
         "29_qwen35_9b_grpo_lr5e5_train.log"),
        ("GSPO G=8 lr1e-5", "gspo_lr1e5_train_result.json",
         "29_gspo_lr1e5_train.log"),
        ("GSPO G=8 lr5e-5", "qwen35_9b_gspo_v1_train_result.json",
         "29_grpo_train.log"),
        ("GSPO G=32 lr5e-5", "qwen35_9b_gspo_g32_train_result.json",
         "29_qwen35_9b_gspo_g32_train.log"),
        ("GSPO G=32 lr1e-5", "qwen35_9b_gspo_g32_lr1e5_train_result.json",
         "29_qwen35_9b_gspo_g32_lr1e5_train.log")]


def real_argv(p):
    if p is None or not p.is_file():
        return None
    txt = p.read_text(encoding="utf-8", errors="replace")
    out = {}
    for a in ARGS:
        m = re.search(r"--" + a + r"[= ]([^\s,]+)", txt)
        if m:
            out[a] = m.group(1)
    return out or None


table, wrong = [], []
for label, recname, logname in RUNS:
    rp = REPORTS / recname
    rec = (json.loads(rp.read_text(encoding="utf-8")).get("config", {})
           if rp.is_file() else None)
    real = real_argv(LOGS / logname if logname else None)
    row = {"run": label}
    if real:
        g = int(real["num_generations"])
        a = int(real["gradient_accumulation_steps"])
        pdb = int(real.get("per_device_train_batch_size", 1))
        row.update({"num_generations": g, "grad_accum_real": a,
                    "per_device_batch": pdb, "effective_batch": g * pdb * a,
                    "learning_rate": real["learning_rate"],
                    "importance_sampling_level": real["importance_sampling_level"],
                    "source": f"logs/{logname}"})
        if rec is not None:
            row["grad_accum_in_record"] = rec.get("grad_accum")
            row["record_agrees"] = rec.get("grad_accum") == a
            if not row["record_agrees"]:
                wrong.append((label, a, rec.get("grad_accum")))
    else:
        row["num_generations"] = rec.get("num_generations") if rec else None
        row["grad_accum_in_record"] = rec.get("grad_accum") if rec else None
        row["note"] = ("no training log retained for this run; its real "
                       "accumulation is not recoverable from the record either, "
                       "because that field is a constant")
    table.append(row)
    print(f"  {label:<20} real={row.get('grad_accum_real', '?'):>4} "
          f"record={row.get('grad_accum_in_record', '?'):>4} "
          f"eff={row.get('effective_batch', '?'):>5}")

rec_out = {
    "recorded": time.strftime("%Y-%m-%d %H:%M:%S"),
    "headline": "The training records' grad_accum field is a hardcoded constant "
                "(4), not a measurement. It is wrong for the GSPO G=8 and G=32 "
                "runs. Separately, accumulation was set equal to group size in "
                "every RL run, so effective batch grew as G^2 and group size was "
                "never varied alone.",
    "how_found": "Read the live swift argv of the running G=32 job via "
                 "/proc/<pid>/cmdline while checking whether a concurrent run "
                 "would fit on the card. The argv showed --gradient_accumulation"
                 "_steps 32 against a record claiming 4.",
    "config_defect": {
        "file": "projects/wafer-defect-vlm/scripts/29_grpo_train.sh",
        "line_contains": '"grad_accum": 4,',
        "nature": "literal constant inside the result-writing heredoc; reads no "
                  "environment variable",
        "fix": 'read os.environ.get("GRAD_ACCUM", 4) instead',
        "deferred_because": "queue 41 is executing this file; editing a running "
                            ".sh shifts bash's resume offset and this project has "
                            "already lost two runs that way. Fix after all runs "
                            "have stopped.",
        "mismatches": [{"run": l, "ran": a, "record_claims": c}
                       for l, a, c in wrong],
    },
    "design_confound": {
        "statement": "accumulation was set equal to group size in every RL run, "
                     "so effective batch = G^2 x per_device_batch(1)",
        "effective_batch_by_G": {"4": 16, "8": 64, "32": 1024},
        "consequence": "every group-size contrast moves group size and effective "
                       "batch size together: G=8 -> G=32 is a 4x change in G and "
                       "a 16x change in effective batch. Section 5.2.5's "
                       "group-size reading and the G=8 -> G=32 paired contrasts "
                       "cannot attribute their difference to group size alone.",
        "what_survives": "the learning-rate contrasts, which hold G -- and hence "
                         "accumulation and effective batch -- fixed within each "
                         "comparison: GRPO G=4 at both lrs, GSPO G=8 at both lrs, "
                         "GSPO G=32 at both lrs.",
        "relationship_to_the_other_confound": "independent of the IS-level "
            "confound already recorded (token-level always G=4, sequence-level "
            "always G=8 or G=32). The two together mean no contrast in this "
            "project varies IS level alone, or group size alone.",
    },
    "runs": table,
    "queue_41_config_is_consistent": "queue 41 passes GRAD_ACCUM=32 for G=32, "
        "matching the completed G=32 lr5e-5 run's real accum of 32. Its lr "
        "contrast at G=32 is therefore as clean as the other lr contrasts, and "
        "the run was NOT restarted.",
    "downstream_effect_on_documents": "section 5.2 must state the confound "
        "wherever group size is read as a cause; the reward-std table's values "
        "are group-internal statistics and are not affected by accumulation, but "
        "any claim that they explain the SCORE differences is.",
}

OUT.write_text(json.dumps(rec_out, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nwrote {OUT}")
print(f"records wrong: {len(wrong)}")
