"""Close the provenance open item, and record how it can silently reopen.

The item said GRPO at lr 5e-5 was the missing control cell and had not been run.
It has now been run. Left as-is, provenance.json would tell every downstream
reader -- including the report generator -- that a gap exists which was filled.

The reopen hazard is worth recording in the file itself: the deviations are
hand-edited into this JSON, but the script that regenerates it
(26_after_training.sh) still passes the ORIGINAL wrong text for the first entry
("GSPO is not offered by this ms-swift commit"). So re-running that script does
not merely drop this resolution -- it restores a claim already proven false.
Anyone who regenerates provenance must reapply both edits.
"""
import json
from pathlib import Path

P = Path("/root/autodl-fs/wafer-vlm/outputs/reports/provenance.json")
d = json.loads(P.read_text(encoding="utf-8"))

OLD = ("Open item: GRPO was run only at lr 1e-5. The claim that the learning rate "
       "rather than the algorithm caused the lr 5e-5 degradation is therefore "
       "supported only within GSPO (which was run at both 5e-5 and 1e-5). A GRPO "
       "run at lr 5e-5 is the missing control cell and has not been run; see "
       "LIMITATIONS 5.2.2.")

NEW = ("CLOSED 2026-09-16 -- GRPO at lr 5e-5 was the missing control cell, and it "
       "has been run. Records: qwen35_9b_grpo_lr5e5_train_result.json (training), "
       "qwen35_9b_grpo_lr5e5__report.json (benchmark), paired_significance.json "
       "(tests). It degrades the same way GSPO does: macro-F1 0.6197 -> 0.5535, "
       "accuracy -0.0556 with paired McNemar p = 0.0436 over the same 252 rows. "
       "The 'learning rate, not algorithm' reading therefore now has "
       "cross-algorithm replication rather than resting on GSPO alone. "
       "What is still NOT resolved is a different and narrower claim: the paper's "
       "own proposition -- that sequence-level normalisation tolerates a higher "
       "learning rate -- is neither supported nor refuted, because every "
       "token-level run is G=4 and every sequence-level run is G=8, so IS level "
       "and group size move together and no contrast here separates them "
       "(cross-algorithm at lr 5e-5: p = 0.40). See LIMITATIONS 5.2.2.")

if d["deviations_from_spec"].count(OLD) != 1:
    raise SystemExit(f"the open item appears {d['deviations_from_spec'].count(OLD)} times")
d["deviations_from_spec"][d["deviations_from_spec"].index(OLD)] = NEW

d["note"] = (d.get("note", "") +
             " TWO ENTRIES ARE HAND-EDITED AND WILL BE LOST IF provenance.py IS "
             "RE-RUN: the first deviation (26_after_training.sh still passes the "
             "retracted wording 'GSPO is not offered by this ms-swift commit', "
             "corrected in LIMITATIONS 5.2.1) and the GRPO lr 5e-5 entry above "
             "(26_after_training.sh does not pass it at all). Re-running that "
             "script restores a claim already proven false. Reapply both edits.")

P.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
print("provenance open item closed; reopen hazard recorded in 'note'")
print("deviations:", len(d["deviations_from_spec"]))
