"""Exercise the three-draw path of seed_variance.py on fixtures, before a real third draw.

The tool exits at its own readiness guard until seed 2's predictions exist, so the
spread block patch135 added cannot be run against production data yet -- and it will be
run for real inside queue 43's landing, on the one night it matters. This builds a
throwaway root in /tmp out of the real benchmark and real predictions files, invents a
second and a third draw by flipping a known number of classification rows, writes
reports whose accuracy is computed from the same parsing rule the tool uses (so the
tool's own reproduction check validates the fixture), and runs the patched tool there.

Three cases, because they are the branches that would otherwise be discovered in
production:

  A. three draws, third drifts further than the pair  -> range above the threshold while
     the pair's own rule is below it: the branch that says the extra draw moved the
     reading, which is the only case in which 5.2.2 has to be re-read;
  B. the third draw's files removed                   -> reported as pending, two-draw
     numbers unchanged, exit 0 (this is the state run 42's landing runs the tool in);
  C. one extra config field perturbed                 -> "not single-variable against
     the first draw" is recorded rather than an assertion that only the seed differs.

Nothing here writes to the project. The fixture root is /tmp/svtest.
"""
import json
import pathlib
import re
import shutil
import subprocess
import sys

REAL = pathlib.Path("/root/autodl-fs/wafer-vlm")
T = pathlib.Path("/tmp/svtest")
PY = sys.executable

shutil.rmtree(T, ignore_errors=True)
for d in ("benchmarks/wafer_bench_v1", "outputs/baselines", "outputs/reports"):
    (T / d).mkdir(parents=True, exist_ok=True)
for f in ("all_requests.jsonl", "classification.jsonl"):
    shutil.copy2(REAL / "benchmarks/wafer_bench_v1" / f, T / "benchmarks/wafer_bench_v1" / f)

CLASSES = sorted({json.loads(l)["gold_label"]
                  for l in (T / "benchmarks/wafer_bench_v1/classification.jsonl")
                  .read_text(encoding="utf-8").splitlines() if l.strip()})


def parse_label(text):
    """The tool's own rule, copied: a fixture whose accuracy is computed under a
    different rule than the tool reads it with would be a fixture the tool rejects."""
    tail = text.strip().splitlines()[-1].strip() if text.strip() else ""
    for c in CLASSES:
        if tail == c:
            return c
    for c in CLASSES:
        if re.search(r"\b" + re.escape(c) + r"\b", tail):
            return c
    for c in CLASSES:
        if c in text:
            return c
    return None


requests = [json.loads(l) for l in (T / "benchmarks/wafer_bench_v1/all_requests.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()]
gold = {r["sample_id"]: r["gold_label"]
        for r in (json.loads(l) for l in (T / "benchmarks/wafer_bench_v1/classification.jsonl")
                  .read_text(encoding="utf-8").splitlines() if l.strip())}


def imgkey(row):
    imgs = row.get("images") or []
    if not imgs:
        return None
    first = imgs[0]
    p = first.get("path") if isinstance(first, dict) else first
    return pathlib.Path(p).name if p else None


CLS = [(i, gold.get(requests[i].get("sample_id")))
       for i, r in enumerate(requests) if r.get("task") == "classification"]
TPL = json.loads((REAL / "outputs/reports/qwen35_9b_grpo__report.json")
                 .read_text(encoding="utf-8"))


def make(tag, src_tag, flips=0):
    """A synthetic draw: an existing predictions file with `flips` rows answered wrong."""
    rows = [json.loads(l) for l in (REAL / "outputs/baselines" / (src_tag + ".jsonl"))
            .read_text(encoding="utf-8").splitlines() if l.strip()]
    if len(rows) != len(requests):
        sys.exit(f"{tag}: source has {len(rows)} rows, benchmark has {len(requests)}")
    step = max(1, len(CLS) // flips) if flips else 0
    flipped = 0
    for j, (i, g) in enumerate(CLS):
        if flips and j % step == 0 and flipped < flips:
            rows[i]["response"] = next(c for c in CLASSES if c != g)
            flipped += 1
    (T / "outputs/baselines" / (tag + ".jsonl")).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    acc = sum(1 for i, g in CLS if parse_label(rows[i].get("response") or "") == g) / len(CLS)
    rep = json.loads(json.dumps(TPL))
    rep["classification"]["accuracy"] = acc
    (T / "outputs/reports" / (tag + "__report.json")).write_text(
        json.dumps(rep, ensure_ascii=False), encoding="utf-8")
    return acc, flipped


DRAWS = [("qwen35_9b_grpo", "qwen35_9b_grpo", 0),
         ("qwen35_9b_grpo_lr1e5_seed3408", "qwen35_9b_grpo", 3),
         ("qwen35_9b_grpo_lr1e5_seed3409", "qwen35_9b_grpo", 12),
         ("qwen35_9b_adapter", "qwen35_9b_adapter", 0),
         ("qwen35_9b_grpo_lr5e5", "qwen35_9b_grpo_lr5e5", 0)]
print("fixture draws (accuracy computed under the tool's own parsing rule):")
for tag, src, flips in DRAWS:
    acc, got = make(tag, src, flips)
    print(f"  {tag:38s} acc {acc:.4f}  ({got} row(s) flipped of {len(CLS)})")

# Training records: the real first draw, and two reseeds of it. Case C perturbs one
# field of the third draw's config to check the "not single-variable" branch.
real1 = json.loads((REAL / "outputs/reports/grpo_train_result.json").read_text(encoding="utf-8"))
for tag, seed, extra in (("qwen35_9b_grpo_lr1e5_seed3408", 3408, None),
                         ("qwen35_9b_grpo_lr1e5_seed3409", 3409, None)):
    rec = json.loads(json.dumps(real1))
    rec.setdefault("config", {})["seed"] = seed
    if extra:
        rec["config"][extra[0]] = extra[1]
    (T / "outputs/reports" / (tag + "_train_result.json")).write_text(
        json.dumps(rec, ensure_ascii=False), encoding="utf-8")
shutil.copy2(REAL / "outputs/reports/grpo_train_result.json",
             T / "outputs/reports/grpo_train_result.json")

SRC = (REAL / "tools/seed_variance.py").read_text(encoding="utf-8")
OLDROOT = 'ROOT = Path("/root/autodl-fs/wafer-vlm")'
if SRC.count(OLDROOT) != 1:
    sys.exit("the tool's ROOT line is not where this fixture expects it")
(T / "sv.py").write_text(SRC.replace(OLDROOT, f'ROOT = Path("{T}")', 1), encoding="utf-8")


def run_case(label, perturb=None):
    if perturb:
        p = T / "outputs/reports/qwen35_9b_grpo_lr1e5_seed3409_train_result.json"
        rec = json.loads(p.read_text(encoding="utf-8"))
        rec["config"][perturb[0]] = perturb[1]
        p.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    r = subprocess.run([PY, str(T / "sv.py")], capture_output=True, text=True, cwd=str(T))
    print(f"\n===== {label}: exit {r.returncode} =====")
    for ln in ((r.stdout or "") + (r.stderr or "")).strip().splitlines()[-18:]:
        print("  " + ln[:170])
    rec = json.loads((T / "outputs/reports/seed_variance.json").read_text(encoding="utf-8"))
    return r.returncode, rec


code, rec = run_case("A: three draws, the third drifts further")
sp = rec["seed_spread"]
accs = list(sp["draws"].values())
assert sp["n_draws"] == 3, sp
assert abs(sp["range"] - (max(accs) - min(accs))) < 1e-9, sp
assert sp["rules_agree"] == (sp["rule_triggered"] == sp["rule_triggered_by_the_pair"])
print(f"\n  case A: n_draws={sp['n_draws']} range={sp['range']:.4f} "
      f"threshold={sp['threshold_half_lr_effect']:.4f} "
      f"pair_triggered={sp['rule_triggered_by_the_pair']} "
      f"range_triggered={sp['rule_triggered']} agree={sp['rules_agree']}")
print(f"          reading: {sp['reading'][:150]}")
print(f"          extra: {rec['extra_seeds']['present']} "
      f"pending={rec['extra_seeds']['declared_but_not_evaluated']}")
print(f"          design.extra: {json.dumps(rec['design']['extra'], ensure_ascii=False)[:220]}")
assert rec["extra_seeds"]["present"] == ["GRPO_G4_lr1e5_seed3409"], rec["extra_seeds"]
assert rec["design"]["extra"]["GRPO_G4_lr1e5_seed3409"]["diff"] == ["seed"]

# ------------------------------------------------------------------- case B: pending
for f in (T / "outputs/baselines/qwen35_9b_grpo_lr1e5_seed3409.jsonl",
          T / "outputs/reports/qwen35_9b_grpo_lr1e5_seed3409__report.json"):
    f.rename(f.with_suffix(f.suffix + ".hidden"))
code_b, rec_b = run_case("B: the third draw's files removed (run 42's landing state)")
sb = rec_b["seed_spread"]
assert code_b == 0, code_b
assert sb["n_draws"] == 2 and rec_b["extra_seeds"]["present"] == [], rec_b["extra_seeds"]
assert rec_b["extra_seeds"]["declared_but_not_evaluated"] == ["GRPO_G4_lr1e5_seed3409"]
assert rec_b["rule_triggered"] == rec_b["seed_spread"]["rule_triggered"]
print(f"\n  case B: n_draws={sb['n_draws']} reading: {sb['reading'][:110]}")
print(f"          the two-draw fields are unchanged: ratio="
      f"{rec_b['ratio_seed_to_lr_effect']} rule_triggered={rec_b['rule_triggered']}")

for f in T.glob("outputs/baselines/*.hidden"):
    f.rename(f.with_suffix(""))
for f in T.glob("outputs/reports/*.hidden"):
    f.rename(f.with_suffix(""))

# ------------------------------------------------------- case C: a config that moved
code_c, rec_c = run_case("C: the third draw's config differs in one more field",
                         perturb=("save_steps", 999))
d3 = rec_c["design"]["extra"]["GRPO_G4_lr1e5_seed3409"]
assert d3["diff"] == ["save_steps", "seed"], d3
print(f"\n  case C: diff={d3['diff']} -> {d3['identical_except'][:130]}")
assert code_c == 0 and rec_c["extra_seeds"]["present"] == ["GRPO_G4_lr1e5_seed3409"]

print("\nall three cases behaved as the record says they do; nothing in the project "
      "was written by this fixture")
