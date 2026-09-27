#!/usr/bin/env bash
# Full teacher-distillation run: DeepSeek V4.1 Flash first pass, Qwen3.5-397B
# re-labels failures/boundary/15% audit, then the documented selection rule.
set -uo pipefail
F=/root/autodl-fs/wafer-vlm
V=/root/autodl-tmp/wafer-vlm/venvs/wafer/bin/python
T=$F/data/teacher_v1
M=$F/data/prepared_v1/manifest.jsonl
MV=$F/data/prepared_v1/manifest_trainval.jsonl
CONC=${CONC:-20}
mkdir -p "$T"
cd "$F/projects/wafer-defect-vlm"
step(){ echo; echo "########## $* ##########"; date "+%F %T"; }

step "train+val manifest"
$V "$F/tools/filter_split.py" "$M" "$MV" train,val

for S in train val; do
  step "primary teacher (deepseek family) split=$S"
  $V -m wafer_vlm.annotate --manifest "$M" --output "$T/primary_deepseek.jsonl" \
    --env-file "$F/secrets/api.env" --provider dashscope --model deepseek-v3.2 \
    --split "$S" --include-label --concurrency "$CONC" --max-tokens 1024 \
    --temperature 0.0 --retries 5 --timeout 180 2>&1 | tail -3
done

step "primary done"
$V - "$T/primary_deepseek.jsonl" <<'PY'
import json,sys
rows=[json.loads(l) for l in open(sys.argv[1])]
ok=[r for r in rows if r["status"]=="ok"]
u=[(r.get("usage") or {}) for r in rows]
print("rows",len(rows),"ok",len(ok),"err",len(rows)-len(ok))
print("total_tokens",sum(x.get("total_tokens",0) for x in u))
print("prompt_tokens",sum(x.get("prompt_tokens",0) for x in u))
print("completion_tokens",sum(x.get("completion_tokens",0) for x in u))
PY

step "curate plan (audit selection)"
$V -m wafer_vlm.curate plan --manifest "$MV" --primary "$T/primary_deepseek.jsonl" \
  --output "$T/plan.jsonl" --ids-output "$T/audit_ids.txt" \
  --audit-fraction 0.15 --seed 3407 2>&1 | tail -12
echo "audit ids: $(wc -l < "$T/audit_ids.txt")"

step "secondary qwen3.5-397b-a17b on audit ids"
$V -m wafer_vlm.annotate --manifest "$M" --output "$T/secondary_qwen.jsonl" \
  --env-file "$F/secrets/api.env" --provider dashscope --model qwen3.5-397b-a17b \
  --split all --ids-file "$T/audit_ids.txt" --include-label \
  --concurrency "$CONC" --max-tokens 1024 --temperature 0.0 --retries 5 --timeout 180 2>&1 | tail -3

step "curate finalize"
$V -m wafer_vlm.curate finalize --manifest "$MV" \
  --primary "$T/primary_deepseek.jsonl" --secondary "$T/secondary_qwen.jsonl" \
  --output-dir "$F/data/curated_v1" 2>&1 | tail -30

step "token totals (both teachers)"
$V - "$T" <<'PY'
import json,sys,pathlib
for p in sorted(pathlib.Path(sys.argv[1]).glob("*.jsonl")):
    rows=[json.loads(l) for l in p.open()]
    ok=[r for r in rows if r["status"]=="ok"]
    u=[(r.get("usage") or {}) for r in rows]
    print(f"{p.name}: rows={len(rows)} ok={len(ok)} total_tokens={sum(x.get('total_tokens',0) for x in u)}")
PY
echo
echo "ANNOTATE_DONE $(date "+%F %T")"
