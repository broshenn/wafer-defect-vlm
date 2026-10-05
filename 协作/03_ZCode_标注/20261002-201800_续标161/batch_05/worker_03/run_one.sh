#!/bin/bash
# worker_03 逐张检查管线：checker1 -> normalizer -> checker2 -> worker_log.jsonl
# 用法: run_one.sh <sample_id> <image_sha256> <T0_ISO> <T0_MS>
D="D:/pycode/晶圆图研究/协作/03_ZCode_标注/20261002-201800_续标161/batch_05/worker_03"
SID="$1"; ISHA="$2"; T0ISO="$3"; T0MS="$4"
PY="D:/python/python.exe"
CHECKER="D:/pycode/晶圆图研究/协作/02_ClaudeCode_实操/口径修正与准备_20261002-021207/check_answer_v3.py"
NORM="D:/pycode/晶圆图研究/协作/01_Codex_指挥/验收_20261002_GLM首批20/normalize_label_alias.py"
LOG="$D/${SID}_check.log"

: > "$LOG"
echo "=== checker1 ===" >> "$LOG"
"$PY" -X utf8 -B "$CHECKER" "$D/${SID}_raw.json" --json > "$D/_c1.out" 2> "$D/_c1.err"
C1=$?
{ echo "exit_code=$C1"; echo "--- stdout ---"; cat "$D/_c1.out"; echo "--- stderr ---"; cat "$D/_c1.err"; } >> "$LOG"

echo "=== normalizer ===" >> "$LOG"
"$PY" -X utf8 -B "$NORM" --raw "$D/${SID}_raw.json" --out "$D/${SID}_canonical.json" --report "$D/${SID}_normalization.json" > "$D/_n.out" 2> "$D/_n.err"
NE=$?
{ echo "exit_code=$NE"; echo "--- stdout ---"; cat "$D/_n.out"; echo "--- stderr ---"; cat "$D/_n.err"; } >> "$LOG"

C2="null"
if [ -e "$D/${SID}_canonical.json" ]; then
  echo "=== checker2 ===" >> "$LOG"
  "$PY" -X utf8 -B "$CHECKER" "$D/${SID}_canonical.json" --json > "$D/_c2.out" 2> "$D/_c2.err"
  C2=$?
  { echo "exit_code=$C2"; echo "--- stdout ---"; cat "$D/_c2.out"; echo "--- stderr ---"; cat "$D/_c2.err"; } >> "$LOG"
else
  echo "=== checker2 === not run (canonical not generated)" >> "$LOG"
fi

T1ISO=$(date '+%Y-%m-%d %H:%M:%S %z'); T1MS=$(date +%s%3N)

"$PY" -X utf8 - "$SID" "$ISHA" "$T0ISO" "$T0MS" "$T1ISO" "$T1MS" "$C1" "$NE" "$C2" "$D" <<'PYEOF' >> "$D/worker_log.jsonl"
import json, sys, hashlib, os, re
sid, isha, t0iso, t0ms, t1iso, t1ms, c1s, nes, c2s, d = sys.argv[1:11]
c1, ne = int(c1s), int(nes)
c2 = None if c2s == "null" else int(c2s)
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(65536), b''):
            h.update(b)
    return h.hexdigest()
raw_p = os.path.join(d, sid + "_raw.json")
can_p = os.path.join(d, sid + "_canonical.json")
rep_p = os.path.join(d, sid + "_normalization.json")
raw_sha = sha(raw_p)
can_sha = sha(can_p) if os.path.exists(can_p) else None
alias_changes = 0
warnings = []
if os.path.exists(rep_p):
    with open(rep_p, encoding='utf-8') as f:
        rep = json.load(f)
    ch = rep.get("changes", [])
    alias_changes = len(ch) if isinstance(ch, list) else 0
    w = rep.get("content_review_warnings", [])
    if isinstance(w, list) and w:
        warnings.extend([str(x) for x in w])
# checker stdout 中可能也含 content_review_warnings
for outp in [os.path.join(d, "_c1.out"), os.path.join(d, "_c2.out")]:
    if os.path.exists(outp):
        try:
            txt = open(outp, encoding='utf-8').read().strip()
            if txt:
                obj = json.loads(txt)
                w = obj.get("content_review_warnings", [])
                if isinstance(w, list) and w:
                    warnings.extend([str(x) for x in w])
        except Exception:
            pass
# 状态判定
raw_parseable = True
try:
    json.loads(open(raw_p, encoding='utf-8').read())
except Exception:
    raw_parseable = False
if c1 == 0:
    status = "raw_pass"
elif raw_parseable and can_sha is not None and c2 == 0:
    status = "raw_fail_canonical_pass"
elif (not raw_parseable) and ne != 0:
    status = "isolated_bad_json"
else:
    status = "raw_fail"
if warnings:
    status = status  # status 不因警告改变
row = {
    "sample_id": sid,
    "image_sha256": isha,
    "raw_sha256": raw_sha,
    "canonical_sha256": can_sha,
    "checker1_exit": c1,
    "normalizer_exit": ne,
    "checker2_exit": c2,
    "alias_changes": alias_changes,
    "pending_review": (True if warnings else False),
    "review_warnings": warnings,
    "t0": t0iso, "t1": t1iso,
    "duration_ms": int(t1ms) - int(t0ms),
    "status": status,
}
print(json.dumps(row, ensure_ascii=False))
PYEOF

rm -f "$D/_c1.out" "$D/_c1.err" "$D/_n.out" "$D/_n.err" "$D/_c2.out" "$D/_c2.err"
echo "DONE $SID"
