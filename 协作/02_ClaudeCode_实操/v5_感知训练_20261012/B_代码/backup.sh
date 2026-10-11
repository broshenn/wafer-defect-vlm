#!/bin/bash
# 夜跑备份（B 卡 16）：打包各线 checkpoint、v5_config、runtime_gate、reward 审计、评测原答与 metrics。
# 不删除任何原文件。用法：WS_ROOT=<真实根> bash backup.sh
set -u
WS=${WS_ROOT:?必须提供 WS_ROOT}
D=$(date +%Y%m%d)
OUT=$WS/v5/backup/v5_night_$D.tgz
LIST=$WS/v5/backup/_backup_list.txt
mkdir -p "$WS/v5/backup"
: > "$LIST"
for d in "$WS"/v5/runs/*/; do
  [ -d "$d" ] || continue
  last=$(find "$d" -type d -name 'checkpoint-*' | sort -t- -k2 -n | tail -1)
  [ -n "$last" ] && echo "$last" >> "$LIST"
  for f in v5_config.json runtime_gate.json reward_audit.jsonl; do
    [ -f "$d/$f" ] && echo "$d/$f" >> "$LIST"
  done
done
for d in "$WS"/v5/eval/*/*/; do
  [ -d "$d" ] || continue
  for f in "$d"/*_raw.jsonl "$d"/metrics.json "$d"/per_sample.jsonl "$d"/台账_*.jsonl; do
    [ -f "$f" ] && echo "$f" >> "$LIST"
  done
done
for f in "$WS"/v5/night/state.json "$WS"/v5/night/status.md "$WS"/v5/night/winner.json \
         "$WS"/v5/night/passk.json "$WS"/v5/night/night_plan.json; do
  [ -f "$f" ] && echo "$f" >> "$LIST"
done
tar -czf "$OUT" -T "$LIST" 2>/dev/null
"$WS/envs/wafer/bin/python" - "$OUT" > "$WS/v5/backup/backup_manifest.json" <<'PYEOF'
import hashlib, json, sys, tarfile
tarpath = sys.argv[1]
h = hashlib.sha256(open(tarpath, 'rb').read()).hexdigest()
items = []
with tarfile.open(tarpath) as tf:
    for m in tf.getmembers():
        if m.isfile():
            items.append({"name": m.name, "size": m.size})
print(json.dumps({"tar": tarpath, "sha256": h, "n_files": len(items), "files": items},
                 ensure_ascii=False, indent=1))
PYEOF
echo "备份完成 $OUT"
ls -la "$OUT"
