#!/usr/bin/env bash
# Stop, re-push and relaunch the three rebuild drivers after a further edit. Same rule as
# before: bash reads a script by byte offset as it executes, so a live one is never edited
# in place; and the remote copies are md5-compared and `bash -n`-checked before launch, so
# neither a truncated push nor a CRLF-translated one can take the evening chain down.
set -uo pipefail
W=/c/Users/Public/wafer-ssh/wssh2
D="/d/pycode/晶圆图研究"

echo "=== 1. stop them ==="
$W "for p in \$(pgrep -f '44_final_consolidation.sh|post_consolidation.sh|after_queue_rebuild.sh|wait_for_idle.py'); do kill \$p && echo \"  killed \$p\"; done; sleep 3; pgrep -af '44_final|post_consolidation|after_queue_rebuild|wait_for_idle' || echo '  (none left)'"

echo "=== 2. push ==="
for pair in "44_final_consolidation.sh:/root/autodl-fs/wafer-vlm/projects/wafer-defect-vlm/scripts/44_final_consolidation.sh" \
            "post_consolidation.sh:/root/autodl-fs/wafer-vlm/tools/post_consolidation.sh" \
            "after_queue_rebuild.sh:/root/autodl-fs/wafer-vlm/tools/after_queue_rebuild.sh"; do
  src="${pair%%:*}"; dst="${pair##*:}"
  B=$(base64 -w0 "$D/$src")
  want=$(md5sum "$D/$src" | cut -d' ' -f1)
  got=$($W "echo $B | base64 -d > $dst && chmod +x $dst && md5sum $dst | cut -d' ' -f1 && bash -n $dst && echo SYNTAX_OK")
  if [ "$want" != "$(echo "$got" | head -1)" ]; then echo "  $src: MD5 MISMATCH -- stopping"; exit 1; fi
  [ "$(echo "$got" | tail -1)" = "SYNTAX_OK" ] || { echo "  $src: bash -n failed -- stopping"; exit 1; }
  echo "  $src: md5 $want, syntax OK"
done

echo "=== 3. relaunch ==="
$W "setsid nohup bash /root/autodl-fs/wafer-vlm/projects/wafer-defect-vlm/scripts/44_final_consolidation.sh >> /root/autodl-fs/wafer-vlm/logs/44_driver.log 2>&1 < /dev/null & sleep 2; setsid nohup bash /root/autodl-fs/wafer-vlm/tools/post_consolidation.sh >> /root/autodl-fs/wafer-vlm/logs/post_consolidation.log 2>&1 < /dev/null & sleep 2; setsid nohup bash /root/autodl-fs/wafer-vlm/tools/after_queue_rebuild.sh 3101 3699 >> /root/autodl-fs/wafer-vlm/logs/after_queue_rebuild.log 2>&1 < /dev/null & sleep 8; ps -eo pid,etime,cmd | grep -E '44_final|post_consolidation|after_queue_rebuild|wait_for_idle' | grep -v grep"
