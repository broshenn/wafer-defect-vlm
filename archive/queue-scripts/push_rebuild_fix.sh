#!/usr/bin/env bash
# Push the three fixed rebuild drivers and relaunch them in dependency order.
#
# They are stopped before the push: bash reads a script by byte offset as it executes it,
# so a file edited under a live process resumes mid-line. Their work is idempotent -- each
# enumerates the reports on disk and rewrites derived artefacts -- so restarting costs
# only the wait time.
#
# Two failures are worth naming, because both were mine and only one was visible:
# `Path.write_text` on Windows rewrites LF as CRLF, so the first push made all three
# scripts unparseable (`$'\r': command not found`, then a syntax error at the first
# multi-line construct) -- a file that had been running fine minutes earlier. And the
# relaunch's `cd BASE && cmd1 & cmd2 &` applies the `cd` only to the first command in the
# list, so the other two wrote their logs to a path that did not exist. Everything below
# is absolute paths, and the pushed copies are md5-compared before anything is launched.
set -uo pipefail
W=/c/Users/Public/wafer-ssh/wssh2
D="/d/pycode/晶圆图研究"

echo "=== 1. push the fixed files (md5-verified, bash -n on the remote copy) ==="
for pair in "44_final_consolidation.sh:/root/autodl-fs/wafer-vlm/projects/wafer-defect-vlm/scripts/44_final_consolidation.sh" \
            "post_consolidation.sh:/root/autodl-fs/wafer-vlm/tools/post_consolidation.sh" \
            "after_queue_rebuild.sh:/root/autodl-fs/wafer-vlm/tools/after_queue_rebuild.sh"; do
  src="${pair%%:*}"; dst="${pair##*:}"
  B=$(base64 -w0 "$D/$src")
  want=$(md5sum "$D/$src" | cut -d' ' -f1)
  got=$($W "echo $B | base64 -d > $dst && chmod +x $dst && md5sum $dst | cut -d' ' -f1 && bash -n $dst && echo SYNTAX_OK")
  remote_md5=$(echo "$got" | head -1); verdict=$(echo "$got" | tail -1)
  if [ "$want" != "$remote_md5" ]; then echo "  $src: MD5 MISMATCH -- stopping"; exit 1; fi
  if [ "$verdict" != "SYNTAX_OK" ]; then echo "  $src: bash -n failed -- stopping"; exit 1; fi
  echo "  $src: pushed, md5 $remote_md5, syntax OK"
done

echo "=== 2. relaunch in dependency order (absolute paths) ==="
$W "setsid nohup bash /root/autodl-fs/wafer-vlm/projects/wafer-defect-vlm/scripts/44_final_consolidation.sh >> /root/autodl-fs/wafer-vlm/logs/44_driver.log 2>&1 < /dev/null & sleep 2; setsid nohup bash /root/autodl-fs/wafer-vlm/tools/post_consolidation.sh >> /root/autodl-fs/wafer-vlm/logs/post_consolidation.log 2>&1 < /dev/null & sleep 2; setsid nohup bash /root/autodl-fs/wafer-vlm/tools/after_queue_rebuild.sh 3101 3699 >> /root/autodl-fs/wafer-vlm/logs/after_queue_rebuild.log 2>&1 < /dev/null & sleep 8; ps -eo pid,etime,cmd | grep -E '44_final|post_consolidation|after_queue_rebuild|wait_for_idle' | grep -v grep"
