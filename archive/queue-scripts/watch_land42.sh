#!/usr/bin/env bash
# Watch run 42's landing from this machine.
#
# The previous watcher (a background task running /tmp/watch_chain.sh) was pointed at a
# script that does not exist on this server and had been sitting there reporting nothing;
# a watcher that cannot see its target is worse than none, because its silence looks like
# "still running". This one polls the live process table over the working ssh wrapper and
# prints what it saw each minute, so every line in its output is a measurement.
#
# It exits when land42_driver is gone -- that is the checkpoint: patch75, patch118 and the
# landing have all finished, and the driver has either restarted the end-of-day chain or
# stopped with a reason in its log.
W=/c/Users/Public/wafer-ssh/wssh2
for i in $(seq 1 90); do
  out=$("$W" "cd /root/autodl-fs/wafer-vlm
    if pgrep -f land42_driver >/dev/null; then echo DRIVER=running; else echo DRIVER=gone; fi
    if pgrep -f 42_queue_grpo_seed2.sh >/dev/null; then echo QUEUE42=alive; else echo QUEUE42=gone; fi
    if pgrep -f 43_queue_grpo_seed3.sh >/dev/null; then echo QUEUE43=alive; else echo QUEUE43=gone; fi
    if pgrep -f 24_train_grpo >/dev/null; then echo TRAIN=alive; else echo TRAIN=gone; fi
    if pgrep -f 30_eval_grpo >/dev/null; then echo EVAL=alive; else echo EVAL=gone; fi
    echo -n 'LAST_DRIVER_LINE: '; tail -1 logs/land42_driver.log
    echo -n 'QUEUE42_TAIL: '; tail -c 400 logs/42_queue_outer.log | tr '\r' '\n' | grep -v '^$' | tail -1 | cut -c1-120
    echo -n 'DOC_MD5: '; md5sum LIMITATIONS.md | cut -c1-32" 2>&1 | tr '\n' ' ')
  echo "[$(date +%H:%M:%S)] $out"
  case "$out" in
    *DRIVER=gone*) echo "=== run 42's landing phase is over; the driver's last line says why ==="; exit 0;;
  esac
  sleep 60
done
echo "=== watcher timed out after 90 minutes without the driver exiting ==="
