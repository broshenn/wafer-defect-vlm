#!/usr/bin/env bash
# Launched through this wrapper rather than directly: watch_retrieval.sh's work-in-flight
# check matches a run's tag against every process command line, and the shell that starts
# it carries that tag in its own arguments -- so the watcher sees its own launcher as work
# in flight and waits forever. `exec` also keeps the tag out of any surviving parent.
set -uo pipefail
ROOT=/root/autodl-fs/wafer-vlm
exec bash "$ROOT/tools/watch_retrieval.sh" \
  qwen35_9b_grpo_lr1e5_seed3409:Qwen3.5-9B-grpo-lr1e5-seed3409-merged
