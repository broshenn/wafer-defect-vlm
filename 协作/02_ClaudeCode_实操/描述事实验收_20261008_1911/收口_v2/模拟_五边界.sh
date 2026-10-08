#!/bin/bash
# D 段 v2 的**离线边界模拟**：五个边界，**完全不碰 GPU**。
#
# 做法：把 `nvidia-smi` 与 `python` 两个外部依赖**注入替换**成假的，
#       WS_ROOT 指向沙箱。生产脚本一行不改 —— 被模拟的就是真正要跑的那份。
#
# 覆盖 Codex 指定的五个边界：
#   1. 无卡          → 队列等满截止，**不启动**，返回 4
#   2. 查询失败      → 连续失败达阈值即**停止派发**，返回 6（不反复堆探针）
#   3. 锁后占用      → 放弃该卡、**释放锁**、不启动
#   4. 子进程非零    → 记账为 FAILED、耗时计入、**返回非零**
#   5. 预算耗尽      → 剩余模型记 **NOT_RUN（部分完成）**，不谎报完全成功
#
# 跑法：bash 模拟_五边界.sh
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
SIM="$HERE/模拟沙箱"
PASS=0; FAIL=0

ok()   { echo "  ✓ $1"; PASS=$((PASS+1)); }
bad()  { echo "  ✗ $1"; FAIL=$((FAIL+1)); }

# ---------------- 造沙箱 ----------------
setup() {
  rm -rf "$SIM"; mkdir -p "$SIM/envs/wafer/bin" "$SIM/existence/派生图" \
      "$SIM/logs" "$SIM/existence/locks"
  : > "$SIM/existence/eval_existence.py"
  # 把**真正要跑的** v2 脚本放进沙箱 —— 否则队列找不到它（返回 127）
  cp "$HERE/串行跑三模型_v2.sh" "$SIM/existence/串行跑三模型_v2.sh"
  cat > "$SIM/envs/wafer/bin/python" <<'PY'
#!/bin/bash
sleep "${FAKE_PY_SLEEP:-0}"
exit "${FAKE_PY_RC:-0}"
PY
  chmod +x "$SIM/envs/wafer/bin/python"
  cat > "$SIM/fake_nvidia-smi" <<'FAKE'
#!/bin/bash
# 假 nvidia-smi。必须**正确解析 `-i <卡号>`** —— 早先版本误把 $4 当卡号，
# 而 $4 其实是 `--format=csv,noheader`，于是所有卡都被当成「有进程占着」。
IDX=""; prev=""
for a in "$@"; do [ "$prev" = "-i" ] && IDX="$a"; prev="$a"; done
case "$*" in
  *"--query-gpu=uuid"*) echo "GPU-fake-${IDX:-?}"; exit 0 ;;
esac
case "$FAKE_SCENARIO" in
  no_card)          echo "12345, 1024 MiB"; exit 0 ;;
  query_fail)       exit 1 ;;
  locked_occupied)  # 全局第 1 次查是空的（卡看着空闲），之后全是有进程
    k=$(cat "$FAKE_COUNT" 2>/dev/null || echo 0); k=$((k+1)); echo "$k" > "$FAKE_COUNT"
    if [ "$k" -le 1 ]; then echo ""; else echo "999, 2048 MiB"; fi; exit 0 ;;
  ok)               # 卡 0 空闲，其余被占
    [ "$IDX" = "0" ] && echo "" || echo "777, 1024 MiB"; exit 0 ;;
  *) exit 1 ;;
esac
FAKE
  chmod +x "$SIM/fake_nvidia-smi"
  rm -f "$SIM/fake_count"; : > "$SIM/ledger.tsv"
}

run_queue() {   # $1=场景 $2=预算秒 $3=截止偏移秒  [$4=假python返回码] [$5=假python睡眠]
  setup
  local sc="$1" bud="$2" off="$3" prc="${4:-0}" psl="${5:-0}"
  FAKE_SCENARIO="$sc" FAKE_COUNT="$SIM/fake_count" \
  NVSMI="$SIM/fake_nvidia-smi" WS_ROOT="$SIM" \
  SLEEP_SECS=0 MAXQFAIL=3 \
  FAKE_PY_RC="$prc" FAKE_PY_SLEEP="$psl" \
  SPECS="M1=$SIM/m1 M2=$SIM/m2 M3=$SIM/m3" \
  bash "$HERE/队列_v2.sh" "$(( $(date +%s) + off ))" "$bud" \
       "$SIM/ledger.tsv" "$SIM/out" >"$SIM/stdout.txt" 2>&1
  echo $?
}

# ---------------- 1. 无卡 ----------------
echo "【1】无卡：所有卡都有进程"
RC=$(run_queue no_card 600 3)
[ "$RC" = "4" ] && ok "队列等满截止后放弃，返回 4" || bad "期望返回 4，实际 $RC"
[ ! -s "$SIM/ledger.tsv" ] && ok "账本为空（没有启动任何模型）" || bad "不该有账本记录"
grep -q "未启动" "$SIM/logs/_queue_v2.log" && ok "日志写明未启动" || bad "日志缺少未启动记录"

# ---------------- 2. 查询失败 ----------------
echo "【2】查询失败：nvidia-smi 返回非零"
RC=$(run_queue query_fail 600 3)
[ "$RC" = "6" ] && ok "连续失败达阈值即停止派发，返回 6" || bad "期望返回 6，实际 $RC"
grep -q "停止派发" "$SIM/logs/_queue_v2.log" && ok "日志写明停止派发" || bad "日志缺少停止派发"
n=$(wc -l < "$SIM/logs/_queue_v2.log")
[ "$n" -le 6 ] && ok "没有堆积探针（日志仅 $n 行）" || bad "日志行数 $n，疑似反复堆探针"
[ ! -s "$SIM/ledger.tsv" ] && ok "账本为空" || bad "不该有账本记录"

# ---------------- 3. 锁后占用 ----------------
echo "【3】锁后占用：第一次查是空的，锁上之后别人占了"
RC=$(run_queue locked_occupied 600 3)
[ "$RC" = "4" ] && ok "放弃该卡后等到截止，返回 4（未启动）" || bad "期望返回 4，实际 $RC"
grep -q "锁后复查发现" "$SIM/logs/_queue_v2.log" && ok "日志记录了锁后复查发现占用" || bad "没有锁后复查记录"
[ ! -s "$SIM/ledger.tsv" ] && ok "账本为空（没有误启动）" || bad "不该有账本记录"
[ -z "$(ls -A "$SIM/existence/locks" 2>/dev/null)" ] && ok "锁已释放，没有残留" || bad "锁目录有残留"

# ---------------- 4. 子进程非零 ----------------
echo "【4】子进程非零：模型跑失败"
RC=$(run_queue ok 600 20 3 0)
[ "$RC" = "1" ] && ok "有失败即返回 1（不再谎报完成）" || bad "期望返回 1，实际 $RC"
grep -q "FAILED" "$SIM/ledger.tsv" && ok "账本记 FAILED" || bad "账本缺少 FAILED"
awk -F'\t' '$2=="FAILED"{ if ($3+0>0) found=1 } END{exit !found}' "$SIM/ledger.tsv" \
  && ok "失败耗时已计入预算" || bad "失败耗时未计入"

# ---------------- 5. 预算耗尽 ----------------
echo "【5】预算耗尽：预算 3 秒，每个模型 2 秒"
RC=$(run_queue ok 3 20 0 2)
[ "$RC" = "7" ] && ok "部分完成返回 7（不谎报完全成功），实际 $RC" || bad "期望返回 7，实际 $RC"
grep -q "NOT_RUN" "$SIM/ledger.tsv" && ok "剩余模型记为 NOT_RUN" || bad "缺少 NOT_RUN"
n=$(awk -F'\t' '$2=="OK"' "$SIM/ledger.tsv" | wc -l)
[ "$n" -ge 1 ] && ok "已跑的 $n 个记为 OK（部分完成）" || bad "应有至少一个 OK"
grep -q "预算耗尽" "$SIM/logs/_queue_v2.log" "$SIM/stdout.txt" 2>/dev/null \
  && ok "明确写出预算耗尽" || bad "缺少预算耗尽说明"

echo
echo "=================================================="
echo "边界模拟：$PASS 通过 / $FAIL 失败"
[ "$FAIL" -eq 0 ] && echo "全部通过" || echo "**有失败项**"
exit $([ "$FAIL" -eq 0 ] && echo 0 || echo 1)
