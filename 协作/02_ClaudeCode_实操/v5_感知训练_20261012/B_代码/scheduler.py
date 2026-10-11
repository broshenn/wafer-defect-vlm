#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v5 夜跑调度器（B 卡 14，服务器端常驻）。
启动：setsid nohup /WS/envs/wafer/bin/python scheduler.py --plan /WS/v5/night/night_plan.json \
      --state /WS/v5/night/state.json < /dev/null > /WS/v5/night/scheduler.log 2>&1 &
规则：每 20 秒一轮：收敛运行中任务（DONE/FAILED/TIMEOUT，重试）→ 找就绪任务（依赖满足；
      依赖失败则连锁 SKIPPED）→ 选卡（本调度器无占用 且 nvidia-smi 该 UUID 下零进程）→ 启动
      （CUDA_VISIBLE_DEVICES=uuid 由父进程在 Python 前设置；start_new_session；日志落盘）。
      hard_stop（默认 08:30 本地）到点：对 training=true 的运行任务发 SIGTERM（取已保存最大
      checkpoint），未启动的训练任务记 NOT_RUN；评测/打分等照跑。
      全部进入终态 → 写 ALL_DONE 并退出。收到 SIGTERM：只杀自己启动的进程组后退出。
就绪判断/连锁跳过/选卡写成纯函数（tests/test_scheduler_logic.py 测）。
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

TERMINAL = ("DONE", "FAILED", "TIMEOUT", "SKIPPED", "NOT_RUN")


def dep_satisfied(status, timeout_ok, after_mode):
    if status == "DONE":
        return True
    if status == "TIMEOUT" and timeout_ok:
        return True
    if after_mode == "all_finished" and status in TERMINAL:
        return True
    return False


def dep_failed_hard(status, timeout_ok, after_mode):
    """依赖处于失败侧且不可原谅 → 连锁 SKIPPED（只在 after_mode=all_ok 时适用）。"""
    if after_mode == "all_finished":
        return False
    if status == "DONE":
        return False
    if status == "TIMEOUT" and timeout_ok:
        return False
    if status in ("FAILED", "SKIPPED", "NOT_RUN"):
        return True
    if status == "TIMEOUT" and not timeout_ok:
        return True
    return False


def job_readiness(job, states, running_gpu_owner):
    """返回 (ready: bool, reason: str)。states: {job_id: 状态}。"""
    if states.get(job["id"], "PENDING") != "PENDING":
        return False, "not_pending"
    for dep in job.get("after", []):
        st = states.get(dep)
        if st is None or st == "PENDING" or st == "RUNNING":
            return False, f"wait:{dep}"
        if dep_failed_hard(st, job.get("dep_timeout_ok", {}).get(dep, False), job.get("after_mode", "all_ok")):
            return False, f"skip:{dep}={st}"
        if not dep_satisfied(st, job.get("dep_timeout_ok", {}).get(dep, False), job.get("after_mode", "all_ok")):
            return False, f"wait:{dep}={st}"
    return True, "ready"


def pick_gpu(job_gpus, busy_uuids, gpu_uuids, external_busy):
    """job_gpus 为候选卡名列表；返回 (卡名, uuid) 或 (None, None)；'cpu' 直接返回 ('cpu', None)。"""
    if job_gpus == "cpu" or job_gpus == ["cpu"]:
        return "cpu", None
    for name in job_gpus:
        uuid = gpu_uuids.get(name)
        if uuid is None:
            continue
        if name in busy_uuids:
            continue
        if uuid in external_busy:      # nvidia-smi 上该 UUID 有他人进程
            continue
        return name, uuid
    return None, None


def _now():
    return datetime.datetime.now()


def _iso(t):
    return t.isoformat(timespec="seconds")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--poll", type=int, default=20)
    a = ap.parse_args()
    plan = json.loads(Path(a.plan).read_text(encoding="utf-8"))
    state_path = Path(a.state)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    gpu_uuids = plan["gpus"]
    hard_stop = plan.get("hard_stop", "08:30")
    jobs = {j["id"]: j for j in plan["jobs"]}
    started_at = _now()
    # 依赖的 timeout_ok 注入（供连锁判断用）
    for j in plan["jobs"]:
        j["dep_timeout_ok"] = {d: bool(jobs[d].get("timeout_ok", False)) for d in j.get("after", [])}

    state = {"started": _iso(started_at), "hard_stop": hard_stop, "updated": None,
             "jobs": {jid: {"status": "PENDING", "gpu": None, "pid": None, "pgid": None,
                            "started": None, "ended": None, "seconds": None,
                            "attempts": 0, "log": j.get("log"), "reason": ""}
                      for jid, j in jobs.items()}}
    running = {}   # job_id -> (Popen, 卡名, uuid, 启动时刻, 结束限刻)

    def write_state(note=""):
        state["updated"] = _iso(_now())
        tmp = state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1) + "\n",
                       encoding="utf-8", newline="\n")
        os.replace(tmp, state_path)
        lines = [f"# v5 夜跑状态（{_iso(_now())}）{note}", "", "| 任务 | 状态 | 卡 | 开始 | 耗时s | 尝试 | 日志 |",
                 "|---|---|---|---|---|---|---|"]
        for jid in [j["id"] for j in plan["jobs"]]:
            s = state["jobs"][jid]
            lines.append(f"| {jid} | {s['status']} | {s['gpu'] or ''} | {s['started'] or ''} | "
                         f"{s['seconds'] if s['seconds'] is not None else ''} | {s['attempts']} | {s['log'] or ''} |")
        Path(state_path.parent, "status.md").write_text("\n".join(lines) + "\n",
                                                        encoding="utf-8", newline="\n")

    def kill_own(sig=signal.SIGTERM):
        for jid, (p, cname, uuid, t0, tlim) in list(running.items()):
            try:
                os.killpg(p.pid, sig)
            except ProcessLookupError:
                pass

    def on_term(signum, frame):
        kill_own(signal.SIGTERM)
        write_state("调度器收到 SIGTERM，已终止自有进程组")
        sys.exit(0)

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)

    def external_busy_uuids():
        try:
            out = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                  "--format=csv,noheader"], capture_output=True, text=True,
                                 timeout=30).stdout
        except Exception:
            return set(), True
        busy = set()
        for line in out.splitlines():
            if "," in line:
                busy.add(line.split(",")[0].strip())
        return busy, False

    def stop_training_now():
        hh, mm = hard_stop.split(":")
        hs = started_at.replace(hour=int(hh), minute=int(mm), second=0, microsecond=0)
        return _now() >= hs

    while True:
        # 1) 收敛运行中任务
        for jid, (p, cname, uuid, t0, tlim) in list(running.items()):
            rc = p.poll()
            st = state["jobs"][jid]
            if rc is not None:
                secs = round((_now() - t0).total_seconds(), 1)
                if rc == 0:
                    st.update(status="DONE", ended=_iso(_now()), seconds=secs, pid=None, pgid=None)
                else:
                    st.update(status="FAILED", ended=_iso(_now()), seconds=secs, pid=None, pgid=None,
                              reason=f"rc={rc}")
                    if jobs[jid].get("retries", 0) > st["attempts"]:
                        st.update(status="PENDING", reason=f"retry after rc={rc}")
                        print(f"[sched] {jid} 失败 rc={rc}，重新排队", flush=True)
                del running[jid]
                print(f"[sched] {jid} → {st['status']} ({secs}s)", flush=True)
            else:
                hard = stop_training_now() and jobs[jid].get("training", False)
                over = _now() > tlim
                if hard or over:
                    why = "hard_stop" if hard else "timeout"
                    try:
                        os.killpg(p.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        p.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(p.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        p.wait(timeout=30)
                    secs = round((_now() - t0).total_seconds(), 1)
                    st.update(status="TIMEOUT", ended=_iso(_now()), seconds=secs, pid=None,
                              pgid=None, reason=why)
                    del running[jid]
                    print(f"[sched] {jid} TIMEOUT({why})", flush=True)

        states = {jid: s["status"] for jid, s in state["jobs"].items()}
        # 2) 就绪 → 启动
        for j in plan["jobs"]:
            jid = j["id"]
            if states.get(jid) != "PENDING" or jid in running:
                continue
            hh, mm = hard_stop.split(":")
            hs = started_at.replace(hour=int(hh), minute=int(mm), second=0, microsecond=0)
            if _now() >= hs and j.get("training", False):
                state["jobs"][jid].update(status="NOT_RUN", reason="hard_stop 后不再启动训练")
                print(f"[sched] {jid} NOT_RUN（hard_stop）", flush=True)
                continue
            ready, reason = job_readiness(j, states, running)
            if not ready:
                if reason.startswith("skip:"):
                    state["jobs"][jid].update(status="SKIPPED", reason=reason)
                    print(f"[sched] {jid} SKIPPED（{reason}）", flush=True)
                continue
            ext_busy, nvsmi_fail = external_busy_uuids()
            if nvsmi_fail:
                print("[sched] nvidia-smi 查询失败，本轮不启动新任务", flush=True)
                continue
            busy_names = {cname for (_, cname, _, _, _) in running.values()}
            cname, uuid = pick_gpu(j.get("gpus", []), busy_names, gpu_uuids, ext_busy)
            if cname is None:
                continue
            log = j.get("log") or f"/WS/v5/night/logs/{jid}.log"
            Path(log).parent.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = uuid if uuid else ""
            t0 = _now()
            tlim = t0 + datetime.timedelta(minutes=j.get("timeout_min", 120))
            with open(log, "a", encoding="utf-8") as lf:
                lf.write(f"\n===== [sched] {jid} 于 {_iso(t0)} 在 {cname} 启动 =====\n")
                lf.flush()
                try:
                    p = subprocess.Popen(j["cmd"], env=env, stdout=lf, stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL, start_new_session=True,
                                         cwd=j.get("cwd") or None)
                except Exception as exc:
                    state["jobs"][jid].update(status="FAILED", reason=f"启动失败 {exc}")
                    print(f"[sched] {jid} 启动失败 {exc}", flush=True)
                    continue
            running[jid] = (p, cname, uuid, t0, tlim)
            st = state["jobs"][jid]
            st.update(status="RUNNING", gpu=cname, pid=p.pid, pgid=p.pid, started=_iso(t0),
                      attempts=st["attempts"] + 1)
            print(f"[sched] {jid} 启动 pid={p.pid} 卡={cname}", flush=True)

        write_state()
        if all(s["status"] in TERMINAL for s in state["jobs"].values()):
            Path(state_path.parent, "ALL_DONE").write_text(
                json.dumps({"finished": _iso(_now()),
                            "status": {jid: s["status"] for jid, s in state["jobs"].items()}},
                           ensure_ascii=False, indent=1), encoding="utf-8")
            write_state("全部终态")
            print("[sched] 全部任务终态，退出", flush=True)
            return 0
        time.sleep(a.poll)


if __name__ == "__main__":
    sys.exit(main())
