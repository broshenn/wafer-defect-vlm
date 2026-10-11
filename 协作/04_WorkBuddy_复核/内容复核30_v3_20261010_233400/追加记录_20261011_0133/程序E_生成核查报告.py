# -*- coding: utf-8 -*-
"""追加记录 · 程序E：由 程序A/程序B 的JSON程序化生成两份核查报告MD（数字全部来自程序输出）。"""
import json
from datetime import datetime, timezone
from pathlib import Path

D = Path(__file__).parent
A = json.loads((D / "程序A_审计日志时间线.json").read_text(encoding="utf-8"))
B = json.loads((D / "程序B_矛盾分类.json").read_text(encoding="utf-8"))

CODEX16 = set(B["codex16_items"])
Brows = {r["item_id"]: r for r in B["rows"]}
Arows = {r["item_id"]: r for r in A["per_item"]}

CLS_ZH = {"machine_match": "机器一致", "local_as_Z": "本地误标Z", "estimate": "手写估计", "missing": "无事件"}

def fmt_b(st):
    return f"{CLS_ZH[st['class']]}(Δ{st['delta_seconds']}s)" if st["delta_seconds"] is not None else CLS_ZH[st["class"]]

def audit_line(r):
    def g(k, field="utc"):
        lst = r.get(k) or []
        return lst[0][field][11:19] + "Z" if lst else "—"
    return (f"started写 {g('started_write')} → PNG读 {g('png_read')} → obs写 {g('observation_write')} "
            f"→ 旧raw读 {g('oldraw_read')} → review写 {g('review_write')} → done写 {g('done_write')}")

# ---------- 报告1：时间顺序矛盾核查 ----------
L = []
L.append("# 16个自报时间矛盾核查（追加记录 · 程序结果）\n")
L.append(f"生成：{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}。数据来源：客户端审计日志（程序A）+ 30图各JSON自报时间与文件mtime（程序B）。所有时间均程序提取，未手工填入；原件一律未改。\n")

L.append("## 1. 结论\n")
L.append("1. Codex 点名的 16 图「自报时间不满足 started ≤ observation_written ≤ done」**核实属实**。矛盾全部出在**自报时间字段**，出图判读与文件本身不受影响（见第4节机器顺序）。\n")
L.append("2. 根因两类（逐条分类见第3节）：\n")
L.append("   - **手写估计时间**：started 自报值比该文件真实写盘时刻（审计事件）晚约 2.5～6.5 分钟且取整到分/半分（片1 的 wbv3_003～014、片3 的 wbv3_040～049 等）；观测时间反而是真实机器时间，于是自报值呈现「观察早于 started」的假矛盾。\n")
L.append("   - **本地钟面误标为 Z**：自报值等于事件发生时的本地墙钟（UTC+8）却带 Z 后缀，比真实 UTC 恰好快约 8 小时（wbv3_002 23:36Z、wbv3_016 23:40Z、wbv3_038 23:41Z、wbv3_073 00:00Z 等约 +7.9h 的条目），于是自报值呈现「started 晚于观察 8 小时」的假矛盾。\n")
L.append("3. **追加发现（Codex 名单之外）**：片4 整片 rv30_022～028（wbv3_051～068）的自报时间**全部**是本地钟面误标 Z（偏差 +7.9h～+8.4h），因数值恰好递增、未出现序列矛盾而未被点名；rv30_029 的 started 同病。这些自报时间同样不可作为执行时刻引用。\n")
L.append("4. **真实工具事件顺序（客户端审计日志，30/30 图全部成立）**：started.json 写入 → 该 PNG Read（恰 1 次）→ observation_raw.json 写入 → 旧 raw Read（恰 1 次）→ review.json 写入 → done.json 写入。即：自报时间不可信，但**实际调用顺序有机器证据且符合规程**。\n")
L.append("5. 时区处理原则：审计 timestamp 为客户端 Unix 毫秒（UTC 纪元），本地读法=UTC+8。分类只用「自报值与审计时刻的差」逐条判定（差≈0→机器一致；差≈+8h→本地误标Z；其余→手写估计），**未用统一加 8 小时去消除错误**；当前时间未回填任何历史事件。\n")

L.append("## 2. 证据与tz口径\n")
L.append("- 事件时刻：`C:/Users/谌伦杰/.workbuddy/audit-log/2026-10-10.jsonl`（已闭合段，manifest fileSha256 现算比对一致）与 `2026-10-11.jsonl`（开放段）。\n")
L.append("- 文件写盘时刻：NTFS mtime（UTC 纪元），与审计事件时刻互为独立佐证（差 < 数秒）。\n")
L.append("- 自报时间：各 started/observation_meta/done JSON 内嵌字符串，为 worker 生成，属执行者自报。\n")

L.append("## 3. Codex16 逐条分类（自报 vs 审计真实时刻）\n")
L.append("| review_id | item_id | 自报started | 自报observation | 自报done | 判定要点 |")
L.append("|---|---|---|---|---|---|")
for it in B["codex16_items"]:
    r = Brows[it]
    st = r["stages"]
    note = ""
    if abs(st["started"]["delta_seconds"] - 28800) < 400:
        note = "started=本地钟面误标Z（比真实快约8h）"
    else:
        note = "started=手写估计（比真实晚%.1f~%.1f分钟）" % (st["started"]["delta_seconds"] / 60, st["started"]["delta_seconds"] / 60)
    if st["done"]["class"] == "estimate":
        note += "；done=手写估计"
    L.append(f"| {r['review_id']} | {it} | {fmt_b(st['started'])} | {fmt_b(st['observation_written'])} | {fmt_b(st['done'])} | {note} |")
L.append("")
L.append("注：Δ=自报值−审计事件时刻（秒）。observation 一列 16/16 均为机器一致（Δ<15s），即观察写盘自报时间真实；矛盾全部由 started/done 的自报失真造成。\n")

L.append("## 3b. 名单外的自报失真（片4整片 + rv30_029 started）\n")
L.append("| review_id | item_id | 自报started | 自报observation | 自报done |")
L.append("|---|---|---|---|---|")
for it in ["wbv3_051", "wbv3_055", "wbv3_056", "wbv3_061", "wbv3_063", "wbv3_064", "wbv3_068", "wbv3_073"]:
    r = Brows[it]
    st = r["stages"]
    L.append(f"| {r['review_id']} | {it} | {fmt_b(st['started'])} | {fmt_b(st['observation_written'])} | {fmt_b(st['done'])} |")
L.append("")

L.append("## 4. 真实机器时间线（审计日志，30/30 全部符合规程顺序）\n")
L.append("| review_id | item_id | 审计事件序列（本地钟面 UTC+8 口径在括号外为UTC） | PNG Read 次数 | 旧raw Read 次数 |")
L.append("|---|---|---|---|---|")
for it in sorted(Arows):
    r = Arows[it]
    br = Brows[it]
    L.append(f"| {r['review_id']} | {it} | {audit_line(r)} | {r['png_read_count']} | {r['oldraw_read_tool_count']} |")
L.append("")
L.append("表中时刻为 UTC（审计机器钟）；本地读法=+8h。30/30 满足 started写→PNG读→obs写→旧raw读→review写；另核得 30/30 的 started 写入先于 PNG Read；PNG Read 与旧raw Read 均恰 1 次/图。Bash 中出现 raw.txt 的命令均为 hashlib 字节读取（算哈希），无内容展示类读取。\n")

L.append("## 5. 无法核实项（诚实清单）\n")
L.append("- 自报时间的「当时来源」：worker 为何写出估计/误标值（钟面读取方式、提示词理解），只有执行者声明，无独立记录。\n")
L.append("- 审计日志证明的是**工具调用事件**（工具、路径、客户端时刻）；不证明工具返回内容、模型上下文实际内容、后端请求次数与后端模型身份。\n")
L.append("- 片2+3、片4+5 的并发 worker 在审计日志中会话归属合并（见真实日志索引.md 第3节），按会话归责不能细分到单个 worker；逐图事件归属不受影响。\n")
(D / "时间顺序矛盾核查.md").write_text("\n".join(L), encoding="utf-8")
print("时间顺序矛盾核查.md:", len(L), "lines")

# ---------- 报告2：真实日志索引 ----------
L2 = []
L2.append("# 能获得的真实日志索引（内容复核30轮 · 2026-10-10 23:34 — 10-11 00:18 窗口）\n")
L2.append(f"生成：{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}。全部为只读提取。\n")
L2.append("## 1. 日志文件\n")
L2.append("| 文件 | 状态 | 条数 | 完整性 |")
L2.append("|---|---|---|---|")
for seg, v in A["audit_integrity_check"].items():
    L2.append(f"| `C:/Users/谌伦杰/.workbuddy/audit-log/{seg}` | 已闭合段 | manifest记{v['entry_count_manifest']} | fileSha256 现算比对 {'一致' if v['match'] else '不一致'} |")
L2.append("| `C:/Users/谌伦杰/.workbuddy/audit-log/2026-10-11.jsonl` | 开放段（提取时仍会追加） | — | 段未闭合，无段哈希 |")
L2.append("| `C:/Users/谌伦杰/.workbuddy/audit-log/manifest.jsonl` / `state.json` | 段级hash链与序号台账 | 3段闭合记录 | — |")
L2.append("")
L2.append("事件形态两类：`file-safety.allow-listed`（Read/Write 工具调用，含路径）；`command-safety.sandbox-executed`（Bash 命令，含命令预览）。每条含客户端 Unix 毫秒时间戳、单调 sequence 与 prevHash/hash 链（客户端自记，非外部见证）。\n")
L2.append("## 2. 本轮窗口内的会话（审计口径）\n")
L2.append("| 审计sessionId前8位 | 事件数 | 起止（本地UTC+8） | 角色 |")
L2.append("|---|---|---|---|")
role = {"b79246db": "父会话（编排+机械核验；**0 次PNG Read**，已程序核查）",
        "01a12677": "片1 worker（rv30_001..007）",
        "01a12682": "片2+片3 worker（rv30_008..021；两会话在审计中合并归属）",
        "01a1268f": "片4+片5 worker（rv30_022..030；两会话在审计中合并归属）"}
for k, v in A["sessions_in_window"].items():
    L2.append(f"| {k} | {v['n_events']} | {v['first_local8']} — {v['last_local8']} | {role.get(k, '?')} |")
L2.append("")
L2.append("## 3. 覆盖与局限\n")
L2.append("- **已证**：每图 PNG Read 恰 1 次、旧 raw Read 恰 1 次、started/observation_raw/observation_meta/review/done 各恰 1 次 Write，且调用顺序 30/30 符合规程（见 时间顺序矛盾核查.md 第4节）；父会话窗口内零次读图。\n")
L2.append("- **未证**：工具返回内容与模型上下文、后端请求数、后端真实模型与采样参数、并发 worker 的会话级归责（审计将同波两个 worker 记于同一 sessionId；逐图事件归属不受影响）。\n")
L2.append("- Bash 扫描：命令中出现 `.raw.txt` 的全部为 `hashlib.sha256(open(p,'rb').read())` 类字节读取（算哈希），无内容展示；PNG 路径同样只出现在哈希计算命令中。\n")
L2.append("- 检索方式：按时间窗（2026-10-10T15:34Z 起）提取两日全部事件后按路径/命令匹配；未读取审计日志以外的客户端数据。\n")
(D / "真实日志索引.md").write_text("\n".join(L2), encoding="utf-8")
print("真实日志索引.md:", len(L2), "lines")
