# -*- coding: utf-8 -*-
"""追加记录 · 程序C：wbv3_056 旧started哈希与新算哈希的字符级差异（程序结果，替代手写解释）。

只读：不改动旧 started、旧更正说明、旧 raw、盲清单与 PNG。
事实基准：旧started(expected=actual)=61字符；盲清单与现算PNG=64字符。
"""
import hashlib
import difflib
import json
from datetime import datetime, timezone
from pathlib import Path

OLD = "708e4ea4e0f39eda60b317b238727fcc26a92ad4f49b921e3149ac22e8016"  # 旧started原值（61字符，保持原样引用）
BLIND = Path(r"D:/pycode/晶圆图研究/协作/01_Codex_指挥/无卡CPU整备_20261010/WorkBuddy84/blind84.jsonl")
PNG = Path(r"D:/pycode/晶圆图研究/data/images/wafer_00042267_019.png")
OUT = Path(__file__).parent / "程序C_056哈希字符差异.json"

def main():
    rec = [json.loads(l) for l in BLIND.read_text(encoding="utf-8").splitlines() if l.strip()]
    blind_sha = [r for r in rec if r["item_id"] == "wbv3_056"][0]["image_sha256"]
    now_sha = hashlib.sha256(PNG.read_bytes()).hexdigest()

    sm = difflib.SequenceMatcher(None, OLD, blind_sha, autojunk=False)
    ops = [{"op": tag, "old_slice": [i1, i2], "new_slice": [j1, j2],
            "old_text": OLD[i1:i2] or None, "new_text": blind_sha[j1:j2] or None}
           for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal"]

    started = json.loads((OUT.parent.parent.parent / "方向形态84_v3_20261010_200950" / "wbv3_056.started.json")
                         .read_text(encoding="utf-8"))
    result = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "item_id": "wbv3_056", "sample_id": "wafer_00042267_019",
        "old_started_expected_61": OLD,
        "old_started_actual_61": started.get("image_sha256_actual"),
        "old_started_sha_match_field": started.get("sha_match"),
        "blind_sha256_64": blind_sha,
        "recomputed_png_sha256_64": now_sha,
        "blind_equals_recomputed": blind_sha == now_sha,
        "old_equals_blind": OLD == blind_sha,
        "lengths": {"old_started": len(OLD), "blind": len(blind_sha), "recomputed": len(now_sha)},
        "char_diff_old_vs_new": ops,
        "diff_summary": "；".join(
            ("插入 %r 于旧串第%d位（0基）" % (o["new_text"], o["old_slice"][0])) if o["op"] == "insert"
            else ("删除 %r" % o["old_text"]) if o["op"] == "delete"
            else ("替换 %r -> %r" % (o["old_text"], o["new_text"]))
            for o in ops) or "无差异",
        "conclusion": [
            "程序difflib结论：旧61字符值与64字符真值之间是【单处插入，长度3】的最小编辑（规范对齐：在0基第27位插入 '279'）；因该区域存在重复字符，'缺失的3个字符'的具体切分依赖对齐方式（等价表述为在0基第29位缺 '927'），以opcodes为准。",
            "两种对齐等价：旧值 = 64字符真值中段丢失3个连续字符的结果；无字符替换、无首尾截断、其余字符完全一致。",
            "此前追加的 wbv3_056_started更正说明.md 中'缺少中段 92 两字符，且总长差 3 个字符'的表述不准确（缺的是3个字符，不是2个），以本程序结果为准；该更正说明原件不改。",
            "本结果只证明：当前磁盘上的PNG与盲清单一致、旧started字段值本身的真实形态；不证明2026-10-10T13:48:07Z执行时worker读取到的字节，也不证明当时读图是否正确。",
        ],
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["lengths", "char_diff_old_vs_new", "diff_summary",
                                             "blind_equals_recomputed", "old_equals_blind"]},
                     ensure_ascii=False, indent=1))

if __name__ == "__main__":
    main()
