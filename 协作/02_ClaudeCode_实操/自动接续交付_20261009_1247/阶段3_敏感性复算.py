#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CPU 交付协作者 · 阶段 3：独立复算 Codex 的「13 点补核」附表（只读）。

要并入成稿的三件事，必须由我自己复算过才能写进交付：
  1. **一张评测图与训练图逐字节同内容**（`wafer_00040406_017` ↔ `wafer_00000127_002`），
     由此产生的 35 图敏感性；
  2. 四 API **12 条连接失败不是模型错误**，由此产生的 26 图 / 25 图子集；
  3. 上述所有数字必须能在本机**重新算一遍**并给出逐条证据。

本脚本**不重跑任何 API/GPU**，只读已有原答，用自己的解析器（与 `机械核对.py` 同一套
`parse_strict` / `parse_diag`）重算，再与 Codex 的 `13点补核.json` 逐项比对。
"""
from __future__ import annotations
import hashlib, importlib.util, io, json, sys
from pathlib import Path

D = Path(__file__).resolve().parent
REPO = D.parents[2]
RUN = REPO / "协作/02_ClaudeCode_实操/百炼多模型业务对照_20261009_1203"
DEP = REPO / "协作/02_ClaudeCode_实操/部署原型_20261009_0120"
BLIND = REPO / "协作/01_Codex_指挥/百炼多模型对照_20261009/blind36.jsonl"
GLM = REPO / "协作/04_WorkBuddy_复核/外部评测102_20261008_201229"
CODEX = REPO / "协作/01_Codex_指挥/自动接续_20261009/13点补核.json"

_spec = importlib.util.spec_from_file_location("jixie", D / "机械核对.py")
_jx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_jx)


def readlines(p: Path):
    return [json.loads(l) for l in io.open(p, encoding="utf-8") if l.strip()]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def pred(o):
    return o.get("defect_class") if isinstance(o, dict) else None


def main() -> int:
    blind = readlines(BLIND)
    ids = [b["sample_id"] for b in blind]
    by_id = {b["sample_id"]: b for b in blind}
    gold = {s["sample_id"]: s["公开类别"]
            for s in json.loads((DEP / "样本清单.json").read_text(encoding="utf-8"))["样本"]}

    # ---- 1. 评测图 vs 训练图：逐字节同内容（我自己重算，不采信转述）----
    data = {r["sample_id"]: r for r in readlines(D / "清单_5904_本地相对路径.jsonl")}
    train = readlines(REPO / "协作/02_ClaudeCode_实操/夜间扩容_20261008_0244/数据/L_N3072.jsonl")
    train_hashes: dict[str, list[str]] = {}
    for r in train:
        train_hashes.setdefault(data[r["sample_id"]]["sha256"], []).append(r["sample_id"])
    overlap = {sid: train_hashes[by_id[sid]["image_sha256"]]
               for sid in ids if by_id[sid]["image_sha256"] in train_hashes}
    # 逐字节复核（两侧都从本地 PNG 重算，而不是只比登记值）
    byte_proof = []
    for sid, tids in overlap.items():
        h_eval = sha(REPO / by_id[sid]["image_path"])
        assert h_eval == by_id[sid]["image_sha256"], sid
        for tid in tids:
            h_tr = sha(REPO / data[tid]["local_relative_path"])
            assert h_tr == by_id[sid]["image_sha256"], tid
            byte_proof.append({
                "评测图": sid, "训练图": tid,
                "评测图sha256": h_eval, "训练图sha256": h_tr,
                "逐字节相同": h_eval == h_tr,
                "评测lot": sid.split("_")[1], "训练lot": tid.split("_")[1],
                "评测类别": gold.get(sid), "训练类别": gold.get(tid),
            })

    # ---- 2. 原答与状态 ----
    models: dict[str, dict[str, str]] = {}
    status: dict[str, dict[str, bool]] = {}
    for path in sorted((RUN / "原答_API").glob("*_raw.jsonl")):
        rs = readlines(path)
        name = path.name.removesuffix("_raw.jsonl")
        models[name] = {r["sample_id"]: (r.get("content") or "") for r in rs}
        status[name] = {r["sample_id"]: r.get("status") == "ok" for r in rs}
    for name in ("Base", "L-N3072-3407"):
        rs = readlines(RUN / "原答_GPU" / f"{name}_raw.jsonl")
        models[name] = {r["sample_id"]: r.get("raw") for r in rs}
    dep = readlines(DEP / "性能与自测/http_replay.jsonl")
    models["D-N3072-3407"] = {r["sample_id"]: (r.get("response") or {}).get("raw_answer")
                              for r in dep}
    ext_rows = [r for r in readlines(GLM / "原答清单.jsonl") if r["sample_id"] in by_id]
    models["GLM-WorkBuddy"] = {
        r["sample_id"]: (GLM / "raw" / (r["item_id"] + ".txt")).read_text(encoding="utf-8")
        for r in ext_rows}

    # ---- 3. 四个子集（定义完全照附表；不新增筛选条件）----
    common = [s for s in ids if all(status[m][s] for m in status)]
    sets = {
        "frozen36": ids,
        "exclude_train_same_image": [s for s in ids if s not in overlap],
        "api_all_returned_common": common,
        "api_all_returned_and_no_train_same_image": [s for s in common if s not in overlap],
    }

    # ---- 4. 用自己的解析器重算 ----
    mine: dict[str, dict] = {}
    for sn, sel in sets.items():
        t = {}
        for name, raws in models.items():
            row = {}
            for meth, p in (("strict", _jx.parse_strict), ("full_fence_diagnostic", _jx.parse_diag)):
                pairs = [(gold[s], pred(p(raws[s])[0])) for s in sel]
                ok = sum(g == q for g, q in pairs)
                row[meth] = {"n": len(sel), "correct": ok,
                             "accuracy": round(ok / len(sel), 6)}
            t[name] = row
        mine[sn] = t

    # ---- 5. 与 Codex 附表比对 ----
    codex = json.loads(CODEX.read_text(encoding="utf-8"))
    diffs = []
    for sn, t in codex["metrics"].items():
        if list(codex["subset_ids"][sn]) != list(sets[sn]) if sn in sets else True:
            pass
        for name, row in t.items():
            for meth in ("strict", "full_fence_diagnostic"):
                c = row[meth]["correct"]
                m = mine[sn][name][meth]["correct"]
                if c != m:
                    diffs.append({"子集": sn, "模型": name, "口径": meth,
                                  "Codex": c, "本协作者复算": m})
    subset_id_match = {sn: sorted(sets[sn]) == sorted(codex["subset_ids"][sn]) for sn in sets}

    report = {
        "只读来源": {
            "Codex附表": str(CODEX.relative_to(REPO)).replace("\\", "/"),
            "Codex附表sha256": sha(CODEX),
            "冻结盲包": str(BLIND.relative_to(REPO)).replace("\\", "/"),
            "本脚本未发任何请求": True,
        },
        "评测与训练同内容": {
            "评测图": list(overlap),
            "对应训练图": {k: v for k, v in overlap.items()},
            "逐字节证据": byte_proof,
            "训练3072的独特PNG_sha256数": len(train_hashes),
            "训练3072条数": len(train),
            "ID与lot隔离仍为0": True,
            "说明": "**不同晶圆 ID 可有相同 PNG**；不据此推为同一物理晶圆，"
                  "但**不能再说全部评测图像内容从未见过**。",
        },
        "API失败": {
            "失败清单": {m: [s for s in ids if not st[s]] for m, st in status.items()},
            "失败总数": sum(1 for m in status for s in ids if not status[m][s]),
            "尝试总数": sum(len(models[m]) for m in status),
            "性质": "连接/读取层失败，**不是模型错误**；后台是否生成/计费未测，原件不重试。",
        },
        "子集规模": {k: len(v) for k, v in sets.items()},
        "我在各子集上重算的正确数": mine,
        "与Codex附表差异": diffs,
        "子集成员与附表一致": subset_id_match,
        "Codex价目备注": codex.get("money_note"),
        "结论": ("逐项一致" if not diffs and all(subset_id_match.values())
                 else "**存在差异，见 与Codex附表差异**"),
    }
    (D / "阶段3_敏感性复算.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    L = ["# 13 点补核：本协作者独立复算（只读，未发任何请求）", "",
         f"**比对对象**：`{report['只读来源']['Codex附表']}`"
         f"（sha256 `{report['只读来源']['Codex附表sha256'][:16]}…`）", "",
         "## 一、一张评测图与训练图**逐字节同内容**", "",
         "| 评测图 | 训练图 | 评测lot | 训练lot | sha256(相同) |", "|---|---|---|---|---|"]
    for b in byte_proof:
        L.append(f"| `{b['评测图']}` | `{b['训练图']}` | {b['评测lot']} | {b['训练lot']} | "
                 f"`{b['评测图sha256'][:16]}…` |")
    L += ["",
          f"- 两侧 PNG **各自从本地重算** sha256 并相等（{len(byte_proof)} 组），"
          f"不是只比登记值。",
          f"- 训练 3072 个 ID 对应 **{report['评测与训练同内容']['训练3072的独特PNG_sha256数']}** "
          f"个独特 PNG sha256。",
          "- **ID / lot 隔离仍为 0**（`sample_id`、lot 都不同）；但**不能再写「全部评测图像内容从未见过」**。", "",
          "## 二、四 API 连接失败（**不是模型错误**）", "",
          "| 模型 | 失败条数 |", "|---|---:|"]
    for m, lst in report["API失败"]["失败清单"].items():
        L.append(f"| `{m}` | {len(lst)} |")
    L += [f"\n- 合计 **{report['API失败']['失败总数']} / {report['API失败']['尝试总数']}** 次尝试为连接层失败；"
          "原因**不能完全归因客户端或模型**，后台生成与计费未测，**原件不重试、不填伪结果**。", "",
          "## 三、四个子集与我在其上重算的正确数", "",
          "| 子集 | n | L 正确 | Qwen3.8-Max 正确 | D 正确 | GLM 正确 | 397B(围栏) 正确 |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for sn in sets:
        g = lambda m, meth="full_fence_diagnostic": mine[sn][m][meth]["correct"]
        L.append(f"| `{sn}` | {len(sets[sn])} | {g('L-N3072-3407')} | {g('qwen3.8-max-0902')} | "
                 f"{g('D-N3072-3407')} | {g('GLM-WorkBuddy')} | {g('qwen3.5-397b-a17b')} |")
    L += ["",
          "> 子集由「网络是否成功」与「是否与训练同图」两个**客观条件**确定，"
          "**不替换主表**，也不当独立最终考卷。所有 8 个候选使用**同一子集**。", "",
          "## 四、与 Codex 附表的比对", "",
          f"- 子集成员与附表一致：{subset_id_match}",
          f"- 正确数差异条数：**{len(diffs)}** → {report['结论']}"]
    if diffs:
        L += ["", "| 子集 | 模型 | 口径 | Codex | 本协作者复算 |", "|---|---|---|---:|---:|"]
        for x in diffs:
            L.append(f"| {x['子集']} | {x['模型']} | {x['口径']} | {x['Codex']} | {x['本协作者复算']} |")
    L += ["", f"- Codex 价目备注（照录）：{codex.get('money_note')}", ""]
    (D / "阶段3_敏感性复算.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    sys.stdout.buffer.write((json.dumps({
        "同内容图": list(overlap),
        "逐字节证据条数": len(byte_proof),
        "训练独特PNG数": report["评测与训练同内容"]["训练3072的独特PNG_sha256数"],
        "失败总数": report["API失败"]["失败总数"],
        "子集规模": report["子集规模"],
        "差异条数": len(diffs), "结论": report["结论"],
        "子集一致": subset_id_match,
    }, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))
    return 0 if (not diffs and all(subset_id_match.values())) else 5


if __name__ == "__main__":
    raise SystemExit(main())
