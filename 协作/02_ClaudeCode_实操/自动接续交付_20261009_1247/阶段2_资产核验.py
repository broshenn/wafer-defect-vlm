#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CPU 交付协作者 · 阶段 2：公开数据资产的**真实图数**库存核验（只读）。

口径纪律：
  · 只读；**不写主会话目录**；不改原件；历史绝对路径只在**新清单**里映射成本地相对路径。
  · **不凑大数**：4292 / 1200 都是派生集合，不与 5904 / 3072 相加。
  · **跑不动就报跑不动**：本地没有的文件（4292 原始 train.jsonl）明确写「本地无法独立复算」。

核验内容：
  A. 5904：逐文件存在性 + 字节数 + sha256 + **图片内容级重复统计**（本项目首次做全库）
  B. split × lot 交集（三划分两两）
  C. 3072：唯一图数、L/D 输入逐图相同、⊂ 5904、与评测 102 的样本/lot 隔离
  D. 1200：唯一 item / 唯一图 / 唯一 sha、与 3072 的重叠（是否 1200/1200）
  E. 4292：本地可得性判定（预期「不可独立复算」）
  F. 新清单：5904 的本地相对路径 manifest（原件不动）

输出：阶段2_资产核验.json / .md、清单_5904_本地相对路径.jsonl、清单_1200_本地相对路径.jsonl
"""
from __future__ import annotations
import hashlib, io, json, sys
from collections import Counter, defaultdict
from pathlib import Path

D = Path(__file__).resolve().parent
ROOT = D.parents[2]                      # D:\pycode\晶圆图研究
IMG = ROOT / "data" / "images"
MAN = ROOT / "data" / "manifest.jsonl"
SUM = ROOT / "data" / "manifest_summary.json"
SRC = ROOT / "data" / "source_stats.json"
INV = ROOT / "协作/01_Codex_指挥/本地准备_20261002_断连期间/public_data_upload_inventory.jsonl"
N3072 = ROOT / "协作/02_ClaudeCode_实操/夜间扩容_20261008_0244/数据"
GAP1200 = ROOT / "协作/02_ClaudeCode_实操/夜间扩容_20261008_0244/客户端盲包/WorkBuddy_训练缺口1200"
EVAL102 = ROOT / "协作/02_ClaudeCode_实操/5090同图ABC_20261008_0100/评测/评测集_102.jsonl"


def jl(p: Path):
    with io.open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    out, md = {}, []

    # ================= A. 5904 库存 =================
    rows = list(jl(MAN))
    ids = [r["sample_id"] for r in rows]
    uniq_ids = set(ids)
    sha_by_id, missing, size_by_id = {}, [], {}
    for sid in sorted(uniq_ids):        # 排序遍历：set 的迭代顺序随哈希种子变化，
                                        # 会让等长重复组在两次运行间换位（同计数、不同顺序）

        p = IMG / f"{sid}.png"
        if not p.exists():
            missing.append(sid)
            continue
        size_by_id[sid] = p.stat().st_size
        sha_by_id[sid] = sha256_file(p)

    sha2ids = defaultdict(list)
    for sid, s in sha_by_id.items():
        sha2ids[s].append(sid)
    dup_groups = {s: sorted(v) for s, v in sha2ids.items() if len(v) > 1}
    content_dups = sum(len(v) - 1 for v in dup_groups.values())

    # 与公开上传清单对照
    inv_rows = list(jl(INV))
    inv_png = {r["relative_path"]: r for r in inv_rows if r["relative_path"].endswith(".png")}
    inv_mismatch = [
        sid for sid, s in sha_by_id.items()
        if (inv_png.get(f"data/images/{sid}.png") or {}).get("sha256") not in (None, s)
    ]

    A = {
        "manifest行数": len(rows),
        "唯一sample_id": len(uniq_ids),
        "本地图片存在": len(sha_by_id),
        "缺失图片": missing,
        "唯一图片sha256数": len(sha2ids),
        "图片内容重复组数": len(dup_groups),
        "重复冗余张数(超出首张)": content_dups,
        "图片总字节": sum(size_by_id.values()),
        "图片字节最小/最大": [min(size_by_id.values()), max(size_by_id.values())] if size_by_id else None,
        "公开清单条数(含元数据)": len(inv_rows),
        "公开清单PNG条数": len(inv_png),
        "公开清单与本地sha不符": inv_mismatch,
        "重复组明细(最多列20组)": [
            {"sha256前16": s[:16], "张数": len(v), "sample_ids": v[:8]}
            for s, v in sorted(dup_groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:20]
        ],
    }
    out["A_5904库存"] = A
    md.append("# 阶段 2：公开数据资产核验（只读·不凑大数）\n")
    md.append("\n## A. 5904 库存核验\n\n")
    md.append(f"- manifest 行数 **{len(rows)}**，唯一 `sample_id` **{len(uniq_ids)}** → "
              f"{'**主键唯一**' if len(rows) == len(uniq_ids) else '**存在重复主键**'}\n")
    md.append(f"- 本地图片存在 **{len(sha_by_id)}/{len(uniq_ids)}**"
              f"{'（无缺图）' if not missing else f'，**缺 {len(missing)} 张**：{missing[:5]}'}\n")
    md.append(f"- **唯一图片 sha256 数 {len(sha2ids)}** → 图片内容重复组 **{len(dup_groups)}** 组，"
              f"冗余 **{content_dups}** 张\n")
    md.append(f"- 图片合计 **{sum(size_by_id.values())/1048576:.2f} MiB**，"
              f"单张 {min(size_by_id.values())}–{max(size_by_id.values())} 字节\n")
    md.append(f"- 与公开上传清单对照：清单 PNG 条 **{len(inv_png)}**，"
              f"sha 不符 **{len(inv_mismatch)}** 张\n")
    if dup_groups:
        md.append("\n### 图片内容重复明细（同 sha、不同 sample_id）\n\n")
        md.append("| sha256前16 | 张数 | sample_ids |\n|---|---:|---|\n")
        for s, v in sorted(dup_groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:20]:
            md.append(f"| `{s[:16]}` | {len(v)} | {', '.join(v[:8])}"
                      f"{'…' if len(v) > 8 else ''} |\n")
    else:
        md.append("\n> **5904 全库图片内容级去重：无重复** —— 此前「未测」的这一项**本阶段已实测**。\n")

    # ================= B. split × lot 交集 =================
    lot_of = defaultdict(set)
    split_counts = Counter()
    for r in rows:
        lot_of[r["split"]].add(r.get("lot_name"))
        split_counts[r["split"]] += 1
    lots = {k: v for k, v in lot_of.items()}
    inter = {
        f"{a}&{b}": sorted(lots.get(a, set()) & lots.get(b, set()))
        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    B = {
        "各划分样本数": dict(split_counts),
        "各划分唯一lot数": {k: len(v) for k, v in lots.items()},
        "lot交集": {k: {"个数": len(v), "样例": v[:5]} for k, v in inter.items()},
        "全部lot数": len(set().union(*lots.values())) if lots else 0,
        "单lot最大张数": max(Counter(r.get("lot_name") for r in rows).values()),
    }
    out["B_split_lot交集"] = B
    md.append("\n## B. split × lot 交集（本机复算）\n\n")
    md.append("| 划分 | 样本数 | 唯一 lot |\n|---|---:|---:|\n")
    for k in ("train", "val", "test"):
        md.append(f"| {k} | {B['各划分样本数'].get(k, 0)} | {B['各划分唯一lot数'].get(k, 0)} |\n")
    md.append(f"\n- lot 交集：train∩val **{len(inter['train&val'])}**、"
              f"train∩test **{len(inter['train&test'])}**、"
              f"val∩test **{len(inter['val&test'])}** → "
              f"{'**三划分 lot 完全隔离**' if not any(inter.values()) else '**存在 lot 重叠**'}\n")
    md.append(f"- 全库唯一 lot **{B['全部lot数']}**，单 lot 最多 **{B['单lot最大张数']}** 张\n")

    # ================= C. 3072 =================
    L = list(jl(N3072 / "L_N3072.jsonl"))
    Dd = list(jl(N3072 / "D_N3072.jsonl"))
    l_ids = [r["sample_id"] for r in L]
    d_ids = [r["sample_id"] for r in Dd]
    meta3072 = json.loads((N3072 / "训练数据_N3072.json").read_text(encoding="utf-8"))
    n3072 = set(l_ids)
    ev102 = list(jl(EVAL102))
    e102_ids = {r["sample_id"] for r in ev102}
    e102_lots = {r.get("lot_name") for r in ev102}
    train_lots = {r.get("lot_name") for r in rows if r["split"] == "train"}
    img_paths = {r["images"][0] for r in L if r.get("images")}
    C = {
        "L行数": len(L), "D行数": len(Dd),
        "L唯一sample_id": len(n3072), "D唯一sample_id": len(set(d_ids)),
        "L_D输入的sample_id序列完全相同": l_ids == d_ids,
        "L本地sha256": sha256_file(N3072 / "L_N3072.jsonl"),
        "D本地sha256": sha256_file(N3072 / "D_N3072.jsonl"),
        "登记L本地sha256": meta3072.get("L", {}).get("本地sha256"),
        "登记D本地sha256": meta3072.get("D", {}).get("本地sha256"),
        "sha与登记一致": (sha256_file(N3072 / "L_N3072.jsonl") == meta3072.get("L", {}).get("本地sha256")
                          and sha256_file(N3072 / "D_N3072.jsonl") == meta3072.get("D", {}).get("本地sha256")),
        "3072⊂5904": len(n3072 - uniq_ids) == 0,
        "3072不在5904中的sample_id数": len(n3072 - uniq_ids),
        "3072来源描述": meta3072.get("来源"),
        "评测102条数": len(ev102), "评测102唯一sample_id": len(e102_ids),
        "3072∩评测102": len(n3072 & e102_ids),
        "3072lot数": len({r.get("lot_name") for r in rows if r["sample_id"] in n3072}),
        "3072lot∩评测102lot": len({r.get("lot_name") for r in rows if r["sample_id"] in n3072} & e102_lots),
        "历史绝对路径样例": sorted(img_paths)[:3],
    }
    out["C_3072"] = C
    md.append("\n## C. 3072（本轮训练池）\n\n")
    md.append(f"- L/D 行数 **{len(L)} / {len(Dd)}**，唯一 `sample_id` **{len(n3072)} / {len(set(d_ids))}**\n")
    md.append(f"- L 与 D 的 `sample_id` 序列**逐图相同** = **{C['L_D输入的sample_id序列完全相同']}**"
              f"（这是「只改监督答案」对照可解释的前提）\n")
    md.append(f"- 本地 sha256 与登记一致 = **{C['sha与登记一致']}**\n")
    md.append(f"- **3072 ⊂ 5904 = {C['3072⊂5904']}**（不在 5904 中的：{C['3072不在5904中的sample_id数']} 个）\n")
    md.append(f"- 与评测 102：样本交集 **{C['3072∩评测102']}**，lot 交集 **{C['3072lot∩评测102lot']}**\n")
    md.append(f"- 训练池涉及唯一 lot **{C['3072lot数']}**（评测 102 为 "
              f"{len(e102_lots)} lot，二者交集为 {C['3072lot∩评测102lot']}）\n")
    md.append(f"- 历史绝对路径样例（**只在新清单中映射、原件不动**）：`{C['历史绝对路径样例'][0]}`\n")

    # ================= D. 1200 =================
    # 只读「分片」文件；`WB_gap1200_索引.jsonl` 是同一批记录的索引副本，读进来会翻倍
    g_rows, g_shards, g_index = [], [], None
    for p in sorted(GAP1200.glob("*分片*.jsonl")):
        n = 0
        for r in jl(p):
            g_rows.append(r)
            n += 1
        g_shards.append({"分片": p.name, "条数": n})
    idxp = GAP1200 / "WB_gap1200_索引.jsonl"
    if idxp.exists():
        idx_ids = {r["item_id"] for r in jl(idxp)}
        shard_ids = {r["item_id"] for r in g_rows}
        g_index = {"文件": idxp.name, "行数": len(list(jl(idxp))),
                   "唯一item_id": len(idx_ids),
                   "与分片集合同一": idx_ids == shard_ids,
                   "说明": "索引与 172 个分片是**同一批记录的两份存放**；只按分片计数，避免翻倍"}
    g_items = {r["item_id"] for r in g_rows}
    g_sids = {r["sample_id"] for r in g_rows}
    g_sha = {r.get("image_sha256") for r in g_rows if r.get("image_sha256")}
    g_sha2sid = defaultdict(set)
    for r in g_rows:
        g_sha2sid[r.get("image_sha256")].add(r["sample_id"])
    g_dupsha = {s: sorted(v) for s, v in sorted(g_sha2sid.items()) if len(v) > 1 and s}
    D_ = {
        "分片数": len(g_shards), "记录数": len(g_rows),
        "唯一item_id": len(g_items), "唯一sample_id": len(g_sids),
        "唯一图片sha256": len(g_sha),
        "同图不同ID组数": len(g_dupsha),
        "索引文件": g_index,
        "同图不同ID明细": [{"sha前16": s[:16], "sample_ids": v} for s, v in g_dupsha.items()],
        "1200∩3072(sample_id)": len(g_sids & n3072),
        "1200唯一图是否全部⊂3072": len(g_sids & n3072) == len(g_sids),
        "1200唯一图是否全部⊂5904": len(g_sids & uniq_ids) == len(g_sids),
        "1200∩评测102": len(g_sids & e102_ids),
        "分片清单": g_shards[:8] + [{"…": f"共 {len(g_shards)} 片"}],
    }
    out["D_1200"] = D_
    md.append("\n## D. 1200（训练缺口候选，**未入训**）\n\n")
    md.append(f"- 分片 **{len(g_shards)}**，记录 **{len(g_rows)}**，唯一 `item_id` **{len(g_items)}**，"
              f"唯一 `sample_id` **{len(g_sids)}**，唯一图片 sha **{len(g_sha)}**\n")
    md.append(f"- 同图不同 ID **{len(g_dupsha)}** 组：{D_['同图不同ID明细']}\n")
    md.append(f"- **1200 ∩ 3072 = {len(g_sids & n3072)} / {len(g_sids)}** → "
              f"{'**整批落在已训训练池内（不得与 3072 相加）**' if D_['1200唯一图是否全部⊂3072'] else '**部分在池外**'}\n")
    md.append(f"- 1200 ∩ 评测 102 = **{len(g_sids & e102_ids)}**\n")
    if g_index:
        md.append(f"- 索引文件 `{g_index['文件']}`：{g_index['行数']} 行、唯一 item "
                  f"{g_index['唯一item_id']}，与分片集合同一 = **{g_index['与分片集合同一']}**"
                  f"（{g_index['说明']}）\n")

    # ================= E. 4292 =================
    cand = list(ROOT.rglob("curated_v2/splits/train.jsonl"))
    E = {
        "本地是否存在 4292 原始文件": bool(cand),
        "本地路径": [str(p.relative_to(ROOT)) for p in cand],
        "判定": ("**本地无法独立复算** —— 本机与两台学校服务器均无副本，"
                 "恢复目标为租用实例；**本阶段不做 SSH**，故只能引用《旧资产恢复.md》登记值"),
        "登记值(引用)": {"行数": 12876, "唯一晶圆": 4292,
                        "sha256": "8dfb6b0f8660306b4db9ae05481ef7dfe5c95e49607030fa06b158882fb71898"},
        "可间接佐证": "3072 的来源字段写明「旧 curated_v2/splits/train.jsonl 的 structured 任务（4292 晶圆）」"
                      "且 3072 ⊂ 5904 已本机复算为真",
    }
    out["E_4292"] = E
    md.append("\n## E. 4292（旧 SFT train 划分）\n\n")
    md.append(f"- 本地存在原始文件：**{E['本地是否存在 4292 原始文件']}**\n")
    md.append(f"- 判定：{E['判定']}\n")
    md.append(f"- 登记值（**引用，非本机复算**）：行数 12876 = 4292 晶圆 × 3 任务，"
              f"sha256 `{E['登记值(引用)']['sha256'][:16]}…`\n")
    md.append(f"- 间接佐证：{E['可间接佐证']}\n")

    # ================= F. 新清单（本地相对路径） =================
    newman = D / "清单_5904_本地相对路径.jsonl"
    with io.open(newman, "w", encoding="utf-8") as f:
        for r in rows:
            sid = r["sample_id"]
            rec = {
                "sample_id": sid,
                "split": r.get("split"),
                "failure_type": r.get("failure_type"),
                "lot_name": r.get("lot_name"),
                "label_source": r.get("label_source"),
                "local_relative_path": f"data/images/{sid}.png",
                "source_absolute_path_historical": r.get("image_path"),
                "bytes": size_by_id.get(sid),
                "sha256": sha_by_id.get(sid),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    newg = D / "清单_1200_本地相对路径.jsonl"
    with io.open(newg, "w", encoding="utf-8") as f:
        for r in g_rows:
            f.write(json.dumps({
                "item_id": r["item_id"], "sample_id": r["sample_id"],
                "local_relative_path": f"data/images/{r['sample_id']}.png",
                "source_absolute_path_historical": r.get("image_path"),
                "image_sha256": r.get("image_sha256"),
            }, ensure_ascii=False) + "\n")
    F = {
        "清单_5904_本地相对路径.jsonl": {"行数": len(rows), "sha256": sha256_file(newman)},
        "清单_1200_本地相对路径.jsonl": {"行数": len(g_rows), "sha256": sha256_file(newg)},
        "说明": "历史绝对路径保留在 *_historical 字段仅供溯源；**新清单主字段是本地相对路径**；原件未改动",
    }
    out["F_新清单"] = F
    md.append("\n## F. 新清单（本地相对路径，原件不动）\n\n")
    for k, v in F.items():
        if isinstance(v, dict):
            md.append(f"- `{k}`：{v['行数']} 行，sha256 `{v['sha256'][:16]}…`\n")
    md.append(f"\n> {F['说明']}\n")

    (D / "阶段2_资产核验.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (D / "阶段2_资产核验.md").write_text("".join(md), encoding="utf-8")

    summary = {
        "A唯一图sha": A["唯一图片sha256数"], "A重复组": A["图片内容重复组数"],
        "A缺图": len(missing), "B_lot交集": {k: len(v) for k, v in inter.items()},
        "C_3072唯一": len(n3072), "C_3072_in_5904": C["3072⊂5904"],
        "C_3072_and_102": C["3072∩评测102"], "D_1200唯一图": len(g_sids),
        "D_1200_and_3072": D_["1200∩3072(sample_id)"], "E_4292本地有": E["本地是否存在 4292 原始文件"],
    }
    sys.stdout.buffer.write(
        (json.dumps(summary, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
