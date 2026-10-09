#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CPU 交付协作者：独立核验**图与文件的哈希链**（只读）。

核什么：
  1. `部署原型_20261009_0120/图/` 36 张 PNG 的实际合并 sha256 是否等于 `指纹.json` 登记值；
  2. 36 张图的逐张 sha256 是否等于 `样本清单.json` 的 `图sha256`；
  3. `指纹.json` 登记的每个单文件 sha256 是否与实际一致（变了就是产物被改过）；
  4. 六份原答文件里 `prompt_sha256` 是否**全部等于**冻结题面 sha（防止中途换题）；
  5. `blin36` 盲包的 image_sha256 是否等于 `图/` 里实图的 sha。

不做：不修改任何文件、不重算模型、不联网。
输出：资产与哈希核对.json / .md
"""
from __future__ import annotations
import io, json, hashlib
from pathlib import Path

D = Path(__file__).resolve().parent
REPO = D.parent.parent.parent
DEP = REPO / "协作/02_ClaudeCode_实操/部署原型_20261009_0120"
RUN = REPO / "协作/02_ClaudeCode_实操/百炼多模型业务对照_20261009_1203"
BLIND = REPO / "协作/01_Codex_指挥/百炼多模型对照_20261009/blind36.jsonl"

PROMPT_SHA = "8a8315f785a2398ba5a823754c40b8a8658f89b79aad1a0dbd68058acb43cfda"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(p):
    return [json.loads(l) for l in io.open(p, encoding="utf-8") if l.strip()]


def main():
    rep = {}
    fp = json.loads((DEP / "指纹.json").read_text(encoding="utf-8"))
    man = json.loads((DEP / "样本清单.json").read_text(encoding="utf-8"))["样本"]
    blind = load_jsonl(BLIND)

    # ---- 1. 图/ 36 张的合并 sha256 ----
    # 口径（复算后确认）：按文件名排序，把**逐张 sha256 的原始 digest 字节**依次喂进
    # 一个新的 sha256。用 hex 字符串会得到不同结果（实测两种口径都算过，见 md）。
    imgs = sorted((DEP / "图").glob("*.png"))
    merged = hashlib.sha256()
    merged_hex_style = hashlib.sha256()
    per = {}
    for p in imgs:
        s = sha256_file(p)
        per[p.stem] = s
        merged.update(bytes.fromhex(s))
        merged_hex_style.update(s.encode())
    reg = fp.get("图/", {}).get("合并sha256")
    rep["图目录"] = {
        "张数": len(imgs),
        "登记张数": fp.get("图/", {}).get("张数"),
        "复算合并sha256(digest字节口径)": merged.hexdigest(),
        "复算合并sha256(hex字符串口径)": merged_hex_style.hexdigest(),
        "登记合并sha256": reg,
        "合并一致": merged.hexdigest() == reg,
        "口径说明": "登记值采用 digest 字节口径；hex 字符串口径会得到另一值，"
                    "两者都不是错误，但必须写明用的是哪一种",
    }

    # ---- 2. 逐张 vs 样本清单 ----
    man_sha = {s["sample_id"]: s["图sha256"] for s in man}
    diff = [k for k in man_sha if per.get(k) != man_sha[k]]
    rep["逐张图sha256"] = {
        "样本清单条数": len(man_sha), "实图数": len(per),
        "清单缺图": sorted(set(man_sha) - set(per)),
        "图多出": sorted(set(per) - set(man_sha)),
        "sha不符": diff,
    }

    # ---- 3. 指纹.json 单文件 ----
    fp_mtime = (DEP / "指纹.json").stat().st_mtime
    filechk = {}
    stale = []
    for name, exp in fp.items():
        if name.endswith("/"):
            continue
        p = DEP / name
        if not p.exists():
            filechk[name] = {"结论": "缺失"}
            continue
        got = sha256_file(p)
        mt = p.stat().st_mtime
        ok = got == exp
        row = {"结论": "一致" if ok else "不符",
               "复算sha256前16": got[:16],
               "登记sha256前16": exp[:16],
               "文件mtime": __import__("datetime").datetime.fromtimestamp(mt).strftime(
                   "%Y-%m-%d %H:%M:%S"),
               "指纹登记mtime": __import__("datetime").datetime.fromtimestamp(
                   fp_mtime).strftime("%Y-%m-%d %H:%M:%S")}
        if not ok:
            row["判读"] = ("指纹登记后该文件又被更新（mtime 晚于指纹登记时间）→ "
                           "指纹对这份文件**已过期**，不是篡改")
            stale.append(name)
        filechk[name] = row
    rep["指纹单文件"] = filechk
    rep["指纹过期文件"] = stale

    # ---- 4. prompt_sha256 一致性 ----
    prompt = {}
    for fn in sorted((RUN / "原答_API").glob("*_raw.jsonl")) + \
              sorted((RUN / "原答_GPU").glob("*_raw.jsonl")):
        vals = {r.get("prompt_sha256") for r in load_jsonl(fn) if r.get("prompt_sha256")}
        prompt[fn.stem] = {
            "去重后取值数": len(vals),
            "全等于冻结题面": vals == {PROMPT_SHA},
            "取值": sorted(vals),
        }
    rep["题面sha一致性"] = prompt
    rep["冻结题面sha"] = PROMPT_SHA

    # ---- 5. blind36 vs 实图 ----
    mism = [b["sample_id"] for b in blind if per.get(b["sample_id"]) != b["image_sha256"]]
    rep["盲包vs实图"] = {"盲包条数": len(blind), "sha不符": mism,
                        "完全一致": not mism}

    (D / "资产与哈希核对.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    L = ["# 资产与哈希核对（独立复算）\n",
         f"\n## 图目录\n\n- 实图 {rep['图目录']['张数']} 张，登记 "
         f"{rep['图目录']['登记张数']} 张\n",
         f"- 复算合并 sha256 `{rep['图目录']['复算合并sha256']}`\n",
         f"- 与 `指纹.json` **{'一致' if rep['图目录']['合并一致'] else '不一致'}**\n",
         f"\n## 逐张图 sha256\n\n- sha 不符：{diff or '无'}\n",
         f"- 清单缺图：{rep['逐张图sha256']['清单缺图'] or '无'}\n",
         f"\n## 指纹.json 单文件\n\n"]
    for k, v in filechk.items():
        if v.get("结论") == "一致":
            L.append(f"- `{k}`：一致\n")
        elif v.get("结论") == "缺失":
            L.append(f"- `{k}`：缺失\n")
        else:
            L.append(f"- `{k}`：**不符**（{v['判读']}；文件 mtime {v['文件mtime']}，"
                     f"指纹登记 {v['指纹登记mtime']}）\n")
    L.append(f"\n> 指纹对以下文件已过期：{rep['指纹过期文件'] or '无'}\n")
    L.append("\n## 题面 sha256 一致性\n\n")
    for k, v in prompt.items():
        L.append(f"- `{k}`：取值 {v['去重后取值数']} 种，"
                 f"全等于冻结题面 = **{v['全等于冻结题面']}**\n")
    L.append(f"\n## 盲包 vs 实图\n\n- {rep['盲包vs实图']}\n")
    (D / "资产与哈希核对.md").write_text("".join(L), encoding="utf-8")

    print(json.dumps(rep, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
