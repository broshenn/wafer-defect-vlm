#!/bin/bash
# 5090 同图 A/B/C —— CPU 门槛（0 GPU）
# A 数据与路径 · B 真实模板编码 · C batch 张量上的监督证据 · D 正式 rank16 配置
set -u
R=/WS
export CUDA_VISIBLE_DEVICES=""          # 必须在 Python 启动前设
export IMAGE_MAX_TOKEN_NUM=256
export PYTHONNOUSERSITE=1
export PYTHONIOENCODING=utf-8
exec $R/envs/wafer/bin/python - <<'PY'
import hashlib, io, json, os, re, sys
from collections import Counter

R = "/WS"
MODEL = R + "/models/Qwen3.5-9B"
OUT = R + "/logs"
EMPTY_THINK = re.compile(r"^\s*<think\b[^>]*>\s*</think\s*>\s*", re.S)
END_MARK = re.compile(r"(?:<\|im_end\|>|<\|endoftext\|>)\s*$")
SPECIAL = re.compile(r"<\|[A-Za-z_]+\|>")

res = []
def ck(n, ok, d=""):
    res.append({"项": n, "通过": bool(ok), "说明": str(d)})
    print(f"  {'OK  ' if ok else 'FAIL'} {n}  {d}")

groups, qses = {}, set()
for g in "ABC":
    p = f"{R}/datasets/abc_{g}.server.jsonl"
    rows = [json.loads(l) for l in io.open(p, encoding="utf-8") if l.strip()]
    groups[g] = rows
    qses.add(rows[0]["messages"][0]["content"])

print("=" * 76)
print("A. 数据与路径")
print("=" * 76)
print(f"  A/B/C 行数：{ {g: len(v) for g, v in groups.items()} }")
ck("三组各 356 行", all(len(v) == 356 for v in groups.values()))
ck("三组题面完全相同", len(qses) == 1, f"{len(qses)} 种")
Q = next(iter(qses))
qsha = hashlib.sha256(Q.encode()).hexdigest()
print(f"  题面 {len(Q)} 字符（含 <image>）  sha256 {qsha}")
ck("含 <image> 题面哈希 == a9eddba0…",
   qsha == "a9eddba0bddd1c1cd23c79b8ef6d80745112e98c5ab5ea09be98c4230036b1df", qsha[:16])
qpure = Q[len("<image>"):]
ck("纯题面哈希 == 6f710f5a…",
   hashlib.sha256(qpure.encode()).hexdigest() ==
   "6f710f5a63ed6f646ea5a16b9d7e1111392b671c2859c8a571c19d6e98c4ae6b")
ids = [set(r["sample_id"] for r in v) for v in groups.values()]
ck("三组 ID 集合一致", ids[0] == ids[1] == ids[2])
ck("每组 356 张唯一图", all(len(s) == 356 for s in ids))
ck("三组顺序一致", all([r["sample_id"] for r in groups[g]] ==
                      [r["sample_id"] for r in groups["A"]] for g in "ABC"))
for g in "ABC":
    miss = [r["sample_id"] for r in groups[g] if not os.path.exists(r["images"][0])]
    ck(f"{g} 组图片全部存在", not miss, f"缺 {len(miss)}")
# B 的字段名已改，A/C 未改
bt = json.loads(groups["B"][0]["messages"][1]["content"])["geometry"]
ck("B 已用新字段名", set(bt) == {"filtered_defect_centroid_radial_zone",
                                "filtered_defect_die_ratio", "extent_r", "clock_direction"},
   str(sorted(bt)))
ck("B 的 ratio 是 JSON 数字", isinstance(bt["filtered_defect_die_ratio"], (int, float))
   and not isinstance(bt["filtered_defect_die_ratio"], bool), str(bt["filtered_defect_die_ratio"]))
ck("A 目标只有 defect_class",
   set(json.loads(groups["A"][0]["messages"][1]["content"])) == {"defect_class"})
ck("C 目标含 caption_zh",
   "caption_zh" in json.loads(groups["C"][0]["messages"][1]["content"]))

print()
print("=" * 76)
print("B. 真实模板编码（逐条上报，无样本顶替）")
print("=" * 76)
from swift.model import get_model_processor
from swift.template import get_template
from swift import InferRequest
proc = get_model_processor(MODEL, model_type="qwen3_5", torch_dtype=None)[1]
tpl = get_template(proc, template_type="qwen3_5")
tok = proc.tokenizer
ck("thinking 已关闭", tpl.enable_thinking is False, f"enable_thinking={tpl.enable_thinking}")
for g in "ABC":
    errs, img, lens = [], Counter(), []
    for r in groups[g]:
        try:
            e = tpl.encode(InferRequest(messages=r["messages"], images=r["images"]),
                           return_length=True)
            img[sum(1 for t in e["input_ids"] if t == tpl.image_token_id)] += 1
            lens.append(len(e["input_ids"]))
        except Exception as ex:
            errs.append((r["sample_id"], f"{type(ex).__name__}: {ex}"))
    ck(f"{g} 组 356 条编码零失败", not errs, f"失败 {len(errs)}")
    ck(f"{g} 组每条 196 视觉 token", set(img) == {196}, dict(img))
    ck(f"{g} 组全部 < max_length 2048", bool(lens) and max(lens) < 2048, f"最长 {max(lens)}")

print()
print("=" * 76)
print("C. batch 张量上的监督证据（真实 collator）")
print("=" * 76)
import torch
from datasets import Dataset
from swift.dataset.utils import LazyLLMDataset
tpl.set_mode("train")
ck("模板已置 train 模式", tpl.mode == "train", tpl.mode)
sel = [(g, r["sample_id"]) for g, k in (("A", 7), ("B", 7), ("C", 6))
       for r in groups[g][:k]]
ck("20 条代表覆盖三组", len(sel) == 20 and len(set(g for g, _ in sel)) == 3,
   str(dict(Counter(g for g, _ in sel))))
byid = {(g, r["sample_id"]): r for g, rows in groups.items() for r in rows}
raw = Dataset.from_list([{"messages": byid[k]["messages"], "images": byid[k]["images"]} for k in sel])
enc = [LazyLLMDataset(raw, tpl.encode, strict=True, random_state=3407)[i] for i in range(len(sel))]
batch = tpl.data_collator(enc)
ck("collator 产出 input_ids/labels/attention_mask",
   all(k in batch for k in ("input_ids", "labels", "attention_mask")), str(sorted(batch.keys())))
BI, BL, AM = batch["input_ids"], batch["labels"], batch["attention_mask"]
print(f"  batch 形状 {tuple(BI.shape)}")
bad = []
for i, (g, sid) in enumerate(sel):
    real = AM[i].bool(); sup = real & (BL[i] != -100)
    if not bool(sup.any()): bad.append((sid, "无监督")); continue
    if not bool((BL[i][sup] == BI[i][sup]).all()): bad.append((sid, "labels错位"))
    if bool(((BI[i] == tpl.image_token_id) & real & (BL[i] != -100)).any()): bad.append((sid, "视觉被监督"))
    if not bool((BL[i][~real] == -100).all()): bad.append((sid, "padding非-100"))
    txt = END_MARK.sub("", EMPTY_THINK.sub("", tok.decode([int(t) for t in BI[i][sup].tolist()])))
    tgt = next(x for x in groups[g] if x["sample_id"] == sid)["messages"][1]["content"]
    if SPECIAL.search(txt): bad.append((sid, f"残留特殊标记 {SPECIAL.search(txt).group(0)}"))
    if txt != tgt: bad.append((sid, f"非精确相等 {txt[:40]!r} vs {tgt[:40]!r}"))
ck("20 条：对齐/仅assistant/视觉-100/padding-100/无残留/精确相等", not bad, f"异常 {bad[:2]}")

print()
print("=" * 76)
print("D. 正式 rank16 配置（CPU 建模）")
print("=" * 76)
VIS = re.compile(r"visual|vision|vit|aligner|merger|patch_embed", re.I)
from swift import SftArguments
from swift.pipelines import SwiftSft
args = SftArguments(model=MODEL, model_type="qwen3_5", template="qwen3_5",
                    dataset=f"{R}/datasets/abc_A.server.jsonl",
                    val_dataset=f"{R}/datasets/abc_A.server.jsonl",
                    tuner_type="lora", target_modules=["all-linear"],
                    freeze_vit=True, freeze_aligner=True, torch_dtype="bfloat16",
                    lora_rank=16, lora_alpha=32, lora_dropout=0.05,
                    max_length=2048, seed=3407, data_seed=3407, use_cpu=True,
                    output_dir=R + "/tmp/gate_probe")
ck("target_modules 是列表", isinstance(args.target_modules, list), str(args.target_modules))
pipe = SwiftSft(args)
m = pipe.prepare_model(args, pipe.model, template=pipe.template)
tr = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
vis_tr = [n for n, _ in tr if VIS.search(n)]
lora = [n for n, _ in tr if "lora_" in n.lower()]
ntrn = sum(p.numel() for _, p in tr)
tot = sum(p.numel() for p in m.parameters())
print(f"  总参数 {tot/1e6:.1f}M | 可训练 {ntrn/1e6:.3f}M | 张量 {len(tr)} | lora {len(lora)} | 视觉侧可训练 {len(vis_tr)}")
ck("视觉侧完全冻结", not vis_tr, f"{len(vis_tr)}")
ck("496 个可训练张量全是 LoRA", len(lora) == len(tr) == 496, f"{len(lora)}/{len(tr)}")
A_ = [(n, tuple(p.shape)) for n, p in tr if "lora_A" in n]
B_ = [(n, tuple(p.shape)) for n, p in tr if "lora_B" in n]
ck("实际张量确认 rank=16", set(s[0] for _, s in A_) == {16} and set(s[1] for _, s in B_) == {16},
   f"A首维{dict(Counter(s[0] for _, s in A_))} B次维{dict(Counter(s[1] for _, s in B_))}")
exp = sum(a*b for _, (a, b) in A_) + sum(a*b for _, (a, b) in B_)
ck("实测参数量 == 由形状推算", exp == ntrn, f"{exp} vs {ntrn}")
osz = R + "/tmp/gate_adapter"
os.makedirs(osz, exist_ok=True)
m.save_pretrained(osz)
cfg = json.load(io.open(osz + "/adapter_config.json", encoding="utf-8"))
ck("adapter_config r/alpha/dropout 正确",
   cfg.get("r") == 16 and cfg.get("lora_alpha") == 32 and abs(float(cfg.get("lora_dropout", -1)) - 0.05) < 1e-9,
   f"r={cfg.get('r')} a={cfg.get('lora_alpha')} d={cfg.get('lora_dropout')}")

fails = [x for x in res if not x["通过"]]
print()
print("=" * 76)
print(f"通过 {len(res)-len(fails)}/{len(res)}")
for x in fails: print(f"  未通过：{x['项']}  {x['说明']}")
os.makedirs(OUT, exist_ok=True)
json.dump({"通过": f"{len(res)-len(fails)}/{len(res)}", "检查": res,
           "题面_含image_sha256": qsha, "题面_纯_sha256": hashlib.sha256(qpure.encode()).hexdigest(),
           "note": "5090 CPU 门槛，0 GPU；未验证真实前反向与训练收敛，"
                   "那由正式训练的运行内 2 步门槛证明。"},
          io.open(OUT + "/abc_gate_5090.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"写出 {OUT}/abc_gate_5090.json")
sys.exit(0 if not fails else 1)
PY
