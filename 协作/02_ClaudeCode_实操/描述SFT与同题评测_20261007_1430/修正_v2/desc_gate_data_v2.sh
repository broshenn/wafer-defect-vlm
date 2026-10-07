#!/bin/bash
# 描述 SFT CPU 门槛（修正版 v2，Codex 2026-10-07 §2）—— 不建模型，0 GPU
# A 数据与路径 · B 真实模板编码 · C **batch 张量**上的监督证据
# `/WS` 是真实私有工作区的别名，服务器上的可执行副本用真实路径。
set -u
R=/WS
export CUDA_VISIBLE_DEVICES=""           # 必须在 Python 启动前设，否则 device_map='auto' 会占共享卡
export IMAGE_MAX_TOKEN_NUM=256
export PYTHONIOENCODING=utf-8
exec $R/envs/wafer/bin/python - <<'PY'
import hashlib, io, json, os, re, sys
from collections import Counter

R = "/WS"
DATA = R + "/datasets/abc/desc_v2.server.jsonl"
SHA = R + "/datasets/abc/desc_v2_sid_sha.tsv"
TYP = R + "/datasets/abc/desc_v2_typing.jsonl"
MODEL = R + "/models/Qwen3.5-9B"
OUT = R + "/logs"

EMPTY_THINK = re.compile(r"^\s*<think\b[^>]*>\s*</think\s*>\s*", re.S)
# 模板正常结束标记。实测监督段末尾为 ..."}<|im_end|>\n —— 结束标记之后还有换行，
# 故不能按「最后一个 token 是 eos」判断，必须按标记文本剥离。
END_MARK = re.compile(r"(?:<\|im_end\|>|<\|endoftext\|>)\s*$")
SPECIAL = re.compile(r"<\|[A-Za-z_]+\|>")

res = []
def ck(n, ok, d=""):
    res.append({"项": n, "通过": bool(ok), "说明": str(d)})
    print(f"  {'OK  ' if ok else 'FAIL'} {n}  {d}")

rows = [json.loads(l) for l in io.open(DATA, encoding="utf-8") if l.strip()]
typ = {json.loads(l)["sample_id"]: json.loads(l) for l in io.open(TYP, encoding="utf-8") if l.strip()}
sha_tab = dict(l.rstrip("\n").split("\t") for l in io.open(SHA, encoding="utf-8") if l.strip())

print("=" * 76)
print("A. 数据与路径")
print("=" * 76)
ck("行数 356", len(rows) == 356, str(len(rows)))
ck("ID 唯一", len({r["sample_id"] for r in rows}) == len(rows))
ck("分型覆盖全部行", set(typ) == {r["sample_id"] for r in rows})

miss, bad, wrongpath = [], [], []
for r in rows:
    p = r["images"][0]
    if p != f"{R}/datasets/expand_v2/images/{r['sample_id']}.png":
        wrongpath.append((r["sample_id"], p))
    if not os.path.exists(p):
        miss.append(r["sample_id"]); continue
    if hashlib.sha256(open(p, "rb").read()).hexdigest() != sha_tab.get(r["sample_id"]):
        bad.append(r["sample_id"])
ck("路径全部规范", not wrongpath, f"异常 {len(wrongpath)}")
ck("356 张图片全部存在", not miss, f"缺 {len(miss)} {miss[:3]}")
ck("356 张图片 sha256 与 固定356条 逐条一致", not bad, f"不符 {len(bad)} {bad[:3]}")

print()
print("=" * 76)
print("B. 真实模板编码（逐条上报，不允许换样本）")
print("=" * 76)
from swift.model import get_model_processor
from swift.template import get_template
from swift import InferRequest

proc = get_model_processor(MODEL, model_type="qwen3_5", torch_dtype=None)[1]
tpl = get_template(proc, template_type="qwen3_5")
tok = proc.tokenizer
ck("thinking 已关闭（实际模板值）", tpl.enable_thinking is False, f"enable_thinking={tpl.enable_thinking}")

errs, img_tok, lens = [], Counter(), []
for r in rows:
    try:
        e = tpl.encode(InferRequest(messages=r["messages"], images=r["images"]), return_length=True)
        img_tok[sum(1 for t in e["input_ids"] if t == tpl.image_token_id)] += 1
        lens.append(len(e["input_ids"]))
    except Exception as ex:
        errs.append((r["sample_id"], f"{type(ex).__name__}: {ex}"))
ck("356 条逐条编码零失败（无样本顶替）", not errs, f"失败 {len(errs)} {errs[:2]}")
ck("每条恰好 196 个 image token", set(img_tok) == {196}, dict(img_tok))
ck("全部短于 max_length 2048", bool(lens) and max(lens) < 2048, f"最长 {max(lens) if lens else '-'}")

print()
print("=" * 76)
print("C. batch 张量上的监督证据（修正版：用 collator 输出，不用 encode 结果）")
print("=" * 76)
import torch
from datasets import Dataset
from swift.dataset.utils import LazyLLMDataset

tpl.set_mode("train")
ck("模板已置 train 模式（labels 的来源）", tpl.mode == "train", tpl.mode)

sel = []
for kind, want in (("结构化_含不确定", 8), ("结构化_无不确定", 6), ("类别Caption辅助", 6)):
    sel += [r["sample_id"] for r in rows if typ[r["sample_id"]]["题面种类"] == kind][:want]
dir_rows = [r["sample_id"] for r in rows
            if isinstance(json.loads(r["messages"][1]["content"]).get("clock_direction"), str)]
for s in dir_rows:
    if s not in sel:
        sel[-1] = s
        break
sel = sel[:20]
byid = {r["sample_id"]: r for r in rows}
print(f"  {len(sel)} 条：{dict(Counter(typ[s]['题面种类'] for s in sel))}"
      f"  含方向有值 {sum(1 for s in sel if s in dir_rows)}")
ck("20 条覆盖三种任务", len(sel) == 20 and len(set(typ[s]["题面种类"] for s in sel)) == 3)

raw = Dataset.from_list([{"messages": byid[s]["messages"], "images": byid[s]["images"]} for s in sel])
lazy = LazyLLMDataset(raw, tpl.encode, strict=True, random_state=3407)
enc = [lazy[i] for i in range(len(sel))]
batch = tpl.data_collator(enc)
ck("collator 产出 input_ids/labels/attention_mask/pixel_values/image_grid_thw",
   all(k in batch for k in ("input_ids", "labels", "attention_mask",
                            "pixel_values", "image_grid_thw")), str(sorted(batch.keys())))

BI, BL, AM = batch["input_ids"], batch["labels"], batch["attention_mask"]
print(f"  batch 形状 input_ids {tuple(BI.shape)}  labels {tuple(BL.shape)}  "
      f"attention_mask {tuple(AM.shape)}  (batch={len(sel)})")
ck("input_ids / labels / attention_mask 形状一致", BI.shape == BL.shape == AM.shape)
print(f"  模板结束标记 eos_token_id = {tok.eos_token_id} ({tok.decode([tok.eos_token_id])!r})", flush=True)

bad_align, bad_vis, bad_pad, bad_exact = [], [], [], []
detail = []
for i, sid in enumerate(sel):
    ids, lab, am = BI[i], BL[i], AM[i]
    real = am.bool()
    pad = ~real
    # 1) padding 位置必须全 -100
    if int(pad.sum()) and not bool((lab[pad] == -100).all()):
        bad_pad.append(sid)
    # 2) 监督位置：真实位置且 labels != -100
    sup = real & (lab != -100)
    if not bool(sup.any()):
        bad_exact.append((sid, "无监督位置")); continue
    # 3) 非 -100 的 labels 必须等于同位置的 input_ids
    if not bool((lab[sup] == ids[sup]).all()):
        bad_align.append(sid)
    # 4) 视觉 token 位置必须全 -100
    vmask = real & (ids == tpl.image_token_id)
    if bool(vmask.any()) and bool((lab[vmask] != -100).any()):
        bad_vis.append(sid)
    # 5) 监督内容精确等于 assistant 目标（只剥离模板的正常空 think 与结束标记）
    #    实测监督段末尾是 ..."}<|im_end|>\n —— 结束标记之后还有换行，
    #    故不能按「最后一个 token 是 eos」判断，须按模板结束标记剥离。
    keep = [int(t) for t in ids[sup].tolist()]
    raw_txt = tok.decode(keep)
    txt = END_MARK.sub("", EMPTY_THINK.sub("", raw_txt))
    tgt = byid[sid]["messages"][1]["content"]
    if txt != tgt:
        k = next((j for j, (x, y) in enumerate(zip(txt, tgt)) if x != y), min(len(txt), len(tgt)))
        bad_exact.append((sid, f"长度 {len(txt)} vs {len(tgt)} 首差位置 {k} "
                               f"片段 {txt[max(0,k-8):k+8]!r} vs {tgt[max(0,k-8):k+8]!r}"))
    if SPECIAL.search(txt):
        bad_exact.append((sid, f"剥离后仍含特殊标记 {SPECIAL.search(txt).group(0)!r}"))
    detail.append({"sample_id": sid, "题面种类": typ[sid]["题面种类"],
                   "batch_len": int(real.sum()), "监督token数": int(sup.sum()),
                   "监督段原样尾部": raw_txt[-24:],
                   "剥离后解码": txt,
                   "与目标精确相等": txt == tgt})

ck("padding 位置全 -100", not bad_pad, f"异常 {bad_pad[:2]}")
ck("非 -100 的 labels 等于同位置 input_ids", not bad_align, f"异常 {bad_align[:2]}")
ck("视觉 token 位置全 -100", not bad_vis, f"异常 {bad_vis[:2]}")
ck("监督内容精确等于 assistant 目标（剥离空 think 与 eos）", not bad_exact,
   f"异常 {bad_exact[:2]}")

print("\n  三种任务各取一条的实际 batch 解码：")
for want in ("结构化_含不确定", "结构化_无不确定", "类别Caption辅助"):
    d = next((x for x in detail if x["题面种类"] == want), None)
    if d:
        print(f"    [{want}] {d['sample_id']}  batch长 {d['batch_len']}  "
              f"监督 {d['监督token数']} token  精确相等={d['与目标精确相等']}")
        print(f"      剥离前尾部: {d['监督段原样尾部']!r}")
        print(f"      剥离后解码: {d['剥离后解码'][:100]}")

fails = [x for x in res if not x["通过"]]
print()
print("=" * 76)
print(f"通过 {len(res)-len(fails)}/{len(res)}")
for f in fails:
    print(f"  未通过：{f['项']}  {f['说明']}")
os.makedirs(OUT, exist_ok=True)
json.dump({"通过": f"{len(res)-len(fails)}/{len(res)}", "检查": res,
           "batch形状": {"input_ids": list(BI.shape), "labels": list(BL.shape),
                         "attention_mask": list(AM.shape)},
           "三种任务实际解码": detail,
           "note": "CPU 门槛，0 GPU；证据取自 collator 之后的 batch 张量。"
                   "未验证多步训练稳定性与真实收敛。"},
          io.open(OUT + "/desc_gate_data_v2.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print(f"写出 {OUT}/desc_gate_data_v2.json")
sys.exit(0 if not fails else 1)
PY
