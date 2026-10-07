#!/bin/bash
# 说明：`/WS` 是真实私有工作区的**别名**，真实值只存在于仓库外的接入资料与服务器上的
# 可执行副本中。本文件入库前已按约定脱敏。
# 描述 SFT CPU 门槛（数据/模板/labels）—— 不建模型，0 GPU
set -u
R=/WS
export CUDA_VISIBLE_DEVICES=""          # 必须在 Python 启动前设，否则 device_map='auto' 会占共享卡
export IMAGE_MAX_TOKEN_NUM=256
export PYTHONIOENCODING=utf-8
exec $R/envs/wafer/bin/python - <<'PY'
import hashlib, io, json, os, sys
from collections import Counter
from functools import partial

R = "/WS"
DATA = R + "/datasets/abc/desc_v2.server.jsonl"
SHA = R + "/datasets/abc/desc_v2_sid_sha.tsv"
TYP = R + "/datasets/abc/desc_v2_typing.jsonl"
MODEL = R + "/models/Qwen3.5-9B"
OUT = R + "/logs"

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
print(f"  {len(rows)} 行 / {len(typ)} 条分型 / {len(sha_tab)} 条哈希")
ck("行数 356", len(rows) == 356, str(len(rows)))
ck("ID 唯一", len({r["sample_id"] for r in rows}) == len(rows))
ck("分型覆盖全部行", set(typ) == {r["sample_id"] for r in rows})

miss, bad, wrongpath = [], [], []
for r in rows:
    p = r["images"][0]
    want = f"/WS/datasets/expand_v2/images/{r['sample_id']}.png"
    if p != want:
        wrongpath.append((r["sample_id"], p))
    if not os.path.exists(p):
        miss.append(r["sample_id"]); continue
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    if h != sha_tab.get(r["sample_id"]):
        bad.append((r["sample_id"], h[:16], (sha_tab.get(r["sample_id"]) or "")[:16]))
ck("路径全部规范", not wrongpath, f"异常 {len(wrongpath)} {wrongpath[:2]}")
ck("356 张图片全部存在", not miss, f"缺 {len(miss)} {miss[:3]}")
ck("356 张图片 sha256 与 固定356条 逐条一致", not bad, f"不符 {len(bad)} {bad[:2]}")

print()
print("=" * 76)
print("B. 真实模板编码（strict=True，不允许换样本）")
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
        # 逐行直接调用 template.encode 并自行捕获异常：换样本只发生在 LazyLLMDataset
        # 内部，这里逐条上报，任何一条失败都原样计入，不会被顶替掩盖。
        e = tpl.encode(InferRequest(messages=r["messages"], images=r["images"]),
                       return_length=True)
        n = sum(1 for t in e["input_ids"] if t == tpl.image_token_id)
        img_tok[n] += 1
        lens.append(len(e["input_ids"]))
    except Exception as ex:
        errs.append((r["sample_id"], f"{type(ex).__name__}: {ex}"))
ck("356 条逐条编码零失败（无样本顶替）", not errs, f"失败 {len(errs)} {errs[:2]}")
ck("每条恰好 196 个 image token", set(img_tok) == {196}, dict(img_tok))
ck("全部短于 max_length 2048", bool(lens) and max(lens) < 2048, f"最长 {max(lens) if lens else '-'}")

print()
print("=" * 76)
print("C. 真实训练路径的 labels（LazyLLMDataset + template.data_collator）")
print("=" * 76)
from datasets import Dataset
from swift.dataset.utils import LazyLLMDataset

# labels 只在 train 模式下由 encode 产出（默认 mode='transformers' 没有）——
# 与 _prepare_template 里的 template.set_mode('train') 一致。
tpl.set_mode("train")
print(f"  模板 mode = {tpl.mode}")
ck("模板已置 train 模式（labels 的来源）", tpl.mode == "train", tpl.mode)

# 20 条代表：覆盖 结构化含不确定 / 结构化缺不确定 / 仅类别Caption；并含方向有值
sel = []
for kind, want in (("结构化_含不确定", 8), ("结构化_无不确定", 6), ("类别Caption辅助", 6)):
    pool = [r["sample_id"] for r in rows if typ[r["sample_id"]]["题面种类"] == kind]
    sel += pool[:want]
dir_rows = [r["sample_id"] for r in rows
            if isinstance(json.loads(r["messages"][1]["content"]).get("clock_direction"), str)]
for s in dir_rows:
    if s not in sel:
        sel[-1] = s                      # 保底塞入一条方向有值的
        break
sel = sel[:20]
print(f"  选中 {len(sel)} 条：{dict(Counter(typ[s]['题面种类'] for s in sel))}")
print(f"  含方向有值 {sum(1 for s in sel if s in dir_rows)} 条")
ck("20 条覆盖三种任务", len(sel) == 20 and len(set(typ[s]["题面种类"] for s in sel)) == 3,
   str(dict(Counter(typ[s]['题面种类'] for s in sel))))

byid = {r["sample_id"]: r for r in rows}
raw = Dataset.from_list([{"messages": byid[s]["messages"], "images": byid[s]["images"]} for s in sel])
lazy = LazyLLMDataset(raw, tpl.encode, strict=True, random_state=3407)
enc = [lazy[i] for i in range(len(sel))]
ck("LazyLLMDataset 产出 labels", all("labels" in e for e in enc),
   str(sorted(enc[0].keys())))

batch = tpl.data_collator(enc)
ck("collator 产出 labels / pixel_values / image_grid_thw",
   all(k in batch for k in ("labels", "pixel_values", "image_grid_thw")),
   str(sorted(batch.keys())))
import torch
lab = batch["labels"]
ck("labels 是 tensor 且 2 维", torch.is_tensor(lab) and lab.dim() == 2, str(tuple(lab.shape)))

bad_sup, bad_prompt, bad_end, bad_img = [], [], [], []
detail = []
for i, e in enumerate(enc):
    ids = list(e["input_ids"])
    L = list(e["labels"])
    if len(L) != len(ids):
        bad_sup.append((sel[i], f"labels {len(L)} != input_ids {len(ids)}")); continue
    sup = [j for j, v in enumerate(L) if v != -100]
    if not sup:
        bad_sup.append((sel[i], "无监督段")); continue
    seg = ids[sup[0]:sup[-1] + 1]
    segtxt = tok.decode(seg)
    tgt = byid[sel[i]]["messages"][1]["content"]
    # 1) 监督段必须能解出训练目标
    if tgt not in segtxt:
        bad_sup.append((sel[i], "监督段不含目标"))
    # 2) 监督段之前（问题+图像）必须全 -100
    if any(v != -100 for v in L[:sup[0]]):
        bad_prompt.append(sel[i])
    # 3) 监督段里不该再出现 image token
    if any(t == tpl.image_token_id for t in seg):
        bad_img.append(sel[i])
    # 4) 模板结束标记（eos）必须被监督
    if ids[-1] not in seg:
        bad_end.append((sel[i], "eos 未被监督"))
    # 5) labels 里非 -100 的值必须等于该位置的 input_ids（不是错位的）
    if any(L[j] != ids[j] for j in sup):
        bad_sup.append((sel[i], "labels 与 input_ids 错位"))
    detail.append({"sample_id": sel[i], "题面种类": typ[sel[i]]["题面种类"],
                   "总长": len(ids), "prompt长": sup[0], "监督长": len(sup),
                   "监督文本": segtxt})

ck("20 条监督段都含目标且与 input_ids 对齐", not bad_sup, f"异常 {bad_sup[:2]}")
ck("问题与视觉 token 全 -100", not bad_prompt, f"异常 {bad_prompt[:2]}")
ck("图像 token 不落在监督段", not bad_img, f"异常 {bad_img[:2]}")
ck("模板结束标记落在监督段内", not bad_end, f"异常 {bad_end[:2]}")

think_pref = sum(1 for d in detail if d["监督文本"].lstrip().startswith("<think"))
print(f"\n  监督段以空 think 前缀开头的：{think_pref}/{len(detail)}"
      f"（模板在 train 模式下自带，属正常渲染；评测解析已剥离）")
print("\n  逐条监督区间：")
for d in detail:
    print(f"    {d['sample_id']}  {d['题面种类']:12s} 总 {d['总长']:4d} "
          f"prompt {d['prompt长']:4d} 监督 {d['监督长']:3d}")
print("\n  首条监督文本前 100 字：")
print("    " + repr(detail[0]["监督文本"][:100]))

fails = [x for x in res if not x["通过"]]
print()
print("=" * 76)
print(f"通过 {len(res)-len(fails)}/{len(res)}")
for f in fails:
    print(f"  未通过：{f['项']}  {f['说明']}")
os.makedirs(OUT, exist_ok=True)
json.dump({"通过": f"{len(res)-len(fails)}/{len(res)}", "检查": res, "明细": detail,
           "note": "CPU 门槛，0 GPU；未验证多步训练稳定性与真实收敛"},
          io.open(OUT + "/desc_gate_data.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print(f"写出 {OUT}/desc_gate_data.json")
sys.exit(0 if not fails else 1)
PY
