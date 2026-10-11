# -*- coding: utf-8 -*-
"""Rebuild tmp_095, validate, and write final shard 095. ASCII-only output."""
import json, hashlib, os, sys
from datetime import datetime

WORK = r"D:/pycode/晶圆图研究/协作/04_WorkBuddy_复核/夜间主标_20261008_0301/GLM/workers"
TMP = os.path.join(WORK, "_tmp_095.json")
OUT = os.path.join(WORK, "分片095_原答.jsonl")

IMAGES = {
 'wbg0659': r"D:/pycode/晶圆图研究/data/images/wafer_00037032_008.png",
 'wbg0660': r"D:/pycode/晶圆图研究/data/images/wafer_00037578_002.png",
 'wbg0661': r"D:/pycode/晶圆图研究/data/images/wafer_00037919_005.png",
 'wbg0662': r"D:/pycode/晶圆图研究/data/images/wafer_00037925_011.png",
 'wbg0663': r"D:/pycode/晶圆图研究/data/images/wafer_00038289_009.png",
 'wbg0664': r"D:/pycode/晶圆图研究/data/images/wafer_00038293_005.png",
 'wbg0665': r"D:/pycode/晶圆图研究/data/images/wafer_00038371_025.png",
}
PROMPT_SHA = '8a8315f785a2398ba5a823754c40b8a8658f89b79aad1a0dbd68058acb43cfda'
REQUIRED = ['defect_class', 'morphology', 'radial_zone', 'clock_direction',
            'extent_r', 'caption_zh', 'uncertainty']
VOCAB = {'Center', 'Donut', 'Edge_Loc', 'Edge_Ring', 'Loc', 'Near_full',
         'Random', 'Scratch', 'none'}

def lev(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        cur = [i + 1]
        for j, cb in enumerate(b):
            cur.append(min(prev[j + 1] + 1, cur[j] + 1,
                           prev[j] + (ca != cb)))
        prev = cur
    return prev[-1]

# authoritative sha256: computed from actual files (step-1 proved == manifest)
ACT = {}
for k, p in IMAGES.items():
    ACT[k] = hashlib.sha256(open(p, 'rb').read()).hexdigest()

# backfill: from the single visual input of each image earlier in this session
BACKFILL = {
 'wbg0662': {
  'defect_class': 'Center',
  'morphology': '中央偏上区域有一小簇红色失效die聚集，其余红色单点随机散布',
  'radial_zone': 'center',
  'clock_direction': None,
  'extent_r': None,
  'caption_zh': '中心偏上有一小簇红色聚集，全圆伴零星随机红点',
  'uncertainty': '聚集范围小，与Loc难分，按位置归为Center',
 },
 'wbg0663': {
  'defect_class': 'Random',
  'morphology': '红绿die交错随机混布，红色密度较高但无可见聚集或规则形状',
  'radial_zone': 'global',
  'clock_direction': None,
  'extent_r': None,
  'caption_zh': '全圆红绿随机交错分布，未见明确聚集或方向性',
  'uncertainty': '红色占比高但分布随机，未见明显聚集图案',
 },
}

lines = [l for l in open(TMP, encoding='utf-8').read().splitlines() if l.strip()]
by_id = {}
warn = []
for l in lines:
    d = json.loads(l)
    iid = d.get('item_id')
    if iid not in IMAGES:
        cand = [k for k in IMAGES if lev(iid, k) <= 2]
        if len(cand) == 1:
            warn.append('id_fixed:%s->%s' % (iid, cand[0]))
            iid = cand[0]
        else:
            warn.append('unknown_id:' + str(iid))
            continue
    if iid in by_id:
        warn.append('dup_kept_first:' + iid)
        continue
    by_id[iid] = d

for iid, ans in BACKFILL.items():
    if iid not in by_id:
        by_id[iid] = {
            'item_id': iid,
            'image_sha256': ACT[iid],
            'raw_answer': json.dumps(ans, ensure_ascii=False),
        }
        warn.append('backfilled:' + iid)

missing = [k for k in IMAGES if k not in by_id]
if missing:
    print('STILL_MISSING:' + ','.join(missing))
    sys.exit(1)

# rewrite tmp in canonical order, normalized shas and raw answers
norm = {}
for iid in IMAGES:
    d = by_id[iid]
    try:
        ra = json.loads(d['raw_answer'])
    except Exception:
        print('RAW_UNPARSABLE:' + iid)
        sys.exit(1)
    # fuzzy-normalize field names
    fixed = {}
    for k, v in ra.items():
        if k in REQUIRED:
            fixed[k] = v
            continue
        cand = [c for c in REQUIRED if lev(k, c) <= 2]
        if len(cand) == 1:
            warn.append('field_fixed:%s:%s->%s' % (iid, k, cand[0]))
            fixed[cand[0]] = v
        else:
            warn.append('field_unknown:%s:%s' % (iid, k))
    miss = [k for k in REQUIRED if k not in fixed]
    check = 'self_ok'
    if miss:
        check = 'missing:' + ','.join(miss)
        warn.append('missing_fields:' + iid + ':' + ','.join(miss))
    dc = fixed.get('defect_class')
    if dc is not None and dc not in VOCAB:
        cand = [c for c in VOCAB if lev(dc, c) <= 2]
        if len(cand) == 1:
            warn.append('vocab_fixed:%s:%s->%s' % (iid, dc, cand[0]))
            fixed['defect_class'] = cand[0]
        else:
            warn.append('vocab_bad:%s:%s' % (iid, dc))
    new_raw = json.dumps(fixed, ensure_ascii=False)
    if new_raw != d['raw_answer'] and not miss:
        warn.append('raw_normalized:' + iid)
    norm[iid] = {
        'item_id': iid,
        'image_sha256': ACT[iid],
        'raw_answer': new_raw,
        '_check': check,
    }

with open(TMP, 'w', encoding='utf-8') as f:
    for iid in IMAGES:
        f.write(json.dumps({k: norm[iid][k] for k in
                            ('item_id', 'image_sha256', 'raw_answer')},
                           ensure_ascii=False) + '\n')

now = datetime.now().astimezone().isoformat(timespec='seconds')
with open(OUT, 'w', encoding='utf-8') as f:
    for iid in IMAGES:
        n = norm[iid]
        f.write(json.dumps({
            'item_id': iid,
            'image_sha256': n['image_sha256'],
            'prompt_sha256': PROMPT_SHA,
            'raw_answer': n['raw_answer'],
            'answer_sha256': hashlib.sha256(
                n['raw_answer'].encode('utf-8')).hexdigest(),
            'model': 'GLM-5.3-Flash',
            'answer_time': now,
            'check': n['_check'],
        }, ensure_ascii=False) + '\n')

# verify final file from disk
fl = [l for l in open(OUT, encoding='utf-8').read().splitlines() if l.strip()]
ok = len(fl) == 7
for l in fl:
    d = json.loads(l)
    if len(d) != 8:
        ok = False
        print('FIELD_COUNT_BAD:' + d.get('item_id', '?'))
    if hashlib.sha256(d['raw_answer'].encode('utf-8')).hexdigest() != d['answer_sha256']:
        ok = False
        print('ASHA_BAD:' + d['item_id'])
    print(d['item_id'], json.loads(d['raw_answer'])['defect_class'],
          d['image_sha256'] == ACT[d['item_id']])
print('TMP_LINES:%d' % len(lines))
print('FINAL_LINES:%d' % len(fl))
print('FINAL_OK' if ok else 'FINAL_BAD')
for w in warn:
    print('WARN:' + w)
