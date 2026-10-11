# -*- coding: utf-8 -*-
import json, hashlib, sys, os
from datetime import datetime

TMP = r"D:/pycode/晶圆图研究/协作/04_WorkBuddy_复核/夜间主标_20261008_0301/GLM/workers/_tmp_095.json"
OUT = r"D:/pycode/晶圆图研究/协作/04_WorkBuddy_复核/夜间主标_20261008_0301/GLM/workers/分片095_原答.jsonl"

MANIFEST = {
 'wbg0659': 'cb96a25fbdb0ad9187bdf93035526ea0dfd57a439c9f9dfa3884f2efad6ee94f',
 'wbg0660': '43e2af7bead32eab0890c5424503552d77ce35099508b7bc7145af7f61b8cbf1',
 'wbg0661': '99fd2aaeec22235f19f04c8c163d61ce65a5ab2f2d0e3a37bb40aeb1bbac7b9b',
 'wbg0662': '803ff7a2d98d5e57724d25b9372bd93d8d87613190a8c1ff12ce8cadb8ebd7b9',
 'wbg0663': 'dec06bb7b36933033be81a8e9fe226542d7378561bca7325a4a95d31113b2f48',
 'wbg0664': 'b5f7d1721e7edf8eef3306de77542d8ca60a3dd2ecb647825473f0a890c11403',
 'wbg0665': 'dbda74ff7e8e0fc18f9660b05d6cd2671f78117864369022f7ea3d93a78647d5',
}
PROMPT_SHA = '8a8315f785a2398ba5a823754c40b8a8658f89b79aad1a0dbd68058acb43cfda'
REQUIRED = ['defect_class', 'morphology', 'radial_zone', 'clock_direction',
            'extent_r', 'caption_zh', 'uncertainty']

lines = [l for l in open(TMP, encoding='utf-8').read().splitlines() if l.strip()]
recs = [json.loads(l) for l in lines]

errors = []
seen = set()
for r in recs:
    iid = r.get('item_id')
    if iid in seen:
        errors.append('dup:' + str(iid))
    seen.add(iid)
    if r.get('image_sha256') != MANIFEST.get(iid):
        errors.append('sha_mismatch:' + str(iid))
    try:
        ra = json.loads(r['raw_answer'])
    except Exception:
        errors.append('raw_unparsable:' + str(iid))
        continue
    miss = [k for k in REQUIRED if k not in ra]
    if miss:
        errors.append('missing:' + str(iid) + ':' + ','.join(miss))

if len(recs) != 7:
    errors.append('count:' + str(len(recs)))
if set(MANIFEST) - seen:
    errors.append('absent:' + ','.join(sorted(set(MANIFEST) - seen)))

if errors:
    print('VALIDATE_FAIL')
    for e in errors:
        print(e)
    sys.exit(1)

now = datetime.now().astimezone().isoformat(timespec='seconds')
with open(OUT, 'w', encoding='utf-8') as f:
    for r in recs:
        rec = {
            'item_id': r['item_id'],
            'image_sha256': r['image_sha256'],
            'prompt_sha256': PROMPT_SHA,
            'raw_answer': r['raw_answer'],
            'answer_sha256': hashlib.sha256(r['raw_answer'].encode('utf-8')).hexdigest(),
            'model': 'GLM-5.3-Flash',
            'answer_time': now,
            'check': 'self_ok',
        }
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')

flines = [l for l in open(OUT, encoding='utf-8').read().splitlines() if l.strip()]
ok = True
for l in flines:
    d = json.loads(l)
    keys = list(d.keys())
    if len(keys) != 8:
        ok = False
        print('FIELD_COUNT_BAD:' + d.get('item_id', '?'))
    dc = json.loads(d['raw_answer'])['defect_class']
    print(d['item_id'], dc)
print('FINAL_LINES:' + str(len(flines)))
print('SIZE:' + str(os.path.getsize(OUT)))
print('FINAL_OK' if (ok and len(flines) == 7) else 'FINAL_BAD')
