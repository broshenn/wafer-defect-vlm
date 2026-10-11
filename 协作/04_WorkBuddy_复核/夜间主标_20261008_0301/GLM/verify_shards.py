# -*- coding: utf-8 -*-
"""分片核验脚本：python verify_shards.py 81 82
校验：行数=7、item_id 无重复且在盲包内、image_sha256 对照盲包、
prompt_sha256 对照锁定哈希、answer_sha256 复算、片内答案哈希查重、全局跨片审计。"""
import json, hashlib, glob, re, sys
from collections import Counter

W = r'D:/pycode/晶圆图研究/协作/04_WorkBuddy_复核/夜间主标_20261008_0301/GLM/workers'
B = r'D:/pycode/晶圆图研究/协作/02_ClaudeCode_实操/夜间扩容_20261008_0244/客户端盲包/WorkBuddy_训练缺口1200'
PROMPT_SHA = '8a8315f785a2398ba5a823754c40b8a8658f89b79aad1a0dbd68058acb43cfda'

idx = {}
shard_items = {}
for p in glob.glob(B + '/WB_gap1200_分片*_共*图.jsonl'):
    m = re.search(r'分片(\d+)_共(\d+)图', p)
    n_sh, cnt = int(m.group(1)), int(m.group(2))
    ids = []
    for l in open(p, encoding='utf-8'):
        d = json.loads(l)
        idx[d['item_id']] = d['image_sha256']
        ids.append(d['item_id'])
    shard_items[n_sh] = (ids, cnt)

for a in sys.argv[1:]:
    n = int(a)
    rows = [json.loads(l) for l in open(W + '/分片%03d_原答.jsonl' % n, encoding='utf-8') if l.strip()]
    ids = [r['item_id'] for r in rows]
    exp_ids, exp_cnt = shard_items[n]
    assert len(rows) == exp_cnt, (n, 'rows', len(rows), 'expect', exp_cnt)
    assert len(set(ids)) == len(ids) and sorted(ids) == sorted(exp_ids), (n, 'item set mismatch')
    ah = [r['answer_sha256'] for r in rows]
    assert not set(x for x in ah if ah.count(x) > 1), (n, 'in-shard answer dup')
    for r in rows:
        assert r['item_id'] in idx, (n, r['item_id'], 'not in blind list')
        assert r['image_sha256'] == idx[r['item_id']], (n, r['item_id'], 'sha mismatch')
        assert r['prompt_sha256'] == PROMPT_SHA, (n, r['item_id'], 'prompt sha')
        assert hashlib.sha256(r['raw_answer'].encode('utf-8')).hexdigest() == r['answer_sha256'], (n, r['item_id'], 'ans hash calc')
    print(n, 'OK', ids[0], '-', ids[-1], 'checks:', sorted(set(r['check'] for r in rows)))

allr = []
for p in sorted(glob.glob(W + '/分片*_原答.jsonl')):
    allr += [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]
print('total rows:', len(allr), 'unique item:', len(set(r['item_id'] for r in allr)))
dups = {k: v for k, v in Counter(r['answer_sha256'] for r in allr).items() if v > 1}
for k in dups:
    print('dup answer group:', [r['item_id'] for r in allr if r['answer_sha256'] == k])
print('global audit done (known历史: wbg0129/0130, wbg0146/0147)')
