"""从现有可见记录核查新考卷候选；不生成模型预测或更改历史划分。"""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json
import random
import re

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
def rows(p):
    return [json.loads(l) for l in p.read_text(encoding='utf-8-sig').splitlines() if l.strip()]
manifest = rows(ROOT / 'data/manifest.jsonl')
byid = {r['sample_id']: r for r in manifest}
inventory = rows(ROOT / '协作/02_ClaudeCode_实操/自动接续交付_20261009_1247/清单_5904_本地相对路径.jsonl')
paths = {r['sample_id']: r['local_relative_path'] for r in inventory}
hashes = {r['sample_id']: r['sha256'] for r in inventory}
train = rows(ROOT / '协作/02_ClaudeCode_实操/夜间扩容_20261008_0244/数据/L_N3072.jsonl')
trainids = {r['sample_id'] for r in train}
trainlots = {byid[s]['lot_name'] for s in trainids}
trainsha = {hashes[s] for s in trainids}
pattern = re.compile(r'wafer_\d+_\d+')
sources = []
evidence = set()
for base in ['results/raw_outputs','benchmark','协作/02_ClaudeCode_实操','协作/03_ZCode_标注','协作/04_WorkBuddy_复核']:
    for p in sorted((ROOT / base).rglob('*')):
        if not p.is_file() or p.suffix.lower() not in {'.json','.jsonl','.txt','.csv'}:
            continue
        relative = p.relative_to(ROOT).as_posix()
        if base not in ['results/raw_outputs','benchmark'] and not re.search(r'raw|原答|/meta/|/done/|/started/|http_replay|请求日志|historical', relative, re.I):
            continue
        content = p.read_text(encoding='utf-8-sig', errors='replace')
        found = set(pattern.findall(content + ' ' + p.name)) & byid.keys()
        if found:
            evidence.update(found)
            sources.append({'path': relative, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'ids': sorted(found)})
exposed_lots = {byid[s]['lot_name'] for s in evidence}
exposed_sha = {hashes[s] for s in evidence}
candidates = []
excluded = defaultdict(list)
for r in manifest:
    if r['split'] != 'test':
        continue
    sid = r['sample_id']
    reasons = []
    if sid in trainids: reasons.append('train_id')
    if r['lot_name'] in trainlots: reasons.append('train_lot')
    if hashes[sid] in trainsha: reasons.append('train_png_content')
    if sid in evidence: reasons.append('visible_historical_answer_or_frozen_benchmark_id')
    if r['lot_name'] in exposed_lots: reasons.append('visible_historical_answer_or_frozen_benchmark_lot')
    if hashes[sid] in exposed_sha: reasons.append('visible_historical_answer_or_frozen_benchmark_png_content')
    if reasons:
        excluded[sid] = reasons
    else:
        candidates.append(r)
classes = ['Center','Donut','Edge_Loc','Edge_Ring','Loc','Near_full','Random','Scratch','none']
rng = random.Random(3407)
selected = []
used_lots, used_sha = set(), set()
for label in classes:
    options = sorted([r for r in candidates if r['failure_type'] == label], key=lambda r:r['sample_id'])
    rng.shuffle(options)
    count = 0
    for r in options:
        sid = r['sample_id']
        if r['lot_name'] in used_lots or hashes[sid] in used_sha:
            continue
        assert hashlib.sha256((ROOT / paths[sid]).read_bytes()).hexdigest() == hashes[sid]
        used_lots.add(r['lot_name']); used_sha.add(hashes[sid])
        selected.append({'item_id':f'H{len(selected)+1:03d}','sample_id':sid,'image_path':paths[sid],'image_sha256':hashes[sid]})
        count += 1
        if count == 8: break
counts = Counter(byid[r['sample_id']]['failure_type'] for r in selected)
report = {'scope':'可见历史证据排除后的test候选准备，非最终模型成绩；未核到的历史教师/旧训练全量接触仍未知。',
          'seed':3407,'target_per_class':8,'target_total':72,
          'test_before_exclusion':dict(Counter(r['failure_type'] for r in manifest if r['split']=='test')),
          'eligible_after_exclusion':dict(Counter(r['failure_type'] for r in candidates)),
          'selected_counts':dict(counts),'selected_n':len(selected),
          'shortfalls':{c:8-counts[c] for c in classes if counts[c]<8},
          'selected_unique_lots':len(used_lots),'selected_unique_png':len(used_sha),
          'visible_exposure_ids':len(evidence),'visible_exposure_lots':len(exposed_lots),
          'train3072_id_overlap':len({r['sample_id'] for r in selected}&trainids),
          'train3072_lot_overlap':len(used_lots&trainlots),'train3072_png_overlap':len(used_sha&trainsha),
          'status':'candidate_package_prepared_not_model_run',
          'independent_final_test_completed':False,'new_gpu_inference':0,'new_business_api_attempts':0,
          'limitations':['未在新候选上运行Base/L/D或外部模型，成绩为未测。','现有源记录不完整，不能保证此前任何模型从未接触过候选。','没有真人gold；没有把开发36/35图改名成独立最终考卷。']}
(OUT/'候选考卷_盲清单.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected), encoding='utf-8')
(OUT/'考卷隔离核查.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(OUT/'可见历史接触证据.json').write_text(json.dumps({'sources':sources,'excluded_test_ids':excluded},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
