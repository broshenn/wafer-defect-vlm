"""Build auditable CPU pools and a blind WorkBuddy84 packet, never alter sources.

Supports dry-run, fixed seed, byte idempotence and safe resume of generated files.
PNG/lot connected groups prevent duplicate images leaking across pools.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re

import numpy as np
from PIL import Image
from core_v3 import HERE, REPO, load_geometry, measure_v3, reference, schema_errors

OLD = REPO/'协作/02_ClaudeCode_实操/多字段RL_v2_CPU准备_20261009'
SEED = 3407
PROTECTED = [
    'benchmark/core.jsonl',
    '协作/02_ClaudeCode_实操/百炼多模型业务对照_20261009_1203/盲号对照_key.json',
    '协作/02_ClaudeCode_实操/56图候选补测与最终结项_20261009/冻结清单.json',
    '协作/02_ClaudeCode_实操/收尾_RL90与报告_20261006/R0_dev18.jsonl',
    '协作/02_ClaudeCode_实操/收尾_RL90与报告_20261006/R0_val90.jsonl',
    '协作/02_ClaudeCode_实操/多字段RL_v2_CPU准备_20261009/清单_确认池.jsonl',
    '协作/02_ClaudeCode_实操/多字段RL_v2_CPU准备_20261009/清单_规则开发池.jsonl',
]
TRAIN_SOURCE = REPO/'协作/02_ClaudeCode_实操/夜间扩容_20261008_0244/数据/D_N3072.jsonl'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    if not path.exists():
        raise FileNotFoundError(path.relative_to(REPO))
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]


def ids_from(path):
    return set(re.findall(r'wafer_\d{8}_\d+', path.read_text(encoding='utf-8')))


def stable_write(path, text):
    raw = text.encode('utf-8')
    if path.exists():
        if path.read_bytes() != raw:
            raise RuntimeError('Frozen output differs; use a new output version: '+str(path.name))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def json_text(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2)+'\n'


def rows_text(rows):
    return ''.join(json.dumps(r, ensure_ascii=False, sort_keys=True)+'\n' for r in rows)


def choose(rows, n):
    """Priority direction originals, then deterministic round robin over classes."""
    rng = random.Random(SEED)
    rows = sorted(rows, key=lambda r:r['sample_id'])
    rng.shuffle(rows)
    positives = [r for r in rows if r['old_direction_candidate']]
    out, used = positives[:n], {r['sample_id'] for r in positives[:n]}
    bins = defaultdict(list)
    for r in rows:
        if r['sample_id'] not in used:
            bins[r['failure_type']].append(r)
    while len(out) < n and any(bins.values()):
        for cls in sorted(bins):
            if bins[cls] and len(out) < n:
                out.append(bins[cls].pop())
    return out


def build(dry_run=False):
    manifest_path = REPO/'data/manifest.jsonl'
    geom_path = OLD/'几何特征表.jsonl'
    manifest = read_rows(manifest_path)
    geom = {r['sample_id']:r for r in read_rows(geom_path)}
    if len({r['sample_id'] for r in manifest}) != len(manifest):
        raise RuntimeError('Duplicate sample ID in manifest')
    prior_train = ids_from(TRAIN_SOURCE)
    history_ids, source_audit = set(), []
    for name in PROTECTED:
        p = REPO/name
        if not p.exists():
            raise FileNotFoundError(name)
        ids = ids_from(p)
        if not ids:
            raise RuntimeError('Protected source yielded no IDs: '+name)
        history_ids |= ids
        source_audit.append({'path':name,'sha256':digest(p),'unique_ids':len(ids)})

    # All original records participate in connectivity: a duplicate of a test PNG
    # must not sneak into train even when its lot differs.
    parent = {r['sample_id']:r['sample_id'] for r in manifest}
    def find(x):
        while x != parent[x]:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(x,y):
        x,y = find(x),find(y)
        if x != y:
            parent[max(x,y)] = min(x,y)
    lot_first, hash_first = {}, {}
    for r in manifest:
        sid = r['sample_id']
        for key, table in [(r['lot_name'],lot_first),(geom[sid]['png_sha256'],hash_first)]:
            if key in table:
                union(sid,table[key])
            else:
                table[key] = sid
    groups = defaultdict(list)
    for r in manifest:
        groups[find(r['sample_id'])].append(r)
    blocked = {key for key, rows in groups.items() if any(r['sample_id'] in history_ids for r in rows)}
    eligible_train, eligible_test = [], []
    for key, rows in sorted(groups.items()):
        if key in blocked or len({r['split'] for r in rows}) != 1:
            continue
        s = rows[0]['split']
        selected = []
        png_seen = set()
        for r in sorted(rows,key=lambda r:r['sample_id']):
            sid = r['sample_id']
            g = geom[sid]
            if (r['label_source'] != 'ground_truth' or g['status'] not in ['ok','no_red']
                    or g.get('outline',{}).get('anomalous') or g['png_sha256'] in png_seen):
                continue
            png_seen.add(g['png_sha256'])
            selected.append({**r,'image_path':f'data/images/{sid}.png',
                             'source_path_mapping':'original manifest server path mapped to local PNG by stable sample_id; verified by SHA256',
                             'group_key':key,'image_sha256':g['png_sha256'],
                             'old_direction_candidate':g['direction_type'] in ['single','axis'],
                             'seen_by_D_id':r['sample_id'] in prior_train})
        if selected and s == 'train':
            eligible_train.append(selected)
        elif selected and s == 'test':
            eligible_test.append(selected)

    # Lock group assignment before recomputing v3 facts or inspecting descriptions.
    # Train allows reuse of old supervised examples with explicit contact metadata.
    pool_groups = {'adapt': [], 'rl': [], 'rule_dev': []}
    groups_with_direction = [rows for rows in eligible_train if any(r['old_direction_candidate'] for r in rows)]
    ordinary = [rows for rows in eligible_train if rows not in groups_with_direction]
    rng = random.Random(SEED)
    rng.shuffle(groups_with_direction); rng.shuffle(ordinary)
    assigned = set()
    for i, rows in enumerate(groups_with_direction):
        pool = 'adapt' if i % 2 == 0 else 'rl'
        pool_groups[pool].extend(rows); assigned.add(rows[0]['group_key'])
    targets = {'adapt':512,'rl':256,'rule_dev':30}
    for pool in ['rule_dev','rl','adapt']:
        for rows in ordinary:
            key = rows[0]['group_key']
            if key not in assigned and len(pool_groups[pool]) < targets[pool]:
                pool_groups[pool].extend(rows); assigned.add(key)
    pools = {k:choose(pool_groups[k],n) for k,n in targets.items()}
    pools['confirmation'] = choose([r for rows in eligible_test for r in rows],120)
    if any(len(pools[k]) != n for k,n in {**targets,'confirmation':120}.items()):
        raise RuntimeError('Insufficient isolated inventory: '+str({k:len(v) for k,v in pools.items()}))
    report = {'status':'CPU_candidates_not_training_authorization','seed':SEED,
              'sources':source_audit + [{'path':'data/manifest.jsonl','sha256':digest(manifest_path)},
                                      {'path':str(geom_path.relative_to(REPO)).replace('\\','/'),'sha256':digest(geom_path)},
                                      {'path':str(TRAIN_SOURCE.relative_to(REPO)).replace('\\','/'),'sha256':digest(TRAIN_SOURCE)}],
              'threshold_origin':'Frozen v2 train-calibrated thresholds, not freshly tuned on confirmation',
              'confirmation_status':'Original test-side candidate subset; prior-contact ledger limited to named files; no human gold; not a full final benchmark',
              'pools':{},'intersections':{},'images_verified':0}
    for a in pools:
        for b in pools:
            if a < b:
                report['intersections'][a+'__'+b] = {
                    'sample':len({r['sample_id'] for r in pools[a]} & {r['sample_id'] for r in pools[b]}),
                    'lot':len({r['lot_name'] for r in pools[a]} & {r['lot_name'] for r in pools[b]}),
                    'PNG':len({r['image_sha256'] for r in pools[a]} & {r['image_sha256'] for r in pools[b]})}
    if any(any(x.values()) for x in report['intersections'].values()):
        raise RuntimeError('Pool leak')
    if dry_run:
        print(json.dumps({'dry_run':True,'candidate_counts':{k:len(v) for k,v in pools.items()},
                          'intersections':report['intersections']},ensure_ascii=False))
        return

    # Recompute direction with the corrected hour contract from actual matrices.
    old_module = load_geometry()
    for name, rows in pools.items():
        for r in rows:
            p = REPO/r['image_path']
            if digest(p) != r['image_sha256']:
                raise RuntimeError('Image fingerprint mismatch: '+r['sample_id']+' expected '+r['image_sha256']+' actual '+digest(p))
            rec = old_module.recover_matrix(p, hint_hw=tuple(r['matrix_shape']))
            if not rec['roundtrip_validated']:
                raise RuntimeError('PNG roundtrip failed')
            g = measure_v3(rec['matrix'])
            obj = reference(g,r['failure_type'])
            if schema_errors(obj):
                raise RuntimeError('Generated reference schema error')
            r['reference'] = obj
            r['geometry_source'] = 'Program silver reference on lossless PNG-recovered grid; not physical die geometry'
            r['seen_by_D_lot'] = r['lot_name'] in {x['lot_name'] for x in manifest if x['sample_id'] in prior_train}
            report['images_verified'] += 1
        report['pools'][name] = {'unique_wafer_IDs':len(rows),'unique_PNGs':len({r['image_sha256'] for r in rows}),
                                'lots':len({r['lot_name'] for r in rows}),
                                'classes':dict(Counter(r['failure_type'] for r in rows)),
                                'angular':dict(Counter(r['reference']['angular_type'] for r in rows)),
                                'seen_by_D_id':sum(r['seen_by_D_id'] for r in rows),
                                'seen_by_D_lot':sum(r['seen_by_D_lot'] for r in rows)}
        stable_write(HERE/'pools'/f'{name}.jsonl',rows_text(rows))

    prompt = (HERE/'题面_v3.txt').read_text(encoding='utf-8').strip()
    def ms_row(r, answer=False, gold=False):
        obj = {'sample_id':r['sample_id'],'messages':[{'role':'user','content':'<image>\n'+prompt}],
               'images':[r['image_path']]}
        if answer:
            obj['messages'].append({'role':'assistant','content':json.dumps(r['reference'],ensure_ascii=False,separators=(',',':'))})
        if gold:
            obj['reference_json'] = json.dumps(r['reference'],ensure_ascii=False,separators=(',',':'))
        return obj
    for filename, rows in {
        'adapt512_sft.jsonl':[ms_row(r,True) for r in pools['adapt']],
        'group256_sft4.jsonl':[ms_row(r,True) for r in pools['rl'] for _ in range(4)],
        'group256_grpo.jsonl':[ms_row(r,gold=True) for r in pools['rl']],
        'confirmation120_requests.jsonl':[ms_row(r) for r in pools['confirmation']],
        'confirmation120_reference.jsonl':[{'sample_id':r['sample_id'],'reference':r['reference']} for r in pools['confirmation']],
    }.items():
        stable_write(HERE/'data_v3'/filename,rows_text(rows))

    # Visual annotation emphasizes failure modes, not invented extra wafers.
    by_id = {r['sample_id']:r for r in pools['adapt']}
    priority = sorted(by_id.values(),key=lambda r:(
        r['reference']['angular_type'] not in ['single','opposed'],
        not (r['failure_type']=='none' and geom[r['sample_id']]['n_red']>0),
        r['reference']['coverage_band']!='half_or_more',r['sample_id']))
    wb_rows = priority[:42]
    rest = choose([r for r in pools['adapt'] if r not in wb_rows],42)
    wb_rows.extend(rest)
    wb_rows.sort(key=lambda r:r['sample_id'])
    blind = [{'item_id':f'wbv3_{i:03d}','sample_id':r['sample_id'],
              'image_path':str((REPO/r['image_path']).resolve()),'image_sha256':r['image_sha256']}
             for i,r in enumerate(wb_rows,1)]
    packet = HERE/'WorkBuddy84'
    stable_write(packet/'blind84.jsonl',rows_text(blind))
    shards = []
    for i in range(12):
        path = packet/f'batch_{i+1:02d}.jsonl'
        stable_write(path,rows_text(blind[i*7:(i+1)*7]))
        shards.append({'file':path.name,'count':7,'sha256':digest(path)})
    stable_write(packet/'index.json',json_text({'version':'WB84_v3','count':84,'first_ever_annotation':'not_claimed',
        'blind_manifest_sha256':digest(packet/'blind84.jsonl'),'batches':shards,
        'prompt_file':'标注题面.txt','prompt_sha256':digest(packet/'标注题面.txt'),
        'checker_file':'../workbuddy_schema_check.py','checker_sha256':digest(HERE/'workbuddy_schema_check.py'),
        'checker_dependency_file':'../core_v3.py','checker_dependency_sha256':digest(HERE/'core_v3.py')}))
    # Selection reasons belong outside the blind directory; never give to workers.
    stable_write(HERE/'WorkBuddy84_来源审计.json',json_text({'counts':dict(Counter(r['failure_type'] for r in wb_rows)),
        'angular':dict(Counter(r['reference']['angular_type'] for r in wb_rows)),
        'training_subset':True,'unique_PNGs':len({r['image_sha256'] for r in wb_rows}),
        'answer_source':'future model visual annotation; not gold; old outputs unchanged',
        'item_map':[{'item_id':b['item_id'],'sample_id':r['sample_id'],'prior_D_seen':r['seen_by_D_id']} for b,r in zip(blind,wb_rows)]}))

    # Rotation recipes do not become extra unique wafers or unseen holdout samples.
    recipes = []
    for name in ['adapt','rl']:
        for r in pools[name]:
            if r['reference']['angular_type'] in ['single','opposed']:
                m = old_module.recover_matrix(REPO/r['image_path'],hint_hw=tuple(r['matrix_shape']))['matrix']
                for k in [1,2,3]:
                    rotated = reference(measure_v3(np.rot90(m,-k)),r['failure_type'])
                    expected = sorted((s-1+3*k)%12+1 for s in r['reference']['clock_sectors'])
                    if sorted(rotated['clock_sectors']) != expected:
                        raise RuntimeError('Real-image rotation sector mismatch')
                    recipes.append({'derived_id':r['sample_id']+f'__cw{k*90}', 'original_sample_id':r['sample_id'],
                                    'split_pool':name,'clockwise_degrees':k*90,'reference':rotated,
                                    'original_sha256':r['image_sha256'],'new_unique_wafer':False,
                                    'free_text_caption_rotated':False})
    stable_write(HERE/'rotation_recipes.jsonl',rows_text(recipes))
    report['rotation_derived_tasks'] = len(recipes)
    report['rotation_images_materialized'] = False
    report['workbuddy_plan_images'] = 84
    report['GPU_gate'] = 'not_run'
    report['new_GPU_training'] = 'not_run'
    report['WorkBuddy_execution'] = 'not_dispatched'
    stable_write(HERE/'CPU数据验收.json',json_text(report))
    print(json.dumps({'pools':report['pools'],'images_verified':report['images_verified'],
                      'rotation_recipes':len(recipes),'workbuddy84':True},ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run',action='store_true')
    args = parser.parse_args()
    build(args.dry_run)
