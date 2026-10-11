"""Read-only audit of the delivered v3 experiment and WorkBuddy review.

Only this directory receives output. No training, API calls or original edits.
Run with --dry-run to compute without writing. Seed and inputs are fixed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import random
import re
import tarfile

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CPU = REPO / '协作/01_Codex_指挥/无卡CPU整备_20261010'
CLAUDE = REPO / '协作/02_ClaudeCode_实操/一天半优化_20261010'
SERVER = CLAUDE / 'out_v3/服务器产物/out'
WB = REPO / '协作/04_WorkBuddy_复核/内容复核30_v3_20261010_233400'
OLD = WB.parent / '方向形态84_v3_20261010_200950'
TAGS = ['M0_v3', 'M1_v3', 'M2_v3']
FIELDS = {'defect_class', 'coverage_band', 'red_mass_zones', 'angular_type',
          'clock_sectors', 'uncertainty'}
EMPTY = re.compile(r'^\s*<think\b[^>]*>\s*</think\s*>\s*', re.S)
SNAPSHOT = {}


def sha(path):
    p = Path(path)
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    SNAPSHOT[str(p)] = digest
    return digest


def read(path):
    sha(path)
    return Path(path).read_text(encoding='utf-8')


def obj(path):
    return json.loads(read(path))


def jl(path):
    return [json.loads(s) for s in read(path).splitlines() if s.strip()]


def load_module(name, path):
    sha(path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def independent_score(text, ref):
    # Delivered answers have no fences. Do not reuse the producer's score function.
    answer = json.loads(EMPTY.sub('', text.strip()))
    assert set(answer) == FIELDS
    weights = np.array([.3, .2, .2, .2])
    for i, field in enumerate(['coverage_band', 'red_mass_zones', 'angular_type'], 1):
        if ref[field] == 'unknown':
            weights[i] = 0
    parts = np.array([
        answer['defect_class'] == ref['defect_class'],
        answer['coverage_band'] == ref['coverage_band'],
        isinstance(answer['red_mass_zones'], list)
        and isinstance(ref['red_mass_zones'], list)
        and set(answer['red_mass_zones']) == set(ref['red_mass_zones']),
        .6 * (answer['angular_type'] == ref['angular_type'])
        + .4 * (answer['angular_type'] != 'unknown'
                and set(answer['clock_sectors']) == set(ref['clock_sectors']))], float)
    return answer, float(.9 * np.dot(weights, parts) / weights.sum() + .1), parts


def f1_counts(gold, pred, classes):
    return np.column_stack([np.array([
        (gold == c) & (pred == c),
        (gold != c) & (pred == c),
        (gold == c) & (pred != c)], float).T for c in classes])


def macro_from_counts(counts):
    triples = counts.reshape(*counts.shape[:-1], -1, 3)
    num = 2 * triples[..., 0]
    den = num + triples[..., 1] + triples[..., 2]
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0).mean(axis=-1)


def ci(values):
    return [float(x) for x in np.quantile(values, [.025, .975], method='linear')]


def legacy_boot(differences):
    rng = random.Random(3407)
    n = len(differences)
    v = sorted(sum(differences[rng.randrange(n)] for _ in range(n)) / n
               for _ in range(10000))
    return [v[250], v[9750]]


def audit_v3(manifest):
    refs = {r['sample_id']: r['reference'] for r in jl(CPU/'data_v3/confirmation120_reference.jsonl')}
    ids = sorted(refs)
    assert len(ids) == 120
    gold = np.array([refs[s]['defect_class'] for s in ids])
    classes = sorted(set(gold))
    lot_keys = [manifest[s]['lot_name'] for s in ids]
    lots = sorted(set(lot_keys))
    lot_matrix = np.array([[lot == key for key in lot_keys] for lot in lots], float)
    # Paired cluster bootstrap: select whole lots, same sample weights for all models.
    rng = np.random.default_rng(3407)
    lot_draws = rng.multinomial(len(lots), np.full(len(lots), 1/len(lots)), size=10000)
    sample_weights = lot_draws @ lot_matrix
    weight_total = sample_weights.sum(axis=1)
    rng_iid = np.random.default_rng(3407)
    iid_weights = rng_iid.multinomial(len(ids), np.full(len(ids), 1/len(ids)), size=10000)
    result = {'status': 'development_confirmation_only', 'N': len(ids), 'lots': len(lots),
              'classes': classes, 'class_counts': dict(Counter(gold)),
              'seed': 3407, 'bootstrap_replicates': 10000,
              'bootstrap_unit': 'lot; same sampled lots for each paired model',
              'missing_class_resamples': {c: int(((sample_weights @ (gold == c)) == 0).sum())
                                         for c in classes},
              'missing_class_handling': 'fixed seven-class denominator; absent TP/FP/FN gives F1=0',
              'models': {}, 'paired': {}}
    per = {}
    for tag in TAGS:
        rows = jl(SERVER/'评测_v3'/f'{tag}_raw.jsonl')
        assert len(rows) == len(ids) and len({r['sample_id'] for r in rows}) == len(ids)
        assert {r['sample_id'] for r in rows} == set(ids)
        by_id = {r['sample_id']: r for r in rows}
        predictions, scores, parts, raw_ok = [], [], [], 0
        for sid in ids:
            r = by_id[sid]
            assert r['stripped'] == EMPTY.sub('', r['raw'].strip())
            assert r['finish_reason'] == 'stop'
            try:
                json.loads(r['raw']); raw_ok += 1
            except ValueError:
                pass
            answer, score, part = independent_score(r['raw'], refs[sid])
            assert not SCORER.schema_errors(answer)
            delivered = SCORER.score(r['raw'], refs[sid])
            assert abs(score-delivered['reward']) < 1e-12
            predictions.append(answer['defect_class']); scores.append(score); parts.append(part)
        pred = np.array(predictions); scores = np.array(scores); parts = np.array(parts)
        counts = f1_counts(gold, pred, classes)
        macro = float(macro_from_counts(counts.sum(axis=0)))
        lot_macro = macro_from_counts(sample_weights @ counts)
        iid_macro = macro_from_counts(iid_weights @ counts)
        correct = (pred == gold).astype(float)
        angular_known = np.array([refs[s]['angular_type'] != 'unknown' for s in ids])
        result['models'][tag] = {
            'correct': int(correct.sum()), 'accuracy': float(correct.mean()), 'macro_f1': macro,
            'macro_f1_lot_95CI': ci(lot_macro), 'rule_reward': float(scores.mean()),
            'coverage_exact': int(parts[:, 1].sum()), 'zones_exact': int(parts[:, 2].sum()),
            'angular_positive_N': sum(refs[s]['angular_type'] in ['single','opposed'] for s in ids),
            'nondirectional_exact': int((parts[angular_known, 3] == 1).sum()),
            'nondirectional_N': int(angular_known.sum()), 'unknown_N': int((~angular_known).sum()),
            'angular_known_mean': float(parts[angular_known, 3].mean()),
            'angular_all_N_legacy_mean': float(parts[:, 3].mean()),
            'raw_strict_json': raw_ok, 'normalized_json': len(rows),
            'adapter_fingerprints': sorted({r['adapter_sha256'] for r in rows}),
            'per_class': {c: {'support': int((gold == c).sum()),
                             'tp': int(counts[:, i*3].sum()),
                             'fp': int(counts[:, i*3+1].sum()),
                             'fn': int(counts[:, i*3+2].sum())} for i,c in enumerate(classes)}}
        per[tag] = {'macro': macro, 'macro_lot': lot_macro, 'macro_iid': iid_macro,
                    'reward': scores, 'correct': correct}
    for a, b in [('M1_v3','M0_v3'), ('M2_v3','M0_v3'), ('M2_v3','M1_v3')]:
        diff = per[a]['reward']-per[b]['reward']
        lot_reward = (sample_weights @ diff) / weight_total
        result['paired'][f'{a}-{b}'] = {
            'macro_f1_difference': per[a]['macro']-per[b]['macro'],
            'macro_f1_lot_95CI': ci(per[a]['macro_lot']-per[b]['macro_lot']),
            'macro_f1_iid_95CI': ci(per[a]['macro_iid']-per[b]['macro_iid']),
            'rule_reward_difference': float(diff.mean()), 'rule_reward_lot_95CI': ci(lot_reward),
            'rule_reward_iid_legacy_95CI': legacy_boot(diff.tolist()),
            'accuracy_difference': float((per[a]['correct']-per[b]['correct']).mean())}
    pools = {n: jl(CPU/'pools'/f'{n}.jsonl') for n in ['adapt','rl','rule_dev','confirmation']}
    result['pool_intersections'] = {}
    for a,b in itertools.combinations(pools, 2):
        result['pool_intersections'][f'{a}__{b}'] = {
            field: len({r[field] for r in pools[a]} & {r[field] for r in pools[b]})
            for field in ['sample_id','lot_name','image_sha256']}
    assert not any(any(v.values()) for v in result['pool_intersections'].values())
    d_rows = jl(REPO/'协作/02_ClaudeCode_实操/夜间扩容_20261008_0244/数据/D_N3072.jsonl')
    d_ids = {r['sample_id'] for r in d_rows}
    d_lots = {manifest[s]['lot_name'] for s in d_ids}
    result['confirmation_D_prior_contact'] = {'IDs': len(set(ids) & d_ids),
                                            'lots': len(set(lot_keys) & d_lots)}
    train = jl(CPU/'data_v3/group256_sft4.jsonl')
    rl = jl(CPU/'data_v3/group256_grpo.jsonl')
    assert len(train) == 1024 and len(rl) == 256
    assert [train[i]['sample_id'] for i in range(0,1024,4)] == [r['sample_id'] for r in rl]
    assert all(len({r['sample_id'] for r in train[i:i+4]}) == 1 for i in range(0,1024,4))
    audit = jl(SERVER/'M2_v3/reward_audit.jsonl')
    assert len(audit) == 1024
    assert Counter(r['sample_id'] for r in audit) == Counter({r['sample_id']:4 for r in rl})
    groups = {}
    reference_by_id = {r['sample_id']: json.loads(r['reference_json']) for r in rl}
    for r in audit:
        expected = SCORER.score(r['response'], reference_by_id[r['sample_id']])
        assert abs(expected['reward']-r['reward']) < 1e-12
        groups.setdefault(r['sample_id'],[]).append(r['reward'])
    result['training_audit'] = {'candidates':len(audit), 'groups':len(groups),
                                'zero_variance_groups':sum(len(set(v)) == 1 for v in groups.values()),
                                'zero_variance_perfect_groups':sum(len(set(v)) == 1 and abs(v[0]-1) < 1e-12
                                                                   for v in groups.values()),
                                'zero_variance_imperfect_groups':sum(len(set(v)) == 1 and v[0] < 1-1e-12
                                                                     for v in groups.values()),
                                'average_reward':sum(r['reward'] for r in audit)/len(audit),
                                'presented_group_order_matches_SFT':True,
                                'generation_audit_order_matches_frozen_pool':
                                    [r['sample_id'] for r in audit[::4]] == [r['sample_id'] for r in rl]}
    with tarfile.open(CLAUDE/'out_v3/服务器产物/v3results.tgz') as archive:
        names = archive.getnames()
    result['evidence_inventory'] = {
        'archive_members':len(names), 'framework_saved_args_json':[n for n in names if n.endswith('/args.json')],
        'runtime_gates_in_archive':[n for n in names if n.endswith('/runtime_gate.json')],
        'training_logs_in_archive':[n for n in names if 'logging.jsonl' in n or 'trainer_state.json' in n],
        'limitations':['v3_config.trainer_actual_args is the launcher object after the pipeline call',
                       'probe runtime evidence exists; formal M1/M2 runtime reports are absent from delivery',
                       'actual scheduler/warmup and label-mask trace not available in the delivery']}
    result['probe_gate'] = obj(SERVER/'gate_rl/runtime_gate.json')
    result['template_gate'] = obj(SERVER/'template_gate.json')
    result['configs'] = {tag:obj(SERVER/tag/'v3_config.json') for tag in TAGS}
    result['runtime_reward_source_byte_match'] = (sha(CLAUDE/'core_v3_复核修复.py') ==
                                                  sha(CLAUDE/'out_v3/服务器产物/reward_v3.py'))
    # Retain useful settings, but never propagate device UUID/private execution paths.
    for cfg in result['configs'].values():
        for key in ['cuda_visible_devices','model_start','data','start_adapter','checkpoints','final_checkpoint']:
            cfg.pop(key,None)
        actual = cfg['trainer_actual_args']
        actual['adapters'] = ['M0_v3_checkpoint128'] if actual['adapters'] else []
        if actual.get('ref_adapters'):
            actual['ref_adapters'] = ['M0_v3_checkpoint128']
    result['wall_time_notes'] = {'basis':'launcher timers and eval ledgers; not total billing',
                                'eval_device_minutes':sum(jl(SERVER/'评测_v3'/f'台账_{tag}.jsonl')[-1]['总秒']
                                                          for tag in TAGS)/60,
                                'old_1437_and_cash_total':'not independently audited in this pass'}
    return result


def audit_workbuddy():
    blind = {r['item_id']:r for r in jl(CPU/'WorkBuddy84/blind84.jsonl')}
    wanted = '002 003 005 006 009 010 014 016 024 029 031 033 034 037 038 039 040 043 044 046 049 051 055 056 061 063 064 068 073 076'.split()
    expected_ids = ['wbv3_'+s for s in wanted]
    rows = []; check_fields = {'morphology_zh','main_location_zh','location_clock','orientation_clock',
                              'secondary_observations','caption_zh','uncertainty'}
    for p in sorted(WB.glob('rv30_*.review.json')):
        review = obj(p); rid = review['review_id']; iid = review['item_id']
        start = obj(WB/f'{rid}.started.json'); meta = obj(WB/f'{rid}.observation_meta.json')
        done = obj(WB/f'{rid}.done.json'); obs = WB/f'{rid}.observation_raw.json'
        image = REPO/'data/images'/f"{blind[iid]['sample_id']}.png"
        assert sha(image) == blind[iid]['image_sha256'] == review['image_sha256']
        assert sha(obs) == meta['observation_raw_sha256'] == review['observation_raw_sha256']
        assert sha(OLD/f'{iid}.raw.txt') == review['original_raw_sha256']
        assert start['image_sha256_computed'] == blind[iid]['image_sha256']
        assert CHECKER.check(read(obs))['schema_ok']
        assert set(review['checks']) == check_fields
        assert all(v['verdict'] in ['supported','contradicted','unverifiable','not_applicable']
                   and isinstance(v['evidence'],str) and v['evidence'].strip()
                   for v in review['checks'].values())
        assert review['overall'] in ['usable','requires_review','reject']
        times = [datetime.fromisoformat(x.replace('Z','+00:00')) for x in
                 [start['started_at_utc'], meta['written_at_utc'], done['done_at_utc']]]
        rows.append({'review_id':rid,'item_id':iid,'sample_id':review['sample_id'],
                     'overall':review['overall'],'recorded_started':start['started_at_utc'],
                     'recorded_observation_written':meta['written_at_utc'],
                     'recorded_done':done['done_at_utc'],
                     'time_sequence_consistent':times[0] <= times[1] <= times[2],
                     'mtime_observation_before_review':obs.stat().st_mtime <= p.stat().st_mtime,
                     'contradicted_fields':[k for k,v in review['checks'].items() if v['verdict']=='contradicted'],
                     'unverifiable_fields':[k for k,v in review['checks'].items() if v['verdict']=='unverifiable']})
    assert [r['item_id'] for r in rows] == expected_ids
    old_schema = sum(CHECKER.check(read(OLD/f'{iid}.raw.txt'))['schema_ok'] for iid in blind)
    png_ok = sum(sha(REPO/'data/images'/f"{r['sample_id']}.png") == r['image_sha256'] for r in blind.values())
    manifest_mismatch = []
    for line in read(WB/'raw_sha256_manifest.txt').splitlines():
        digest, name = line.split('  ',1)
        p = WB/name
        if not p.is_file() or sha(p) != digest:
            manifest_mismatch.append(name)
    return {'counts':{'reviews':len(rows),'observations':len(rows),'originals_schema_ok':old_schema,
                      'current_PNG_SHA_ok':png_ok},
            'model_review_distribution':dict(Counter(r['overall'] for r in rows)),
            'timestamps_inconsistent_items':[r['item_id'] for r in rows if not r['time_sequence_consistent']],
            'mtime_reversed_items':[r['item_id'] for r in rows if not r['mtime_observation_before_review']],
            'producer_SHA_manifest_mismatches':manifest_mismatch,
            'independent_before_old_answer_open':'unverified: file mtime does not record old raw open event',
            'visual_tool_reads_and_backend_requests':'not independently evidenced by this package',
            'per_item':rows,
            'followup_allocation':{
                'model_usable_candidates':[r['item_id'] for r in rows if r['overall']=='usable'],
                'review_or_reject':[r['item_id'] for r in rows if r['overall']!='usable'],
                'remaining_unreviewed':[iid for iid in blind if iid not in expected_ids],
                'semantic_hold_in_usable':['wbv3_064'],
                'rule':'CPU candidate copy only; no new training or auto-gold; preserve original annotations'}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run',action='store_true')
    args = parser.parse_args()
    global SCORER, CHECKER
    SCORER = load_module('audited_v3_score',CLAUDE/'core_v3_复核修复.py')
    # Format checker depends on its own core_v3 module; import from its frozen directory.
    import sys
    sys.path.insert(0,str(CPU))
    CHECKER = load_module('audited_wb_schema',CPU/'workbuddy_schema_check.py')
    manifest = {r['sample_id']:r for r in jl(REPO/'data/manifest.jsonl')}
    v3 = audit_v3(manifest)
    wb = audit_workbuddy()
    assert SCORER.self_test()['all_passed']
    inputs = {str(Path(p).relative_to(REPO)).replace('\\','/'):h for p,h in SNAPSHOT.items()}
    changed = [p for p,h in SNAPSHOT.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
    assert not changed, 'Source bytes changed while auditing'
    outputs = {'v3_独立核验.json':v3,'WorkBuddy30_独立核验.json':wb,
               '输入指纹.json':{'seed':3407,'sources_unchanged_during_audit':True,'inputs':inputs}}
    if not args.dry_run:
        for name,data in outputs.items():
            (HERE/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'dry_run':args.dry_run,'v3':v3['models'],'paired':v3['paired'],
                      'training':v3['training_audit'],'WB_distribution':wb['model_review_distribution'],
                      'WB_time_inconsistent_N':len(wb['timestamps_inconsistent_items']),
                      'WB_manifest_mismatches':wb['producer_SHA_manifest_mismatches']},ensure_ascii=False))


if __name__ == '__main__':
    main()
