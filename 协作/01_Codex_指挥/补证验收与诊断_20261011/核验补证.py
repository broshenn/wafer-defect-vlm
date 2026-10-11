"""Verify delivered supplements and diagnose existing outputs. CPU, read-only.

Private WorkBuddy log contents are not copied into project outputs.
--dry-run computes without writing; fixed inputs and seed; originals unchanged.
"""
from collections import Counter
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
FIRST = HERE.parent/'接手续验收_20261011'
CPU = HERE.parent/'无卡CPU整备_20261010'
CC = REPO/'协作/02_ClaudeCode_实操/一天半优化_20261010'
EV = CC/'补证_20261011/ev'
WB = REPO/'协作/04_WorkBuddy_复核/内容复核30_v3_20261010_233400'
ADD = WB/'追加记录_20261011_0133'
OLD = WB.parent/'方向形态84_v3_20261010_200950'
SNAP = {}


def read(p):
    p = Path(p); data = p.read_bytes()
    SNAP[p] = hashlib.sha256(data).hexdigest()
    return data.decode('utf-8')


def obj(p):
    return json.loads(read(p))


def jl(p):
    return [json.loads(s) for s in read(p).splitlines() if s.strip()]


def digest(p):
    p = Path(p); h = hashlib.sha256(p.read_bytes()).hexdigest(); SNAP[p] = h
    return h


def formal():
    out = {}; args = {}
    fingerprints = {}
    for tag in ['M0_v3','M1_v3','M2_v3']:
        ap = next((EV/tag).rglob('args.json')); a = obj(ap); args[tag] = a
        s = obj(next((EV/tag).rglob('trainer_state.json')))
        steps = 128 if tag=='M0_v3' else 256
        assert s['global_step'] == s['max_steps'] == steps
        assert s['epoch'] == 1.0 and s['num_train_epochs'] == 1
        fp = read(EV/tag/'adapter_model.sha256').split()[0]
        fingerprints[tag] = fp
        out[tag] = {'final_step':steps,'actual_epoch':s['epoch'],
                    'epoch_arg_default':a['num_train_epochs'],'max_steps_arg':a['max_steps'],
                    'loss_scale':a['loss_scale'],'adapter_sha256':fp}
        if tag!='M0_v3':
            g = obj(EV/f'{tag}_runtime_gate.json')
            assert g['model_loading_gate'] == g['two_step_gate'] == 'passed'
            assert g['policy_equal_M0_tensor_count'] == 496
            assert g['policy_changed_tensors'] == g['nonzero_gradient_tensors'] == 496
            if tag=='M2_v3':
                assert g['reference_equal_M0_tensor_count'] == g['reference_unchanged_M0_tensor_count'] == 496
                assert g['forwards']['reference'] > 0
            out[tag]['formal_runtime_gate'] = g
    assert all(out[tag]['formal_runtime_gate']['M0_sha256']==fingerprints['M0_v3']
               for tag in ['M1_v3','M2_v3'])
    m1,m2 = args['M1_v3'],args['M2_v3']
    shared = ['max_steps','learning_rate','lr_scheduler_type','warmup_ratio','warmup_steps','optim',
              'adam_beta1','adam_beta2','adam_epsilon','weight_decay','max_grad_norm','tuner_type',
              'target_modules','lora_rank','lora_alpha','lora_dropout','freeze_vit','freeze_aligner',
              'torch_dtype','bf16','per_device_train_batch_size','gradient_accumulation_steps',
              'gradient_checkpointing','max_length','seed','data_seed','dataset_shuffle',
              'train_dataloader_shuffle','dataloader_drop_last','save_strategy','save_steps',
              'save_total_limit','eval_strategy','report_to','resume_only_model','template',
              'add_non_thinking_prefix','disable_ignore_empty_think','padding_side','packing']
    assert all(m1[k]==m2[k] for k in shared)
    assert m1['model']==m2['model'] and m1['adapters']==m2['adapters']
    # ms-swift saved the policy as an absolute path, reference as a relative path.
    # The identical checkpoint suffix and runtime M0 SHA corroborate the target.
    norm_adapter=lambda p:str(p).replace('\\','/').split('out/M0_v3/')[-1]
    assert [norm_adapter(p) for p in m2['ref_adapters']]==[norm_adapter(p) for p in m2['adapters']]
    previous = obj(FIRST/'v3_独立核验.json')
    assert all(previous['models'][tag]['adapter_fingerprints']==[fp] for tag,fp in fingerprints.items())
    out['comparison'] = {'common_settings':{k:m1[k] for k in shared},
                         'same_base_path':True,'same_M0_adapter_path':True,
                         'M2_reference_path_matches_M0':True,
                         'loss_scale_difference':{'M1':m1['loss_scale'],'M2':m2['loss_scale']},
                         'batch_labels_mask_evidence':'not_recorded',
                         'full_base_shard_SHA':'not_obtained; config SHA is not full weight SHA'}
    return out


def workbuddy():
    # Inspect only relevant events; logs are private and can contain unrelated commands.
    private = Path(os.environ['USERPROFILE'])/'.workbuddy/audit-log'
    manifests = [json.loads(s) for s in (private/'manifest.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    relevant = []; closed = None
    for day in ['2026-10-10','2026-10-11']:
        p=private/f'{day}.jsonl'; data=p.read_bytes()
        rows=[json.loads(s) for s in data.decode('utf-8').splitlines() if s.strip()]
        if day=='2026-10-10':
            entry=next(r for r in manifests if r['segment']==p.name)
            assert hashlib.sha256(data).hexdigest()==entry['fileSha256']
            assert len(rows)==entry['entryCount']
            assert all(b['prevHash']==a['hash'] and b['sequence']==a['sequence']+1
                       for a,b in zip(rows,rows[1:]))
            closed={'file_SHA_matches_local_manifest':True,'entry_count':len(rows),
                    'adjacent_chain_and_sequences_match':True,
                    'scope':'local client records; not an external cryptographic witness'}
        for r in rows:
            if 1791646440000 <= r.get('timestamp',0) <= 1791649800000:
                relevant.append(r)
    relevant.sort(key=lambda r:(r['timestamp'],r['sequence']))
    def paths(r):
        return str((r.get('messageParams') or {}).get('paths','')).replace('\\','/')
    def is_read(r):
        return 'readWithRule' in str(r.get('messageKey',''))
    def is_write(r):
        return 'writeWithRule' in str(r.get('messageKey',''))
    reported = obj(ADD/'程序A_审计日志时间线.json')['per_item']
    rows=[]
    for item in reported:
        rid,iid,sid = [item[k] for k in ['review_id','item_id','sample_id']]
        stages=[('started',lambda r:is_write(r) and f'/{rid}.started.json' in paths(r)),
                ('PNG',lambda r:is_read(r) and f'data/images/{sid}.png' in paths(r)),
                ('observation',lambda r:is_write(r) and f'/{rid}.observation_raw.json' in paths(r)),
                ('old_raw',lambda r:is_read(r) and f'方向形态84_v3_20261010_200950/{iid}.raw.txt' in paths(r)),
                ('review',lambda r:is_write(r) and f'/{rid}.review.json' in paths(r))]
        matches={k:[r for r in relevant if test(r)] for k,test in stages}
        assert all(len(v)==1 for v in matches.values())
        ts=[matches[k][0]['timestamp'] for k,_ in stages]
        assert ts==sorted(ts)
        assert matches['observation'][0]['id']==item['observation_write'][0]['id']
        rows.append({'review_id':rid,'item_id':iid,'ordered_client_tool_events':True,
                     'PNG_Read_tool_events':1,'old_raw_Read_tool_events':1,
                     'source_event_sequences':{k:v[0]['sequence'] for k,v in matches.items()}})
    assert len(rows)==30
    new_manifest=[]
    for line in read(ADD/'新清单_sha256.txt').splitlines():
        if not line.strip() or line.startswith('#'):continue
        h,name=line.split('  ',1)
        assert digest(ADD/name)==h
        new_manifest.append(name)
    original_report=obj(ADD/'程序D_核验结果.json')['originals_current_fingerprints']['files']
    originals=sum(digest(WB/name)==h for group in original_report.values() for name,h in group.items())
    assert originals==150
    before=obj(FIRST/'输入指纹.json')['inputs']
    protected={path:h for path,h in before.items() if '/方向形态84_v3_' in path or '/rv30_' in path}
    assert all(digest(REPO/path)==h for path,h in protected.items())
    return {'client_log_verification':closed,'reconstructed_items':len(rows),
            'client_dispatch_order_verified':True,'matched_new_manifest_entries':len(new_manifest),
            'current_original_fingerprints_verified':originals,
            'originals_match_first_Codex_review':True,
            'limitations':['Read/Write events record dispatched tool operations, not their returned contents',
                           'no backend request count, model identity, context isolation or first cognition proof'],
            'per_item':rows}


def candidates():
    rows=jl(CC/'WorkBuddy候选分账_20261011/候选清单.jsonl')
    assert len(rows)==84 and len({r['item_id'] for r in rows})==84
    assert all(r['gold_eligible'] is False for r in rows)
    adapt_ids={r['sample_id'] for r in jl(CPU/'pools/adapt.jsonl')}
    d_ids={r['sample_id'] for r in jl(REPO/'协作/02_ClaudeCode_实操/夜间扩容_20261008_0244/数据/D_N3072.jsonl')}
    assert {r['sample_id'] for r in rows}<=adapt_ids
    for r in rows:
        assert digest(OLD/f"{r['item_id']}.raw.txt")==r['old_raw_sha256']
    return {'allocation':dict(Counter(r['allocation'] for r in rows)),
            'all_gold_eligible_false':True,'all_in_adapt':True,
            'seen_by_previous_D':sum(r['sample_id'] in d_ids for r in rows),
            'green_target_flags':[r['item_id'] for r in rows if r['target_object'].startswith('绿色合格区')]}


def diagnose():
    spec=importlib.util.spec_from_file_location('review_reward',CC/'core_v3_复核修复.py')
    core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
    ref={r['sample_id']:r['reference'] for r in jl(CPU/'data_v3/confirmation120_reference.jsonl')}
    by={m:{r['sample_id']:core.parse(r['raw']) for r in jl(CC/f'out_v3/服务器产物/out/评测_v3/{m}_raw.jsonl')}
        for m in ['M0_v3','M1_v3','M2_v3']}
    fields=['defect_class','coverage_band','red_mass_zones']
    count={}; cases=[]
    for field in fields:
        good=lambda a,b: set(a)==set(b) if field=='red_mass_zones' else a==b
        lost=[s for s,r in ref.items() if good(by['M1_v3'][s][field],r[field]) and not good(by['M2_v3'][s][field],r[field])]
        gained=[s for s,r in ref.items() if not good(by['M1_v3'][s][field],r[field]) and good(by['M2_v3'][s][field],r[field])]
        count[field]={'M1_correct_M2_wrong':len(lost),'M1_wrong_M2_correct':len(gained),
                      'net_correct_change':len(gained)-len(lost)}
        for sid in lost:
            cases.append({'sample_id':sid,'field':field,'reference':ref[sid][field],
                          'M0':by['M0_v3'][sid][field],'M1':by['M1_v3'][sid][field],
                          'M2':by['M2_v3'][sid][field],
                          'usage':'development diagnosis; never tune the final test on these cases'})
    def pred_hist(m,field):
        return dict(Counter(json.dumps(v[field],ensure_ascii=False,sort_keys=True) for v in by[m].values()))
    hist={m:{f:pred_hist(m,f) for f in ['coverage_band','angular_type']} for m in by}
    reference_hist={f:dict(Counter(json.dumps(r[f],ensure_ascii=False) for r in ref.values()))
                    for f in ['coverage_band','angular_type']}
    # Examine all training groups; mask eligibility is dictated by frozen references.
    rl=jl(CPU/'data_v3/group256_grpo.jsonl')
    training_refs={r['sample_id']:json.loads(r['reference_json']) for r in rl}
    group={}
    for r in jl(CC/'out_v3/服务器产物/out/M2_v3/reward_audit.jsonl'):
        group.setdefault(r['sample_id'],[]).append(r)
    same_wrong=Counter(); examples=[]
    for sid,rs in group.items():
        if len({r['reward'] for r in rs})!=1 or rs[0]['reward']>=1-1e-12:continue
        r=training_refs[sid]
        answers=[core.parse(x['response']) for x in rs]
        for field in fields+['angular_type']:
            if r[field]=='unknown':continue
            norm=lambda x:tuple(sorted(x)) if isinstance(x,list) else x
            values=[norm(a[field]) for a in answers]
            if len(set(values))==1 and values[0]!=norm(r[field]):same_wrong[field]+=1
        examples.append({'sample_id':sid,'reward':rs[0]['reward'],
                         'distinct_parsed_answers':len({json.dumps(a,sort_keys=True) for a in answers}),
                         'eligibility':rs[0]['eligible_weights']})
    return {'field_changes_M2_vs_M1':count,'prediction_distributions':hist,
            'reference_distributions':reference_hist,
            'same_reward_imperfect_groups_same_wrong_field':dict(same_wrong),
            'regression_cases':cases,'imperfect_group_diagnostics':examples,
            'interpretation':'counts show behavior changes; do not establish training mechanism causality'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    out={'formal_training':formal(),'workbuddy':workbuddy(),'candidates':candidates(),'diagnosis':diagnose()}
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==h for p,h in SNAP.items())
    out['scope']={'new_GPU_runs':0,'new_model_requests':0,'sources_unchanged':True,
                  'server_connections':0,'seed':3407}
    if not a.dry_run:
        (HERE/'补证独立核验与诊断.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        (HERE/'输入指纹.json').write_text(json.dumps({str(p.relative_to(REPO)).replace('\\','/'):h for p,h in SNAP.items()},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'formal_runtime_gates':'M1/M2 passed','WB_order':out['workbuddy']['reconstructed_items'],
                      'new_manifest':out['workbuddy']['matched_new_manifest_entries'],
                      'candidates':out['candidates'],'changes':out['diagnosis']['field_changes_M2_vs_M1'],
                      'same_wrong':out['diagnosis']['same_reward_imperfect_groups_same_wrong_field']},ensure_ascii=False))


if __name__=='__main__':main()
