from pathlib import Path
from collections import Counter
import hashlib
import importlib.util
import json
import random
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
RUN=ROOT/'协作/02_ClaudeCode_实操/多字段RL_v2_CPU准备_20261009'
DATA=RUN/'out_v2'
def rows(p):return [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]
spec=importlib.util.spec_from_file_location('multifield_reward',RUN/'奖励_v2.py')
reward=importlib.util.module_from_spec(spec);spec.loader.exec_module(reward)
gold=rows(DATA/'confirm120_gold.jsonl');bygold={r['sample_id']:r for r in gold}
assert len(gold)==len(bygold)==120
ids=[r['sample_id'] for r in gold]
classes=sorted({r['gold_class'] for r in gold})
class_index={c:i for i,c in enumerate(classes)}
unknown_index=len(classes)
truth=np.array([class_index[r['gold_class']] for r in gold])
N=len(ids)
def macro_for(t,p):
    vals=[]
    for i in range(len(classes)):
        tp=np.count_nonzero((t==i)&(p==i));fp=np.count_nonzero((t!=i)&(p==i));fn=np.count_nonzero((t==i)&(p!=i))
        vals.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    return float(np.mean(vals))

points={};preds={};scores={};flags={};direction_details=[]
for tag in ['M0','M1','M2']:
    rr=rows(DATA/'评测'/f'{tag}_raw.jsonl');br={r['sample_id']:r for r in rr}
    assert len(rr)==len(br)==120 and set(br)==set(ids)
    ps=[];rs=[];parts=[];format_scores=[];parsed=0
    direction=Counter();direction_den=Counter()
    for sid in ids:
        r=br[sid];g=bygold[sid];obj,how=reward.parse_answer(r['raw'])
        ref=reward.parse_reference(g,0);out=reward.score_one(obj,ref);fm,_=reward.check_format(obj,how)
        pc=obj.get('defect_class') if isinstance(obj,dict) else None
        ps.append(class_index.get(pc,unknown_index));rs.append(.9*out['fact']+.1*fm)
        parts.append(out['parts']);format_scores.append(fm);parsed+=obj is not None
        dt=g['gold_direction_type'];pred_dt=obj.get('direction_type') if obj else None;sectors=obj.get('clock_sectors') if obj else None
        if dt in ['single','axis']:
            direction_den['positive_clock']+=1
            direction['positive_type_correct']+=pred_dt==dt
            direction['positive_fully_correct']+=out['parts']['direction']==1
            direction['positive_weighted_sum']+=out['parts']['direction']
            direction_details.append({'model':tag,'sample_id':sid,'ref_type':dt,'ref_sectors':json.loads(g['gold_sectors']) if isinstance(g['gold_sectors'],str) else g['gold_sectors'],
                                      'pred_type':pred_dt,'pred_sectors':sectors,'weighted_score':out['parts']['direction']})
        elif dt=='none':
            direction_den['nondirectional']+=1
            direction['nondirectional_fully_correct']+=out['parts']['direction']==1
        else:direction_den['reference_unknown']+=1
    p=np.array(ps);preds[tag]=p;scores[tag]=np.array(rs)
    points[tag]={'n':N,'strict_json_parsed':parsed,'correct':int(np.count_nonzero(p==truth)),
                 'accuracy':float(np.mean(p==truth)),'macro_f1_fixed_actual_classes':macro_for(truth,p),
                 'coverage_fully_correct':sum(r['coverage_level']==1 for r in parts),
                 'zones_fully_correct':sum(r['zones']==1 for r in parts),
                 'mean_rule_reward':float(np.mean(rs)),'format_mean':float(np.mean(format_scores)),
                 'direction_counts':dict(direction),'direction_denominators':dict(direction_den),
                 'positive_direction_weighted_mean':direction['positive_weighted_sum']/direction_den['positive_clock'] if direction_den['positive_clock'] else None}

rnd=random.Random(3407);B=10000
ix=np.asarray([[rnd.randrange(N) for _ in range(N)] for _ in range(B)],dtype=np.int16)
boot_macros={}
for tag,p in preds.items():
    vals=np.zeros(B)
    tx=truth[ix];px=p[ix]
    for c in range(len(classes)):
        tp=((tx==c)&(px==c)).sum(axis=1);fp=((tx!=c)&(px==c)).sum(axis=1);fn=((tx==c)&(px!=c)).sum(axis=1)
        den=2*tp+fp+fn;vals+=np.divide(2*tp,den,out=np.zeros(B),where=den!=0)/len(classes)
    boot_macros[tag]=vals
def interval(a):
    s=np.sort(a);return [float(s[int(.025*B)]),float(s[int(.975*B)])]
pairs={}
for a,b in [('M1','M0'),('M2','M0'),('M2','M1')]:
    d=(preds[a]==truth).astype(float)-(preds[b]==truth).astype(float)
    dr=scores[a]-scores[b]
    dm=boot_macros[a]-boot_macros[b]
    pairs[a+' minus '+b]={'class_accuracy_diff':float(d.mean()),'class_accuracy_ci95':interval(d[ix].mean(axis=1)),
                          'primary_macro_f1_diff':points[a]['macro_f1_fixed_actual_classes']-points[b]['macro_f1_fixed_actual_classes'],
                          'primary_macro_f1_ci95':interval(dm),'secondary_rule_reward_diff':float(dr.mean()),'secondary_rule_reward_ci95':interval(dr[ix].mean(axis=1))}

audit=rows(DATA/'服务器产物/out/M2/reward_audit.jsonl')
order=rows(DATA/'group60_grpo.jsonl')
groups=[audit[i:i+4] for i in range(0,len(audit),4)]
assert len(audit)==240 and len(groups)==60
assert all(len({r['sample_id'] for r in g})==1 for g in groups)
actual_order=[g[0]['sample_id'] for g in groups]
assert actual_order==[r['sample_id'] for r in order]
first={r['sample_id'] for g in groups[:10] for r in g};last={r['sample_id'] for g in groups[-10:] for r in g}
testgeom={r['sample_id']:r for r in rows(RUN/'几何特征表.jsonl')}
first_cov=Counter(r['gold_coverage_level'] for r in order[:10]);last_cov=Counter(r['gold_coverage_level'] for r in order[-10:])
trial_audit={'rows':len(audit),'groups':len(groups),'groups_all_same_id':True,'matches_frozen_group60_order':True,
             'first10_last10_id_overlap':len(first&last),'first10_reference_coverage':dict(first_cov),'last10_reference_coverage':dict(last_cov),
             'first10_candidate_reward_mean':float(np.mean([r['reward'] for g in groups[:10] for r in g])),
             'last10_candidate_reward_mean':float(np.mean([r['reward'] for g in groups[-10:] for r in g])),
             'direction_reference_counts':dict(Counter(r['gold_direction_type'] for r in order)),
             'first10_direction_reference_counts':dict(Counter(r['gold_direction_type'] for r in order[:10])),
             'last10_direction_reference_counts':dict(Counter(r['gold_direction_type'] for r in order[-10:])),
             'same_reward_groups':sum(len({r['reward'] for r in g})==1 for g in groups),
             'interpretation':'首尾10组是不同图，不是固定训练集的前后同题比较，奖励均值趋势不能单独证明记忆/过拟合或训练改善。'}
old_train_ids={x['sample_id'] for x in rows(ROOT/'协作/02_ClaudeCode_实操/夜间扩容_20261008_0244/数据/L_N3072.jsonl')}
old_train_lots={r['lot_name'] for r in rows(ROOT/'data/manifest.jsonl') if r['sample_id'] in old_train_ids}
result={'primary_metric':'公开类别Macro-F1，分母固定为确认池实际8类，Near_full缺席不补0混入主分。',
        'classes':classes,'class_support':dict(Counter(r['gold_class'] for r in gold)),
        'models':points,'paired_bootstrap':pairs,'direction_positive_cases':direction_details,
        'M2_group_audit':trial_audit,
        'confirmation_vs_oldD_training_lot_overlap':len({r['lot_name'] for r in gold}&old_train_lots),
        'scope':'独立CPU重算，无GPU/API；规则奖励属于次指标；不改原答与旧主表。'}
(HERE/'独立复算.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='direction_positive_cases'},ensure_ascii=False,indent=2))
