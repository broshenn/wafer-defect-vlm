"""Repaired v2 result analysis. Requires all 360 outputs; never scores not-run as zero.

Keep v2 rules frozen, primary class Macro-F1, direction three denominators.
Input original remote artifacts should remain private; this analyzes sanitized copies.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import numpy as np

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
OLD=REPO/'协作/02_ClaudeCode_实操/多字段RL_v2_CPU准备_20261009'


def rows(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    spec=importlib.util.spec_from_file_location('_frozen_v2_reward',OLD/'奖励_v2.py')
    reward=importlib.util.module_from_spec(spec);spec.loader.exec_module(reward)
    gold=rows(OLD/'out_v2/confirm120_gold.jsonl')
    bygold={r['sample_id']:r for r in gold};ids=[r['sample_id'] for r in gold]
    assert len(ids)==len(bygold)==120
    classes=sorted({r['gold_class'] for r in gold})
    index={c:i for i,c in enumerate(classes)}
    truth=np.array([index[r['gold_class']] for r in gold])
    def macro(t,p):
        vals=[]
        for c in range(len(classes)):
            tp=((t==c)&(p==c)).sum();fp=((t!=c)&(p==c)).sum();fn=((t==c)&(p!=c)).sum()
            vals.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
        return float(np.mean(vals))
    pred,values,models={}, {}, {}
    for tag in ['M0_actual','M1_fix','M2_fix']:
        path=args.input/f'{tag}_raw.jsonl'
        rr=rows(path);br={r['sample_id']:r for r in rr}
        if len(rr)!=len(br) or set(br)!=set(ids):
            raise RuntimeError('Incomplete/duplicate raw IDs: '+tag)
        ps, rs, parts, formats=[],[],[],[]
        direction=Counter();den=Counter();strict_json=0;normalized_json=0;empty_think=0
        for sid in ids:
            raw=br[sid]['raw'];g=bygold[sid]
            try:
                parsed=json.loads(raw)
                strict_json+=isinstance(parsed,dict)
            except (ValueError,TypeError):
                pass
            obj,how=reward.parse_answer(raw);ref=reward.parse_reference(g,0)
            normalized_json+=obj is not None
            empty_think+=raw.lstrip().startswith('<think>')
            result=reward.score_one(obj,ref);fmt,_=reward.check_format(obj,how)
            pc=obj.get('defect_class') if isinstance(obj,dict) else None
            ps.append(index.get(pc,len(classes)));rs.append(.9*result['fact']+.1*fmt)
            parts.append(result['parts']);formats.append(fmt)
            dt=g['gold_direction_type']
            bucket='positive' if dt in ['single','axis'] else ('nondirectional' if dt=='none' else 'unknown')
            den[bucket]+=1
            if bucket!='unknown':
                direction[bucket+'_fully_correct']+=result['parts']['direction']==1
        p=np.array(ps);pred[tag]=p;values[tag]=np.array(rs)
        models[tag]={'n':120,'class_correct':int((p==truth).sum()),'class_accuracy':float((p==truth).mean()),
                     'primary_class_macro_f1':macro(truth,p),'strict_JSON_objects':strict_json,
                     'normalized_JSON_objects_frozen_parser':normalized_json,
                     'think_prefix_count':empty_think,
                     'frozen_v2_checker_full_count':sum(f==1 for f in formats),
                     'frozen_v2_checker_mean':float(np.mean(formats)),
                     'coverage_correct':sum(r['coverage_level']==1 for r in parts),
                     'zones_correct':sum(r['zones']==1 for r in parts),
                     'secondary_rule_reward':float(np.mean(rs)),
                     'direction_denominators':dict(den),'direction_correct':dict(direction),
                     'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    B=10000;rnd=random.Random(3407)
    ix=np.array([[rnd.randrange(120) for _ in range(120)] for _ in range(B)],dtype=np.int16)
    boot={}
    for tag,p in pred.items():
        tx,px=truth[ix],p[ix];val=np.zeros(B)
        for c in range(len(classes)):
            tp=((tx==c)&(px==c)).sum(axis=1);fp=((tx!=c)&(px==c)).sum(axis=1);fn=((tx==c)&(px!=c)).sum(axis=1)
            d=2*tp+fp+fn
            val+=np.divide(2*tp,d,out=np.zeros(B),where=d!=0)/len(classes)
        boot[tag]=val
    def ci(val):
        s=np.sort(val)
        return [float(s[int(.025*B)]),float(s[int(.975*B)])]
    pairs={}
    for a,b in [('M1_fix','M0_actual'),('M2_fix','M0_actual'),('M2_fix','M1_fix')]:
        delta=(pred[a]==truth).astype(float)-(pred[b]==truth).astype(float)
        reward_delta=values[a]-values[b]
        pairs[a+' minus '+b]={'primary_macro_f1_diff':models[a]['primary_class_macro_f1']-models[b]['primary_class_macro_f1'],
                              'primary_macro_f1_CI95':ci(boot[a]-boot[b]),
                              'class_accuracy_diff':float(delta.mean()),'class_accuracy_CI95':ci(delta[ix].mean(axis=1)),
                              'secondary_rule_reward_diff':float(reward_delta.mean()),'secondary_rule_reward_CI95':ci(reward_delta[ix].mean(axis=1))}
    result={'run_id':'v2_fix_s3407_20261010','classes':classes,'support':dict(Counter(r['gold_class'] for r in gold)),
            'primary':'Public class Macro-F1 over fixed actual confirmation classes; not rule reward',
            'category_extraction':'Frozen v2 parse_answer normalization; raw strict JSON is reported separately',
            'bootstrap':'sample-paired 10000, seed3407; single seed; development confirmation',
            'models':models,'paired':pairs,'caption_evaluation':'not_run',
            'limitations':['v2 legacy clock/axis/coverage definitions retained for controlled repair',
                           'old confirmation120 is development and overlaps old D training lots',
                           'No independent human gold; no claim of universal RL improvement']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'models':models,'paired':pairs},ensure_ascii=False))


if __name__=='__main__':
    main()
