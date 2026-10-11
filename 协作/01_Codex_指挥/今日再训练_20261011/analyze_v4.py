"""Evaluate frozen v4 and cloned M0 raw outputs locally; no GPU or requests."""
from collections import Counter
import importlib.util
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
CPU=HERE.parent/'无卡CPU整备_20261010'
RAW=HERE/'产物/eval'


def jl(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]


def main():
    spec=importlib.util.spec_from_file_location('frozen_reward',REPO/'协作/02_ClaudeCode_实操/一天半优化_20261010/core_v3_复核修复.py')
    R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)
    refs={r['sample_id']:r['reference'] for r in jl(CPU/'data_v3/confirmation120_reference.jsonl')}
    manifest={r['sample_id']:r for r in jl(REPO/'data/manifest.jsonl')}
    ids=sorted(refs);classes=sorted({r['defect_class'] for r in refs.values()})
    gold=np.array([refs[s]['defect_class'] for s in ids]);lot_keys=[manifest[s]['lot_name'] for s in ids]
    lots=sorted(set(lot_keys));membership=np.array([[a==b for b in lot_keys] for a in lots],float)
    draw=np.random.default_rng(3407).multinomial(len(lots),np.full(len(lots),1/len(lots)),size=10000)
    weights=draw@membership;totals=weights.sum(axis=1)
    def counts(pred):
        return np.column_stack([np.array([(gold==c)&(pred==c),(gold!=c)&(pred==c),(gold==c)&(pred!=c)],float).T for c in classes])
    def mf(c):
        t=c.reshape(*c.shape[:-1],-1,3);n=2*t[...,0];d=n+t[...,1]+t[...,2]
        return np.divide(n,d,out=np.zeros_like(n),where=d>0).mean(axis=-1)
    ci=lambda x:[float(v) for v in np.quantile(x,[.025,.975])]
    report={'status':'development_only_no_human_gold','N':len(ids),'classes':classes,
            'seed':3407,'lot_N':len(lots),'replicates':10000,'models':{}}
    stash={}
    for tag in ['M0_clone','SFT_replay_v4b']:
        rows=jl(RAW/f'{tag}_raw.jsonl');by={r['sample_id']:r for r in rows}
        assert len(rows)==len(by)==len(ids) and set(by)==set(ids),'Frozen evaluation incomplete'
        pred=[];reward=[];coverage=0;zones=0;raw_ok=0;format_ok=0;nondir=0;unknown=0;angular_positive=0
        by_zone={};by_coverage={};parsed={}
        for sid in ids:
            text=by[sid]['raw']
            try:answer=R.parse(text)
            except (ValueError,TypeError):answer={}
            try:R.parse_raw_strict(text);raw_ok+=1
            except (ValueError,TypeError):pass
            result=R.score(text,refs[sid]);format_ok+=int(result['format_ok'])
            parsed[sid]=answer
            zone='|'.join(refs[sid]['red_mass_zones']) if isinstance(refs[sid]['red_mass_zones'],list) else 'unknown'
            z=by_zone.setdefault(zone,{'N':0,'exact_correct':0});z['N']+=1;z['exact_correct']+=int(result['parts']['zones']==1)
            band=refs[sid]['coverage_band'];z=by_coverage.setdefault(band,{'N':0,'exact_correct':0});z['N']+=1;z['exact_correct']+=int(result['parts']['coverage']==1)
            pred.append(answer.get('defect_class'));reward.append(result['reward'])
            coverage+=int(result['parts']['coverage']==1);zones+=int(result['parts']['zones']==1)
            if refs[sid]['angular_type']=='nondirectional':nondir+=int(result['parts']['angular']==1)
            elif refs[sid]['angular_type']=='unknown':unknown+=1
            else:angular_positive+=1
        pred=np.array(pred,dtype=object);rewards=np.array(reward);c=counts(pred)
        f1=float(mf(c.sum(axis=0)));bootstrap_f1=mf(weights@c)
        report['models'][tag]={'class_correct':int((pred==gold).sum()),'accuracy':float((pred==gold).mean()),
            'macro_f1':f1,'macro_f1_lot_CI':ci(bootstrap_f1),'coverage_correct':coverage,
            'zones_correct':zones,'rule_reward':float(rewards.mean()),'strict_raw_JSON':raw_ok,
            'normalized_schema_pass':format_ok,'nondirectional_correct':nondir,
            'nondirectional_N':sum(r['angular_type']=='nondirectional' for r in refs.values()),
            'positive_direction_N':angular_positive,'unknown_direction_N':unknown,
            'finish_reason_counts':dict(Counter(by[s].get('finish_reason') for s in ids)),
            'by_zone_reference':by_zone,'by_coverage_reference':by_coverage,
            'per_class':{str(cl):{'N':int((gold==cl).sum()),'tp':int(c[:,i*3].sum()),
                          'fp':int(c[:,i*3+1].sum()),'fn':int(c[:,i*3+2].sum())} for i,cl in enumerate(classes)}}
        stash[tag]={'bootstrap_f1':bootstrap_f1,'reward':rewards,'raw':by,'parsed':parsed}
    b=report['models']['M0_clone'];v=report['models']['SFT_replay_v4b']
    diff=stash['SFT_replay_v4b']['reward']-stash['M0_clone']['reward']
    report['paired_v4_minus_M0']={'macro_f1_difference':v['macro_f1']-b['macro_f1'],
        'macro_f1_lot_CI':ci(stash['SFT_replay_v4b']['bootstrap_f1']-stash['M0_clone']['bootstrap_f1']),
        'rule_reward_difference':float(diff.mean()),'rule_reward_lot_CI':ci((weights@diff)/totals),
        'coverage_correct_difference':v['coverage_correct']-b['coverage_correct'],
        'zones_correct_difference':v['zones_correct']-b['zones_correct']}
    old_path=REPO/'协作/02_ClaudeCode_实操/一天半优化_20261010/out_v3/服务器产物/out/评测_v3/M0_v3_raw.jsonl'
    old={r['sample_id']:r for r in jl(old_path)}
    report['cloned_M0_vs_old_raw_equal']=sum(stash['M0_clone']['raw'][s]['raw']==old[s]['raw'] for s in ids)
    changes=[]
    for sid in ids:
        old_answer=stash['M0_clone']['parsed'][sid];new_answer=stash['SFT_replay_v4b']['parsed'][sid]
        for field in ['defect_class','coverage_band','red_mass_zones']:
            norm=lambda x:tuple(sorted(x)) if isinstance(x,list) else x
            before=norm(old_answer.get(field))==norm(refs[sid][field]);after=norm(new_answer.get(field))==norm(refs[sid][field])
            if before!=after:changes.append({'sample_id':sid,'field':field,'change':'improved' if after else 'regressed',
                'reference':refs[sid][field],'M0':old_answer.get(field),'v4':new_answer.get(field)})
    report['changed_cases']=changes
    report['application_gate_passed']=(v['macro_f1']>=b['macro_f1'] and
       v['coverage_correct']>=b['coverage_correct'] and v['zones_correct']>=b['zones_correct'] and
       max(v['coverage_correct']-b['coverage_correct'],v['zones_correct']-b['zones_correct'])>=2 and
       v['normalized_schema_pass']==len(ids))
    report['limitations']=['single_seed','seven_covered_classes','development_confirmation_not_final_blind_test',
                          'no_positive_direction_reference','same_rule_compliance_is_not_independent_fact_gold',
                          'caption_not_evaluated','lot_bootstrap_fixed_class_denominator_absent_class_F1_zero']
    (HERE/'最终比较.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# v4b 实际同题结果','',
        '| 模型 | 类别正确 | Macro-F1 | 覆盖正确 | 区带正确 | 规则分 |',
        '|---|---:|---:|---:|---:|---:|']
    for tag,a in report['models'].items():
        lines.append(f"| {tag} | {a['class_correct']}/120 | {a['macro_f1']:.4f} | {a['coverage_correct']}/120 | {a['zones_correct']}/120 | {a['rule_reward']:.4f} |")
    p=report['paired_v4_minus_M0'];lines+=['',f"主指标差 {p['macro_f1_difference']:+.4f}，按lot区间 {p['macro_f1_lot_CI']}。",
        f"规则分差 {p['rule_reward_difference']:+.4f}，按lot区间 {p['rule_reward_lot_CI']}。",
        f"克隆M0与旧M0原答完全相同：{report['cloned_M0_vs_old_raw_equal']}/120。",
        f"预设应用门槛：{'通过' if report['application_gate_passed'] else '未通过'}。",'',
        '以上为单种子七类开发比较；没有正向方向参考、caption或人审金标，不能升级为最终能力结论。']
    (HERE/'实际结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
