"""Codex: 核对格式结果的语义与路径指向，保留首次严格字符串比较产物。"""
from pathlib import Path
import json
import hashlib
from collections import Counter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BATCH = ROOT / '协作/03_ZCode_标注/20261002-024921_十八张开发试标'
PREP = ROOT / '协作/02_ClaudeCode_实操/口径修正与准备_20261002-021207'

def resolve_record_path(value):
    p = Path(value)
    return (p if p.is_absolute() else BATCH / p).resolve()

def main():
    target = HERE / '最终核验.json'
    if target.exists():
        raise SystemExit('Existing evidence retained; use a new directory to rerun.')
    # 相对/绝对指向同一文件应通过，指向另一文件必须失败。
    example = BATCH / 'wafer_00043234_006_raw.json'
    assert resolve_record_path(example.name) == resolve_record_path(str(example))
    assert resolve_record_path('different_raw.json') != resolve_record_path(str(example))
    first = json.loads((HERE / '指纹与格式复验.json').read_text(encoding='utf-8'))
    audit = [json.loads(s) for s in (PREP / 'audit_selected.jsonl').read_text(encoding='utf-8').splitlines() if s]
    amap = {r['sample_id']:r for r in audit}
    manifest = [json.loads(s) for s in (ROOT/'data/manifest.jsonl').read_text(encoding='utf-8').splitlines() if s]
    rows=[]
    for r in first['rows']:
        sid=r['sample_id']
        stored=json.loads((BATCH/(sid+'_check_stdout.txt')).read_text(encoding='utf-8-sig'))
        current=r['checker_result']
        path_same=resolve_record_path(stored['path']) == resolve_record_path(current['path']) == (BATCH/(sid+'_raw.json')).resolve()
        fields_same={k:v for k,v in stored.items() if k!='path'} == {k:v for k,v in current.items() if k!='path'}
        rawfile=BATCH/(sid+'_raw.json')
        assert hashlib.sha256(rawfile.read_bytes()).hexdigest()==r['raw_sha256']
        obj=json.loads(rawfile.read_text(encoding='utf-8-sig'))
        a=amap[sid]
        source=manifest[a['manifest_line']-1]
        assert source['sample_id']==sid
        source_class=source['failure_type'].replace('-','_')
        assert source_class==a['failure_type_normalized']
        assert source['label_source']==a['label_source']=='ground_truth'
        keys=['png_matches_blind','png_matches_tsv','raw_matches_tsv','fingerprint_records_match','strict_json_and_seven_fields','stderr_empty']
        passed=all(r[k] for k in keys) and path_same and fields_same and r['checker_exit']==r['stored_exit']==0
        rows.append({'item_id':r['item_id'],'sample_id':sid,'passed':passed,
                     'checker_semantics_same':fields_same,'checker_paths_same_target':path_same,
                     'manifest_line':a['manifest_line'],'original_class':source_class,
                     'label_source':source['label_source'],'prediction':obj['defect_class'],
                     'original_label_agreement':source_class==obj['defect_class']})
    classes=['Center','Donut','Edge_Loc','Edge_Ring','Loc','Near_full','Random','Scratch','none']
    recalls={c:{'total':sum(r['original_class']==c for r in rows),'agree':sum(r['original_class']==c and r['original_label_agreement'] for r in rows)} for c in classes}
    summary=dict(first['summary'])
    summary.update({'rows_pass':sum(r['passed'] for r in rows),'path_lexical_only_differences':sum(not r['stored_stdout_matches_current'] for r in first['rows']),
                    'original_label_agreements':sum(r['original_label_agreement'] for r in rows),'original_label_agreement_rate':sum(r['original_label_agreement'] for r in rows)/len(rows),
                    'original_label_counts':dict(Counter(r['original_class'] for r in rows)),
                    'label_agreement_is_visual_accuracy':False,'review_kind':'nonblind_model_review'})
    result={'summary':summary,'per_class_agreement':recalls,'rows':rows,
            'note':'原始类标经 manifest 核对；一致率不是人审视觉准确率。首次0/18是路径字符串比较过严，18个格式检查退出码本来均为0。'}
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))
    print(json.dumps(recalls,ensure_ascii=False))
    return 0 if summary['rows_pass']==18 else 1

if __name__=='__main__':
    raise SystemExit(main())
