"""独立核对导出内容；不 import 构建脚本，不加载模型。"""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
SOURCE=ROOT/'协作/02_ClaudeCode_实操/学校训练准备_20261002-025848'

def read(path):
    return [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l]

def main():
    meta=json.loads((HERE/'preparation_report.json').read_text(encoding='utf-8'))
    classes={'Center','Donut','Edge_Loc','Edge_Ring','Loc','Near_full','Random','Scratch','none'}
    prompt=(HERE/'classification_prompt_v3.txt').read_text(encoding='utf-8').removesuffix('\n')
    assert all(c in prompt for c in classes) and 'unknown' in prompt
    system=(HERE/'system_prompt.txt').read_text(encoding='utf-8').removesuffix('\n')
    for name,h in meta['output_sha256'].items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==h
    byid={r['sample_id']:r for r in read(ROOT/'data/manifest.jsonl')}
    sets={}; total=0
    for name,n in [('train_180',180),('dev_18',18),('smoke_20',20)]:
        rows=read(HERE/(name+'.ms_swift.jsonl'))
        source=read(SOURCE/(name+'.jsonl'))
        assert len(rows)==n==len(source)
        assert [r['sample_id'] for r in rows]==[r['sample_id'] for r in source]
        for row,old in zip(rows,source):
            sid=row['sample_id']; gold=byid[sid]
            assert set(row)=={'sample_id','messages','images'}
            assert row['messages'][0]=={'role':'system','content':system}
            assert row['messages'][1]=={'role':'user','content':'<image>\n'+prompt}
            assert row['messages'][2]['role']=='assistant'
            answer=json.loads(row['messages'][2]['content'])
            assert set(answer)=={'defect_class'} and answer['defect_class']==old['label']==gold['failure_type']
            assert old['label_source']==gold['label_source']=='ground_truth'
            assert row['images']==[str((ROOT/'data/images'/(sid+'.png')).resolve())]
            assert Path(row['images'][0]).is_file()
            assert row['messages'][1]['content'].count('<image>')==1
            total+=1
        sets[name]={r['sample_id'] for r in rows}
    assert sets['smoke_20']<=sets['train_180']
    assert not sets['train_180']&sets['dev_18']
    zids={r['sample_id'] for r in read(ROOT/'协作/02_ClaudeCode_实操/口径修正与准备_20261002-021207/blind_inputs.jsonl')}
    excluded_lots={byid[x]['lot_name'] for x in zids}
    for name in sets:
        assert not {byid[x]['lot_name'] for x in sets[name]}&excluded_lots
    assert not {byid[x]['lot_name'] for x in sets['train_180']}&{byid[x]['lot_name'] for x in sets['dev_18']}
    print(json.dumps({'rows_verified':total,'files':3,'all_pass':True,'template_mask_and_gpu_training':'not run'},ensure_ascii=False))

if __name__=='__main__':main()
