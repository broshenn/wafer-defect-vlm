"""Codex：核验既有清单，固定九类题面并另存 ms-swift 数据；不重新选样。"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / '协作/02_ClaudeCode_实操/学校训练准备_20261002-025848'
BLIND = ROOT / '协作/02_ClaudeCode_实操/口径修正与准备_20261002-021207/blind_inputs.jsonl'
CLASSES = ['Center','Donut','Edge_Loc','Edge_Ring','Loc','Near_full','Random','Scratch','none']
PROMPT = ('这是公开晶圆 BIN 图。黑色为晶圆外，绿色为合格 die，红色为失效 die。'
          '请根据主要可见图案，从 Center、Donut、Edge_Loc、Edge_Ring、Loc、Near_full、Random、Scratch、none 中选择最符合的一类；'
          '无法可靠判断时填 unknown。只输出一个 JSON 对象，且仅包含 defect_class 字段，不写推理过程或其他文字。')
SYSTEM = '仅根据晶圆图中的可见结构判读，不推测工艺根因。'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def loadl(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8-sig').splitlines() if s.strip()]

def jbytes(obj):
    return (json.dumps(obj,ensure_ascii=False,indent=2)+'\n').encode('utf-8')

def canonical_split(s):
    return 'val' if s in ('val','validation') else s

def emit(path, data, resume):
    if path.exists():
        if resume and path.read_bytes()==data:
            return 'unchanged'
        raise ValueError('拒绝覆盖已有文件：'+path.name)
    path.write_bytes(data)
    return 'created'

def prepare():
    plan=json.loads((SOURCE/'dataset_plan.json').read_text(encoding='utf-8'))
    manifest=loadl(ROOT/'data/manifest.jsonl')
    byid={r['sample_id']:r for r in manifest}
    assert len(byid)==len(manifest), 'manifest重复ID'
    assert sha(ROOT/'data/manifest.jsonl')==plan['manifest_sha256'], 'manifest版本漂移'
    zids={r['sample_id'] for r in loadl(BLIND)}
    zlots={byid[s]['lot_name'] for s in zids}
    bids=set()
    for path in (ROOT/'benchmark').glob('*.jsonl'):
        for row in loadl(path):
            if isinstance(row,dict):
                for k in ('sample_id','query_id','gallery_id'):
                    sid=row.get(k)
                    if isinstance(sid,str) and sid.startswith('wafer_'):
                        # 鲁棒性 ID 带 __rotate/__resize 等后缀，仍属于同一原晶圆。
                        bids.add(sid.split('__',1)[0])
    assert bids <= byid.keys(), 'benchmark ID不在manifest'
    blots={byid[s]['lot_name'] for s in bids}
    testlots={r['lot_name'] for r in manifest if canonical_split(r['split'])=='test'}
    selections={}; summaries={}; outputs={}; pngrows=[]
    for name,n,split in [('train_180',180,'train'),('dev_18',18,'val'),('smoke_20',20,'train')]:
        sourcepath=SOURCE/(name+'.jsonl')
        assert sha(sourcepath)==plan['files'][sourcepath.name], '清单版本漂移'
        selected=loadl(sourcepath)
        assert len(selected)==n
        assert len({r['sample_id'] for r in selected})==n
        rows=[]; imagehashes=[]
        for selected_row in selected:
            sid=selected_row['sample_id']; original=byid[sid]
            assert selected_row['label_source']==original['label_source']=='ground_truth'
            assert selected_row['label']==original['failure_type'] in CLASSES
            assert selected_row['lot_name']==original['lot_name']
            assert canonical_split(selected_row['split'])==canonical_split(original['split'])==split
            assert manifest[selected_row['manifest_line']-1]['sample_id']==sid
            path=(ROOT/'data/images'/(sid+'.png')).resolve()
            assert Path(selected_row['image_path']).resolve()==path
            assert path.exists() and path.read_bytes()[:8]==b'\x89PNG\r\n\x1a\n'
            digest=sha(path); imagehashes.append(digest)
            if name!='smoke_20':
                pngrows.append({'sample_id':sid,'image_sha256':digest,'image_path':str(path),'lot_name':original['lot_name'],'label_source':'ground_truth'})
            rows.append({'sample_id':sid,'messages':[
                {'role':'system','content':SYSTEM},
                {'role':'user','content':'<image>\n'+PROMPT},
                {'role':'assistant','content':json.dumps({'defect_class':selected_row['label']},ensure_ascii=False,separators=(',',':'))}],
                'images':[str(path)]})
        counts=dict(Counter(r['label'] for r in selected))
        if name!='smoke_20':
            assert counts==dict.fromkeys(CLASSES,n//9), '类别计数不符'
        assert len(set(imagehashes))==n, '同一清单图片字节重复'
        lots={r['lot_name'] for r in selected}
        assert not lots & blots, 'benchmark lot泄漏'
        assert not lots & testlots, 'test lot泄漏'
        assert not lots & zlots, 'ZCode lot泄漏'
        selections[name]=selected
        summaries[name]={'rows':n,'unique_ids':n,'unique_lots':len(lots),'unique_image_hashes':len(set(imagehashes)),'per_class':counts,'source_sha256':sha(sourcepath)}
        outputs[name+'.ms_swift.jsonl']=(''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in rows)).encode('utf-8')
    trainids={r['sample_id'] for r in selections['train_180']}
    smokeids={r['sample_id'] for r in selections['smoke_20']}
    assert smokeids<=trainids, '小测试不是主集子集'
    trainlots={r['lot_name'] for r in selections['train_180']}
    devlots={r['lot_name'] for r in selections['dev_18']}
    assert not trainlots & devlots, 'train/dev lot泄漏'
    assert len({r['image_sha256'] for r in pngrows})==198, 'train/dev图片字节重复'
    stats={'seed':3407,'selection_unchanged':True,'source_manifest_sha256':sha(ROOT/'data/manifest.jsonl'),
           'sets':summaries,'unique_train_dev_images':len(pngrows),'smoke_is_train_subset':True,
           'lot_intersections':{'train_dev':len(trainlots&devlots),'train_test':len(trainlots&testlots),'dev_test':len(devlots&testlots),'train_benchmark':len(trainlots&blots),'dev_benchmark':len(devlots&blots),'train_zcode':len(trainlots&zlots),'dev_zcode':len(devlots&zlots)},
           'classification_prompt_sha256':hashlib.sha256(PROMPT.encode()).hexdigest(),
           'source_prompt_missing_candidates':[c for c in CLASSES if c not in plan['question']],
           'prompt_version':'classification_v3_codex_20261002','server_baseline_3_of_18_reusable':False,
           'note':'服务器基座v2题面原件未取得，不能假定与新题面字节相同；新题面需同题基座输出后再比较。'}
    outputs['classification_prompt_v3.txt']=(PROMPT+'\n').encode('utf-8')
    outputs['system_prompt.txt']=(SYSTEM+'\n').encode('utf-8')
    outputs['selected_image_fingerprints.jsonl']=(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in pngrows)).encode('utf-8')
    stats['output_sha256']={n:hashlib.sha256(b).hexdigest() for n,b in outputs.items()}
    outputs['preparation_report.json']=jbytes(stats)
    return outputs,stats

def main():
    ap=argparse.ArgumentParser()
    mode=ap.add_mutually_exclusive_group(); mode.add_argument('--dry-run',action='store_true'); mode.add_argument('--write',action='store_true')
    ap.add_argument('--resume',action='store_true'); ap.add_argument('--seed',type=int,default=3407)
    a=ap.parse_args()
    if a.seed!=3407:
        ap.error('本次不重抽样；固定原选择seed=3407')
    outputs,stats=prepare()
    written=0
    if a.write:
        # 预检查全部输出，避免覆盖已有产物。
        for name,data in outputs.items():
            path=HERE/name
            if path.exists() and not(a.resume and path.read_bytes()==data):
                raise ValueError('拒绝覆盖：'+name)
        for name,data in outputs.items(): written+=emit(HERE/name,data,a.resume)=='created'
    print(json.dumps({'mode':'write' if a.write else 'dry-run','unique_images':stats['unique_train_dev_images'],'rows':{k:v['rows'] for k,v in stats['sets'].items()},'lot_intersections':stats['lot_intersections'],'source_prompt_missing_candidates':stats['source_prompt_missing_candidates'],'files_written':written,'existing_outputs_matched':len(outputs)-written if a.write else 0},ensure_ascii=False))

if __name__=='__main__':main()
