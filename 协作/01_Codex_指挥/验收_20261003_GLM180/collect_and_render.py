"""Codex：核验180原答并排版续标161原图供逐张模型复核；原图不变。"""
from collections import Counter,defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
import check_answer_v4 as V4

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
PREP=ROOT/'协作/02_ClaudeCode_实操/GLM训练标注准备_20261002'
FIRST=ROOT/'协作/03_ZCode_标注/20261002-192902_batch01_并行20'
REST=ROOT/'协作/03_ZCode_标注/20261002-201800_续标161'
NORMALIZER=ROOT/'协作/01_Codex_指挥/验收_20261002_GLM首批20/normalize_label_alias.py'

def mod(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def loadl(path):return [json.loads(s) for s in path.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def save(name,value):
    data=(json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf-8');path=HERE/name
    if path.exists() and path.read_bytes()!=data:raise ValueError('拒绝覆盖：'+name)
    if not path.exists():path.write_bytes(data)

def main():
    normal=mod('normal',NORMALIZER)
    qty=mod('qty',ROOT/'协作/02_ClaudeCode_实操/离线修正_20261002-202300/cn_quantity_audit.py')
    master=loadl(PREP/'blind_inputs_all_180.jsonl');gold={r['sample_id']:r for r in loadl(PREP/'audit_selected.jsonl')}
    manifest={r['sample_id']:r for r in loadl(ROOT/'data/manifest.jsonl')}
    indexpath=ROOT/'协作/01_Codex_指挥/GLM续标_20261002_无答案输入/inputs.json'
    assert sha(indexpath)=='60762f2ee9c1018cc83e5d1ebb167832e0534e2eb57c5cc3b2ac482da59f55e0'
    index=json.loads(indexpath.read_text(encoding='utf-8'))
    for entry in index['groups']:
        assert sha(Path(entry['input_path']))==entry['input_sha256']
    for key in ['prompt','checker','normalizer']:
        e=index[key];assert sha(Path(e['path']))==e['sha256']
    old=mod('old_checker',Path(index['checker']['path']))
    files={};source_counts=Counter()
    for source in [FIRST,REST]:
        for p in source.glob('**/*_raw.json'):
            sid=p.name.removesuffix('_raw.json');assert sid not in files,sid
            files[sid]=p;source_counts[source.name]+=1
    assert len(files)==180 and set(files)=={r['sample_id'] for r in master}
    records=[];duplicate=defaultdict(list);counts=Counter();numeric=[]
    for b in master:
        sid=b['sample_id'];p=files[sid];g=gold[sid];m=manifest[sid]
        assert g['label']==m['failure_type'] and g['label_source']==m['label_source']=='ground_truth'
        assert g['lot_name']==m['lot_name'] and g['split']==m['split']=='train'
        assert sha(Path(b['image_path']))==b['image_sha256']==g['image_sha256']
        raw=p.read_bytes();answer=json.loads(raw.decode('utf-8-sig'),object_pairs_hook=normal.unique_object)
        canonical,changes=normal.normalize(raw)
        old_result=old.check_text(raw.decode('utf-8-sig'));new_raw=V4.check_object(answer);new_canonical=V4.check_object(canonical)
        counts['old_raw_'+str(old_result['exit_code'])]+=1;counts['v4_raw_'+str(new_raw['exit_code'])]+=1
        counts['v4_canonical_'+str(new_canonical['exit_code'])]+=1
        duplicate[hashlib.sha256(raw).hexdigest()].append(sid)
        is_rest=REST in p.parents
        if is_rest:
            cp=p.with_name(p.name.replace('_raw.json','_canonical.json'))
            np=p.with_name(p.name.replace('_raw.json','_normalization.json'))
            nd=json.loads(np.read_text(encoding='utf-8'),object_pairs_hook=normal.unique_object)
            assert json.loads(cp.read_text(encoding='utf-8'),object_pairs_hook=normal.unique_object)==canonical
            assert nd['raw_sha256']==sha(p) and nd['normalized_sha256']==sha(cp) and nd['changes']==changes==[]
        reminder=qty.audit_file(p)
        if reminder.get('**unverified_count_claims**'):numeric.append({'sample_id':sid,'flags':reminder['**unverified_count_claims**']})
        records.append({**b,'original_label':g['label'],'label_source':g['label_source'],'lot_name':g['lot_name'],
            'source_raw':p.relative_to(ROOT).as_posix(),'raw_sha256':sha(p),'is_continuation':is_rest,
            'answer':answer,'canonical_answer':canonical,'alias_changes':changes,
            'old_check':old_result,'v4_raw_check':new_raw,'v4_canonical_check':new_canonical,
            'class_matches_raw':answer['defect_class']==g['label'],'class_matches_canonical':canonical['defect_class']==g['label'],
            'quantity_reminder':reminder.get('**unverified_count_claims**',{}),'content_review':'pending_nonblind_model_review'})
    rest=[r for r in records if r['is_continuation']]
    assert len(rest)==161
    fixed=[r for r in rest if r['old_check']['exit_code']!=0 and r['v4_raw_check']['exit_code']==0]
    assert len(fixed)==8 and all(r['old_check']['exit_code']==8 for r in fixed)
    per_class={}
    for c in ['Center','Donut','Edge_Loc','Edge_Ring','Loc','Near_full','Random','Scratch','none']:
        rows=[r for r in records if r['original_label']==c]
        assert len(rows)==20
        per_class[c]={'n':len(rows),'label_matches':sum(r['class_matches_canonical'] for r in rows)}
    report={'author':'Codex','date':'2026-10-03','source_counts':dict(source_counts),'all_ids':len(records),
        'format_counts':dict(counts),'recovered_clock_false_failures':[r['sample_id'] for r in fixed],
        'class_matches_raw':sum(r['class_matches_raw'] for r in records),
        'class_matches_canonical':sum(r['class_matches_canonical'] for r in records),
        'per_class':per_class,'predictions_continuation':dict(Counter(r['canonical_answer']['defect_class'] for r in rest)),
        'identical_raw_pairs':{h:s for h,s in duplicate.items() if len(s)>1},'quantity_flags':numeric,
        'old_prompt_checker_input_unchanged':True,'teacher_api_requests_added':0,'gpu_added':0,
        'note':'格式和原类别一致率不是人工内容准确率；内容接收仍须逐图模型复核。'}
    save('机械核验.json',report);save('180条原答对照.json',records)
    out=HERE/'原图排版';out.mkdir(exist_ok=True)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
    for start in range(0,len(rest),8):
        subset=rest[start:start+8];canvas=Image.new('RGB',(936,2096),'white');draw=ImageDraw.Draw(canvas)
        for j,r in enumerate(subset):
            col=j%2;row=j//2;x=12+col*468;y=10+row*524
            draw.text((x,y),r['item_id']+' '+r['sample_id'],font=font,fill='black')
            draw.text((x,y+24),'orig='+r['original_label']+' pred='+r['canonical_answer']['defect_class'],font=font,fill='black')
            with Image.open(r['image_path']) as im:
                assert im.size==(448,448);canvas.paste(im.convert('RGB'),(x,y+50))
        name=f'page_{start//8+1:02d}.png';path=out/name
        if not path.exists():canvas.save(path)
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
