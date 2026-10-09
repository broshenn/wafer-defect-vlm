from pathlib import Path
from collections import Counter
import hashlib
import importlib.util
import json
import random
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
RUN=ROOT/'协作/02_ClaudeCode_实操/56图候选补测与最终结项_20261009'
def rows(p):return [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('locked56',RUN/'schema_check.py')
sc=importlib.util.module_from_spec(spec);spec.loader.exec_module(sc)
assert sha(RUN/'schema_check.py')=='7d46c37f72bb8fd09488dd73754ad75e6a1b711d91a6f08f82e5b34026c0c0ff'
assert sha(RUN/'prompt_7f.txt')=='8a8315f785a2398ba5a823754c40b8a8658f89b79aad1a0dbd68058acb43cfda'
manifest={r['sample_id']:r for r in rows(ROOT/'data/manifest.jsonl')}
frozen=json.loads((RUN/'冻结清单.json').read_text(encoding='utf-8'))['逐图']
original=rows(ROOT/'协作/01_Codex_指挥/AI验收补版_20261009/候选考卷_盲清单.jsonl')
assert len(frozen)==len(original)==56
assert [(r['item_id'],r['sample_id'],r['image_sha256']) for r in frozen]==[(r['item_id'],r['sample_id'],r['image_sha256']) for r in original]
classes=['Center','Donut','Edge_Loc','Edge_Ring','Loc','Random','Scratch']
def macro(pairs):
    fs=[]
    for c in classes:
        tp=sum(g==c and p==c for g,p in pairs);fp=sum(g!=c and p==c for g,p in pairs);fn=sum(g==c and p!=c for g,p in pairs)
        fs.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
    return sum(fs)/7
expected={'Base':None,'L-N3072-3407':'40144b18d143e5991c5f49972183e52d7375244f4ca1965e8c15fa9437973368',
          'D-N3072-3407':'babf37ecced78d37f56cc6cdea4d4dd839bce9b8324ef8caf4c7bb7b45fb018b'}
result={};correct={};class_correct={};coupling=[]
for tag in expected:
    data=rows(RUN/'out'/f'{tag}_raw.jsonl');byid={r['sample_id']:r for r in data}
    assert len(data)==len(byid)==56 and set(byid)=={r['sample_id'] for r in frozen}
    gated=[];independent=[];fmt=Counter()
    for fr in frozen:
        sid=fr['sample_id'];r=byid[sid];image=ROOT/'data/images'/f'{sid}.png'
        assert sha(image)==fr['image_sha256']==r['image_sha256']
        assert r['prompt_sha256']==sha(RUN/'prompt_7f.txt')
        assert r['checker_sha256']==sha(RUN/'schema_check.py')
        assert r['adapter_sha256']==expected[tag]
        truth=manifest[sid]['failure_type'];assert truth==fr['failure_type']
        checked=sc.check_answer(r['raw']);obj,status=sc.parse_strict(r['raw'])
        pred=obj.get('defect_class') if status=='ok' and isinstance(obj,dict) else None
        if pred not in sc.CLASSES:pred=None
        guarded=pred if checked['schema_ok'] else None
        gated.append((truth,guarded));independent.append((truth,pred))
        fmt['strict_json']+=checked['strict_json'];fmt['schema_ok']+=checked['schema_ok'];fmt['with_contract_warning']+=bool(checked['contract_warnings'])
        if guarded!=pred:coupling.append({'tag':tag,'item_id':fr['item_id'],'public_label':truth,'parsed_class':pred,'schema_problems':checked['schema_problems']})
    correct[tag]=[int(g==p) for g,p in gated];class_correct[tag]=[int(g==p) for g,p in independent]
    result[tag]={'records':len(data),'format':dict(fmt),
                 'legacy_full_schema_gated':{'correct':sum(correct[tag]),'accuracy':sum(correct[tag])/56,'macro_f1_seven':macro(gated)},
                 'class_field_with_strict_json':{'correct':sum(class_correct[tag]),'accuracy':sum(class_correct[tag])/56,'macro_f1_seven':macro(independent)}}
def paired(v):
    rnd=random.Random(3407);n=10000
    boot=sorted(sum(v[rnd.randrange(56)] for _ in range(56))/56 for _ in range(n))
    return {'mean':sum(v)/56,'ci95':[boot[int(.025*n)],boot[int(.975*n)-1]],'n':56,'bootstrap':n}
pairs={}
for a,b in [('L-N3072-3407','D-N3072-3407'),('L-N3072-3407','Base'),('D-N3072-3407','Base')]:
    pairs[a+' minus '+b]={'legacy_full_schema_gated':paired([x-y for x,y in zip(correct[a],correct[b])]),
                         'class_field_with_strict_json':paired([x-y for x,y in zip(class_correct[a],class_correct[b])])}
diagnostics={}
for item in ['H003','H049']:
    fr=next(r for r in frozen if r['item_id']==item)
    rgb=np.asarray(Image.open(ROOT/'data/images'/f"{fr['sample_id']}.png").convert('RGB'))
    red=(rgb==[255,0,0]).all(axis=2);green=(rgb==[0,255,0]).all(axis=2);valid=red|green
    y,x=np.where(valid);cy=(y.min()+y.max())/2;cx=(x.min()+x.max())/2;radius=max(y.max()-y.min(),x.max()-x.min())/2
    yr,xr=np.where(red);r=np.hypot(yr-cy,xr-cx)/radius
    diagnostics[item]={'sample_id':fr['sample_id'],'public_label':fr['failure_type'],'render_red_ratio':float(red.sum()/valid.sum()),
                       'all_raw_red_bbox_circle_radial_span':float(r.max()-r.min()),
                       'scope':'PNG像素计数；bbox近似圆归一化仅作反例诊断，不是校准尺寸gold。缺陷点少不决定径向跨度。'}
ai=rows(RUN/'AI描述验收.jsonl')
report={'model_results':result,'paired':pairs,'schema_class_coupling_examples':coupling,
        'two_image_diagnostics':diagnostics,'ai_verdict_rows':len(ai),
        'complete_students_raw':True,'same_blind56':True,
        'resource_source':'运行合同与资源账.json：报告9.02设备分钟；租用实例用户自行提供，总累计与实际账单未独立核清。',
        'scope':'独立CPU复算；历史主成绩保留，另列类别字段口径；未运行新模型、API或训练。'}
(HERE/'独立核验.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
