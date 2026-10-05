"""Codex：导出原标签训练集与模型审核描述候选；不训练、不联网。"""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import zipfile
import check_answer_v4 as CHECK

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
OUT=ROOT/'协作/01_Codex_指挥/训练数据就绪_20261003'
MATCHED=ROOT/'协作/01_Codex_指挥/本地准备_20261002_断连期间'
CAPTION_PROMPT='请仅根据晶圆图描述主要可见形态、位置及外围散点。黑色为背景，绿色为合格die，红色为失效die。不推测工艺根因，不给未经验证的数量、尺寸或覆盖比例。'

def sha(data):return hashlib.sha256(data).hexdigest()
def loadl(path):return [json.loads(s) for s in path.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def lb(rows):return ''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in rows).encode('utf-8')
def emit(name,data):
    p=OUT/name
    if p.exists() and p.read_bytes()!=data:raise ValueError('拒绝覆盖：'+name)
    if not p.exists():p.write_bytes(data)

def main():
    OUT.mkdir(exist_ok=True)
    records=json.loads((HERE/'180条原答对照.json').read_text(encoding='utf-8'))
    reviews=json.loads((HERE/'review_all_161.json').read_text(encoding='utf-8'))
    first_full=json.loads((ROOT/'协作/01_Codex_指挥/验收_20261002_GLM首批20/接收候选_10条.json').read_text(encoding='utf-8'))
    first_neutral=json.loads((ROOT/'协作/01_Codex_指挥/验收_20261002_GLM首批20/中性描述候选_2条.json').read_text(encoding='utf-8'))
    ff={r['sample_id']:r for r in first_full};fn={r['sample_id']:r for r in first_neutral}
    full=[];captions=[];quarantine=[];provenance=[]
    for r in records:
        sid=r['sample_id'];base={'sample_id':sid,'image_path':r['image_path'],'image_sha256':r['image_sha256'],
            'original_label':r['original_label'],'label_source':'ground_truth','teacher_prediction':r['canonical_answer']['defect_class'],
            'raw_path':r['source_raw'],'raw_sha256':r['raw_sha256'],'review_kind':'Codex非盲模型复核','human_review':'not run'}
        caption=None;answer=None;change=[]
        if not r['is_continuation']:
            if sid in ff:
                answer=ff[sid]['answer'];caption=answer['caption_zh'];decision='full_candidate'
                change=ff[sid]['model_revision']
            elif sid in fn:caption=fn[sid]['caption_zh'];decision='caption_only';change=[{'source':'前轮Codex中性描述'}]
            else:decision='quarantine_full'
            note='沿用上一轮模型复核决定'
        else:
            review=reviews[r['item_id']];decision=review['decision'];note=review['note']
            if decision.startswith('full_'):
                answer=copy.deepcopy(r['canonical_answer'])
                if decision=='full_with_model_revision':
                    for field in ['morphology','caption_zh']:
                        change.append({'field':field,'before':answer[field],'after':review['scene'],'source':'Codex原图复核摘要'})
                        answer[field]=review['scene']
                    if r['item_id'] in ['item_039','item_054','item_056','item_103']:
                        change.append({'field':'clock_direction','before':answer['clock_direction'],'after':None,'reason':'不强定唯一精确方向'})
                        answer['clock_direction']=None
                    if r['item_id']=='item_056':answer['radial_zone']='unknown'
                    if r['item_id']=='item_165':answer['clock_direction']='约1至2点钟'
                assert answer['defect_class']==r['original_label']
                assert CHECK.check_object(answer)['exit_code']==0
                caption=answer['caption_zh']
            elif decision.startswith('caption_only'):
                caption=review['scene'];change=[{'field':'caption_zh','before':r['canonical_answer']['caption_zh'],
                    'after':caption,'source':'Codex原图复核中性摘要；不接收教师类别'}]
        if answer is not None:full.append({**base,'answer':answer,'model_revision':change,'status':'draft_model_assisted_candidate'})
        if caption is not None:
            captions.append({'sample_id':sid,'messages':[{'role':'user','content':'<image>\n'+CAPTION_PROMPT},
                {'role':'assistant','content':caption}],'images':[r['image_path']]})
        else:quarantine.append({**base,'reason':note})
        provenance.append({**base,'decision':decision,'review_note':note,'caption_included':caption is not None,
            'structured_candidate_included':answer is not None,'model_revision':change,
            'caption_source':'GLM草稿经Codex模型复核/必要重写' if caption is not None else 'not accepted'})
    assert len(full)==90 and len(captions)==158 and len(quarantine)==22 and len(provenance)==180
    match_report=json.loads((MATCHED/'matched_preparation_report.json').read_text(encoding='utf-8'))
    inputs={'sft_a_train_180.jsonl':'train_180.matched.ms_swift.jsonl','sft_a_dev_18.jsonl':'dev_18.matched.ms_swift.jsonl',
            'sft_a_smoke_20.jsonl':'smoke_20.matched.ms_swift.jsonl'}
    outputs={}
    for dst,src in inputs.items():
        data=(MATCHED/src).read_bytes();assert sha(data)==match_report['output_sha256'][src];outputs[dst]=data
    train=loadl(MATCHED/inputs['sft_a_train_180.jsonl'])
    assert {r['sample_id'] for r in train}=={r['sample_id'] for r in records}
    for row in train:
        ref=next(r for r in records if r['sample_id']==row['sample_id'])
        assert json.loads(row['messages'][-1]['content'])['defect_class']==ref['original_label']
    outputs['caption_train_158.jsonl']=lb(captions)
    outputs['multitask_candidate_338.jsonl']=lb(train+captions)
    outputs['structured_candidates_90.jsonl']=lb(full)
    outputs['provenance_180.jsonl']=lb(provenance)
    outputs['caption_quarantine_22.jsonl']=lb(quarantine)
    outputs['caption_prompt_zh_v1.txt']=(CAPTION_PROMPT+'\n').encode('utf-8')
    summary={'author':'Codex','date':'2026-10-03','classification_train':180,'classification_dev':18,'classification_smoke':20,
        'caption_candidates':158,'structured_candidates':90,'caption_quarantined':22,'multitask_rows':338,
        'unique_train_images':180,'new_human_annotator_count':0,'teacher_prediction_used_as_class_gold':False,
        'review_scope':'180张全部逐图模型复核；原19复用上轮结论，续161原图逐页复核并逐图记依据',
        'caption_source':'GLM+Codex模型审核/重写，不称纯GLM或人工金标',
        'training_status':'not run','primary_authorized_training':'原20步/60步标签SFT-A',
        'other_datasets':'描述/多任务/结构化仅就绪候选，不自动新增训练预算',
        'remote_paths':'当前images为本地路径；到服务器后需仅重映射并校验图片hash',
        'per_class_train':dict(Counter(r['original_label'] for r in records)),
        'files_sha256':{name:sha(data) for name,data in outputs.items()}}
    outputs['data_summary.json']=(json.dumps(summary,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
    for name,data in outputs.items():emit(name,data)
    # 图像字节仅按原样加入传输包；不修改原图，不加入私有配置、凭据或权重。
    zip_path=OUT/'训练数据包_20261003.zip'
    if not zip_path.exists():
        with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for name in outputs:z.write(OUT/name,'datasets/'+name)
            image_paths={p for row in train+loadl(MATCHED/inputs['sft_a_dev_18.jsonl']) for p in row['images']}
            assert len(image_paths)==198
            for image in sorted(image_paths):z.write(image,'images/'+Path(image).name)
    with zipfile.ZipFile(zip_path) as z:
        assert z.testzip() is None
        for name,data in outputs.items():assert z.read('datasets/'+name)==data
    summary['zip_sha256']=sha(zip_path.read_bytes());summary['zip_bytes']=zip_path.stat().st_size
    emit('交付指纹.json',(json.dumps(summary,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    print(json.dumps({k:v for k,v in summary.items() if k!='files_sha256'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
