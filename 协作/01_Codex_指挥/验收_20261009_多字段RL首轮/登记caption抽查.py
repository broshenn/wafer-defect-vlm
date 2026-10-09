from pathlib import Path
from collections import Counter
import json

HERE=Path(__file__).resolve().parent
source=json.loads((HERE/'caption12_输入.json').read_text(encoding='utf-8'))['cases']
M={'S':'SUPPORTED','P':'PARTIAL','C':'CONTRADICTED','U':'UNCERTAIN'}
# 四项依次：形态、位置、主要结构覆盖、无依据断言；只评价caption本身，不按公开类别代看图。
DATA=[
('C01',['SPPS','SPPS','SPPP'],'内外均有散点，中部有小块；仅中心限定不充分，6点主方向并不明确。'),
('C02',['PPPS','PPPP','PPPP'],'中央有较密但不规则/断续的弧环红块，均匀连续或覆盖全部径向的限定偏强。'),
('C03',['PPCS','PPCS','PPCS'],'外缘红块成立，但下中部有较明显纵向红色结构，稀疏边缘点的描述遗漏它。'),
('C04',['SSPS','SSSS','SSSS'],'外缘连续红带明显，内部伴散点；后两份包含内部信息，第一份略有遗漏。'),
('C05',['SSSS','SSSS','PPPP'],'主要红区在上半部并连向中部，12点延伸成立；并非连续轴向贯穿全片。'),
('C06',['PPPS','PPPS','PSPS'],'大范围不规则红块与散点存在，随机可部分支持；无清晰单一方向。'),
('C07',['SPPS','SCPP','PCPP'],'主要粗斜红带从左下向右上延伸，约2点/8点轴；10点或12点至6点的caption方向均不符。'),
('C08',['CUCC','CUCC','SSSS'],'图上可见多处红色失效点和黑色空洞；无可见缺陷不符，M2提及少量散点较贴合。'),
('C09',['SSPS','PSPS','PPPS'],'右中部短连通红块和左下散点并存；笼统随机/中心限定均不完整，3点附近有部分依据。'),
('C10',['SSSS','SSSS','SPPS'],'中央大相连红块和外围散点存在；M2只写整个中心区域略过强，主团块本身有依据。'),
('C11',['SPSS','SSSP','SSSS'],'中央环带与绿色小内孔明显，M1均匀限定偏强，M0所有径向限定需收窄。'),
('C12',['SSPS','SSSS','SSSS'],'左下外侧约7点有较密聚集，片内散点较多；后两份同时提及背景散点。'),
]
out=[]
for r,(rid,codes,evidence) in zip(source,DATA):
    assert r['review_id']==rid and len(codes)==3
    for a,code in zip(r['answers'],codes):
        out.append({'review_id':rid,'sample_id':r['sample_id'],'image_sha256':r['image_sha256'],
                    'model':a['model'],'raw_sha256':a['raw_sha256'],'caption_zh':a['caption_zh'],
                    'verdicts':dict(zip(['shape','position','major_structure_coverage','unsupported_assertion'],[M[k] for k in code])),
                    'visual_evidence':evidence,'reviewer':'Codex当前会话AI，继承项目上下文且知道模型名，非严格盲评。',
                    'human_gold':None,'scope':'12图caption抽查，不能当120图全量描述准确率。'})
assert len(out)==36
(HERE/'caption12_AI复核.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in out),encoding='utf-8')
summary={m:{d:dict(Counter(r['verdicts'][d] for r in out if r['model']==m)) for d in out[0]['verdicts']} for m in ['M0','M1','M2']}
(HERE/'caption12_AI汇总.json').write_text(json.dumps({'images':12,'answers':36,'field_counts':summary,'no_combined_accuracy':True},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
lines=['# v2首轮caption抽查（12图，36回答）','',
       '实际看图后独立登记AI意见；不是真人gold，继承项目上下文，非严格盲评。每类随机1图加固定seed3407随机4图，先选图后看原答。不是120图全量准确率。','',
       '|图|模型|形态|位置|结构覆盖|无依据断言|依据|','|---|---|---|---|---|---|---|']
for r in out:lines.append('|'+ '|'.join([r['review_id'],r['model'],*r['verdicts'].values(),r['visual_evidence']])+'|')
lines+=['','观察：C07三份caption都给了不符的线方向；C08的M0/M1误写无可见缺陷，而M2补出了散点。混合表现不支持单凭这12图宣布GRPO描述能力提升或退化。公开类别none可以表示无显著特定缺陷模式，不能直接当成红色像素数为0；程序caption模板也须避免这一混同。']
(HERE/'caption12_AI复核.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('Saved actual 12-image/36-answer AI caption check; no model/API reruns.')
