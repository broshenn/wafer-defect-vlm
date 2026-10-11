"""Codex: persist the completed visual model review, preserving all raw files."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
RUN = ROOT / '协作/03_ZCode_标注/接续49_20261006_20261007-000123'
INPUT = ROOT / '协作/01_Codex_指挥/ZCode接续49_20261006'
rows = [json.loads(x) for x in (INPUT/'blind_remaining49.jsonl').read_text(encoding='utf-8').splitlines()]
manifest = {r['sample_id']: r for x in (ROOT/'data/manifest.jsonl').read_text(encoding='utf-8').splitlines() for r in [json.loads(x)]}
spec = importlib.util.spec_from_file_location('review49_checker',ROOT/'协作/01_Codex_指挥/验收_20261003_GLM180/check_answer_v4.py')
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)

# Corrections were determined by direct original-image inspection in this turn.
edits = {
 'zc_150': {'morphology':'红色失效die以散点及局部相邻小簇分布于晶圆各处，未见明显的中心大簇、完整环带或贯穿性长条。'},
 'zc_155': {'morphology':'红色失效die覆盖晶圆大部分区域，下半部有绿色区域，右缘与上缘也可见绿色散点，内部还可见黑色格块。','caption_zh':'晶圆以红色失效die为主，下半部残留绿色区域，上缘和右缘也有绿色散点。'},
 'zc_164': {'morphology':'中央及中部红色失效die较密集，形成不规则块状及相邻聚集，周围绿色区域中仍散布红色点和小簇。','caption_zh':'晶圆中央及中部红色失效较密集，周围仍有红色散点和小簇。'},
 'zc_165': {'morphology':'红色失效die以散点为主，局部相邻形成小簇，分布于晶圆内部及边缘；未见明显的大面积主聚集区或完整环带。'},
 'zc_166': {'clock_direction':'约5至6点钟方向','morphology':'晶圆中部偏右下有不规则红色聚集，含块状区域和短条状分支，周围还分布红色散点。','caption_zh':'晶圆中部偏右下有不规则红色聚集，其余位置散布红色点。'},
 'zc_167': {'morphology':'晶圆内分布红色散点、短条及不规则小簇，外围红色分布较多，未见单一的贯穿性长条。','caption_zh':'红色散点、短条和小簇分布于晶圆多处，外围也有较多红色die。'},
 'zc_171': {'morphology':'红色失效die以散点和局部相邻小簇分布于晶圆多处，中央附近也有小块聚集，未见明显贯穿性长线或完整环带。'},
 'zc_172': {'morphology':'红色失效die以散点及短小相邻区域分布于晶圆多处，中央附近可见短条状小聚集，未见明显完整环带。'},
 'zc_173': {'morphology':'红色失效die以散点和少量相邻短段分布于晶圆多处，未见明显的大面积主聚集区或完整环带。'},
 'zc_174': {'clock_direction':None,'morphology':'红色失效die散布于晶圆内部及边缘，顶部边缘有局部较密的断续红色区域；全局散点没有唯一稳定主方向。'},
 'zc_176': {'radial_zone':'edge','morphology':'红色失效die在外围呈较密集的断续弧状及块状分布，中央较大片区域为绿色，并有较稀的红色散点。','caption_zh':'晶圆外围红色失效较密，呈断续弧状和小块分布，中央绿色较多并散有红点。'},
 'zc_178': {'morphology':'红色失效die覆盖晶圆大部分，绿色die以散点和相邻小簇穿插在内部及边缘。','caption_zh':'晶圆以红色失效为主，内部和边缘穿插绿色散点与小簇。'},
 'zc_180': {'morphology':'绿色die占多数，红色失效die以稀疏散点及少量相邻短段分布于晶圆内部和边缘，未见明显大面积主聚集区。'},
 'zc_181': {'morphology':'绿色die占多数，红色散点及相邻小簇分布于晶圆内部和边缘，未见明显大面积主聚集区。','caption_zh':'绿色die占多数，红色散点及小簇分布于晶圆内部和边缘。'},
 'zc_182': {'morphology':'绿色die占多数，红色散点分布于晶圆多处，中央附近有相邻红色die形成的小块，未见大片主聚集区。'},
 'zc_183': {'morphology':'绿色die占多数，红色散点和局部相邻小簇分布于晶圆内部及边缘，未见明显大面积主聚集区、贯穿性长线或完整环带。','caption_zh':'绿色die占多数，红色散点和局部小簇分布于晶圆多处。','uncertainty':'稀疏散点在none与Random标签之间的界限不能仅凭本次视觉描述确定；未判断晶圆整体是否合格。'},
 'zc_184': {'morphology':'绿色die占多数，红色散点和局部相邻小簇分布于晶圆内部及边缘，未见明显大面积主聚集区、贯穿性长线或完整环带。','caption_zh':'绿色die占多数，红色散点和局部小簇分布于晶圆多处。','uncertainty':'稀疏散点在none与Random标签之间的界限不能仅凭本次视觉描述确定；未判断晶圆整体是否合格。'},
 'zc_185': {'morphology':'红色失效die以散点和小簇分布于晶圆内部及边缘，未见明显的大面积主聚集区、完整环带或贯穿性长线。'},
 'zc_186': {'morphology':'右侧中部有水平红色条带与右边缘相接，中央另有分离的小红簇，条带与中央小簇之间可见绿色间隔，其余有红色散点。','caption_zh':'右侧中部的水平红色条带接到右缘，中央另有分离的小红簇，其余位置散布红点。'},
 'zc_188': {'morphology':'红色散点和局部小簇分布于晶圆多处；上部有短横向红段，右下区域有分叉状相邻小红簇。'},
 'zc_189': {'morphology':'红色散点分布于晶圆多处，中下部有竖向相邻小簇，底部边缘也可见红点，未见完整环带。'},
 'zc_191': {'morphology':'红色失效die覆盖晶圆大部分，绿色die以散点和相邻小块分布于内部与边缘。'},
 'zc_193': {'radial_zone':'middle','morphology':'红色die在中外部形成不规则环状带，中央绿色较集中，环带外也有红色散点和小簇。','caption_zh':'晶圆中外部有不规则红色环带，中央绿色较多，外围另有红色散点。'},
 'zc_194': {'morphology':'红色失效die以散点和相邻小簇分布于晶圆多处，局部呈短横段或竖段，未见明显完整环带或贯穿性长线。'},
 'zc_197': {'morphology':'顶部和右侧边缘有局部红色条带，右上区域还有红色小簇，整体呈断续的弧状分布，其余位置有红色散点。','caption_zh':'顶部和右侧边缘有局部红色条带，沿右上区域断续排列，其余位置散布红点。'},
}
review = []
candidates = []
original_pass = 0
class_agree = 0
for row in rows:
    raw = next(RUN.glob('worker_*/'+row['item_id']+'_raw.json'))
    answer = json.loads(raw.read_text(encoding='utf-8-sig'))
    original_result = checker.check_text(raw.read_text(encoding='utf-8-sig'))
    original_pass += original_result['exit_code'] == 0
    candidate = copy.deepcopy(answer)
    replacements = edits.get(row['item_id'],{})
    candidate.update(replacements)
    # Green majority cannot establish whole-wafer acceptance.
    cap = candidate['caption_zh']
    for phrase in ['晶圆整体大面积合格，','晶圆整体合格，']:
        cap = cap.replace(phrase,'绿色合格die占多数，')
    candidate['caption_zh'] = cap
    gold = manifest[row['sample_id']]['failure_type']
    class_agree += answer['defect_class'] == gold
    teacher_class = candidate['defect_class']
    candidate['defect_class'] = gold
    assert checker.check_object(candidate)['exit_code'] == 0
    changed = {k:{'raw':answer[k],'candidate':candidate[k]} for k in answer if answer[k]!=candidate[k]}
    review.append({**row,'raw_path':str(raw.relative_to(ROOT)),
        'raw_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),
        'original_format_exit_code':original_result['exit_code'],
        'review_kind':'nonblind_model_visual_review','reviewer':'Codex main conversation',
        'human_review':'not_done','direct_original_image_inspected':True,
        'teacher_class':answer['defect_class'],'public_class_label':gold,
        'class_source':'data/manifest.jsonl failure_type, not teacher or reviewer prediction',
        'changes':changed,'status':'model_assisted_candidate',
        'geometry_gold':'not_confirmed','size_gold':'not_measured'})
    candidates.append({**row,'answer':candidate,'teacher_class':answer['defect_class'],
        'label_source':'public_dataset_original_class','description_source':'GLM_raw_with_Codex_model_visual_review',
        'gold_source':'class_only_public_label; descriptive_fields_are_model_candidates',
        'human_annotator_count':0})

def write_new(name,obj,jsonl=False):
    path=OUT/name
    if path.exists():raise FileExistsError(path)
    text=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in obj) if jsonl else json.dumps(obj,ensure_ascii=False,indent=2)+'\n'
    path.write_text(text,encoding='utf-8',newline='\n')

write_new('逐图模型复核49.jsonl',review,True)
write_new('模型辅助候选49.jsonl',candidates,True)
summary={'date':'2026-10-07','raw_count':49,'raw_format_pass':original_pass,'raw_format_failed':49-original_pass,
    'candidate_count':len(candidates),'candidate_format_pass':49,'visual_reviewed_images':49,
    'teacher_class_public_label_agreement':class_agree,'teacher_class_agreement_is_not_human_visual_accuracy':True,
    'changed_candidate_count':sum(bool(r['changes']) for r in review),
    'human_review_count':0,'size_gold_not_measured':True,
    'duplicate_183_184':'Same broad pattern supports similar wording; does not prove independent execution. Remove wafer-level acceptance claim and unverified counts.',
    'original_raw_150':'Retained isolated format failure; candidate dequantified after model review, not original pass.'}
write_new('验收统计.json',summary)
print(json.dumps(summary,ensure_ascii=False))
