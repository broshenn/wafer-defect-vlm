"""Codex：首批19原答/日志只读复算，另存模型复核后的训练候选。"""
from collections import Counter
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
from PIL import Image
from normalize_label_alias import normalize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT/'协作/03_ZCode_标注/20261002-192902_batch01_并行20'
PREP = ROOT/'协作/02_ClaudeCode_实操/GLM训练标注准备_20261002'
CHECKER = ROOT/'协作/02_ClaudeCode_实操/口径修正与准备_20261002-021207/check_answer_v3.py'
FULL_ACCEPT = {'item_001','item_002','item_003','item_004','item_005','item_006','item_007','item_009','item_010','item_019'}
REASONS = {
 'item_008':'原Donut/模型Loc；局部聚集有依据，环形/块状判据不确定，完整答案隔离',
 'item_011':'原Scratch/模型Random；左下方短斜带被否定为无条状结构，主要结构遗漏',
 'item_012':'原Loc/模型Random；右下主簇可见，模型描述到了但主次取舍不符',
 'item_013':'原none/模型Random；散点可见，完整类别不接收，另存中性描述候选',
 'item_014':'原Scratch/模型Edge_Loc；沿左下边缘斜带有依据，类别边界未定',
 'item_015':'原Random/模型Near_full；有大面积绿色，几乎全红/少量绿的描述过强',
 'item_016':'原Edge_Ring/模型Random；外围红点增强和弧带被否定，主要结构遗漏',
 'item_017':'原none/模型Random；散点可见，完整类别不接收，另存中性描述候选',
 'item_018':'原Edge_Loc/模型Random；顶部边缘聚集已提及但主次取舍不符'}


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def loadl(path): return [json.loads(s) for s in path.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def emit(name, value):
    data = (json.dumps(value, ensure_ascii=False, indent=2)+'\n').encode('utf-8')
    path = HERE/name
    if path.exists() and path.read_bytes()!=data: raise ValueError('拒绝覆盖：'+name)
    if not path.exists(): path.write_bytes(data)


def main():
    assert sha(PREP/'zcode_batch_01_inputs.json')=='641f75212839e5c847edf0d7bb53225460b03f99f4e16c10a1343f47cd97ffe7'
    assert sha(CHECKER)=='794a254e005eaa232362c539817abde98a1c2406ecb52574b9e9edb35303ec55'
    index=json.loads((PREP/'zcode_batch_01_inputs.json').read_text(encoding='utf-8'))
    for name in ['batch_manifest','prompt','checker']:
        entry=index[name];assert sha(Path(entry['path']))==entry['sha256']
    spec=importlib.util.spec_from_file_location('checker',CHECKER);check=importlib.util.module_from_spec(spec);spec.loader.exec_module(check)
    gold={r['sample_id']:r for r in loadl(PREP/'audit_selected.jsonl')}
    records=[];accepted=[];neutral=[];raw_exits=Counter();norm_exits=Counter();seen=set();not_run=[]
    for worker in index['workers']:
        wp=SOURCE/worker['worker_id'];shard=loadl(Path(worker['input_path']))
        assert sha(Path(worker['input_path']))==worker['input_sha256']
        logs=loadl(wp/'worker_log.jsonl');log_by_id={r['sample_id']:r for r in logs}
        assert len(log_by_id)==len(logs)
        files=list(wp.glob('*_raw.json'))
        assert {f.name.removesuffix('_raw.json') for f in files}==set(log_by_id)
        for item in shard:
            sid=item['sample_id'];image=Path(item['image_path']);assert sha(image)==item['image_sha256']
            rawfile=wp/(sid+'_raw.json')
            if not rawfile.exists():
                assert sid not in log_by_id;not_run.append({'item_id':item['item_id'],'sample_id':sid});continue
            assert sid not in seen;seen.add(sid)
            raw=rawfile.read_bytes();raw_text=raw.decode('utf-8-sig')
            original=json.loads(raw_text);answer,changes=normalize(raw)
            raw_check=check.check_text(raw_text);raw_exits[raw_check['exit_code']]+=1
            canon_check=check.check_object(answer);norm_exits[canon_check['exit_code']]+=1
            log=log_by_id[sid];assert log['image_sha256']==item['image_sha256'] and log['check_exit_code']==raw_check['exit_code']
            if 't0_ms' in log: assert log['duration_ms']==log['t1_ms']-log['t0_ms']
            gt=gold[sid]['label'];itemid=item['item_id'];decision='model_review_candidate_accept' if itemid in FULL_ACCEPT else 'quarantine_full_answer'
            record={'item_id':itemid,'sample_id':sid,'original_label':gt,'label_source':gold[sid]['label_source'],
                'raw_prediction':original['defect_class'],'canonical_prediction':answer['defect_class'],
                'raw_label_match':original['defect_class']==gt,'canonical_label_match':answer['defect_class']==gt,
                'raw_check_exit':raw_check['exit_code'],'canonical_check_exit':canon_check['exit_code'],
                'raw_sha256':hashlib.sha256(raw).hexdigest(),'image_sha256':item['image_sha256'],
                'raw_path':rawfile.relative_to(ROOT).as_posix(),'normalization_changes':changes,'decision':decision,
                'reason':REASONS.get(itemid,'主要形态与位置有图像依据；尺寸未测；非盲模型复核通过'),
                'human_review':'not run','review_kind':'Codex非盲模型复核'}
            if itemid in FULL_ACCEPT:
                assert answer['defect_class']==gt and canon_check['exit_code']==0
                revision=[]
                if itemid=='item_019':
                    before=answer['morphology'];answer['morphology']=before.replace('零星十几颗绿色合格die','少量绿色合格die')
                    assert answer['morphology']!=before
                    revision.append({'field':'morphology','before':before,'after':answer['morphology'],'reason':'删除未经验证的中文数量；保留定性绿色散点'})
                assert check.check_object(answer)['exit_code']==0
                accepted.append({**{k:record[k] for k in ['item_id','sample_id','original_label','label_source','raw_path','raw_sha256','image_sha256','human_review','review_kind']},
                                 'status':'draft_accepted_for_model_assisted_training','teacher_prediction':record['canonical_prediction'],
                                 'answer':answer,'normalization_changes':changes,'model_revision':revision})
                record['model_revision']=revision
            if itemid in {'item_013','item_017'}:
                caption='绿色晶圆区域中散布红色失效点和少量相邻小块，未见明显的连续环带、主导团簇或长条状结构。'
                neutral.append({'item_id':itemid,'sample_id':sid,'source_raw':record['raw_path'],'source_raw_sha256':record['raw_sha256'],
                                'caption_zh':caption,'source':'GLM原答与图像经Codex模型复核改写','status':'caption_only_candidate',
                                'class_supervision_source':'原类别独立保留；不接收GLM完整类别答案','human_review':'not run'})
            records.append(record)
    assert len(records)==19 and len(not_run)==1 and len(accepted)==10
    duplicate_keys=[]
    def hook(pairs):
        d={}
        for k,v in pairs:
            if k in d:duplicate_keys.append(k)
            d[k]=v
        return d
    parent=json.loads((SOURCE/'batch_log.json').read_text(encoding='utf-8'),object_pairs_hook=hook)
    image_stats={}
    for sid in ['wafer_00018231_019','wafer_00043570_016','wafer_00046030_024']:
        colors=Counter(Image.open(ROOT/'data/images'/(sid+'.png')).convert('RGB').getdata())
        red=colors[(255,0,0)];green=colors[(0,255,0)]
        image_stats[sid]={'red_pixels':red,'green_pixels':green,'red_share':red/(red+green),'kind':'rendered_pixel_ratio_not_matrix_die_ratio'}
    report={'author':'Codex','date':'2026-10-02','planned':20,'attempted':19,'raw_format_pass':raw_exits[0],
        'raw_format_fail':19-raw_exits[0],'raw_check_exit_counts':dict(raw_exits),'canonical_check_exit_counts':dict(norm_exits),
        'raw_class_matches_all_attempts':sum(r['raw_label_match'] for r in records),
        'canonical_class_matches_all_attempts':sum(r['canonical_label_match'] for r in records),
        'full_answer_candidates_accepted':len(accepted),'full_answer_quarantined':len(records)-len(accepted),
        'neutral_caption_only_candidates':len(neutral),'not_run':not_run,'duplicate_keys_in_parent_log':duplicate_keys,
        'parent_fresh_session_requirement_met':False,'child_fresh_context':'client_tool_contract_declared_not_independently_verified',
        'reported_subagent_tokens_unknown_unit':sum(v['client_reported_subagent_tokens'] for v in parent['per_worker'].values()),
        'rendered_color_checks':image_stats,'input_index_sha256':sha(PREP/'zcode_batch_01_inputs.json'),
        'normalizer_sha256':sha(HERE/'normalize_label_alias.py'),'teacher_api_requests_added':0,'gpu_training_added':0,
        'actual_annotation_cost':'not provided; current Codex review use not separately provided',
        'note':'类别一致不是人工视觉准确率；原始格式失败保留，规范化仅改变明确别名；接收为模型辅助草稿，不解除真人审核。'}
    emit('独立核验.json',report);emit('逐图模型复核.json',records);emit('接收候选_10条.json',accepted);emit('中性描述候选_2条.json',neutral)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
