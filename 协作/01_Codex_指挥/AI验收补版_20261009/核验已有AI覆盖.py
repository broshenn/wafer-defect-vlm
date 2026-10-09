from pathlib import Path
import hashlib
import importlib.util
import json

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
RUN=ROOT/'协作/02_ClaudeCode_实操/百炼多模型业务对照_20261009_1203'
PACK=ROOT/'协作/02_ClaudeCode_实操/自动接续交付_20261009_1247'
spec=importlib.util.spec_from_file_location('grid_reader',PACK/'阶段2_网格逐格核.py')
reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
key=json.loads((RUN/'盲号对照_key.json').read_text(encoding='utf-8'))
mapping={r['sample_id']:r['盲号对照'] for r in key}
grid={};sources=[]
for p in sorted((RUN/'复核结果').glob('分片*_复核.json')):
    for sid,blind,answer in reader.flat_answers(json.loads(p.read_text(encoding='utf-8'))):
        model=mapping[sid][blind]
        for dim in reader.DIMS:
            k=(sid,model,dim)
            val=reader.norm_val(dim,answer.get(dim))
            if k in grid:assert grid[k]==val
            grid[k]=val
    sources.append({'path':p.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
published=json.loads((RUN/'描述对照_枚举.json').read_text(encoding='utf-8'))
other={(r['sample_id'],r['模型'],r['维度']):r['判定'] for r in published['逐条']}
assert grid==other
ids={k[0] for k in grid};models={k[1] for k in grid}
assert len(ids)==36 and len(models)==8 and len(grid)==1728
new=[json.loads(l) for l in (HERE/'Codex_D36_逐图复核.jsonl').read_text(encoding='utf-8').splitlines()]
assert len(new)==len({r['sample_id'] for r in new})==36
assert {r['sample_id'] for r in new}==ids
dimensions={'morphology':'形态','position':'位置','main_structure_coverage':'遗漏','unsupported_assertions':'断言'}
disagreements=[]
for row in new:
    for fresh,prior in dimensions.items():
        a=grid[(row['sample_id'],'D-N3072-3407',prior)];b=row['verdicts'][fresh]
        if a!=b:disagreements.append({'review_id':row['review_id'],'sample_id':row['sample_id'],'dimension':prior,'prior':a,'Codex':b})
result={'prior_review_models':sorted(models),'prior_review_images':len(ids),'prior_review_cells':len(grid),'missing_prior_cells':0,
        'original_shards_match_published_cells':True,'source_files':sources,
        'new_Codex_D_reviewed_images':len(new),'new_visual_cells':144,'visual_cell_disagreements':len(disagreements),
        'disagreement_details':disagreements,'scope':'分歧不代表某个AI一定正确；保留两轮原判断，不能平均成真人gold。格式严重度口径不同，未混入视觉四维比较。'}
(HERE/'AI覆盖与分歧核验.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ['source_files','disagreement_details']},ensure_ascii=False,indent=2))
