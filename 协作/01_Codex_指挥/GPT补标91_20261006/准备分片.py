import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / '协作/02_ClaudeCode_实操/ZCode标注准备_20261006'
OLD = ROOT / '协作/03_ZCode_标注/扩容198_20261006_165855'
index_path = SOURCE / 'zcode_input_index.json'
assert hashlib.sha256(index_path.read_bytes()).hexdigest() == 'c2405a22ca2a3a8a71eef59773b13340e78514c5b83c67044adf277118a1db55'
index = json.loads(index_path.read_text(encoding='utf-8'))
existing = {p.name.removesuffix('_raw.json') for p in OLD.glob('worker_*/*_raw.json')}
assert len(existing) == 107
remaining = [r for r in index['items'] if r['item_id'] not in existing]
assert len(remaining) == 91
parts = []
for start in range(0, 91, 7):
    number = start // 7 + 1
    path = OUT / f'blind_{number:02d}.jsonl'
    if path.exists():
        raise FileExistsError(path)
    rows = remaining[start:start + 7]
    assert all(set(r) == {'item_id','sample_id','image_path','image_sha256'} for r in rows)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows),encoding='utf-8',newline='\n')
    worker = OUT / f'worker_{number:02d}'
    worker.mkdir(exist_ok=False)
    parts.append({'number':number,'path':str(path.relative_to(ROOT)),
                  'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'n':len(rows)})
plan = {'date':'2026-10-06','author':'Codex','original_saved':107,'to_complete':91,
        'glm_previous_attempt_uncertain':18,'new_gpt_annotation':True,
        'requested_model':'gpt-6.1-sol','requested_reasoning_effort':'ultra',
        'source_index_sha256':hashlib.sha256(index_path.read_bytes()).hexdigest(),
        'parts':parts,
        'source_note':'New GPT model annotation, not GLM, human annotation, or human review.'}
path = OUT / 'plan.json'
assert not path.exists()
path.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'parts':len(parts),'n':sum(p['n'] for p in parts),'n_per_part':[p['n'] for p in parts]},ensure_ascii=False))
