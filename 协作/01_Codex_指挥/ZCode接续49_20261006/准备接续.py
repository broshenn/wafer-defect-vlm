import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
INPUT = ROOT / '协作/02_ClaudeCode_实操/ZCode标注准备_20261006'
GLM = ROOT / '协作/03_ZCode_标注/扩容198_20261006_165855'
GPT = ROOT / '协作/01_Codex_指挥/GPT补标91_20261006'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

index = json.loads((INPUT / 'zcode_input_index.json').read_text(encoding='utf-8'))
assert sha(INPUT / 'zcode_input_index.json') == 'c2405a22ca2a3a8a71eef59773b13340e78514c5b83c67044adf277118a1db55'
spec = importlib.util.spec_from_file_location('resume49_v4', ROOT / index['checker_path'])
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)
done = {}
counts = {}
for source, directory in [('GLM_original', GLM), ('GPT_supplement', GPT)]:
    paths = sorted(directory.glob('worker_*/*_raw.json'))
    counts[source] = len(paths)
    for p in paths:
        item = p.name.removesuffix('_raw.json')
        assert item not in done
        check = checker.check_text(p.read_text(encoding='utf-8-sig'))
        assert check['exit_code'] == 0
        done[item] = {'item_id': item, 'source': source, 'raw_path': str(p.relative_to(ROOT)),
                      'raw_sha256': sha(p), 'format_exit_code': 0, 'fact_review': 'pending'}
assert counts == {'GLM_original': 107, 'GPT_supplement': 42}
rows = [r for r in index['items'] if r['item_id'] not in done]
assert len(rows) == 49 and [r['item_id'] for r in rows] == [f'zc_{n:03d}' for n in range(150,199)]
for r in rows:
    assert set(r) == {'item_id','sample_id','image_path','image_sha256'}
    assert sha(Path(r['image_path'])) == r['image_sha256']

def write(name, obj, lines=False):
    p = OUT / name
    if p.exists():
        raise FileExistsError(p)
    text = ''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in obj) if lines else json.dumps(obj,ensure_ascii=False,indent=2)+'\n'
    p.write_text(text,encoding='utf-8',newline='\n')
    return {'file':name,'sha256':sha(p),'n':len(obj)}

all49 = write('blind_remaining49.jsonl',rows,True)
parts = [write(f'batch_{n//7+1:02d}.jsonl',rows[n:n+7],True) for n in range(0,49,7)]
snapshot = write('已保存149_保留索引.json',list(done.values()))
metadata = {'date':'2026-10-06','author':'Codex','n_items':49,
            'original_items':198,'existing_glm':107,'existing_gpt':42,
            'original_index_path':str((INPUT/'zcode_input_index.json').relative_to(ROOT)),
            'original_index_sha256':sha(INPUT/'zcode_input_index.json'),
            'blind_inputs':all49,'batches':parts,'preserved_raw_index':snapshot,
            'prompt_path':index['prompt_path'],'prompt_sha256':index['prompt_sha256'],
            'checker_path':index['checker_path'],'checker_sha256':index['checker_sha256'],
            'checker_v3_path':index['checker_v3_path'],'checker_v3_sha256':index['checker_v3_sha256'],
            'permission':'User explicitly transferred remaining unfinished work back to ZCode.',
            'gpt_interrupted_no_raw_items':[f'zc_{n:03d}' for n in range(150,157)],
            'prior_gpt_attempt_note':'No saved raw in interrupted worker07; previous visual attempt state not independently confirmed. New GLM annotation authorized, do not rewrite past counts.',
            'teacher_source_note':'Mixed GLM and GPT model annotations; never human gold or pure GLM dataset.'}
locked = write('zcode_resume49_index.json',metadata)
write('移交状态.json',{'state':'transferred_to_zcode_by_user','gpt_dispatch':'remain_stopped','remaining':49,'new_index_sha256':locked['sha256']})
print(json.dumps({'counts':counts,'remaining':49,'batch_counts':[p['n'] for p in parts],'index_sha256':locked['sha256']},ensure_ascii=False))
