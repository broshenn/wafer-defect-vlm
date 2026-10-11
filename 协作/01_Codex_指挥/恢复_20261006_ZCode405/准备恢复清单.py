"""Codex: readonly audit of stopped workers and new answer-free resume inputs."""
import collections
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
INPUT = ROOT / '协作/02_ClaudeCode_实操/ZCode标注准备_20261006'
RUN = ROOT / '协作/03_ZCode_标注/扩容198_20261006_165855'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def pairs(values):
    obj = {}
    for key, value in values:
        if key in obj:
            raise ValueError('duplicate key')
        obj[key] = value
    return obj

index_path = INPUT / 'zcode_input_index.json'
assert sha(index_path) == 'c2405a22ca2a3a8a71eef59773b13340e78514c5b83c67044adf277118a1db55'
index = json.loads(index_path.read_text(encoding='utf-8'), object_pairs_hook=pairs)
mapping = {r['item_id']: r for r in index['items']}
spec = importlib.util.spec_from_file_location('locked_v4', ROOT / index['checker_path'])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
saved = {}
for path in sorted(RUN.glob('worker_*/*_raw.json')):
    item = path.name.removesuffix('_raw.json')
    assert item in mapping and item not in saved
    check = module.check_text(path.read_text(encoding='utf-8-sig'))
    assert check['exit_code'] == 0
    entry_path = path.with_name(item + '_entry.json')
    if entry_path.exists():
        entry = json.loads(entry_path.read_text(encoding='utf-8-sig'), object_pairs_hook=pairs)
        assert entry['item_id'] == item
        assert entry['sample_id'] == mapping[item]['sample_id']
        assert entry['image_sha256'] == mapping[item]['image_sha256']
        assert entry['raw_sha256'] == sha(path)
    saved[item] = {
        'item_id': item, 'sample_id': mapping[item]['sample_id'],
        'raw_path': str(path.relative_to(ROOT)), 'raw_sha256': sha(path),
        'format_exit_code': 0, 'entry_present': entry_path.exists(),
        'fact_review': 'pending_model_review',
    }

# User-provided dispatch transcript stops at wave 6 (workers 16/17/18).
# Do not infer never attempted just from absence of raw in those workers.
later = [r for r in index['items'] if int(r['item_id'].split('_')[1]) >= 126]
uncertain = [r for r in index['items']
             if 105 <= int(r['item_id'].split('_')[1]) <= 125
             and r['item_id'] not in saved]
assert not ({r['item_id'] for r in later} & set(saved))
assert len(saved) == 107 and len(later) == 73 and len(uncertain) == 18
assert set(saved) | {r['item_id'] for r in later + uncertain} == set(mapping)

def write_new(name, value, jsonl=False):
    path = OUT / name
    if path.exists():
        raise FileExistsError(path)
    text = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in value) if jsonl else json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    path.write_text(text, encoding='utf-8', newline='\n')
    return {'file': name, 'sha256': sha(path), 'n': len(value)}

completed = write_new('已保存107_恢复核验.json', list(saved.values()))
all_pending = write_new('续标73_无答案.jsonl', later, True)
uncertain_info = write_new('状态待核18_不可自动重试.jsonl', uncertain, True)
chunks = []
for start in range(0, len(later), 7):
    chunks.append(write_new(f'resume_{start // 7 + 1:02d}.jsonl', later[start:start + 7], True))
resume = {
    'date': '2026-10-06', 'author': 'Codex',
    'original_index_path': str(index_path.relative_to(ROOT)),
    'original_index_sha256': sha(index_path),
    'original_run': str(RUN.relative_to(ROOT)),
    'completed': completed, 'resume_inputs': all_pending,
    'uncertain_do_not_retry': uncertain_info,
    'batches': chunks,
    'scope': 'Only 73 undispatched items according to user-provided dispatch transcript; do not repeat 107 saved answers or 18 uncertain attempts.',
    'prompt_path': index['prompt_path'], 'prompt_sha256': index['prompt_sha256'],
    'checker_path': index['checker_path'], 'checker_sha256': index['checker_sha256'],
    'checker_v3_path': index['checker_v3_path'], 'checker_v3_sha256': index['checker_v3_sha256'],
    'old_raw_106_note': 'raw saved and format passes; original entry missing, no image retry',
}
locked = write_new('恢复输入索引.json', resume)
print(json.dumps({'saved': len(saved), 'safe_resume': len(later), 'uncertain': len(uncertain),
                  'index': locked, 'batch_counts': [x['n'] for x in chunks]}, ensure_ascii=False))
