"""Read-only checks of GPT raw files; writes final new metadata only on --final."""
import argparse
import collections
import hashlib
import importlib.util
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
INPUT = ROOT / '协作/02_ClaudeCode_实操/ZCode标注准备_20261006'
OLD = ROOT / '协作/03_ZCode_标注/扩容198_20261006_165855'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def pairs(items):
    out = {}
    for key, value in items:
        if key in out:
            raise ValueError('duplicate key: ' + key)
        out[key] = value
    return out

def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'), object_pairs_hook=pairs)

args = argparse.ArgumentParser()
args.add_argument('--final', action='store_true')
options = args.parse_args()
index = load(INPUT / 'zcode_input_index.json')
mapping = {r['item_id']: r for r in index['items']}
plan = load(BASE / 'plan.json')
expected = set()
for part in plan['parts']:
    path = ROOT / part['path']
    assert sha(path) == part['sha256']
    expected.update(json.loads(line)['item_id'] for line in path.read_text(encoding='utf-8').splitlines())
spec = importlib.util.spec_from_file_location('gpt_final_v4', ROOT / index['checker_path'])
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)
rows = []
seen = set()
issues = []
for path in sorted(BASE.glob('worker_*/*_raw.json')):
    item = path.name.removesuffix('_raw.json')
    if item not in expected or item in seen:
        issues.append({'item_id': item, 'problem': 'unexpected_or_duplicate'})
        continue
    seen.add(item)
    result = checker.check_text(path.read_text(encoding='utf-8-sig'))
    meta = path.with_name(item + '_entry.json')
    if not meta.exists():
        issues.append({'item_id': item, 'problem': 'entry_pending'})
    else:
        entry = load(meta)
        for key in ('item_id', 'sample_id', 'image_sha256'):
            if entry.get(key) != mapping[item][key]:
                issues.append({'item_id': item, 'problem': 'metadata_' + key})
        if entry.get('raw_sha256') != sha(path):
            issues.append({'item_id': item, 'problem': 'raw_hash'})
    image = Path(mapping[item]['image_path'])
    assert sha(image) == mapping[item]['image_sha256']
    parsed = load(path) if result['exit_code'] == 0 else {}
    rows.append({
        **mapping[item], 'raw_path': str(path.relative_to(ROOT)), 'raw_sha256': sha(path),
        'format_exit_code': result['exit_code'], 'unknown': parsed.get('defect_class') == 'unknown',
        'model_requested': 'gpt-6.1-sol', 'reasoning_effort_requested': 'ultra',
        'source': 'gpt_model_annotation', 'human_review': 'not_done',
        'fact_review': 'pending_model_review',
    })
summary = {'planned': len(expected), 'saved': len(rows),
           'format_pass': sum(r['format_exit_code'] == 0 for r in rows),
           'unknown': sum(r['unknown'] for r in rows),
           'missing_items': sorted(expected - seen), 'metadata_issues': issues,
           'worker_counts': {p.name: len(list(p.glob('*_raw.json'))) for p in sorted(BASE.glob('worker_*'))}}
print(json.dumps(summary, ensure_ascii=False))
if options.final:
    if len(rows) != 91 or issues:
        raise SystemExit('final incomplete; no output written')
    old_ids = {p.name.removesuffix('_raw.json') for p in OLD.glob('worker_*/*_raw.json')}
    assert len(old_ids) == 107 and not(old_ids & seen) and old_ids | seen == set(mapping)
    for name, value in [('补标91_汇总.json', summary), ('补标91_来源清单.json', rows)]:
        destination = BASE / name
        if destination.exists():
            raise FileExistsError(destination)
        destination.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
