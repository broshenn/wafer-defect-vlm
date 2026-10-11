import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path('D:/pycode/晶圆图研究')
BLIND = HERE.parent / 'blind_06.jsonl'
CHECKER = ROOT / '协作/01_Codex_指挥/验收_20261003_GLM180/check_answer_v4.py'
PYTHON = 'D:/python/python.exe'

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def exclusive(path, text):
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(text)

action, item_id = sys.argv[1:3]
items = [json.loads(line) for line in BLIND.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
item = next(row for row in items if row['item_id'] == item_id)
n = int(item_id.split('_')[1])
w = (n - 21) // 7 + 4
old = ROOT / f'协作/03_ZCode_标注/扩容198_20261006_165855/worker_{w:02d}/{item_id}_raw.json'
raw_path = HERE / f'{item_id}_raw.json'
if action == 'prepare':
    if old.exists():
        print(json.dumps({'item_id': item_id, 'status': 'skip_old_exists'}, ensure_ascii=False))
        sys.exit(0)
    if raw_path.exists():
        raise RuntimeError('Raw exists; refusing another answer')
    actual_sha = hashlib.sha256(Path(item['image_path']).read_bytes()).hexdigest()
    if actual_sha != item['image_sha256']:
        raise RuntimeError('Image SHA mismatch')
    stamp = {'item_id': item_id, 'started_at': now(), 'image_sha256': actual_sha, 'old_raw_exists': False}
    exclusive(HERE / f'{item_id}_start.json', json.dumps(stamp, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': 'ready', **item, **stamp}, ensure_ascii=False))
elif action == 'record':
    raw = sys.stdin.read()
    exclusive(raw_path, raw)
    answer_saved_at = now()
    result = subprocess.run([PYTHON, '-X', 'utf8', '-B', str(CHECKER), str(raw_path)], capture_output=True, text=True, encoding='utf-8')
    exclusive(HERE / f'{item_id}_check.stdout.txt', result.stdout)
    exclusive(HERE / f'{item_id}_check.stderr.txt', result.stderr)
    exclusive(HERE / f'{item_id}_check.exit_code.txt', str(result.returncode) + '\n')
    stamp = json.loads((HERE / f'{item_id}_start.json').read_text(encoding='utf-8'))
    entry = {**item, **stamp, 'raw_path': str(raw_path),
             'raw_sha256': hashlib.sha256(raw_path.read_bytes()).hexdigest(),
             'answer_saved_at': answer_saved_at, 'finished_at': now(),
             'visual_read_count': 1, 'checker_version': 'format_v4_clock_range_20261003',
             'checker_exit_code': result.returncode, 'model': 'gpt-6.1-sol',
             'reasoning_effort': 'ultra', 'source': 'gpt_model_annotation',
             'backend_version': '未提供', 'token_usage': '未提供',
             'prompt_path': str(ROOT / '协作/01_Codex_指挥/prompt_image_only_v2_20261002.txt'),
             'prompt_used_verbatim': True}
    exclusive(HERE / f'{item_id}_entry.json', json.dumps(entry, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'item_id': item_id, 'checker_exit_code': result.returncode, 'raw_sha256': entry['raw_sha256']}, ensure_ascii=False))
else:
    raise RuntimeError('Unknown action')
