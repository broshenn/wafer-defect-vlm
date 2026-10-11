import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path('D:/pycode/晶圆图研究')
WORKER = Path(__file__).resolve().parent
BLIND = WORKER.parent / 'blind_02.jsonl'
CHECKER = ROOT / '协作/01_Codex_指挥/验收_20261003_GLM180/check_answer_v4.py'
PROMPT = ROOT / '协作/01_Codex_指挥/prompt_image_only_v2_20261002.txt'

def now():
    return datetime.now(timezone.utc).isoformat()

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write_new(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(value)

def get_item(item_id):
    rows = [json.loads(line) for line in BLIND.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    matches = [row for row in rows if row['item_id'] == item_id]
    if len(matches) != 1:
        raise RuntimeError('Blind item is not unique')
    return matches[0]

def start(item_id, legacy):
    item = get_item(item_id)
    for root in legacy:
        candidate = Path(root) / (item_id + '_raw.json')
        if candidate.is_file():
            print(json.dumps({'item_id': item_id, 'status': 'skipped_existing_zcode_raw', 'path': str(candidate)}, ensure_ascii=False))
            return
    raw = WORKER / (item_id + '_raw.json')
    if raw.exists():
        raise FileExistsError('Refuse to overwrite existing raw: ' + str(raw))
    actual_hash = digest(Path(item['image_path']))
    if actual_hash != item['image_sha256']:
        raise RuntimeError('IMAGE_SHA256_MISMATCH: ' + item_id)
    record = {**item, 'started_at_utc': now(), 'prompt_sha256': digest(PROMPT),
              'checker_sha256': digest(CHECKER), 'status': 'hash_verified_before_visual_read'}
    write_new(WORKER / (item_id + '_start.json'), json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(record, ensure_ascii=False))

def finish(item_id):
    started = json.loads((WORKER / (item_id + '_start.json')).read_text(encoding='utf-8'))
    raw = WORKER / (item_id + '_raw.json')
    raw_hash = digest(raw)
    result = subprocess.run([sys.executable, '-X', 'utf8', '-B', str(CHECKER), str(raw)], capture_output=True)
    for suffix, value in [('check_stdout.txt', result.stdout.decode('utf-8')), ('check_stderr.txt', result.stderr.decode('utf-8')), ('check_exit_code.txt', str(result.returncode) + '\n')]:
        write_new(WORKER / (item_id + '_' + suffix), value)
    entry = {
        'item_id': item_id,
        'sample_id': started['sample_id'],
        'image_path': started['image_path'],
        'image_sha256': started['image_sha256'],
        'raw_path': str(raw),
        'raw_sha256': raw_hash,
        'visual_read_count': 1,
        'visual_read_method': "tools.view_image(path=original_png, detail='original') then image(result.image_url)",
        'checker_exit_code': result.returncode,
        'checker_version': 'format_v4_clock_range_20261003',
        'checker_path': str(CHECKER),
        'checker_sha256': started['checker_sha256'],
        'prompt_path': str(PROMPT),
        'prompt_sha256': started['prompt_sha256'],
        'started_at_utc': started['started_at_utc'],
        'ended_at_utc': now(),
        'model_call_config': {'model': 'gpt-6.1-sol', 'reasoning_effort': 'ultra', 'backend_model': '未提供', 'temperature': '未提供', 'seed': '未提供', 'tokens': '未提供'},
        'source': 'gpt_model_annotation',
        'retry_count': 0,
        'first_answer_preserved': True,
        'human_review_status': 'not_performed',
        'context_scope': 'blind_02 seven rows and locked image-only prompt; sequential single-image visual reads',
    }
    write_new(WORKER / (item_id + '_entry.json'), json.dumps(entry, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'item_id': item_id, 'checker_exit_code': result.returncode, 'entry_path': str(WORKER / (item_id + '_entry.json'))}, ensure_ascii=False))
    if result.returncode == 1:
        raise RuntimeError('Checker system error; stop shard')

parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=['start', 'finish', 'save'])
parser.add_argument('item_id')
parser.add_argument('--legacy', action='append', default=[])
parser.add_argument('--raw-base64')
args = parser.parse_args()
if args.mode == 'start':
    start(args.item_id, args.legacy)
elif args.mode == 'save':
    write_new(WORKER / (args.item_id + '_raw.json'), base64.b64decode(args.raw_base64).decode('utf-8'))
    finish(args.item_id)
else:
    finish(args.item_id)
