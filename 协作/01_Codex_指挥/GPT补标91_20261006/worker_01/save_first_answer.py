import base64
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BLIND = HERE.parent / 'blind_01.jsonl'
CHECKER = ROOT / '协作/01_Codex_指挥/验收_20261003_GLM180/check_answer_v4.py'
PROMPT = ROOT / '协作/01_Codex_指挥/prompt_image_only_v2_20261002.txt'

def put_new(path, data):
    with path.open('xb') as stream:
        stream.write(data)

item_id, started_at, encoded_raw = sys.argv[1:]
records = [json.loads(line) for line in BLIND.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
item = next(record for record in records if record['item_id'] == item_id)
paths = {name: HERE / (item_id + suffix) for name, suffix in {
    'raw': '_raw.json', 'stdout': '_check.stdout.txt', 'stderr': '_check.stderr.txt',
    'exit': '_check.exit.json', 'entry': '_entry.json'}.items()}
for path in paths.values():
    if path.exists():
        raise FileExistsError('Refusing to overwrite ' + str(path))
image_sha = hashlib.sha256(Path(item['image_path']).read_bytes()).hexdigest()
if image_sha != item['image_sha256']:
    raise RuntimeError('Image SHA256 mismatch: ' + item_id)
raw = base64.b64decode(encoded_raw, validate=True)
put_new(paths['raw'], raw)
result = subprocess.run([r'D:/python/python.exe', '-X', 'utf8', '-B', str(CHECKER), str(paths['raw'])], capture_output=True, check=False)
put_new(paths['stdout'], result.stdout)
put_new(paths['stderr'], result.stderr)
put_new(paths['exit'], (json.dumps({'exit_code': result.returncode}, ensure_ascii=False) + '\n').encode('utf-8'))
entry = {
    'item_id': item_id, 'sample_id': item['sample_id'], 'image_path': item['image_path'],
    'image_sha256': image_sha, 'raw_sha256': hashlib.sha256(raw).hexdigest(),
    'visual_read_count': 1, 'visual_read_tool': 'tools.view_image', 'visual_read_detail': 'original',
    'image_presented_via': 'functions.exec image(result.image_url)',
    'checker_exit_code': result.returncode, 'checker_version': 'format_v4_clock_range_20261003',
    'started_at_utc': started_at, 'finished_at_utc': datetime.now(timezone.utc).isoformat(),
    'model_call_config': {'model': 'gpt-6.1-sol', 'reasoning_effort': 'ultra'},
    'backend_model_id': '未提供', 'input_tokens': '未提供', 'output_tokens': '未提供',
    'source': 'gpt_model_annotation', 'human_reviewed': False,
    'prompt_path': str(PROMPT), 'prompt_sha256': hashlib.sha256(PROMPT.read_bytes()).hexdigest(),
    'checker_path': str(CHECKER), 'checker_sha256': hashlib.sha256(CHECKER.read_bytes()).hexdigest(),
    'raw_path': str(paths['raw']), 'checker_stdout_path': str(paths['stdout']),
    'checker_stderr_path': str(paths['stderr']), 'checker_exit_path': str(paths['exit'])
}
put_new(paths['entry'], (json.dumps(entry, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
print(json.dumps({'item_id': item_id, 'checker_exit_code': result.returncode, 'entry_path': str(paths['entry'])}, ensure_ascii=False))
