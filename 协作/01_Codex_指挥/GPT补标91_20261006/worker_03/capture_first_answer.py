"""Store one first visual answer, run the locked format checker, and record provenance."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
TASK = HERE.parent
ROOT = TASK.parents[2]
BLIND = TASK / "blind_03.jsonl"
PROMPT = ROOT / "协作/01_Codex_指挥/prompt_image_only_v2_20261002.txt"
CHECKER = ROOT / "协作/01_Codex_指挥/验收_20261003_GLM180/check_answer_v4.py"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_new(path, content):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(content)


item_id, started_at, legacy_path_text = sys.argv[1:4]
items = [json.loads(line) for line in BLIND.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
item = next(row for row in items if row["item_id"] == item_id)
legacy = Path(legacy_path_text)
if legacy.exists():
    raise RuntimeError("Legacy original appeared; stop before saving duplicate: " + item_id)
image_sha256 = hashlib.sha256(Path(item["image_path"]).read_bytes()).hexdigest()
if image_sha256 != item["image_sha256"]:
    raise RuntimeError("Image hash mismatch: " + item_id)
raw_path = HERE / (item_id + "_raw.json")
raw_text = sys.stdin.read().strip() + "\n"
write_new(raw_path, raw_text)
raw_saved_at = utc_now()
raw_sha256 = hashlib.sha256(raw_path.read_bytes()).hexdigest()
command = ["D:/python/python.exe", "-X", "utf8", "-B", str(CHECKER), str(raw_path), "--json"]
checked = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
write_new(HERE / (item_id + "_check_stdout.txt"), checked.stdout)
write_new(HERE / (item_id + "_check_stderr.txt"), checked.stderr)
write_new(HERE / (item_id + "_check_exit_code.txt"), str(checked.returncode) + "\n")
if hashlib.sha256(raw_path.read_bytes()).hexdigest() != raw_sha256:
    raise RuntimeError("Raw changed after saving: " + item_id)
metadata = {
    "item_id": item["item_id"],
    "sample_id": item["sample_id"],
    "image_path": item["image_path"],
    "image_sha256": image_sha256,
    "raw_path": str(raw_path),
    "raw_sha256": raw_sha256,
    "visual_read_count": 1,
    "visual_read_tool": "tools.view_image(path=original_png, detail=original); functions.exec image(result.image_url)",
    "first_answer_only": True,
    "retry_count": 0,
    "checker_exit_code": checked.returncode,
    "checker_path": str(CHECKER),
    "checker_sha256": hashlib.sha256(CHECKER.read_bytes()).hexdigest(),
    "checker_command": command,
    "started_at": started_at,
    "raw_saved_at": raw_saved_at,
    "finished_at": utc_now(),
    "model_call_config": {"model": "gpt-6.1-sol", "reasoning_effort": "ultra", "backend_model_id": "未提供", "temperature": "未提供", "seed": "未提供", "token_usage": "未提供"},
    "prompt_path": str(PROMPT),
    "prompt_sha256": hashlib.sha256(PROMPT.read_bytes()).hexdigest(),
    "source": "gpt_model_annotation",
    "legacy_original_path_checked_without_reading": str(legacy),
    "legacy_original_existed_at_save": False,
    "context_scope": "blind_03 only, fixed prompt and checker, global project instructions; no original label, lot, geometry, other answers or parent conversation",
    "human_review_status": "not_performed"
}
write_new(HERE / (item_id + "_entry.json"), json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"item_id":item_id, "checker_exit_code":checked.returncode, "entry_path":str(HERE / (item_id + "_entry.json"))}, ensure_ascii=False))
if checked.returncode == 1 or checked.stderr:
    raise RuntimeError("Checker system error; stop shard: " + item_id)
