"""Codex：从执行方源码静态取回修正题面，另存匹配版；不执行来源脚本。"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / '协作/02_ClaudeCode_实操/学校训练准备_20261002-025848/make_swift_dataset.py'
NAMES = ('train_180', 'dev_18', 'smoke_20')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def prepare():
    source_bytes = SOURCE.read_bytes()
    tree = ast.parse(source_bytes.decode('utf-8-sig'))
    assignments = [n for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'QUESTION' for t in n.targets)]
    if len(assignments) != 1:
        raise ValueError('QUESTION 必须恰好定义一次')
    question = ast.literal_eval(assignments[0].value)
    if not isinstance(question, str):
        raise ValueError('QUESTION 不是常量文本')
    base_report = json.loads((HERE / 'preparation_report.json').read_text(encoding='utf-8'))
    outputs = {'classification_prompt_server_v2.txt': question.encode('utf-8')}
    inputs = {}
    counts = {}
    for name in NAMES:
        filename = name + '.ms_swift.jsonl'
        original = (HERE / filename).read_bytes()
        if digest(original) != base_report['output_sha256'][filename]:
            raise ValueError('已核验输入发生漂移：' + filename)
        rows = [json.loads(line) for line in original.decode('utf-8').splitlines() if line.strip()]
        converted = []
        for row in rows:
            if [m['role'] for m in row['messages']] != ['system', 'user', 'assistant']:
                raise ValueError('输入角色不符')
            converted.append({'sample_id': row['sample_id'], 'messages': [
                {'role': 'user', 'content': '<image>' + question},
                row['messages'][-1]], 'images': row['images']})
        outputs[name + '.matched.ms_swift.jsonl'] = ''.join(
            json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n'
            for row in converted).encode('utf-8')
        inputs[filename] = digest(original)
        counts[name] = len(converted)
    report = {'author': 'Codex', 'date': '2026-10-02', 'seed': 3407,
              'source_script': SOURCE.relative_to(ROOT).as_posix(),
              'source_script_sha256': digest(source_bytes),
              'source_question_line': assignments[0].lineno,
              'question_sha256': digest(question.encode('utf-8')),
              'input_sha256': inputs, 'rows': counts,
              'output_sha256': {name: digest(data) for name, data in outputs.items()},
              'message_roles': ['user', 'assistant'], 'extra_system_message': False,
              'user_prefix': '<image>', 'newline_after_image': False,
              'selection_labels_splits_unchanged': True,
              'server_baseline_prompt_verified': False,
              'note': '沿用执行方修正源码题面；实际远端基座原答中的题面与模板仍待核对。'}
    outputs['matched_preparation_report.json'] = (
        json.dumps(report, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    return outputs, report


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--dry-run', action='store_true')
    modes.add_argument('--write', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--seed', type=int, default=3407)
    args = parser.parse_args()
    if args.seed != 3407:
        parser.error('沿用原选样，不改变 seed=3407')
    outputs, report = prepare()
    written = 0
    if args.write:
        for name, data in outputs.items():
            path = HERE / name
            if path.exists() and not (args.resume and path.read_bytes() == data):
                raise ValueError('拒绝覆盖：' + name)
        for name, data in outputs.items():
            path = HERE / name
            if not path.exists():
                path.write_bytes(data)
                written += 1
    print(json.dumps({'mode': 'write' if args.write else 'dry-run', 'rows': report['rows'],
                      'files_written': written,
                      'existing_outputs_matched': len(outputs) - written if args.write else 0,
                      'source_question_line': report['source_question_line']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
