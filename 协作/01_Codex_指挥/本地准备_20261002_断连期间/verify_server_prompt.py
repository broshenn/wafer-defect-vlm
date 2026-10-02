"""Codex：独立核对匹配版218条；不 import 导出器，不运行训练。"""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE_DIR = ROOT / '协作/02_ClaudeCode_实操/学校训练准备_20261002-025848'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def loadl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def main():
    report = json.loads((HERE / 'matched_preparation_report.json').read_text(encoding='utf-8'))
    source = ROOT / report['source_script']
    assert sha(source) == report['source_script_sha256'], '来源源码漂移'
    source_tree = ast.parse(source.read_text(encoding='utf-8-sig'))
    node, = [n for n in source_tree.body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == 'QUESTION' for t in n.targets)]
    question = ast.literal_eval(node.value)
    assert node.lineno == report['source_question_line']
    assert (HERE / 'classification_prompt_server_v2.txt').read_bytes() == question.encode('utf-8')
    for filename, expected in report['output_sha256'].items():
        assert sha(HERE / filename) == expected, '输出指纹不符：' + filename
    manifest = {r['sample_id']: r for r in loadl(ROOT / 'data/manifest.jsonl')}
    image_hashes = {r['sample_id']: r['image_sha256']
                    for r in loadl(HERE / 'selected_image_fingerprints.jsonl')}
    plan = json.loads((SOURCE_DIR / 'dataset_plan.json').read_text(encoding='utf-8'))
    assert sha(ROOT / 'data/manifest.jsonl') == plan['manifest_sha256']
    checked = 0
    idsets = {}
    for name, size, split in [('train_180', 180, 'train'), ('dev_18', 18, 'val'), ('smoke_20', 20, 'train')]:
        assert sha(SOURCE_DIR / (name + '.jsonl')) == plan['files'][name + '.jsonl']
        selected = loadl(SOURCE_DIR / (name + '.jsonl'))
        rows = loadl(HERE / (name + '.matched.ms_swift.jsonl'))
        previous_file = name + '.ms_swift.jsonl'
        assert sha(HERE / previous_file) == report['input_sha256'][previous_file]
        previous = loadl(HERE / previous_file)
        assert len(rows) == len(selected) == len(previous) == size
        assert len({r['sample_id'] for r in rows}) == size
        for row, old, original in zip(rows, previous, selected):
            sid = original['sample_id']
            assert row['sample_id'] == old['sample_id'] == sid
            assert set(row) == {'sample_id', 'messages', 'images'}
            assert row['images'] == old['images'] == [str((ROOT / 'data/images' / (sid + '.png')).resolve())]
            assert sha(Path(row['images'][0])) == image_hashes[sid]
            assert row['messages'] == [
                {'role': 'user', 'content': '<image>' + question},
                {'role': 'assistant', 'content': json.dumps({'defect_class': original['label']},
                   ensure_ascii=False, separators=(',', ':'))}]
            assert original['label'] == manifest[sid]['failure_type']
            assert original['label_source'] == manifest[sid]['label_source'] == 'ground_truth'
            assert original['lot_name'] == manifest[sid]['lot_name']
            assert original['split'] == manifest[sid]['split'] == split
            checked += 1
        idsets[name] = {r['sample_id'] for r in rows}
    assert not idsets['train_180'] & idsets['dev_18']
    assert idsets['smoke_20'] <= idsets['train_180']
    print(json.dumps({'status': 'PASS', 'checked_rows': checked,
                      'unique_images': len(idsets['train_180'] | idsets['dev_18']),
                      'matches_source_question': True, 'extra_system_message': False,
                      'server_baseline_prompt_verified': False,
                      'report_sha256': sha(HERE / 'matched_preparation_report.json')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
