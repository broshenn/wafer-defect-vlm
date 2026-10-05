"""Codex：只读独立复算三教师/GLM开发结果；不联网、不读取凭据。"""
import argparse
import base64
from collections import Counter
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / '协作/02_ClaudeCode_实操/三教师对照_20261002-162722'
PREP = ROOT / '协作/02_ClaudeCode_实操/口径修正与准备_20261002-021207'
GLM = ROOT / '协作/03_ZCode_标注/20261002-024921_十八张开发试标'
PROMPT = ROOT / '协作/01_Codex_指挥/prompt_image_only_v2_20261002.txt'
CLASSES = ['Center', 'Donut', 'Edge_Loc', 'Edge_Ring', 'Loc', 'Near_full', 'Random', 'Scratch', 'none']
LOCKED = {'blind_inputs.jsonl': '70d48e6f5a7ab3c14f4c3d675e94adc11a9cb518f814120fadb8b2e802120b20',
          'prompt_image_only_v2_20261002.txt': '7df73031ee1a30dfb018dbb2417f505ea9ff139280307bce2934de604119954a',
          'check_answer_v3.py': '794a254e005eaa232362c539817abde98a1c2406ecb52574b9e9edb35303ec55'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def loadl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    middle = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return [middle - half, middle + half]


def score(truth, predictions):
    f1 = []
    per_class = {}
    for c in CLASSES:
        tp = sum(t == p == c for t, p in zip(truth, predictions))
        fp = sum(t != c and p == c for t, p in zip(truth, predictions))
        fn = sum(t == c and p != c for t, p in zip(truth, predictions))
        value = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0
        f1.append(value)
        per_class[c] = {'support': truth.count(c), 'correct': tp, 'f1': value}
    hits = sum(t == p for t, p in zip(truth, predictions))
    return {'correct': hits, 'n': len(truth), 'agreement': hits/len(truth),
            'macro_f1': statistics.mean(f1), 'per_class': per_class,
            'prediction_counts': dict(Counter(predictions)),
            'wilson_95_descriptive': wilson(hits, len(truth))}


def audit():
    inputs = [PREP / 'blind_inputs.jsonl', PROMPT, PREP / 'check_answer_v3.py']
    for path in inputs:
        assert sha(path.read_bytes()) == LOCKED[path.name], '固定输入漂移：' + path.name
    spec = importlib.util.spec_from_file_location('locked_format_checker', inputs[-1])
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    blind = loadl(inputs[0])
    gold = {r['item_id']: r for r in loadl(PREP / 'audit_selected.jsonl')}
    prompt = PROMPT.read_text(encoding='utf-8')
    records = {}
    failures = []
    record_count = 0
    source_hashes = {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in inputs}
    for file in sorted((SOURCE / 'runs').glob('*/*__*.json')):
        row = json.loads(file.read_text(encoding='utf-8'))
        record_count += 1
        source_hashes[file.relative_to(ROOT).as_posix()] = sha(file.read_bytes())
        if row['http_status'] != 200:
            failures.append({'file': file.relative_to(ROOT).as_posix(), 'candidate': row['candidate'],
                             'item_id': row['item_id'], 'http_status': row['http_status']})
            continue
        key = row['candidate'], row['item_id']
        assert key not in records, '发现重复成功记录；禁止按最新自动挑选'
        raw = json.loads(row['raw_text'])
        assert raw['model'] == row['returned_model']
        assert raw['usage'] == row['usage']
        assert raw['choices'][0]['finish_reason'] == row['finish_reason'] == 'stop'
        assert raw['choices'][0]['message']['content'] == row['answer_text']
        assert checker.check_text(row['answer_text'])['exit_code'] == 0
        assert row['max_tokens'] == 16384 and row['timeout_seconds'] == 600 and row['client_retries'] == 0
        records[key] = (file, row, json.loads(row['answer_text']))
    output_rows = []
    truth = []
    predictions = {c: [] for c in ['deepseek', 'kimi', 'qwen', 'glm']}
    for b in blind:
        item = b['item_id']
        g = gold[item]
        assert g['sample_id'] == b['sample_id'] and g['label_source'] == 'ground_truth'
        image = Path(b['image_path']).read_bytes()
        assert sha(image) == b['image_sha256']
        t = g['failure_type_normalized']
        truth.append(t)
        combined = {'item_id': item, 'sample_id': b['sample_id'], 'original_label': t,
                    'label_source': 'ground_truth', 'human_review': 'not run', 'models': {}}
        for c in ['deepseek', 'kimi', 'qwen']:
            file, row, answer = records[c, item]
            assert row['sample_id'] == b['sample_id'] and row['image_sha256'] == b['image_sha256']
            assert row['prompt_byte_sha256'] == LOCKED[PROMPT.name]
            body = {'model': row['model_requested'], 'messages': [{'role': 'user', 'content': [
                    {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(image).decode('ascii')}},
                    {'type': 'text', 'text': prompt}]}], 'max_tokens': 16384, 'stream': False, 'n': 1}
            canonical = json.dumps(body, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
            assert sha(canonical) == row['payload_sha256'], '请求体对象指纹不符'
            u = row['usage']
            assert u['prompt_tokens'] + u['completion_tokens'] == u['total_tokens']
            predictions[c].append(answer['defect_class'])
            combined['models'][c] = {'prediction': answer['defect_class'], 'matches_label': answer['defect_class'] == t,
                'raw_record': file.relative_to(ROOT).as_posix(), 'answer': answer}
        glmfile = GLM / (b['sample_id'] + '_raw.json')
        glmraw = glmfile.read_text(encoding='utf-8-sig')
        assert checker.check_text(glmraw)['exit_code'] == 0
        glma = json.loads(glmraw)
        predictions['glm'].append(glma['defect_class'])
        combined['models']['glm'] = {'prediction': glma['defect_class'], 'matches_label': glma['defect_class'] == t,
                                    'raw_record': glmfile.relative_to(ROOT).as_posix(), 'answer': glma}
        source_hashes[glmfile.relative_to(ROOT).as_posix()] = sha(glmfile.read_bytes())
        output_rows.append(combined)
    assert len(records) == 54 and len(output_rows) == 18
    candidates = {c: score(truth, p) for c, p in predictions.items()}
    for c in ['deepseek', 'kimi', 'qwen']:
        rows = [records[c, b['item_id']][1] for b in blind]
        usage = {k: sum(r['usage'][k] for r in rows) for k in ['prompt_tokens', 'completion_tokens', 'total_tokens']}
        usage['reasoning_tokens'] = sum(r['usage']['completion_tokens_details']['reasoning_tokens'] for r in rows)
        candidates[c].update({'usage': usage, 'median_seconds': statistics.median(r['elapsed_seconds'] for r in rows),
            'completion_min': min(r['usage']['completion_tokens'] for r in rows),
            'completion_max': max(r['usage']['completion_tokens'] for r in rows),
            'completion_over_8192': [r['item_id'] for r in rows if r['usage']['completion_tokens'] > 8192],
            'returned_models': sorted({r['returned_model'] for r in rows}),
            'image_tokens_reported': sorted({r['usage'].get('prompt_tokens_details', {}).get('image_tokens')
                    for r in rows if r['usage'].get('prompt_tokens_details', {}).get('image_tokens') is not None})})
    pairwise = {}
    for a, b in itertools.combinations(predictions, 2):
        ao = sum(x == t and y != t for x, y, t in zip(predictions[a], predictions[b], truth))
        bo = sum(y == t and x != t for x, y, t in zip(predictions[a], predictions[b], truth))
        n = ao + bo
        p = min(1.0, 2*sum(math.comb(n, k) for k in range(min(ao, bo)+1))/2**n) if n else 1.0
        pairwise[a + '_vs_' + b] = {'a_only_correct': ao, 'b_only_correct': bo, 'exact_mcnemar_p_descriptive': p}
    unanimous = [r for r in output_rows if len({r['models'][c]['prediction'] for c in ['deepseek', 'kimi', 'qwen']}) == 1]
    wrong_unanimous = [r['item_id'] for r in unanimous if not r['models']['kimi']['matches_label']]
    probe_file = SOURCE / 'text_only_probe.json'
    probes = json.loads(probe_file.read_text(encoding='utf-8'))['results']
    source_hashes[probe_file.relative_to(ROOT).as_posix()] = sha(probe_file.read_bytes())
    probe_usage = {c: {'records': sum(r['candidate'] == c for r in probes),
                      'total_tokens': sum(r['usage']['total_tokens'] for r in probes if r['candidate'] == c)}
                   for c in ['deepseek', 'qwen']}
    ds = candidates['deepseek']['usage']
    pricing = json.loads((SOURCE / 'pricing_verified.json').read_text(encoding='utf-8'))['candidates']['deepseek']
    peak = (ds['prompt_tokens'] * pricing['price_per_mtok']['input_cache_miss']['peak']
            + ds['completion_tokens'] * pricing['price_per_mtok']['output']['peak']) / 1e6
    q = candidates['qwen']['usage']
    image_tokens = sum(records['qwen', b['item_id']][1]['usage']['prompt_tokens_details']['image_tokens'] for b in blind)
    totals = sum(candidates[c]['usage']['total_tokens'] for c in ['deepseek', 'kimi', 'qwen'])
    reasoning = sum(candidates[c]['usage']['reasoning_tokens'] for c in ['deepseek', 'kimi', 'qwen'])
    report = {'author': 'Codex', 'date': '2026-10-02', 'kind': 'development_selection_not_teacher_certification',
        'successful_image_records': len(records), 'format_pass': len(records), 'finish_stop': len(records),
        'image_attempt_records': record_count, 'failure_records': failures,
        'text_probe_records': len(probes), 'minimum_evidenced_requests': record_count + len(probes),
        'report_claimed_requests': 59, 'unreconciled_claimed_requests': 59 - record_count - len(probes),
        'candidates': candidates, 'pairwise_descriptive': pairwise,
        'three_unanimous': len(unanimous), 'three_disagree': len(output_rows) - len(unanimous),
        'three_unanimous_wrong_items': wrong_unanimous,
        'three_teacher_total_tokens': totals, 'three_teacher_reasoning_tokens': reasoning,
        'reasoning_share_of_total': reasoning/totals, 'probe_usage': probe_usage,
        'minimum_evidenced_total_tokens': totals + sum(r['usage']['total_tokens'] for r in probes),
        'deepseek_main_peak_cache_miss_estimate_usd': peak, 'actual_billed_cost': 'not provided',
        'qwen_image_share_of_main_total_tokens': image_tokens/q['total_tokens'],
        'billing_free_quota_remaining': 'not independently verified',
        'image_token_difference_mechanism': 'not established',
        'note': '一致率是与原类别标签一致，不是人审视觉准确率；Wilson区间/配对检验仅为本开发集描述，不能外推。GLM是客户端累积上下文，与独立API有协议差异。payload指纹验证的是规范化对象，不是实际发送字节。',
        'source_sha256': source_hashes}
    return report, output_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    report, rows = audit()
    outputs = {'独立复算.json': (json.dumps(report, ensure_ascii=False, indent=2)+'\n').encode('utf-8'),
               '逐图对照.jsonl': ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows).encode('utf-8')}
    if args.write:
        for name, data in outputs.items():
            p = HERE / name
            if p.exists() and not (args.resume and p.read_bytes() == data):
                raise ValueError('拒绝覆盖：' + name)
        for name, data in outputs.items():
            p = HERE / name
            if not p.exists(): p.write_bytes(data)
    public = {k: v for k, v in report.items() if k not in ['source_sha256']}
    print(json.dumps(public, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
