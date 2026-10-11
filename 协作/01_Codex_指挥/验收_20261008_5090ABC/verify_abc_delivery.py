"""离线从原答重算ABC主分、统一格式诊断与按lot配对区间；不调用模型。"""
from pathlib import Path
import argparse
import hashlib
import json
import re
from collections import Counter

import numpy as np

CLASSES = ['Center', 'Donut', 'Edge_Loc', 'Edge_Ring', 'Loc',
           'Near_full', 'Random', 'Scratch', 'none']
EMPTY_THINK = re.compile(r'^\s*<think\b[^>]*>\s*</think\s*>\s*', re.S)
FENCE = re.compile(r'```(?:json)?\s*\n?(.*?)\n?```', re.S)


class InvalidJSON(ValueError):
    pass


def nodup(pairs):
    d = {}
    for k, v in pairs:
        if k in d:
            raise InvalidJSON('duplicate_key')
        d[k] = v
    return d


def nonfinite(_):
    raise InvalidJSON('non_finite')


def parse(raw, diagnostic=False):
    s = EMPTY_THINK.sub('', raw).strip()
    if diagnostic:
        m = FENCE.fullmatch(s)
        if m:
            s = m.group(1).strip()
    try:
        d = json.loads(s, object_pairs_hook=nodup, parse_constant=nonfinite)
    except json.JSONDecodeError:
        return None, 'not_json'
    except InvalidJSON as e:
        return None, str(e)
    if not isinstance(d, dict):
        return None, 'not_object'
    c = d.get('defect_class')
    if not isinstance(c, str) or c not in CLASSES + ['unknown']:
        return None, 'invalid_class'
    return c, 'ok'


def metric(gold, pred):
    # 第10列保留unknown/未解析，计入FN，九个类别分母始终固定。
    cm = np.bincount(gold * 10 + pred, minlength=90).reshape(9, 10)
    tp = np.diag(cm[:, :9]).astype(float)
    den = cm.sum(1) + cm[:, :9].sum(0)
    f = np.divide(2 * tp, den, out=np.zeros(9), where=den != 0)
    return {'n': len(gold), 'correct': int(tp.sum()),
            'accuracy': float(tp.sum() / len(gold)), 'macro_f1': float(f.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path)
    a = ap.parse_args()
    repo = Path(__file__).resolve().parents[3]
    source = repo / '协作/02_ClaudeCode_实操/5090同图ABC_20261008_0100/结果'
    data = {k: [json.loads(l) for l in (source / f'{k}_raw.jsonl').read_text(
        encoding='utf-8-sig').splitlines() if l.strip()] for k in ['S0', 'A', 'B', 'C']}
    sig = [(r['sample_id'], r['label'], r['part']) for r in data['A']]
    assert len(sig) == 102 and len({s[0] for s in sig}) == 102
    assert all([(r['sample_id'], r['label'], r['part']) for r in rows] == sig
               for rows in data.values())
    sel = np.array([i for i, r in enumerate(data['A']) if r['part'] != 'dev18'])
    assert len(sel) == 84
    manifest = {r['sample_id']: r for r in (json.loads(l) for l in
        (repo / 'data/manifest.jsonl').read_text(encoding='utf-8-sig').splitlines() if l.strip())}
    lots = [manifest[sig[i][0]]['lot_name'] for i in sel]
    assert len(set(lots)) == 84, '此脚本lot重抽实现要求本组每lot一图'
    gold = np.array([CLASSES.index(s[1]) for s in sig])
    out = {'source': str(source.relative_to(repo)), 'n': 102, 'val_n': 84,
           'val_unique_lots': 84, 'seed': 3407, 'replicates': 5000,
           'strict': {}, 'diagnostic': {}, 'paired': {}, 'raw_sha256': {},
           'note': '固定九类；单种子开发集条件区间，未调整多重比较，不作普遍优势或等效结论。'}
    preds = {}
    for mode in ['strict', 'diagnostic']:
        for k, rows in data.items():
            ps = [parse(r['raw'], mode == 'diagnostic') for r in rows]
            p = np.array([CLASSES.index(c) if c in CLASSES else 9 for c, _ in ps])
            out[mode][k] = {'all': metric(gold, p), 'val84': metric(gold[sel], p[sel]),
                            'parse_counts': dict(Counter(h for _, h in ps))}
            if mode == 'strict':
                preds[k] = p[sel]
                out['raw_sha256'][k] = hashlib.sha256((source / f'{k}_raw.jsonl').read_bytes()).hexdigest()
    # val84为84个不同lot，重抽lot与重抽图在本数据上一一对应。
    draws = np.random.default_rng(3407).integers(0, 84, size=(5000, 84))
    g = gold[sel]
    for left, right in [('B', 'A'), ('C', 'A'), ('C', 'B')]:
        pl, pr = preds[left], preds[right]
        f_delta = np.array([metric(g[ii], pl[ii])['macro_f1'] -
                            metric(g[ii], pr[ii])['macro_f1'] for ii in draws])
        acc_delta = (pl == g).astype(float) - (pr == g).astype(float)
        acc_draws = acc_delta[draws].mean(1)
        out['paired'][f'{left}-{right}'] = {
            'f1_diff': metric(g, pl)['macro_f1'] - metric(g, pr)['macro_f1'],
            'f1_ci95': np.quantile(f_delta, [.025, .975]).tolist(),
            'accuracy_diff': float(acc_delta.mean()),
            'accuracy_ci95': np.quantile(acc_draws, [.025, .975]).tolist(),
            'win': int((acc_delta > 0).sum()), 'lose': int((acc_delta < 0).sum())}
    text = json.dumps(out, ensure_ascii=False, indent=2)
    if a.output:
        with a.output.open('x', encoding='utf-8', newline='\n') as f:
            f.write(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
