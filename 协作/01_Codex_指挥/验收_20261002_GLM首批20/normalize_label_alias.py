"""Codex：只规范化明确的类别拼写别名，原答保持不变；不联网。"""
import argparse
import hashlib
import json
from pathlib import Path
import re

ALIASES = {'Near-full': 'Near_full', 'Edge-Loc': 'Edge_Loc', 'Edge-Ring': 'Edge_Ring'}
VERSION = 'class_alias_v1_20261002'


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('重复JSON键：' + key)
        result[key] = value
    return result


def normalize(raw_bytes):
    answer = json.loads(raw_bytes.decode('utf-8-sig'), object_pairs_hook=unique_object)
    if not isinstance(answer, dict):
        raise ValueError('顶层不是对象')
    changes = []
    original = answer.get('defect_class')
    if isinstance(original, str) and original in ALIASES:
        answer['defect_class'] = ALIASES[original]
        changes.append({'field': 'defect_class', 'before': original, 'after': answer['defect_class']})
    return answer, changes


def self_test():
    # 明确别名可恢复；语义未知/非枚举不会被模糊改成合法类别。
    assert normalize(b'{"defect_class":"Near-full"}')[0]['defect_class'] == 'Near_full'
    assert normalize(b'{"defect_class":"Near_full"}')[1] == []
    assert normalize(b'{"defect_class":"near-full"}')[0]['defect_class'] == 'near-full'
    assert normalize(b'{"defect_class":"unknown"}')[0]['defect_class'] == 'unknown'
    assert normalize(b'{"defect_class":"Edge-Loc","caption_zh":"keep me"}')[0]['caption_zh'] == 'keep me'
    for bad in [b'{"defect_class":"Loc","defect_class":"Center"}', b'{"defect_class":', b'[]']:
        try:
            normalize(bad)
        except (ValueError, json.JSONDecodeError):
            continue
        raise AssertionError('坏输入被静默修好')
    print('PASS: 8 alias/identity/unknown/duplicate/truncation tests')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not all([args.raw, args.out, args.report]):
        parser.error('必须提供--raw/--out/--report')
    paths = [p.resolve() for p in [args.raw, args.out, args.report]]
    if len(set(paths)) != 3 or args.out.exists() or args.report.exists():
        raise ValueError('拒绝覆盖原答或既有副本/记录')
    raw = args.raw.read_bytes()
    answer, changes = normalize(raw)
    output = (json.dumps(answer, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    warnings = []
    for key in ['morphology', 'caption_zh', 'uncertainty']:
        value = answer.get(key)
        if isinstance(value, str) and re.search(r'[一二三四五六七八九十百千]+几?(?:颗|枚|格)|百分之[一二三四五六七八九十百]+', value):
            warnings.append({'field': key, 'kind': 'Chinese_quantity_requires_content_review'})
    report = {'version': VERSION, 'raw_sha256': hashlib.sha256(raw).hexdigest(),
              'normalized_sha256': hashlib.sha256(output).hexdigest(), 'changes': changes,
              'content_review_warnings': warnings,
              'note': '仅类别字符串规范化；副本格式通过不代表事实通过，未经验证的中文数值另行审核。'}
    # 目录由执行者事先建立，避免意外扩展输出位置。
    args.out.write_bytes(output)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__': main()
