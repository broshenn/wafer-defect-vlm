#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v3 奖励：我的复核修复版（**不改 Codex 的 core_v3.py**）。

源件：`协作/01_Codex_指挥/无卡CPU整备_20261010/core_v3.py`（我只读、不写）。
本文件保持原件的结构与权重（事实 0.9 + 格式 0.1；类 .3/覆盖 .2/区带 .2/角向 .2），
只修**两处经实测复现的缺陷**，并把修复写成可复跑的合成反例。

## 修复 A：`parse()` 不认"完整空 think 包装"，会把真实原答全部判为格式错误

- 实测：起点修复轮 **360/360** 原答都带 `<think>\n\n</think>\n\n` 前缀（v2 轮同型）。
- 而 v2 的适配目标**同样是裸 JSON**，训练出来的模型仍然带前缀 —— 说明"用裸 JSON 训练"
  并不能可靠地去掉这个包装。
- 后果：若沿用原件的 `parse()`，**每一次 GRPO 生成都会解析失败**，事实分全 0、格式分 0，
  奖励恒为 0，RL 无信号；评测也会显示"格式全灭"。
- 处置：`parse()` 先剥掉**完整空**的 think 包装（非空 think 不放过），与原 v2 冻结口径一致；
  **同时保留 `parse_raw_strict()`**，让"原始全文严格 JSON"这个口径仍可单独统计与报告，
  两个口径分开、不互相遮盖。

## 修复 B："角向答 unknown 但扇区为空"白拿 0.4 的扇区分

- 原件：`angular = .6*(类型对) + .4*(扇区集合相等)`。
- 当参考是 `nondirectional`（扇区为空）时，模型回 `unknown` + `[]`，
  扇区集合**恰好相等** → 白拿 0.4（实测复现：reward 0.880 而只错角向一项；
  全 unknown 策略也能拿 0.180）。
- 处置：扇区分要求模型**确实给了方向类型**（非 unknown）才计入：
  `sectors_ok = (obj_type != 'unknown') and set(obj) == set(ref)`。
  语义：**没回答不算答对**。参考已知时答 unknown 得 0，这条在 v2 就写进了契约，此处补齐。

## 明确不动的东西

- 不放松任何阈值（R1/R2、覆盖档位、区带 10%/2 格、扇区 ±3°）。
- 不改题面、不改已发布 WorkBuddy84 依赖的字段名与 hash。
- 不重写 Codex 或历史文件；本文件是**另存的复核副本**。
"""

from __future__ import annotations

import json
import math
import re

CLASSES = ['Center', 'Donut', 'Edge_Loc', 'Edge_Ring', 'Loc', 'Near_full', 'Random',
           'Scratch', 'none']
COVERAGE = ['zero', 'under_5pct', '5_to_under_20pct', '20_to_under_50pct',
            'half_or_more', 'unknown']
ANGULAR = ['single', 'opposed', 'nondirectional', 'unknown']
ZONES = ['center', 'middle', 'edge']
FIELDS = ['defect_class', 'coverage_band', 'red_mass_zones', 'angular_type',
          'clock_sectors', 'uncertainty']

# ── 修复 A：答案信封（与 v2 冻结口径一致）──────────────────────────────
EMPTY_THINK = re.compile(r'^\s*<think\b[^>]*>\s*</think\s*>\s*', re.S)
FENCE = re.compile(r'^\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*$', re.S)


def strip_envelope(text: str) -> str:
    s = EMPTY_THINK.sub('', text.strip())
    m = FENCE.match(s)
    if m:
        s = m.group(1).strip()
    return s


def _pairs(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError('duplicate_key')
        obj[key] = value
    return obj


def _constant(_):
    raise ValueError('non_finite')


def parse_raw_strict(text: str) -> dict:
    """**原始全文严格 JSON**：不剥任何包装。用于单独报告"裸 JSON"口径。"""
    if not isinstance(text, str):
        raise ValueError('answer_not_string')
    obj = json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant)
    if not isinstance(obj, dict):
        raise ValueError('answer_not_object')
    return obj


def parse(text: str) -> dict:
    """评测/奖励用：剥掉**完整空** think 包装与单个围栏后再解析。"""
    if not isinstance(text, str):
        raise ValueError('answer_not_string')
    return parse_raw_strict(strip_envelope(text))


def schema_errors(obj: dict) -> list[str]:
    errors = []
    if not isinstance(obj, dict):
        return ['not_object']
    if set(obj) != set(FIELDS):
        errors.append('fields')
    if obj.get('defect_class') not in CLASSES + ['unknown']:
        errors.append('class')
    if obj.get('coverage_band') not in COVERAGE:
        errors.append('coverage')
    if obj.get('angular_type') not in ANGULAR:
        errors.append('angular')
    z = obj.get('red_mass_zones')
    if z != 'unknown' and (not isinstance(z, list) or any(x not in ZONES for x in z)
                           or len(z) != len(set(z))):
        errors.append('zones')
    sectors = obj.get('clock_sectors')
    valid_sectors = (isinstance(sectors, list) and
                     all(type(x) is int and 1 <= x <= 12 for x in sectors) and
                     len(sectors) == len(set(sectors)))
    if not valid_sectors:
        errors.append('sectors')
    else:
        a = obj.get('angular_type')
        if ((a == 'single' and len(sectors) != 1) or
            (a == 'opposed' and (len(sectors) != 2 or (sectors[0] - sectors[1]) % 12 != 6)) or
            (a in ['nondirectional', 'unknown'] and sectors)):
            errors.append('angular_sector_consistency')
    if not isinstance(obj.get('uncertainty'), str) or len(obj.get('uncertainty', '')) > 200:
        errors.append('uncertainty')
    return errors


def score(text: str, ref: dict) -> dict:
    """单条打分。权重与原件一致；**修复 B 在这里**。"""
    if schema_errors(ref) or ref['defect_class'] == 'unknown':
        raise ValueError('invalid_reference')
    weights = {'class': .3, 'coverage': .2, 'zones': .2, 'angular': .2}
    if ref['coverage_band'] == 'unknown':
        weights['coverage'] = 0
    if ref['red_mass_zones'] == 'unknown':
        weights['zones'] = 0
    if ref['angular_type'] == 'unknown':
        weights['angular'] = 0
    try:
        obj = parse(text)
        errors = schema_errors(obj)
    except (ValueError, TypeError) as exc:
        obj, errors = {}, [type(exc).__name__]
    parts = dict.fromkeys(weights, 0.)
    if not errors:
        parts['class'] = float(obj['defect_class'] == ref['defect_class'])
        parts['coverage'] = float(obj['coverage_band'] == ref['coverage_band'])
        parts['zones'] = float(isinstance(obj['red_mass_zones'], list) and
                               isinstance(ref['red_mass_zones'], list) and
                               set(obj['red_mass_zones']) == set(ref['red_mass_zones']))
        type_ok = obj['angular_type'] == ref['angular_type']
        # 修复 B：没给方向类型（unknown）就不算扇区答对，杜绝空扇区"恰好相等"白拿 0.4
        sectors_ok = (obj['angular_type'] != 'unknown' and
                      set(obj['clock_sectors']) == set(ref['clock_sectors']))
        parts['angular'] = .6 * type_ok + .4 * sectors_ok
    fact = sum(weights[k] * parts[k] for k in weights) / sum(weights.values())
    return {'reward': .9 * fact + .1 * (not errors), 'format_ok': not errors,
            'errors': errors, 'parts': parts, 'eligible_weights': weights}


class WaferV3Reward:
    """ms-swift 奖励回调。`reference_json` 与 `sample_id` 由数据集列提供。

    与 Codex 原件的差别只有一处：内部走本文件的 `parse()`（剥完整空 think 包装）。
    """

    def __init__(self):
        self.audit = []

    def __call__(self, completions, reference_json=None, sample_id=None, **kwargs):
        if reference_json is None or len(reference_json) != len(completions):
            raise RuntimeError('Missing or misaligned frozen reference')
        if sample_id is None or len(sample_id) != len(completions):
            raise RuntimeError('Missing or misaligned sample IDs')
        out = []
        for text, raw_ref, sid in zip(completions, reference_json, sample_id):
            if not isinstance(text, str):
                raise RuntimeError('Completion contract expects string')
            result = score(text, parse(raw_ref))
            stripped = strip_envelope(text)
            self.audit.append({'sample_id': sid, 'response': text,
                               'had_empty_think_prefix': stripped != text.strip(),
                               'raw_parse_ok': _raw_ok(text), **result})
            out.append(result['reward'])
        return out


def _raw_ok(text: str) -> bool:
    """原始全文严格 JSON 口径是否通过（**只作分口径报告，不参与奖励**）。"""
    try:
        parse_raw_strict(text)
        return True
    except Exception:
        return False


def self_test() -> dict:
    cases = []

    def add(name, got, want, extra=None):
        cases.append({'name': name, 'passed': bool(got == want), 'got': got,
                      'want': want, 'extra': extra})

    base_ref = {'defect_class': 'Edge_Ring', 'coverage_band': '20_to_under_50pct',
                'red_mass_zones': ['middle', 'edge'], 'angular_type': 'nondirectional',
                'clock_sectors': [], 'uncertainty': ''}
    good = json.dumps(base_ref, ensure_ascii=False)
    single_ref = {**base_ref, 'angular_type': 'single', 'clock_sectors': [3]}
    single = json.dumps(single_ref, ensure_ascii=False)

    # 修复 A：信封
    r = score('<think>\n\n</think>\n\n' + good, base_ref)
    add('空think前缀：全对得1.0', round(r['reward'], 6), 1.0, r['errors'])
    r = score('<think>我猜是环</think>\n' + good, base_ref)
    add('非空think：不放过', r['format_ok'], False, r['errors'])
    r = score('```json\n' + good + '\n```', base_ref)
    add('单围栏：容错', round(r['reward'], 6), 1.0, r['errors'])
    raised = False
    try:
        parse_raw_strict('<think>\n\n</think>\n\n' + good)
    except ValueError:
        raised = True
    add('raw严格口径：带前缀必须失败（用于分开口径报告）', raised, True)

    # 修复 B：unknown + 空扇区不得白拿
    bad = json.dumps({**base_ref, 'angular_type': 'unknown'}, ensure_ascii=False)
    r = score(bad, base_ref)
    add('角向答unknown：角向分必须为0', r['parts']['angular'], 0.0, r['parts'])
    add('角向答unknown：总分不再0.88', round(r['reward'], 3) < 0.88, True,
        round(r['reward'], 3))
    allunknown = json.dumps({'defect_class': 'unknown', 'coverage_band': 'unknown',
                             'red_mass_zones': 'unknown', 'angular_type': 'unknown',
                             'clock_sectors': [], 'uncertainty': ''}, ensure_ascii=False)
    r = score(allunknown, base_ref)
    add('全unknown：奖励<=0.1', r['reward'] <= 0.1 + 1e-9, True, round(r['reward'], 3))
    r = score(json.dumps(base_ref, ensure_ascii=False), base_ref)
    add('nondirectional答对：仍1.0', round(r['reward'], 6), 1.0, None)
    r = score(bad, single_ref)
    add('参考single答unknown：0分角向', r['parts']['angular'], 0.0, r['parts'])
    r = score('{"defect_class":"Edge_Ring","coverage_band":"20_to_under_50pct",'
              '"red_mass_zones":["middle","edge"],"angular_type":"single",'
              '"clock_sectors":[3],"uncertainty":""}', single_ref)
    add('参考single答对：1.0', round(r['reward'], 6), 1.0, None)
    r = score('{"defect_class":"Edge_Ring","coverage_band":"20_to_under_50pct",'
              '"red_mass_zones":["middle","edge"],"angular_type":"single",'
              '"clock_sectors":[9],"uncertainty":""}', single_ref)
    add('类型对扇区错：只拿0.6类型分', r['parts']['angular'], 0.6, r['parts'])

    # 既有反投机：重复键 / 非有限 / 字段集 / 一致性
    dup = '{"defect_class":"Edge_Ring","defect_class":"Loc"}'
    add('重复键：拒收', score(dup, base_ref)['format_ok'], False, None)
    # （不能用字符串替换构造：json.dumps 默认带空格，替换串不匹配就会静默变成"没改"）
    nan_answer = ('{"defect_class":"Edge_Ring","coverage_band":"20_to_under_50pct",'
                  '"red_mass_zones":["middle","edge"],"angular_type":"nondirectional",'
                  '"clock_sectors":[],"uncertainty":NaN}')
    add('非有限常量：拒收', score(nan_answer, base_ref)['format_ok'], False,
        score(nan_answer, base_ref)['errors'])
    add('字段集不符：拒收',
        score(json.dumps({**base_ref, 'extra': 1}, ensure_ascii=False),
              base_ref)['format_ok'], False, None)
    add('single配两扇区：一致性拒收',
        score(json.dumps({**single_ref, 'clock_sectors': [3, 9]}, ensure_ascii=False),
              single_ref)['format_ok'], False, None)
    add('bool冒充扇区：拒收',
        score(json.dumps({**single_ref, 'clock_sectors': [True]}, ensure_ascii=False),
              single_ref)['format_ok'], False, None)
    # 参考侧
    try:
        score(good, {**base_ref, 'defect_class': 'unknown'})
        add('参考类别unknown：必须抛错', 'no-raise', 'raise')
    except ValueError as e:
        add('参考类别unknown：必须抛错', str(e), 'invalid_reference')

    passed = sum(1 for c in cases if c['passed'])
    return {'total': len(cases), 'passed': passed, 'all_passed': passed == len(cases),
            'cases': cases}


if __name__ == '__main__':
    import sys
    res = self_test()
    for c in res['cases']:
        print(f"  {'OK ' if c['passed'] else 'FAIL'} {c['name']:32s} got={c['got']!r}")
    print(f"通过 {res['passed']}/{res['total']}")
    sys.exit(0 if res['all_passed'] else 1)
