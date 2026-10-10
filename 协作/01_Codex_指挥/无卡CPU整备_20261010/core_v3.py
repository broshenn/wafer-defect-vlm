"""v3 CPU draft: explicit measurement scope, strict schema and fixed-reference reward.

The v2 measurement algorithm and thresholds are reused without recalibration.
These are program reference labels, not human gold or main-structure captions.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OLD_GEOM = REPO / '协作/02_ClaudeCode_实操/多字段RL_v2_CPU准备_20261009/几何参考_v2.py'
OLD_GEOM_SHA = 'e07ee78486d0f78bef56f0796f71dc13f1aab6a4a3a8562583c260e57c0cbb1e'
CLASSES = ['Center', 'Donut', 'Edge_Loc', 'Edge_Ring', 'Loc', 'Near_full', 'Random', 'Scratch', 'none']
COVERAGE = ['zero', 'under_5pct', '5_to_under_20pct', '20_to_under_50pct', 'half_or_more', 'unknown']
ANGULAR = ['single', 'opposed', 'nondirectional', 'unknown']
ZONES = ['center', 'middle', 'edge']
FIELDS = ['defect_class', 'coverage_band', 'red_mass_zones', 'angular_type', 'clock_sectors', 'uncertainty']
COV_MAP = dict(zip(['none', 'low', 'medium', 'high', 'near_all', 'unknown'], COVERAGE))
ANG_MAP = dict(zip(['single', 'axis', 'none', 'unknown'], ANGULAR))


def load_geometry():
    if hashlib.sha256(OLD_GEOM.read_bytes()).hexdigest() != OLD_GEOM_SHA:
        raise RuntimeError('Pinned v2 geometry source changed')
    spec = importlib.util.spec_from_file_location('_v2_geometry_pinned', OLD_GEOM)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hour_sector(degrees: float) -> int:
    """Hour centers: 0° -> 12, 90° -> 3. Boundaries are ±15°."""
    return int(math.floor(((degrees + 15) % 360) / 30)) % 12 or 12


def hour_boundary_margin(degrees: float) -> float:
    value = (degrees + 15) % 30
    return min(value, 30 - value)


def measure_v3(matrix):
    # Patch a newly loaded module object only; no old source/cache is rewritten.
    geom = load_geometry()
    geom.sector_of_deg = hour_sector
    geom._sector_boundary_margin_deg = hour_boundary_margin
    result = geom.measure_matrix(matrix)
    result['clock_contract'] = 'hour_centers_v3'
    result['definition_version'] = 'wafer_geom_v3_cpu_draft'
    # Three-degree ambiguity tolerance is unchanged, evaluated at correct boundaries.
    if result.get('direction', {}).get('sector_tie'):
        result['direction_type'], result['clock_sectors'] = 'unknown', []
    return result


def reference(g: dict, public_class: str) -> dict:
    if public_class not in CLASSES:
        raise ValueError('Public class invalid')
    if g.get('status') not in ['ok', 'no_red'] or g.get('outline', {}).get('anomalous'):
        raise ValueError('Geometry not eligible')
    if g['direction_type'] in ['single', 'axis'] and g.get('clock_contract') != 'hour_centers_v3':
        raise ValueError('Directional references require v3 hour recomputation')
    # A new vocabulary for the SAME frozen v2 thresholds, not a new measurement.
    return {
        'defect_class': public_class,
        'coverage_band': COV_MAP[g['coverage_level']],
        'red_mass_zones': g['red_mass_zones'],
        'angular_type': ANG_MAP[g['direction_type']],
        'clock_sectors': list(g['clock_sectors']),
        'uncertainty': ('全红点角向分布证据不足' if g['direction_type'] == 'unknown' else ''),
    }


def _pairs(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError('duplicate_key')
        obj[key] = value
    return obj


def _constant(_):
    raise ValueError('non_finite')


def parse(text: str) -> dict:
    if not isinstance(text, str):
        raise ValueError('answer_not_string')
    obj = json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant)
    if not isinstance(obj, dict):
        raise ValueError('answer_not_object')
    return obj


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
    # Mask depends exclusively on the frozen reference, never the prediction.
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
        parts['angular'] = (.6 * (obj['angular_type'] == ref['angular_type']) +
                            .4 * (set(obj['clock_sectors']) == set(ref['clock_sectors'])))
    fact = sum(weights[k] * parts[k] for k in weights) / sum(weights.values())
    return {'reward': .9 * fact + .1 * (not errors), 'format_ok': not errors,
            'errors': errors, 'parts': parts, 'eligible_weights': weights}


class WaferV3Reward:
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
            self.audit.append({'sample_id': sid, 'response': text, **result})
            out.append(result['reward'])
        return out
