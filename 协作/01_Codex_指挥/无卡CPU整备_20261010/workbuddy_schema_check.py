"""Format-only validation for real-image description candidates. Not image truth."""
import argparse
import hashlib
import json
import re
from pathlib import Path
from core_v3 import parse

FIELDS = ['morphology_zh', 'main_location_zh', 'location_clock', 'orientation_clock',
          'secondary_observations', 'caption_zh', 'uncertainty']


def check(text):
    errors, warnings = [], []
    try:
        obj = parse(text)
    except (ValueError, TypeError) as exc:
        return {'schema_ok': False, 'errors': [type(exc).__name__], 'warnings': [], 'content_verified': False}
    if set(obj) != set(FIELDS):
        errors.append('fields')
    for key in ['morphology_zh', 'main_location_zh', 'caption_zh', 'uncertainty']:
        if not isinstance(obj.get(key), str) or (key != 'uncertainty' and not obj[key].strip()):
            errors.append(key)
        elif len(obj[key]) > 300:
            errors.append(key + '_too_long')
    for key in ['location_clock', 'orientation_clock']:
        sectors = obj.get(key)
        valid = (isinstance(sectors, list) and all(type(s) is int and 1 <= s <= 12 for s in sectors)
                 and len(sectors) == len(set(sectors)))
        if not valid:
            errors.append(key)
        elif key == 'location_clock' and len(sectors) not in [0, 1]:
            errors.append('location_clock_count')
        elif key == 'orientation_clock' and (len(sectors) not in [0, 2] or
                 (len(sectors) == 2 and (sectors[0] - sectors[1]) % 12 != 6)):
            errors.append('orientation_clock_opposed')
    secondary = obj.get('secondary_observations')
    if not isinstance(secondary, list) or any(not isinstance(s, str) or len(s) > 200 for s in secondary):
        errors.append('secondary_observations')
    texts = [obj.get(k, '') for k in ['morphology_zh','main_location_zh','caption_zh','uncertainty']]
    texts += secondary if isinstance(secondary, list) else []
    if any(isinstance(s, str) and re.search(r'\d|百分|比例|半径|工艺|根因|蚀刻|污染|应力', s) for s in texts):
        warnings.append('numeric_or_causal_claim_requires_content_review')
    return {'schema_ok': not errors, 'errors': errors, 'warnings': warnings, 'content_verified': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('raw')
    args = parser.parse_args()
    p = Path(args.raw)
    result = check(p.read_text(encoding='utf-8'))
    result['raw_sha256'] = hashlib.sha256(p.read_bytes()).hexdigest()
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result['schema_ok'] else 8


if __name__ == '__main__':
    raise SystemExit(main())
