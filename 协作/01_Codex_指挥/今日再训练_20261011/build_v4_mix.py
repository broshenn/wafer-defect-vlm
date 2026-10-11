"""Prepare the v4 supervised objective mix without changing v3 sources.

Server-side use for training data; local --dry-run is an inspectable preview.
One frozen wafer group: three original six-field answers, one geometry answer.
No teacher requests, new wafers, rotations, reward changes or test-label edits.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

SEED = 3407
FIELDS = {'defect_class','coverage_band','red_mass_zones','angular_type','clock_sectors','uncertainty'}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--out',type=Path)
    ap.add_argument('--images-dir',type=Path)
    ap.add_argument('--dry-run',action='store_true')
    ap.add_argument('--smoke',type=int,help='preview first N frozen image groups, not N dialogue rows')
    a=ap.parse_args()
    raw=a.source.read_bytes(); before=hashlib.sha256(raw).hexdigest()
    rows=[json.loads(s) for s in raw.decode('utf-8').splitlines() if s.strip()]
    assert len(rows)==1024 and len({r['sample_id'] for r in rows})==256
    groups=[rows[i:i+4] for i in range(0,len(rows),4)]
    original_ids=[g[0]['sample_id'] for g in groups]
    if a.smoke:
        assert a.smoke>0
        groups=groups[:a.smoke]
    output=[]; targets=[]
    for group in groups:
        assert len({json.dumps(r,ensure_ascii=False,sort_keys=True) for r in group})==1
        base=group[0]
        assert [m['role'] for m in base['messages']]==['user','assistant']
        ref=json.loads(base['messages'][1]['content']); assert set(ref)==FIELDS
        prompt=base['messages'][0]['content']
        coverage=next(line for line in prompt.splitlines() if line.startswith('coverage_band：'))
        zones=next(line for line in prompt.splitlines() if line.startswith('red_mass_zones：'))
        # Retain the precise frozen definitions instead of rephrasing thresholds.
        geo_prompt=('＜IMAGE_PLACEHOLDER＞\n这是公开晶圆 BIN 图，黑色为片外，绿色为合格格点，红色为失效格点。\n'
                    '只输出一个 JSON 对象，严格包含 coverage_band 和 red_mass_zones 两个字段。\n'
                    +coverage+'\n'+zones+'\n'
                    '不写 Markdown、推理、类别、角向、Caption、失效原因或额外字段。')
        geo_prompt=geo_prompt.replace('＜IMAGE_PLACEHOLDER＞','<image>')
        joint=[copy.deepcopy(base) for _ in range(3)]
        geo=copy.deepcopy(base)
        geo['messages'][0]['content']=geo_prompt
        target={k:ref[k] for k in ['coverage_band','red_mass_zones']}
        geo['messages'][1]['content']=json.dumps(target,ensure_ascii=False,separators=(',',':'))
        assert geo['images']==base['images'] and geo['sample_id']==base['sample_id']
        output.extend(joint+[geo]); targets.append(target)
    if a.images_dir:
        for r in output:r['images']=[str(a.images_dir/Path(x).name) for x in r['images']]
    assert [r['sample_id'] for r in output[::4]]==original_ids[:len(groups)]
    assert a.source.read_bytes()==raw
    encoded=('\n'.join(json.dumps(r,ensure_ascii=False,separators=(',',':')) for r in output)+'\n').encode('utf-8')
    report={'seed':SEED,'source_sha256':before,'output_sha256':hashlib.sha256(encoded).hexdigest(),
            'image_groups':len(groups),'dialogue_rows':len(output),
            'joint_rows':len(groups)*3,'geometry_rows':len(groups),
            'new_unique_wafers':0,'original_group_order_preserved':True,
            'full_answer_reference_unchanged':True,'geometry_reference_is_original_subset':True,
            'geometry_prompt_sha256':hashlib.sha256(geo_prompt.encode('utf-8')).hexdigest(),
            'coverage_zero_or_unknown_targets':sum(r['coverage_band'] in ['zero','unknown'] for r in targets),
            'geometry_example':{'question':geo_prompt,'answer':targets[0]},'dry_run':a.dry_run}
    if not a.dry_run:
        assert a.out is not None and a.out.resolve()!=a.source.resolve()
        assert a.smoke is None,'Smoke is a preview; do not accidentally write a partial formal dataset'
        a.out.parent.mkdir(parents=True,exist_ok=True)
        if a.out.exists():
            assert a.out.read_bytes()==encoded,'Refuse to overwrite a different existing artifact'
        else:a.out.write_bytes(encoded)
        manifest=a.out.with_suffix('.manifest.json')
        data=(json.dumps(report,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
        if manifest.exists():assert manifest.read_bytes()==data
        else:manifest.write_bytes(data)
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
