"""Prepare reused val/test direction diagnostics; never describe as new blind gold."""
from collections import Counter
import json

from build_cpu_packet import read_rows, digest, stable_write, rows_text, json_text, OLD, PROTECTED
from core_v3 import HERE, REPO, load_geometry, measure_v3, reference


def main():
    manifest=read_rows(REPO/'data/manifest.jsonl')
    geom={r['sample_id']:r for r in read_rows(OLD/'几何特征表.jsonl')}
    old=load_geometry()
    train=read_rows(HERE/'pools/adapt.jsonl')+read_rows(HERE/'pools/rl.jsonl')
    train_lots={r['lot_name'] for r in train}
    train_hashes={r['image_sha256'] for r in train}
    rows, seen_hashes=[],set()
    considered=[]
    for row in manifest:
        if row['split'] not in ['val','test']:
            continue
        g=geom[row['sample_id']]
        if g['direction_type'] not in ['single','axis']:
            continue
        sid=row['sample_id'];p=REPO/'data/images'/f'{sid}.png'
        if digest(p)!=g['png_sha256']:
            raise RuntimeError('PNG fingerprint mismatch')
        rec=old.recover_matrix(p,hint_hw=tuple(row['matrix_shape']))
        if not rec['roundtrip_validated']:
            raise RuntimeError('PNG recovery failed')
        current=measure_v3(rec['matrix'])
        ref=reference(current,row['failure_type'])
        considered.append({'sample_id':sid,'v2_angular':g['direction_type'],'v3_angular':ref['angular_type']})
        if ref['angular_type'] not in ['single','opposed']:
            continue
        if row['lot_name'] in train_lots or g['png_sha256'] in train_hashes:
            raise RuntimeError('Direction diagnostic leaks into training pools')
        if g['png_sha256'] in seen_hashes:
            continue
        seen_hashes.add(g['png_sha256'])
        rows.append({'sample_id':sid,'image_path':f'data/images/{sid}.png','image_sha256':g['png_sha256'],
                     'lot_name':row['lot_name'],'original_split':row['split'],'reference':ref,
                     'prior_evaluation_contact':'may_have_been_seen; intentionally reused diagnostic; not new blind test'})
    stable_write(HERE/'direction_diagnostic.jsonl',rows_text(rows))
    report={'status':'reused_development_diagnostic_not_final_benchmark','selected':len(rows),
            'angular':dict(Counter(r['reference']['angular_type'] for r in rows)),
            'considered_and_recomputed':considered,'human_review':'not_run',
            'GPU_predictions':'not_run','train_lot_PNG_intersections':0,
            'new_confirmation120_positive_direction_N':0,
            'limitation':'New confirmation candidate pool has no positive angular cases. Do not infer direction ability from its nondirectional score. Reused diagnostic is reported separately.'}
    stable_write(HERE/'方向诊断验收.json',json_text(report))
    print(json.dumps({'direction_diagnostic_images':len(rows),'angular':report['angular']},ensure_ascii=False))


if __name__=='__main__':
    main()
