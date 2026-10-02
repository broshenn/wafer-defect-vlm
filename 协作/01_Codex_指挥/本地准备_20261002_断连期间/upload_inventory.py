"""记录现有公开图片/manifest库存指纹，未传输、不复制图片。"""
import argparse
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
META=['manifest.jsonl','manifest_trainval.jsonl','manifest.parquet','manifest_summary.json','source_stats.json']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--write',action='store_true');ap.add_argument('--resume',action='store_true');a=ap.parse_args()
    files=sorted((ROOT/'data/images').glob('*.png'))+[(ROOT/'data'/n) for n in META]
    assert len(files)==5909, '库存变化；先核对而不是静默改变上传范围'
    rows=[]
    for p in files:
        assert p.is_file() and not p.is_symlink()
        b=p.read_bytes()
        rows.append({'relative_path':p.relative_to(ROOT).as_posix(),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
    if a.write:
        path=HERE/'public_data_upload_inventory.jsonl'
        data=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows).encode('utf-8')
        if path.exists():
            if not(a.resume and path.read_bytes()==data):raise ValueError('拒绝覆盖库存指纹')
        else:path.write_bytes(data)
    print(json.dumps({'png_files':5904,'metadata_files':5,'files':len(rows),'total_bytes':sum(r['bytes'] for r in rows),'uploaded':False},ensure_ascii=False))

if __name__=='__main__':main()
