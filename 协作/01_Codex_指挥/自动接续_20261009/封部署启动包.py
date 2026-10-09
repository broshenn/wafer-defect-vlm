from pathlib import Path
import hashlib
import json
import shutil
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / '协作/02_ClaudeCode_实操/部署原型_20261009_0120'
DEST = HERE / '部署恢复'
for name in ('wafer_service.py','demo.html','replay_selftest.py','test_cpu_contract.py','prompt_7f.txt','schema_check.py','样本清单.json'):
    shutil.copyfile(SOURCE / name, DEST / name)
(DEST / '图').mkdir(exist_ok=True)
samples = json.loads((SOURCE / '样本清单.json').read_text(encoding='utf-8'))['样本']
assert len(samples) == 36
for rec in samples:
    name = rec['sample_id'] + '.png'
    src = SOURCE / '图' / name
    assert hashlib.sha256(src.read_bytes()).hexdigest() == rec['图sha256']
    shutil.copyfile(src, DEST / '图' / name)
records = [{'path': p.relative_to(DEST).as_posix(), 'bytes': p.stat().st_size,
            'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
           for p in sorted(DEST.rglob('*')) if p.is_file() and p.name != 'bundle_manifest.json']
(DEST / 'bundle_manifest.json').write_text(json.dumps({'files': records}, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
output = HERE / '晶圆描述服务_恢复启动包.zip'
with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(DEST.rglob('*')):
        if p.is_file():
            z.write(p, p.relative_to(DEST).as_posix())
with zipfile.ZipFile(output) as z:
    assert z.testzip() is None
seal = {'file': output.name, 'bytes': output.stat().st_size,
        'sha256': hashlib.sha256(output.read_bytes()).hexdigest(), 'files_including_manifest': len(records)+1,
        'sample_pngs': 36, 'crc_ok': True, 'new_gpu_or_api_calls': 0}
(HERE / '部署启动包验收.json').write_text(json.dumps(seal, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(json.dumps(seal, ensure_ascii=False))
