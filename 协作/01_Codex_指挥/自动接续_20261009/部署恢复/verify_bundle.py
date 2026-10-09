"""CPU-only package/hash/version validation. Does not import torch or call APIs."""
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError
import argparse
import hashlib
import json

HERE = Path(__file__).resolve().parent
EXPECTED_D = 'babf37ecced78d37f56cc6cdea4d4dd839bce9b8324ef8caf4c7bb7b45fb018b'
CORE = {'torch': '2.8.0', 'transformers': '5.16.1', 'peft': '0.20.0', 'ms-swift': '4.5.3'}

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--adapter')
    ap.add_argument('--files-only', action='store_true')
    args = ap.parse_args()
    manifest = json.loads((HERE / 'bundle_manifest.json').read_text(encoding='utf-8'))
    errors = []
    for r in manifest['files']:
        p = HERE / r['path']
        if not p.is_file() or digest(p) != r['sha256']:
            errors.append('file hash mismatch: ' + r['path'])
    if not args.files_only:
        for package, expected in CORE.items():
            try:
                actual = version(package)
            except PackageNotFoundError:
                errors.append('missing package: ' + package)
                continue
            if actual.split('+')[0] != expected:
                errors.append('package version mismatch: ' + package)
        for package in ('Pillow', 'numpy'):
            try:
                version(package)
            except PackageNotFoundError:
                errors.append('missing package: ' + package)
        if not args.adapter:
            errors.append('--adapter is required for runtime validation')
        else:
            adapter = Path(args.adapter)
            weight = adapter / 'adapter_model.safetensors'
            if not weight.is_file() or digest(weight) != EXPECTED_D:
                errors.append('D adapter weight hash mismatch')
            if not (adapter / 'adapter_config.json').is_file():
                errors.append('missing adapter_config.json')
    print(json.dumps({'files': len(manifest['files']), 'ok': not errors, 'errors': errors}, ensure_ascii=False))
    raise SystemExit(bool(errors))

if __name__ == '__main__':
    main()
