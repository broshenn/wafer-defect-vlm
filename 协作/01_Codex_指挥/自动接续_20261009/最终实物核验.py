"""只读核验交付包、实际 PDF 与 adapter；新输出写在本脚本目录。"""
from pathlib import Path
import hashlib
import json
import re
import zipfile
import fitz

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
DELIVERY = ROOT / '协作/02_ClaudeCode_实操/自动接续交付_20261009_1247'
DEPLOY = ROOT / '协作/02_ClaudeCode_实操/部署原型_20261009_0120'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

pdf = DELIVERY / '项目经历页.pdf'
doc = fitz.open(pdf)
pdftext = '\n'.join(page.get_text() for page in doc)
(OUT / '实际PDF文本.txt').write_text(pdftext, encoding='utf-8')
doc[0].get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).save(OUT / '实际PDF_打印版.png')
pdfcheck = {'path': str(pdf.relative_to(ROOT)), 'sha256': sha(pdf),
            'pages': len(doc), 'text_chars': len(pdftext),
            'pending_personal_fields': pdftext.count('待补'),
            'page_mm': [[round(p.rect.width * 25.4 / 72, 1), round(p.rect.height * 25.4 / 72, 1)] for p in doc],
            'private_patterns_present': bool(re.search(r'(?i)sk-[a-z0-9]{15,}|connect\.[^\s]+|C:[/\\]Users[/\\]', pdftext))}

zpath = DELIVERY / '公开数据恢复包/晶圆图公开数据恢复包_20261009.zip'
with zipfile.ZipFile(zpath) as z:
    files = [i for i in z.infolist() if not i.is_dir()]
    names = [i.filename for i in files]
    unsafe = [n for n in names if Path(n).is_absolute() or '..' in Path(n).parts
              or re.search(r'(?i)(^|/)(\.env|secrets|id_rsa|id_ed25519)(\.|/|$)', n)]
    images = [n for n in names if n.lower().endswith('.png')]
    zcheck = {'path': str(zpath.relative_to(ROOT)), 'sha256': sha(zpath),
              'bytes': zpath.stat().st_size, 'files': len(files), 'png_files': len(images),
              'unique_png_sha256': len({hashlib.sha256(z.read(n)).hexdigest() for n in images}),
              'crc_bad': z.testzip(), 'unsafe_names': unsafe,
              'has_restore_readme': any(n.endswith('README_RESTORE.md') for n in names)}

back = Path('D:/pycode/晶圆图模型备份_20261009')
manifest = json.loads((ROOT / '协作/02_ClaudeCode_实操/适配器备份_20261009_1336/backup_manifest.json').read_text(encoding='utf-8'))
weights = []
for rec in manifest['文件']:
    path = back / rec['路径']
    weights.append({'path': rec['路径'], 'exists': path.is_file(),
                    'bytes_match': path.is_file() and path.stat().st_size == rec['字节'],
                    'sha256_match': path.is_file() and sha(path) == rec['sha256']})

required = ['wafer_service.py','demo.html','replay_selftest.py','test_cpu_contract.py','prompt_7f.txt','schema_check.py','样本清单.json']
runtime = [{'path': n, 'exists': (DEPLOY / n).is_file(), 'tracked': None} for n in required]
report = {'pdf': pdfcheck, 'data_zip': zcheck, 'adapter_files': weights, 'runtime_files': runtime,
          'scope': '只读实物验收；没有重跑 GPU 或 API；PDF 布局须另行目视核对。'}
(OUT / '最终实物核验.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(json.dumps(report, ensure_ascii=False, indent=2))
