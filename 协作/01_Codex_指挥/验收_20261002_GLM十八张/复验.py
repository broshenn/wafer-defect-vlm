"""Codex: 只读重验交付指纹/格式，在自有目录保存验收证据；不导出训练数据。"""
from pathlib import Path
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PREP = ROOT / '协作/02_ClaudeCode_实操/口径修正与准备_20261002-021207'
BATCH = ROOT / '协作/03_ZCode_标注/20261002-024921_十八张开发试标'
INPUTS = {
    PREP / 'blind_inputs.jsonl': '70d48e6f5a7ab3c14f4c3d675e94adc11a9cb518f814120fadb8b2e802120b20',
    ROOT / '协作/01_Codex_指挥/prompt_image_only_v2_20261002.txt': '7df73031ee1a30dfb018dbb2417f505ea9ff139280307bce2934de604119954a',
    PREP / 'check_answer_v3.py': '794a254e005eaa232362c539817abde98a1c2406ecb52574b9e9edb35303ec55',
}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def loadl(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8-sig').splitlines() if x.strip()]

def strict_json(raw):
    def pairs(items):
        obj = {}
        for k, v in items:
            if k in obj:
                raise ValueError('duplicate key: ' + k)
            obj[k] = v
        return obj
    def invalid(v):
        raise ValueError('non-finite JSON: ' + v)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)

def main():
    target = HERE / '指纹与格式复验.json'
    if target.exists():
        raise SystemExit('Existing evidence retained; choose a new output directory to rerun.')
    inputs = [{'path': str(p.relative_to(ROOT)), 'sha256': sha(p), 'expected': h, 'match': sha(p) == h} for p, h in INPUTS.items()]
    blind = loadl(PREP / 'blind_inputs.jsonl')
    tsv = list(csv.DictReader((BATCH / 'consolidated_fingerprints.tsv').open(encoding='utf-8-sig'), delimiter='\t'))
    byid = {r['sample_id']: r for r in tsv}
    checks = []
    answers = []
    fields = {'defect_class','morphology','radial_zone','clock_direction','extent_r','caption_zh','uncertainty'}
    for b in blind:
        sid = b['sample_id']
        raw = BATCH / (sid + '_raw.json')
        rawtext = raw.read_text(encoding='utf-8-sig')
        obj = strict_json(rawtext)
        pngsha, rawsha = sha(Path(b['image_path'])), sha(raw)
        proc = subprocess.run([sys.executable, '-X', 'utf8', '-B', str(PREP / 'check_answer_v3.py'), str(raw), '--json'], capture_output=True, text=True, encoding='utf-8')
        record = json.loads(proc.stdout)
        stored_exit = int((BATCH / (sid + '_check_exit.txt')).read_text(encoding='utf-8-sig').strip())
        hashrecord = (BATCH / (sid + '_fingerprints.txt')).read_text(encoding='utf-8-sig')
        itemhash = (BATCH / (b['item_id'] + '_fingerprints.txt')).read_text(encoding='utf-8-sig')
        row = {'item_id': b['item_id'], 'sample_id': sid, 'png_sha256': pngsha, 'raw_sha256': rawsha,
               'png_matches_blind': pngsha == b['image_sha256'],
               'png_matches_tsv': pngsha == byid[sid]['png_sha256'],
               'raw_matches_tsv': rawsha == byid[sid]['raw_answer_sha256'],
               'fingerprint_records_match': rawsha in hashrecord and pngsha in itemhash,
               'strict_json_and_seven_fields': isinstance(obj, dict) and set(obj) == fields,
               'checker_exit': proc.returncode, 'stored_exit': stored_exit,
               'stored_stdout_matches_current': json.loads((BATCH / (sid + '_check_stdout.txt')).read_text(encoding='utf-8-sig')) == record,
               'stderr_empty': proc.stderr == '' and (BATCH / (sid + '_check_stderr.txt')).stat().st_size == 0,
               'checker_result': record}
        row['passed'] = all(row[k] for k in ['png_matches_blind','png_matches_tsv','raw_matches_tsv','fingerprint_records_match','strict_json_and_seven_fields','stored_stdout_matches_current','stderr_empty']) and proc.returncode == stored_exit == int(byid[sid]['check_exit']) == 0
        checks.append(row)
        answers.append(obj)
    summary = {'planned':18,'blind_rows':len(blind),'raw_files':len(list(BATCH.glob('*_raw.json'))),
               'tsv_rows':len(tsv),'unique_ids':len({b['sample_id'] for b in blind}),
               'input_hashes_pass':all(x['match'] for x in inputs), 'rows_pass':sum(x['passed'] for x in checks),
               'unknown':sum(x['defect_class']=='unknown' for x in answers),'all_extent_null':all(x['extent_r'] is None for x in answers),
               'prediction_counts':dict(Counter(x['defect_class'] for x in answers)),
               'batch_start':(BATCH/'batch_start_time.txt').read_text().strip(),
               'batch_end':(BATCH/'batch_end_time.txt').read_text().strip(),
               'backend_and_calls_verified':False,'human_review':False}
    result = {'summary':summary,'inputs':inputs,'rows':checks,'visual_observations_sha256':sha(HERE/'视觉先行观察.jsonl')}
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))
    return 0 if summary['input_hashes_pass'] and summary['rows_pass'] == summary['unique_ids'] == summary['tsv_rows'] == summary['raw_files'] == 18 else 1

if __name__=='__main__':
    sys.exit(main())
