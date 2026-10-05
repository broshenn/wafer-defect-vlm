"""Codex：在自有探针目录验证修正，不改冻结输入或执行方文件。"""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
SOURCE=ROOT/'协作/02_ClaudeCode_实操/离线修正_20261002-202300'
PREP=ROOT/'协作/02_ClaudeCode_实操/GLM训练标注准备_20261002'
RAW=ROOT/'协作/03_ZCode_标注/20261002-192902_batch01_并行20'
TARGET=ROOT/'协作/01_Codex_指挥/验收_20261002_GLM首批20/normalize_label_alias.py'

def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def run(path,args,expected):
    result=subprocess.run([sys.executable,'-X','utf8','-B',str(path),*args],cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
    assert result.returncode==expected,(path.name,result.returncode,result.stderr[-500:])
    return {'script':path.name,'args':args,'exit_code':result.returncode}
class Obj(list):pass
def unique(pairs):
    result={}
    for k,v in pairs:
        if k in result:raise ValueError('重复键:'+k)
        result[k]=v
    return result
def rebuild_independent(node,path=()):
    if isinstance(node,Obj):
        out={}
        for k,v in node:
            if path==('counts',) and k=='failed':
                key='failure_details' if isinstance(v,Obj) else 'failure_count'
            else:key=k
            assert key not in out
            out[key]=rebuild_independent(v,path+(key,))
        return out
    if isinstance(node,list):return [rebuild_independent(x,path) for x in node]
    return node

def main():
    old={'old_checker':PREP/'verify_glm_preparation.py','old_preparer':PREP/'prepare_glm_annotation.py',
         'old_batch_log':RAW/'batch_log.json','normalizer':TARGET,'first_index':PREP/'zcode_batch_01_inputs.json',
         'running_remaining_index':ROOT/'协作/01_Codex_指挥/GLM续标_20261002_无答案输入/inputs.json'}
    before={k:h(v) for k,v in old.items()}
    assert before['first_index']=='641f75212839e5c847edf0d7bb53225460b03f99f4e16c10a1343f47cd97ffe7'
    assert before['running_remaining_index']=='60762f2ee9c1018cc83e5d1ebb167832e0534e2eb57c5cc3b2ac482da59f55e0'
    for n in range(1000):
        probe=HERE/f'probe_{n:03d}'
        if not probe.exists():probe.mkdir();break
    else:raise ValueError('没有新探针目录')
    commands=[];preparer=SOURCE/'prepare_glm_annotation_v2.py'
    commands.append(run(preparer,['--write','--out',str(probe)],0))
    first={p.name:h(p) for p in probe.iterdir() if p.is_file()}
    commands.append(run(preparer,['--write','--out',str(probe)],3))
    assert first=={p.name:h(p) for p in probe.iterdir() if p.is_file()}
    commands.append(run(preparer,['--write','--resume','--out',str(probe)],0))
    assert first=={p.name:h(p) for p in probe.iterdir() if p.is_file()}
    assert h(probe/'batch_01_blind.jsonl')==h(PREP/'batch_01_blind.jsonl')
    raw_fixture=probe/'synthetic_raw.json'
    raw_fixture.write_text(json.dumps({'defect_class':'Near-full','morphology':'保留中文原文','radial_zone':'global',
        'clock_direction':None,'extent_r':None,'caption_zh':'仅作合成边界测试','uncertainty':''},ensure_ascii=False),encoding='utf-8')
    fixture_hash=h(raw_fixture)
    commands.append(run(TARGET,['--raw',str(raw_fixture),'--out',str(raw_fixture),'--report',str(probe/'refused.json')],1))
    assert h(raw_fixture)==fixture_hash and not (probe/'refused.json').exists()
    canon=probe/'synthetic_canonical.json';report=probe/'synthetic_normalization.json'
    commands.append(run(TARGET,['--raw',str(raw_fixture),'--out',str(canon),'--report',str(report)],0))
    out_hash=h(canon)
    commands.append(run(TARGET,['--raw',str(raw_fixture),'--out',str(canon),'--report',str(probe/'refused2.json')],1))
    assert h(raw_fixture)==fixture_hash and h(canon)==out_hash and not (probe/'refused2.json').exists()
    source=json.loads((RAW/'batch_log.json').read_text(encoding='utf-8'),object_pairs_hook=Obj)
    expected=rebuild_independent(source)
    actual=json.loads((SOURCE/'batch_log_no_dupkey.json').read_text(encoding='utf-8'),object_pairs_hook=unique)
    assert actual==expected and actual['counts']['failure_count']==1
    qty=module('qty',SOURCE/'cn_quantity_audit.py')
    results=[qty.audit_file(p) for p in sorted(RAW.glob('worker_*/*_raw.json'))]
    hits=[r for r in results if r.get('**unverified_count_claims**')]
    assert len(results)==19 and len(hits)==2
    assert {Path(r['path']).name for r in hits}=={'wafer_00046030_024_raw.json','wafer_00042308_024_raw.json'}
    after={k:h(v) for k,v in old.items()};assert before==after
    output={'author':'Codex','date':'2026-10-02','status':'PASS','probe_directory':probe.relative_to(ROOT).as_posix(),
        'prepare_write_then_refuse_then_resume':[0,3,0],'all_17_prepared_outputs_byte_stable':True,
        'raw_path_overwrite_refused':True,'existing_copy_overwrite_refused':True,'schema_values_preserved':True,
        'quantity_files_scanned':19,'quantity_files_flagged':2,'quantity_truth_not_checked':True,
        'frozen_hashes_before':before,'frozen_hashes_after':after,'commands':commands,
        'annotation_api_calls':0,'gpu_use':0,'note':'探针是合成/元数据测试，不是新模型标注；提醒不是答案判错。'}
    (HERE/'独立复验.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in output.items() if k!='commands'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
