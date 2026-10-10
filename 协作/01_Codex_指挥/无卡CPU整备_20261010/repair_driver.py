"""Remote single-GPU v2 repair supervisor, total wall/device upper bound 50 minutes.

No new v3 training, no paid API, no checkpoint selection. Never kills other tasks.
Write outputs to a new dedicated directory. Executed only after resource/CPU gates.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

HERE = Path(__file__).resolve().parent
OLD = Path('/root/autodl-tmp/ws/v2run')
PYTHON = '/root/miniconda3/bin/python'
BASE_MODEL = '/root/autodl-tmp/ws/models/Qwen3.5-9B-Dmerged'
M0 = Path(os.environ['WAFER_M0_PATH'])
OUT = HERE/'run_fix_s3407'
CAP_S = 50*60


def main():
    OUT.mkdir(exist_ok=False)
    proc = subprocess.run(['/usr/bin/nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    if proc.stdout.strip():
        raise RuntimeError('GPU occupied; not starting repair')
    proc = subprocess.run(['/usr/bin/nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True)
    uuids = [x.strip() for x in proc.stdout.splitlines() if x.strip()]
    if len(uuids) != 1:
        raise RuntimeError('Contract requires rented instance with one GPU')
    env = dict(os.environ,CUDA_VISIBLE_DEVICES=uuids[0],IMAGE_MAX_TOKEN_NUM='256',TOKENIZERS_PARALLELISM='false')
    start = time.monotonic()
    status = {'run_id':'v2_fix_s3407_20261010','status':'running','cap_device_minutes':50,
              'start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'jobs':[],
              'cumulative_historical_device_minutes':'unknown_exact_estimate_1437',
              'total_project_cap_device_minutes':1800,'rental_CNY':'unknown_actual_bill',
              'single_GPU':True,'new_v3_training':False}

    def save():
        status['elapsed_device_min_upper'] = round((time.monotonic()-start)/60,4)
        tmp = OUT/'state.tmp'
        tmp.write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        tmp.replace(OUT/'state.json')

    def run_job(tag, cmd):
        remaining = CAP_S-(time.monotonic()-start)
        if remaining < 90:
            raise RuntimeError('Insufficient remaining repair budget')
        job = {'tag':tag,'start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'running'}
        status['jobs'].append(job);save()
        t = time.monotonic()
        with (OUT/(tag+'.log')).open('xb') as log:
            child = subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                                     env=env,start_new_session=True,cwd=HERE)
            job['pid']=child.pid;save()
            try:
                code = child.wait(timeout=max(1,remaining-15))
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGINT)
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL);child.wait()
                code=124
        job.update(exit_code=code,status='finished' if code==0 else 'failed',
                   end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   device_minutes_upper=round((time.monotonic()-t)/60,4))
        save()
        if code != 0:
            raise RuntimeError(tag+' failed; no automatic GPU retry')

    try:
        for tag, mode in [('M1_fix','sft'),('M2_fix','rl')]:
            destination = OUT/tag
            cmd = [PYTHON,str(HERE/'v2_train_fixed.py'),'--mode',mode,'--steps','60','--out',str(destination),
                   '--model',BASE_MODEL,'--start-adapter',str(M0)]
            run_job(tag,cmd)
            gate = json.loads((destination/'runtime_gate.json').read_text())
            if gate['model_loading_gate']!='passed' or gate['two_step_gate']!='passed':
                raise RuntimeError(tag+' runtime gate failed')
        checkpoints={'M0_actual':M0}
        for tag in ['M1_fix','M2_fix']:
            cfg=json.loads((OUT/tag/'v2_config.json').read_text())
            path=cfg['final_checkpoint']
            if not path or not path.endswith('checkpoint-60'):
                raise RuntimeError('Preselected final checkpoint absent')
            checkpoints[tag]=Path(path)
        for tag, adapter in checkpoints.items():
            remaining=int(CAP_S-(time.monotonic()-start)-20)
            run_job(tag+'_eval',[PYTHON,str(OLD/'v2_eval.py'),'--model',BASE_MODEL,'--adapter',str(adapter),
                '--tag',tag,'--dataset',str(OLD/'数据/confirm120.jsonl'),'--images',str(OLD/'图'),
                '--out',str(OUT/'eval'),'--budget-s',str(min(9*60,remaining))])
            raw=OUT/'eval'/f'{tag}_raw.jsonl'
            rows=[json.loads(x) for x in raw.read_text().splitlines() if x.strip()]
            if len(rows)!=120 or len({r['sample_id'] for r in rows})!=120:
                raise RuntimeError('Evaluation incomplete; partial outputs retained without retry')
        status['status']='completed_training_and_raw_evaluation_pending_CPU_analysis'
    except Exception as exc:
        status['status']='stopped'
        status['reason']=str(exc)
    finally:
        status['end_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()


if __name__=='__main__':
    main()
