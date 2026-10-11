"""Single-instance execution with a process-group deadline and durable receipts."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime,timezone


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--budget-seconds',type=int,default=2700);a=ap.parse_args()
    root=a.root;root.mkdir(parents=True,exist_ok=True)
    lock=root/'run.lock'
    try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:raise RuntimeError('Existing v4 run lock; do not duplicate execution')
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    ledger=root/'execution.json';started=time.monotonic();jobs=[]
    report={'executor':'Codex','run_id':'full_distribution_aux_sft_v4c_s3407','status':'preparing',
            'start_utc':datetime.now(timezone.utc).isoformat(),'cap_device_minutes':a.budget_seconds/60,
            'jobs':jobs,'candidate_new_unique_wafers':0}
    def save():
        report['elapsed_wall_seconds']=time.monotonic()-started
        ledger.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    save()
    env=os.environ.copy();env.update(IMAGE_MAX_TOKEN_NUM='256',PYTHONNOUSERSITE='1',
                                     HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    nvidia=subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True)
    cards=nvidia.stdout.strip().splitlines();assert len(cards)==1
    procs=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True)
    assert not procs.stdout.strip(),'GPU already has a compute process; do not share or kill it'
    env['CUDA_VISIBLE_DEVICES']=cards[0]
    def run(tag,cmd,gpu=True):
        job={'tag':tag,'start_utc':datetime.now(timezone.utc).isoformat(),'status':'running'};jobs.append(job)
        report['status']=tag;save();log=root/(tag+'.log');remain=a.budget_seconds-(time.monotonic()-started)-20
        if remain<=0:raise TimeoutError('Overall deadline exhausted')
        job_env=env.copy()
        if not gpu:job_env['CUDA_VISIBLE_DEVICES']=''
        with log.open('wb') as f:
            proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=job_env,start_new_session=True,cwd=root)
            job['pid']=proc.pid;save()
            try:rc=proc.wait(timeout=remain)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                job.update(status='timed_out',end_utc=datetime.now(timezone.utc).isoformat());save();raise
        job.update(status='finished' if rc==0 else 'failed',exit_code=rc,end_utc=datetime.now(timezone.utc).isoformat());save()
        if rc:raise RuntimeError(f'{tag} failed with exit code {rc}; inspect saved log')
    try:
        source=root/'source/group256_sft4.jsonl'
        assert source.exists()
        assert hashlib.sha256(source.read_bytes()).hexdigest()=='3d33a191b1423513298a2d7f052c631cef5f9370807870750994f60dba6e57ba'
        run('build_data',[sys.executable,str(root/'build_v4_mix.py'),'--source',str(source),
                          '--out',str(root/'data/mix_sft4.jsonl'),'--images-dir',str(root/'images')],False)
        run('mask_gate',[sys.executable,str(root/'mask_v4c.py'),'--root',str(root)],False)
        run('probe_sft',[sys.executable,str(root/'train_v4c.py'),'--root',str(root),'--steps','2','--out',str(root/'probe'),'--probe'])
        run('formal_sft',[sys.executable,str(root/'train_v4c.py'),'--root',str(root),'--steps','256','--out',str(root/'model')])
        ckpts=list((root/'model').glob('**/checkpoint-256'));assert len(ckpts)==1
        checkpoint=ckpts[0]
        report['final_adapter_sha256']=hashlib.sha256((checkpoint/'adapter_model.safetensors').read_bytes()).hexdigest();save()
        dataset=str(root/'source/confirmation120_requests.jsonl')
        images=str(root/'images')
        base='/root/autodl-tmp/ws/models/Qwen3.5-9B-Dmerged'
        run('evaluate_v4',[sys.executable,str(root/'eval_v4.py'),'--model',base,'--adapter',str(checkpoint),
                          '--tag','SFT_aux_v4c','--dataset',dataset,'--images',images,'--out',str(root/'eval'),'--budget-s','700'])
        report['baseline_reused']='verified v4b M0_clone raw; unchanged environment and M0 fingerprint'
        report['status']='completed_training_and_evaluation'
    except Exception as exc:
        report['status']='failed';report['error_type']=type(exc).__name__;report['error']=str(exc)
    finally:
        report['end_utc']=datetime.now(timezone.utc).isoformat()
        procs=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True)
        report['GPU_compute_processes_after']=len(procs.stdout.strip().splitlines()) if procs.stdout.strip() else 0
        save()
        lock.unlink(missing_ok=True)
    return 0 if report['status']=='completed_training_and_evaluation' else 1


if __name__=='__main__':sys.exit(main())
