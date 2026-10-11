"""A frozen SFT continuation from M0, with runtime initialization proof."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--steps',type=int,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--probe',action='store_true');a=p.parse_args()
    assert a.steps==(2 if a.probe else 256)
    import torch
    assert os.environ.get('CUDA_VISIBLE_DEVICES') and torch.cuda.device_count()==1
    root=a.root; data=root/'data/replay_sft4.jsonl'
    model=Path('/root/autodl-tmp/ws/models/Qwen3.5-9B-Dmerged')
    adapter=Path('/root/autodl-tmp/ws/v3run/out/M0_v3/v0-20261010-232543/checkpoint-128')
    expected='191c62e842cb93149397d19a96bf25e9cd1e0362cfb68e9af6857e55cd702e09'
    assert hashlib.sha256((adapter/'adapter_model.safetensors').read_bytes()).hexdigest()==expected
    os.environ.update(WAFER_START_ADAPTER=str(adapter),WAFER_FIX_MODE='sft',
                      WAFER_GUARD_OUT=str(a.out/'runtime_gate.json'),WAFER_EXPECT_M0_SHA=expected)
    from runtime_guard_v3 import install
    install()
    from swift import SftArguments
    from swift.pipelines import sft_main
    args=SftArguments(model=str(model),model_type='qwen3_5',template='qwen3_5',
        dataset=str(data),adapters=[str(adapter)],resume_only_model=True,
        tuner_type='lora',target_modules=['all-linear'],freeze_vit=True,freeze_aligner=True,
        torch_dtype='bfloat16',lora_rank=16,lora_alpha=32,lora_dropout=.05,
        learning_rate=1e-5,warmup_ratio=0.,lr_scheduler_type='cosine',
        max_length=2048,per_device_train_batch_size=1,gradient_accumulation_steps=4,
        max_steps=a.steps,gradient_checkpointing=True,seed=3407,data_seed=3407,
        dataset_shuffle=False,train_dataloader_shuffle=False,dataloader_drop_last=False,
        logging_steps=1,save_strategy='steps',save_steps=(1 if a.probe else 128),
        save_total_limit=2,eval_strategy='no',output_dir=str(a.out),report_to=[])
    a.out.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();sft_main(args)
    cfg={'steps':a.steps,'seconds':time.monotonic()-started,
         'data_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),'start_adapter_sha256':expected,
         'selection':'preselected_last_step','peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,
         'versions':{x:importlib.metadata.version(x) for x in ['torch','transformers','ms-swift','peft','trl']}}
    (a.out/'run_record.json').write_text(json.dumps(cfg,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
