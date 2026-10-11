"""CPU-only image integrity and real collator supervision check."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=a.root
    required={'torch':'2.8.0','transformers':'5.16.1','ms-swift':'4.5.3','peft':'0.20.0','trl':'0.29.1'}
    versions={k:importlib.metadata.version(k) for k in required}
    assert all(versions[k].split('+')[0]==v for k,v in required.items())
    hashes=json.loads((root/'image_hashes.json').read_text(encoding='utf-8'))
    images=root/'images'
    for name,want in hashes.items():
        assert hashlib.sha256((images/name).read_bytes()).hexdigest()==want,f'Image mismatch: {name}'
    rows=[json.loads(s) for s in (root/'data/mix_sft4.jsonl').read_text(encoding='utf-8').splitlines()]
    ids={r['sample_id'] for r in rows}
    assert not ids.intersection(json.loads((root/'confirmation_ids.json').read_text(encoding='utf-8')))
    import mask_helpers as helpers
    got=helpers.get_model_processor(helpers.MODEL,model_type='qwen3_5',load_model=False)
    proc=got[1];tok=proc.tokenizer;tmpl=helpers.build_template(proc,'default','train')
    picks=[rows[0],rows[-1],max(rows,key=lambda r:len(r['messages'][-1]['content'])),
           next(r for r in rows if json.loads(r['messages'][-1]['content'])['red_mass_zones']==['edge'])]
    records=[]
    for i,r in enumerate(picks):
        record=helpers.inspect(tmpl,r,f'v4b_{i}',tok)
        sup=''.join(t['token'] for t in record['逐token'] if t['supervised'])
        assert record['有labels'] and record['总token']<2048
        assert r['messages'][-1]['content'] in sup
        assert '<|vision_start|>' not in sup and '<|im_start|>user' not in sup
        assert all(t['label']==-100 for t in record['逐token'][:9])
        records.append(record)
    result={'verdict':'passed','image_hashes_checked':len(hashes),'unique_training_wafers':len(ids),
            'confirmation_ID_intersection':0,'versions':versions,'records':records,
            'model_forward_run':False,'weights_loaded':False}
    (root/'mask_gate.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
