"""Class-preserving, capped radial-combination replay of frozen training groups.

Changes only the training presentation sequence. Four full six-field answers per
group stay intact. Each original wafer can occur at most twice. No holdout input.
"""
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import random

MAJOR=[('edge',),('middle','edge'),('center','middle','edge')]


def construct(rows):
    assert len(rows)==1024
    groups=[rows[i:i+4] for i in range(0,len(rows),4)]
    assert all(len({json.dumps(r,sort_keys=True) for r in g})==1 for g in groups)
    ids=[g[0]['sample_id'] for g in groups]
    assert len(set(ids))==256
    refs=[json.loads(g[0]['messages'][-1]['content']) for g in groups]
    zones=[tuple(r['red_mass_zones']) if isinstance(r['red_mass_zones'],list) else ('unknown',)
           for r in refs]
    classes=[r['defect_class'] for r in refs]
    selection=list(range(256)); use=Counter(selection); counts=Counter(zones)
    initial=counts.copy()
    targets={z:sum(counts[z] for z in MAJOR)/len(MAJOR) for z in MAJOR}
    rng=random.Random(3407)
    priorities=list(range(256));rng.shuffle(priorities)
    rank={x:i for i,x in enumerate(priorities)}
    swaps=[]
    while True:
        options=[]
        for pos,src in enumerate(selection):
            a=zones[src]
            if a not in MAJOR:continue
            for dst in range(256):
                b=zones[dst]
                if b not in MAJOR or a==b or use[dst]>=2 or classes[src]!=classes[dst]:continue
                before=(counts[a]-targets[a])**2+(counts[b]-targets[b])**2
                after=(counts[a]-1-targets[a])**2+(counts[b]+1-targets[b])**2
                improvement=before-after
                if improvement>1e-9:
                    options.append((-improvement,rank[src],rank[dst],pos,src,dst))
        if not options:break
        _,_,_,pos,src,dst=min(options)
        selection[pos]=dst; use[src]-=1;use[dst]+=1
        counts[zones[src]]-=1;counts[zones[dst]]+=1
        swaps.append({'presentation_group':pos,'original_id':ids[src],
                      'replacement_id':ids[dst],'class':classes[src],
                      'from_zones':list(zones[src]),'to_zones':list(zones[dst])})
    out=[r for i in selection for r in groups[i]]
    assert len(out)==1024 and max(Counter(r['sample_id'] for r in out).values())<=8
    assert Counter(classes)==Counter(classes[i] for i in selection)
    assert all(classes[i]==classes[j] for i,j in zip(range(256),selection))
    report={'seed':3407,'input_unique_wafers':256,'presentation_groups':256,
            'dialogue_rows':1024,'output_unique_wafers':len(set(selection)),
            'max_presentations_per_original_wafer':max(use.values()),
            'class_counts_preserved':True,'class_at_each_presentation_position_preserved':True,
            'class_counts':dict(Counter(classes)),'new_unique_wafers':0,
            'original_zone_counts':{'|'.join(z):n for z,n in initial.items()},
            'replayed_zone_counts':{'|'.join(z):n for z,n in counts.items()},
            'changed_presentation_positions':len(swaps),'swaps':swaps,
            'prompt_and_target_per_image_unchanged':True,
            'sampling_basis':'frozen training references only; no confirmation scores or images read'}
    return out,report


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path);p.add_argument('--dry-run',action='store_true')
    p.add_argument('--images-dir',type=Path);a=p.parse_args()
    raw=a.source.read_bytes();rows=[json.loads(s) for s in raw.decode('utf-8').splitlines() if s.strip()]
    out,report=construct(rows)
    if a.images_dir:
        import copy
        out=copy.deepcopy(out)
        for r in out:r['images']=[str(a.images_dir/Path(x).name) for x in r['images']]
        report['image_path_rewrite_only']=True
    encoded=('\n'.join(json.dumps(r,ensure_ascii=False,separators=(',',':')) for r in out)+'\n').encode('utf-8')
    report.update(source_sha256=hashlib.sha256(raw).hexdigest(),output_sha256=hashlib.sha256(encoded).hexdigest())
    assert raw==a.source.read_bytes()
    if not a.dry_run:
        assert a.out and a.out.resolve()!=a.source.resolve()
        a.out.parent.mkdir(parents=True,exist_ok=True)
        if a.out.exists():assert a.out.read_bytes()==encoded
        else:a.out.write_bytes(encoded)
        manifest=a.out.with_suffix('.manifest.json')
        data=(json.dumps(report,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
        if manifest.exists():assert manifest.read_bytes()==data
        else:manifest.write_bytes(data)
    print(json.dumps({k:v for k,v in report.items() if k!='swaps'},ensure_ascii=False))


if __name__=='__main__':main()
