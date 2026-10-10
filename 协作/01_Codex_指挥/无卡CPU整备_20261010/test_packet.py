import hashlib
import json
from pathlib import Path
import unittest

from core_v3 import HERE, REPO, schema_errors, parse
from workbuddy_schema_check import check


def read(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]


class PacketTests(unittest.TestCase):
    def test_pool_integrity_and_isolation(self):
        pools = {k:read(HERE/'pools'/f'{k}.jsonl') for k in ['adapt','rl','rule_dev','confirmation']}
        for name, rows in pools.items():
            self.assertEqual(len(rows),len({r['sample_id'] for r in rows}))
            self.assertEqual(len(rows),len({r['image_sha256'] for r in rows}))
            for r in rows:
                p=REPO/r['image_path']
                self.assertEqual(p.stem,r['sample_id'])
                self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),r['image_sha256'])
                self.assertEqual(schema_errors(r['reference']),[])
                self.assertEqual(r['split'],'test' if name=='confirmation' else 'train')
        for a in pools:
            for b in pools:
                if a<b:
                    for field in ['sample_id','lot_name','image_sha256']:
                        self.assertFalse({r[field] for r in pools[a]} & {r[field] for r in pools[b]})

    def test_teacher_blind_manifest_and_hashes(self):
        p=HERE/'WorkBuddy84'
        index=json.loads((p/'index.json').read_text(encoding='utf-8'))
        allrows=read(p/'blind84.jsonl')
        self.assertEqual(len(allrows),84)
        self.assertEqual(len({r['sample_id'] for r in allrows}),84)
        self.assertEqual(len({r['image_sha256'] for r in allrows}),84)
        shards=[]
        for batch in index['batches']:
            f=p/batch['file']
            self.assertEqual(hashlib.sha256(f.read_bytes()).hexdigest(),batch['sha256'])
            rows=read(f);self.assertEqual(len(rows),7);shards.extend(rows)
        self.assertEqual(shards,allrows)
        for r in allrows:
            self.assertEqual(set(r),{'item_id','sample_id','image_path','image_sha256'})
            self.assertEqual(hashlib.sha256(Path(r['image_path']).read_bytes()).hexdigest(),r['image_sha256'])
        for name,key in [('blind84.jsonl','blind_manifest_sha256'),('标注题面.txt','prompt_sha256'),
                         ('../workbuddy_schema_check.py','checker_sha256'),('../core_v3.py','checker_dependency_sha256')]:
            self.assertEqual(hashlib.sha256((p/name).read_bytes()).hexdigest(),index[key])
        adapt={r['sample_id'] for r in read(HERE/'pools/adapt.jsonl')}
        self.assertTrue({r['sample_id'] for r in allrows} <= adapt)

    def test_student_input_has_no_reference_or_caption(self):
        for name in ['adapt512_sft.jsonl','group256_sft4.jsonl','group256_grpo.jsonl','confirmation120_requests.jsonl']:
            rows=read(HERE/'data_v3'/name)
            prompts={r['messages'][0]['content'] for r in rows}
            self.assertEqual(len(prompts),1)
            for r in rows:
                self.assertNotIn(r['sample_id'],r['messages'][0]['content'])
                for m in r['messages'][1:]:
                    self.assertNotIn('caption_zh',parse(m['content']))
                if name.startswith('confirmation'):
                    self.assertEqual(set(r),{'sample_id','messages','images'})
                    self.assertEqual(len(r['messages']),1)

    def test_teacher_invalid_responses(self):
        good={'morphology_zh':'环状','main_location_zh':'边缘','location_clock':[],
              'orientation_clock':[],'secondary_observations':[],'caption_zh':'边缘可见环状红色结构。','uncertainty':''}
        self.assertTrue(check(json.dumps(good))['schema_ok'])
        for x in [dict(good,location_clock=[True]),dict(good,orientation_clock=[1,3]),
                  dict(good,location_clock=[1,2]),dict(good,secondary_observations='x')]:
            self.assertFalse(check(json.dumps(x))['schema_ok'])
        for text in [json.dumps(good)+'{}','{"x":NaN}','{"x":1,"x":2}']:
            self.assertFalse(check(text)['schema_ok'])
        warned=check(json.dumps(dict(good,caption_zh='跨度约0.98R。')))
        self.assertTrue(warned['schema_ok']);self.assertTrue(warned['warnings'])
        self.assertFalse(warned['content_verified'])


if __name__=='__main__':
    unittest.main(verbosity=2)
