"""Known-output metric contract: primary classification is independent of rule reward."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

from core_v3 import HERE, REPO


class AnalysisContractTests(unittest.TestCase):
    def test_primary_secondary_and_complete_denominator(self):
        p=REPO/'协作/02_ClaudeCode_实操/多字段RL_v2_CPU准备_20261009/out_v2/confirm120_gold.jsonl'
        gold=[json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
        with tempfile.TemporaryDirectory(prefix='wafer-metrics-synthetic-') as directory:
            tmp=Path(directory).resolve()
            assert tmp.is_relative_to(Path(tempfile.gettempdir()).resolve())
            for model in ['M0_actual','M1_fix','M2_fix']:
                rows=[]
                for i,g in enumerate(gold):
                    obj={'defect_class':g['gold_class'],'coverage_level':g['gold_coverage_level'],
                         'red_mass_zones':json.loads(g['gold_zones']) if g['gold_zones']!='unknown' else 'unknown',
                         'direction_type':g['gold_direction_type'],'clock_sectors':json.loads(g['gold_sectors']),
                         'caption_zh':'合成夹具文本，不是模型回答。','uncertainty':''}
                    if model=='M1_fix' and i==0:
                        obj['defect_class']='Loc' if obj['defect_class']!='Loc' else 'Center'
                    if model=='M2_fix':
                        obj['coverage_level']='low' if obj['coverage_level']!='low' else 'high'
                    rows.append({'sample_id':g['sample_id'],'raw':json.dumps(obj,ensure_ascii=False)})
                (tmp/f'{model}_raw.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
            output=tmp/'synthetic_result.json'
            result=subprocess.run([sys.executable,'-X','utf8','-B',str(HERE/'analyze_fix.py'),'--input',str(tmp),'--output',str(output)],capture_output=True,text=True,encoding='utf-8')
            self.assertEqual(result.returncode,0,result.stderr)
            report=json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(report['models']['M0_actual']['class_correct'],120)
            self.assertAlmostEqual(report['models']['M0_actual']['primary_class_macro_f1'],1)
            self.assertEqual(report['models']['M1_fix']['class_correct'],119)
            self.assertLess(report['models']['M1_fix']['primary_class_macro_f1'],1)
            self.assertEqual(report['models']['M2_fix']['primary_class_macro_f1'],1)
            pair=report['paired']['M2_fix minus M0_actual']
            self.assertEqual(pair['primary_macro_f1_CI95'],[0,0])
            self.assertLess(pair['secondary_rule_reward_diff'],0)
            # Missing output is a failure, never silently shrink to surviving cases.
            (tmp/'M2_fix_raw.jsonl').write_text('{}\n',encoding='utf-8')
            failed=subprocess.run([sys.executable,'-X','utf8','-B',str(HERE/'analyze_fix.py'),'--input',str(tmp),'--output',str(output)],capture_output=True,text=True,encoding='utf-8')
            self.assertNotEqual(failed.returncode,0)


if __name__=='__main__':
    unittest.main(verbosity=2)
