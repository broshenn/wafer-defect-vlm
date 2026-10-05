"""Codex：仅修复钟点区间“至”的识别；旧检查器/原答不改写。"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
SOURCE=ROOT/'协作/02_ClaudeCode_实操/口径修正与准备_20261002-021207/check_answer_v3.py'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()=='794a254e005eaa232362c539817abde98a1c2406ecb52574b9e9edb35303ec55'
spec=importlib.util.spec_from_file_location('locked_checker_v3',SOURCE)
BASE=importlib.util.module_from_spec(spec);spec.loader.exec_module(BASE)
original_pattern=BASE.CLOCK_RE.pattern
assert '和及与]' in original_pattern
BASE.CLOCK_RE=re.compile(original_pattern.replace('和及与]','和及与至]'))
VERSION='format_v4_clock_range_20261003'

def check_text(text):return BASE.check_text(text)
def check_object(obj):return BASE.check_object(obj)

def self_test():
    def obj(text):return {'defect_class':'Loc','morphology':text,'radial_zone':'middle',
        'clock_direction':None,'extent_r':None,'caption_zh':text,'uncertainty':''}
    positives=['约10至11点钟方向','约4至5点钟方向','2至3点钟方向','约7至11点钟方向',
               '约5至7点钟方向','4至6点钟方向','6至8点钟方向','约4至5点钟方向',
               '7~10点钟方向','4-5点钟','十二至一点钟方向','散点，无唯一方向']
    negatives=['宽度0.2R，在4至5点钟方向','面积4至5%','约10至11个失效die','4至5点钟方向，跨度0.3R']
    for text in positives:assert check_object(obj(text))['exit_code']==0,text
    for text in negatives:assert check_object(obj(text))['exit_code']==8,text
    print('PASS: 12 clock/identity and 4 real numeric rejection cases')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('path',nargs='?',type=Path)
    parser.add_argument('--json',action='store_true');parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    if args.self_test:self_test();return
    if args.path is None:parser.error('需要答案文件')
    report=check_text(args.path.read_text(encoding='utf-8-sig'))
    print(json.dumps({'checker_version':VERSION,'path':str(args.path),**report},ensure_ascii=False))
    raise SystemExit(report['exit_code'])

if __name__=='__main__':main()
