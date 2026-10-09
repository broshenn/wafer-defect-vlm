"""只写本目录的修复副本，使用AST验证参数传递；不加载torch/模型或运行GPU。"""
from pathlib import Path
from types import SimpleNamespace
import ast
import json

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
SOURCE=ROOT/'协作/02_ClaudeCode_实操/多字段RL_v2_CPU准备_20261009/服务器_v2_train.py'
old=SOURCE.read_text(encoding='utf-8')
assert old.count('output_dir=a.out, report_to=[])')==2
new=old.replace('output_dir=a.out, report_to=[])','output_dir=a.out, report_to=[], **extra)')
guard='''    if a.mode in ("sft", "rl") and not a.start_adapter:
        print("!! sft/rl必须显式提供共同M0 start-adapter，拒绝从默认基座重新起跑")
        return 3
'''
assert new.count('    a = ap.parse_args()\n')==1
new=new.replace('    a = ap.parse_args()\n','    a = ap.parse_args()\n'+guard)
new=new.replace('/root/autodl-tmp/ws/v2run', '/WS/v2run').replace('/root/autodl-tmp/ws/models/Qwen3.5-9B-Dmerged','/WS/models/Qwen3.5-9B-Dmerged')
new=new.replace('BASE = "/WS/v2run"','BASE = os.environ["WAFER_V2_WORKSPACE"]')
new=new.replace('MERGED = "/WS/models/Qwen3.5-9B-Dmerged"','MERGED = os.environ["WAFER_D_MERGED"]')
path=HERE/'v2_train_起点修复草案.py'
path.write_text(new,encoding='utf-8')

def capture(text):
    tree=ast.parse(text)
    a=SimpleNamespace(start_adapter='/SYNTHETIC/M0/checkpoint-64',model='/SYNTHETIC/Dmerged',
                      model_type='qwen3_5',template_type='qwen3_5',lr=None,steps=60,out='/SYNTHETIC/unused',max_completion_length=256)
    env={'a':a,'data':'/SYNTHETIC/data.jsonl','reward':object(),
         'SftArguments':lambda **kw:kw,'RLHFArguments':lambda **kw:kw}
    ex=next(n.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='extra' for t in n.targets))
    env['extra']=eval(compile(ast.Expression(ex),'<extra>','eval'),env)
    result={}
    for n in ast.walk(tree):
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ['SftArguments','RLHFArguments']:
            kw=eval(compile(ast.Expression(n),'<argument-constructor>','eval'),env)
            result[n.func.id]={'adapters_forwarded':kw.get('adapters')==[a.start_adapter],
                              'resume_only_model_forwarded':kw.get('resume_only_model') is True,
                              'steps':kw.get('max_steps'),'model_override_preserved':kw.get('model')==a.model}
    return result
before=capture(old);after=capture(new)
assert all(not r['adapters_forwarded'] for r in before.values())
assert all(r['adapters_forwarded'] and r['resume_only_model_forwarded'] and r['model_override_preserved'] for r in after.values())
report={'legacy_constructor_capture':before,'fixed_constructor_capture':after,
        'required_M0_guard_added':True,'scope':'CPU构造函数实参传递验证，不是ms-swift实际模型加载验证；新代码仍是草案，未训练。',
        'remaining_gate':'真实运行前验证解析后args及加载后的LoRA逐张量等于M0，再允许第一次更新；不得只看打印或训练后相似性。'}
(HERE/'起点修复CPU回归.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
