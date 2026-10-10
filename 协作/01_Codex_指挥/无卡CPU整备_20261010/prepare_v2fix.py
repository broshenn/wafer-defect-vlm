"""Generate a NEW v2 repair script; preserve data/order/steps/reward, fix both adapters.

No GPU. Dry-run and byte-idempotent generation. No historical file is rewritten.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SOURCE = REPO / '协作/02_ClaudeCode_实操/多字段RL_v2_CPU准备_20261009/服务器_v2_train.py'
EXPECTED = '981c6a80e5ec55993547ea05ff7ab2da080f64a19a3925faecd5233a53185c42'


def build():
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != EXPECTED:
        raise RuntimeError('Frozen v2 training script changed')
    code = SOURCE.read_text(encoding='utf-8')
    old = '    if a.start_adapter:\n        import os as _os'
    new = ('    if a.mode not in ("sft", "rl") or not a.start_adapter:\n'
           '        raise RuntimeError("Repair requires sft/rl and explicit M0 adapter")\n'
           '    if a.steps != 60:\n'
           '        raise RuntimeError("Repair contract fixes 60 steps")\n'
           '    if a.start_adapter:\n        import os as _os')
    if code.count(old) != 1:
        raise RuntimeError('Adapter insertion site not unique')
    code = code.replace(old, new)
    old = '             if a.start_adapter else {})\n'
    new = ('             if a.start_adapter else {})\n'
           '    if a.mode == "rl":\n'
           '        extra["ref_adapters"] = [a.start_adapter]\n'
           '    os.environ["WAFER_START_ADAPTER"] = a.start_adapter\n'
           '    os.environ["WAFER_FIX_MODE"] = a.mode\n'
           '    os.environ["WAFER_GUARD_OUT"] = f"{a.out}/runtime_gate.json"\n'
           '    from runtime_guard_v2fix import install\n'
           '    install()\n')
    if code.count(old) != 1:
        raise RuntimeError('Runtime guard insertion site not unique')
    code = code.replace(old, new)
    # BOTH constructors must receive the actual extra keyword dictionary.
    old = '            output_dir=a.out, report_to=[])'
    old_sft = '            eval_strategy="no", output_dir=a.out, report_to=[])'
    if code.count(old) != 1 or code.count(old_sft) != 1:
        raise RuntimeError('Argument constructor sites not unique')
    code = code.replace(old, '            output_dir=a.out, report_to=[], **extra)')
    code = code.replace(old_sft, '            eval_strategy="no", output_dir=a.out, report_to=[], **extra)')
    calls = [node for node in ast.walk(ast.parse(code)) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id in ['SftArguments', 'RLHFArguments']]
    assert len(calls) == 2 and all(any(k.arg is None and isinstance(k.value, ast.Name)
                                      and k.value.id == 'extra' for k in call.keywords) for call in calls)
    return code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    code = build()
    report = {'constructor_forwarding': 'passed_CPU_AST', 'policy_adapter': 'explicit_M0',
              'reference_adapter': 'explicit_M0_for_GRPO', 'steps': 60,
              'source_sha256': EXPECTED, 'runtime_model_loading': 'not_run',
              'GPU_training': 'not_run', 'generated_sha256': hashlib.sha256(code.encode()).hexdigest()}
    if not args.dry_run:
        (HERE/'v2_train_fixed.py').write_text(code, encoding='utf-8', newline='\n')
        (HERE/'起点修复_CPU检查.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
