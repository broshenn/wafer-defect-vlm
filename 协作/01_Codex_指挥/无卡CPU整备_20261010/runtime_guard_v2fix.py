"""Runtime proof for repaired v2 initialization. Import before sft_main/rlhf_main.

Requires environment WAFER_START_ADAPTER, WAFER_FIX_MODE, WAFER_GUARD_OUT.
Preserves Trainer.__init__ signature because swift inspects it.
This module is a prepared guard, not evidence that a GPU run passed.
"""
import functools
import hashlib
import json
import os
from pathlib import Path


def install():
    import torch
    from safetensors.torch import load_file
    from peft import get_peft_model_state_dict
    from transformers import Trainer, TrainerCallback

    adapter = Path(os.environ['WAFER_START_ADAPTER'])
    target = load_file(str(adapter / 'adapter_model.safetensors'), device='cpu')
    sha = hashlib.sha256((adapter / 'adapter_model.safetensors').read_bytes()).hexdigest()
    expected = '43b36375f914ae972e5947f1d742a24e5442d1133b918dea3fc698832174681b'
    if sha != expected:
        raise RuntimeError('M0 file fingerprint mismatch')
    mode = os.environ['WAFER_FIX_MODE']
    out = Path(os.environ['WAFER_GUARD_OUT'])
    out.parent.mkdir(parents=True, exist_ok=True)

    def compare(model, name):
        actual = get_peft_model_state_dict(model, adapter_name=name)
        if set(actual) != set(target):
            raise RuntimeError(f'{name}: M0 tensor key set mismatch')
        equal = sum(torch.equal(actual[k].detach().cpu(), target[k].to(actual[k].dtype)) for k in actual)
        if equal != len(actual):
            raise RuntimeError(f'{name}: initial/reference tensors differ from M0')
        return equal

    class Guard(TrainerCallback):
        def __init__(self, trainer):
            self.trainer = trainer
            self.forward = {'policy': 0, 'reference': 0, 'image': 0}
            self.gradients = set()
            self.report = {'model_loading_gate': 'pending', 'two_step_gate': 'pending',
                           'mode': mode, 'M0_sha256': sha}
            self.handles = []

        def save(self):
            out.write_text(json.dumps(self.report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

        def on_train_begin(self, args, state, control, model=None, **kwargs):
            if torch.cuda.device_count() != 1 or self.trainer.accelerator.num_processes != 1:
                raise RuntimeError('Not a true single GPU process')
            m = self.trainer.accelerator.unwrap_model(model)
            self.model = m
            self.report['policy_equal_M0_tensor_count'] = compare(m, 'default')
            trainable = [n for n, p in m.named_parameters() if p.requires_grad]
            if not trainable or any('lora_' not in n for n in trainable):
                raise RuntimeError('Trainable parameters are not exclusively LoRA')
            if any('visual' in n or 'vision' in n for n in trainable):
                raise RuntimeError('Vision parameters trainable')
            if mode == 'rl':
                if getattr(self.trainer, 'ref_adapter_name', None) != 'ref_adapter':
                    raise RuntimeError('Actual GRPO trainer reference is not named M0 adapter')
                self.report['reference_equal_M0_tensor_count'] = compare(m, 'ref_adapter')
                if any('ref_adapter' in n for n in trainable):
                    raise RuntimeError('Reference adapter trainable')

            def forward_hook(module, inp, result):
                active = list(getattr(m, 'active_adapters', []))
                if 'ref_adapter' in active:
                    self.forward['reference'] += 1
                elif 'default' in active:
                    self.forward['policy'] += 1

            self.handles.append(m.register_forward_hook(forward_hook))
            visual = next((module for name, module in m.named_modules()
                           if name.endswith('.visual') or name == 'visual'), None)
            if visual is None:
                raise RuntimeError('Vision module hook target absent')
            self.handles.append(visual.register_forward_hook(
                lambda module, inp, result: self.forward.__setitem__('image', self.forward['image']+1)))
            for name, p in m.named_parameters():
                if p.requires_grad:
                    def hook(grad, n=name):
                        if not torch.isfinite(grad).all():
                            raise RuntimeError('Nonfinite LoRA gradient')
                        if grad.abs().max().item() > 0:
                            self.gradients.add(n)
                        return grad
                    self.handles.append(p.register_hook(hook))
            self.report['model_loading_gate'] = 'passed'
            self.report['trainable_tensor_count'] = len(trainable)
            self.save()

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step != 2:
                return
            actual = get_peft_model_state_dict(self.model, adapter_name='default')
            changed = sum(not torch.equal(actual[k].detach().cpu(), target[k].to(actual[k].dtype)) for k in actual)
            self.report.update(step=2, policy_changed_tensors=changed,
                               nonzero_gradient_tensors=len(self.gradients), forwards=self.forward)
            if not changed or not self.gradients or not self.forward['image'] or not self.forward['policy']:
                self.report['two_step_gate'] = 'failed'
                self.save()
                raise RuntimeError('Two-step image/gradient/update proof failed')
            if mode == 'rl':
                if not self.forward['reference']:
                    self.report['two_step_gate'] = 'failed'
                    self.save()
                    raise RuntimeError('Actual reference forward not observed')
                self.report['reference_unchanged_M0_tensor_count'] = compare(self.model, 'ref_adapter')
            self.report['two_step_gate'] = 'passed'
            self.save()

    original = Trainer.__init__

    @functools.wraps(original)
    def patched(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.add_callback(Guard(self))

    Trainer.__init__ = patched
