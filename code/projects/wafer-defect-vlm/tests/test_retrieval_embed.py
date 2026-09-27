"""Tests for the retrieval embedding path.

The bug these guard against is a silent one: a processor returns pixel_values
whether or not the prompt contains image placeholders. Without a check, the
embeddings come back text-only, the rankings look plausible, and every metric
derived from them is meaningless.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("PIL")
pytest.importorskip("transformers")

IMAGE_PAD_ID = 151655


def _find_tool(name: str) -> Path:
    for base in Path(__file__).resolve().parents:
        candidate = base / "tools" / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(name)


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), _find_tool(name))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rank = _load("retrieval_rank.py")


class _Tokenizer:
    def convert_tokens_to_ids(self, token):
        return IMAGE_PAD_ID if token == "<|image_pad|>" else 0


class _Batch(dict):
    def to(self, device):
        return self


class _Processor:
    """A processor that records how it was called and returns a fixed batch."""

    def __init__(self, ids):
        self.tokenizer = _Tokenizer()
        self._ids = ids
        self.text_calls = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        self.text_calls.append(messages)
        return ["templated"] * len(messages)

    def __call__(self, text=None, images=None, return_tensors=None, padding=None):
        assert isinstance(text, list), "text must be a batch, not a single string"
        n = len(text)
        return _Batch({
            "input_ids": torch.tensor([self._ids] * n),
            "attention_mask": torch.ones((n, len(self._ids)), dtype=torch.long),
        })


def test_embedding_refuses_a_prompt_without_vision_tokens(tmp_path) -> None:
    # Ten text tokens and no image placeholder: exactly what the naive call
    # produced, and the shape of the failure that motivated the guard.
    processor = _Processor([72240, 279, 21531, 5215, 303, 411, 10138, 776, 2336, 13])
    image = tmp_path / "wafer.png"
    from PIL import Image
    Image.new("RGB", (8, 8)).save(image)

    with pytest.raises(SystemExit, match="no <\\|image_pad\\|> tokens"):
        rank.embed(object(), processor, [str(image)], "cpu", 1)


def test_embedding_goes_through_the_chat_template(tmp_path) -> None:
    # The positive case: image tokens present, and the prompt reached the
    # processor only after the template inserted the image placeholder.
    processor = _Processor([151652, IMAGE_PAD_ID, 151653, 13])
    image = tmp_path / "wafer.png"
    from PIL import Image
    Image.new("RGB", (8, 8)).save(image)

    class _Output:
        hidden_states = (torch.zeros(1, 4, 8),)

    class _Model:
        def __call__(self, **kwargs):
            return _Output()

        def eval(self):
            return self

    vectors = rank.embed(_Model(), processor, [str(image)], "cpu", 1)
    assert vectors.shape == (1, 8)
    assert len(processor.text_calls) == 1
    content = processor.text_calls[0][0][0]["content"]
    assert content[0]["type"] == "image"
    assert content[1]["type"] == "text"
