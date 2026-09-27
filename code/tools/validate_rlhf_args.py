"""Parse a GRPO command line against ms-swift's own argument classes.

The point is to catch an invented or misspelled flag before a real run spends
half an hour discovering it. This is deliberately offline: it builds the parser,
parses the arguments and prints what ms-swift understood. It never loads a
model, so it costs no GPU and can run while training is using the card.

Arguments are passed straight through from the calling script, so there is no
second copy of the flag list that could drift away from the one actually run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load_arguments_class(swift_src: Path):
    """Find RLHFArguments wherever this ms-swift version keeps it."""
    sys.path.insert(0, str(swift_src))
    for module_path, name in (
        ("swift.llm.argument.rlhf_args", "RLHFArguments"),
        ("swift.llm", "RLHFArguments"),
        ("swift", "RLHFArguments"),
    ):
        try:
            module = __import__(module_path, fromlist=[name])
            return getattr(module, name)
        except (ImportError, AttributeError):
            continue
    raise ImportError("could not locate RLHFArguments in this ms-swift checkout")


def main() -> int:
    argv = sys.argv[1:]
    if not argv:
        print("usage: validate_rlhf_args.py <swift rlhf arguments...>", file=sys.stderr)
        return 2

    swift_src = Path(
        __import__("os").environ.get("SWIFT_SRC", "/root/autodl-tmp/wafer-vlm/src/ms-swift")
    )
    from transformers import HfArgumentParser

    rlhf_arguments = load_arguments_class(swift_src)
    parser = HfArgumentParser(rlhf_arguments)

    # HfArgumentParser writes a usage message and exits on a bad flag, which is
    # exactly the signal wanted here.
    parsed = parser.parse_args_into_dataclasses(argv)[0]

    reward_funcs = getattr(parsed, "reward_funcs", None)
    print(json.dumps({
        "parsed": True,
        "rlhf_type": str(getattr(parsed, "rlhf_type", None)),
        "use_vllm": bool(getattr(parsed, "use_vllm", None)),
        "num_generations": getattr(parsed, "num_generations", None),
        "max_completion_length": getattr(parsed, "max_completion_length", None),
        "max_length": getattr(parsed, "max_length", None),
        "reward_funcs": [str(r) for r in reward_funcs] if reward_funcs else reward_funcs,
        "reward_weights": getattr(parsed, "reward_weights", None),
        "external_plugins": [str(p) for p in (getattr(parsed, "external_plugins", None) or [])],
        "adapters": [str(p) for p in (getattr(parsed, "adapters", None) or [])],
        "tuner_type": str(getattr(parsed, "tuner_type", None)),
        "beta": getattr(parsed, "beta", None),
        "learning_rate": getattr(parsed, "learning_rate", None),
        "max_steps": getattr(parsed, "max_steps", None),
    }, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
