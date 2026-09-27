"""Small dispatcher for the wafer-vlm console entry point."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="wafer-vlm")
    parser.add_argument("command", choices=("prepare", "annotate", "curate", "benchmark", "evaluate"))
    args, remainder = parser.parse_known_args()
    if args.command == "prepare":
        from .prepare_data import make_parser
    elif args.command == "annotate":
        from .annotate import make_parser
    elif args.command == "curate":
        from .curate import make_parser
    elif args.command == "benchmark":
        from .benchmark import make_parser
    else:
        from .evaluate import make_parser
    command_args = make_parser().parse_args(remainder)
    if hasattr(command_args, "func"):
        command_args.func(command_args)
    elif args.command == "annotate":
        import asyncio
        from .annotate import run
        asyncio.run(run(command_args))
    elif args.command == "prepare":
        from .prepare_data import build
        build(command_args)
    else:
        from .evaluate import evaluate
        evaluate(command_args)


if __name__ == "__main__":
    main()
