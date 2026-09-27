"""Record what produced the results: environment, commands, hashes, deviations.

The point is that a reader can reconstruct which code, which weights and which
data a number came from. Environment facts are measured here rather than copied
from a README, so the record cannot drift from the machine that did the work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: str | None = None) -> str | None:
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def environment(swift_src: Path) -> dict:
    import torch

    info = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_total_mib": (torch.cuda.get_device_properties(0).total_memory // (1 << 20))
                         if torch.cuda.is_available() else None,
        "nvidia_smi": run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
    }
    for name in ("transformers", "trl", "peft", "bitsandbytes", "datasets", "accelerate", "swift"):
        module = __import__(name)
        info[name] = getattr(module, "__version__", None)
    info["ms_swift_commit"] = run(["git", "rev-parse", "HEAD"], cwd=str(swift_src))
    info["ms_swift_describe"] = run(["git", "describe", "--always", "--dirty"], cwd=str(swift_src))
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--swift-src", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--artifact", action="append", default=[],
                        help="path relative to --root to hash; repeat as needed")
    parser.add_argument("--deviation", action="append", default=[],
                        help="a deliberate departure from the specification, recorded verbatim")
    args = parser.parse_args()

    root = Path(args.root)
    artifacts = {}
    for rel in args.artifact:
        path = root / rel
        artifacts[rel] = {"exists": path.is_file(),
                          "bytes": path.stat().st_size if path.is_file() else None,
                          "sha256": sha256(path) if path.is_file() and path.stat().st_size < (1 << 28) else None}

    # The run scripts are the commands; keep their text so the exact flags survive.
    commands = {}
    for path in sorted(root.glob("projects/wafer-defect-vlm/scripts/*.sh")) + sorted(root.glob("logs/*.sh")):
        commands[str(path.relative_to(root))] = {"sha256": sha256(path)}

    manifest_docs = {}
    for path in sorted(root.glob("manifests/*.json")):
        manifest_docs[path.name] = json.loads(path.read_text(encoding="utf-8"))

    report = {
        "environment": environment(Path(args.swift_src)),
        "artifacts": artifacts,
        "commands": commands,
        "manifest_documents": manifest_docs,
        "deviations_from_spec": args.deviation,
        "note": "hashes are of the artefacts as they exist now; re-run after any regeneration",
    }
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["environment"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
