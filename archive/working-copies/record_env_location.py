"""Record where the environment physically lives, and why it was not moved.

The project rule is that data, models, environments and outputs live under
/root/autodl-fs/wafer-vlm. Everything does, except two entries that are symlinks:

    venvs/wafer     -> /root/autodl-tmp/wafer-vlm/venvs/wafer    (8.0G)
    src/ms-swift    -> /root/autodl-tmp/wafer-vlm/src/ms-swift   (49M)

so the interpreter and the ms-swift source are physically on the autodl-tmp side,
and the editable install maps `swift` to that path
(MAPPING = {'swift': '/root/autodl-tmp/wafer-vlm/src/ms-swift/swift'}).

This is benign and was left alone, for three reasons that are worth being
explicit about because the naive reading is "the rule was broken":

  1. /root/autodl-tmp and /root/autodl-fs are the same block device. /proc/mounts
     shows /dev/md0 mounted at both, differing only by project quota (50G vs
     200G, hence 21.8G vs 129.1G free). Nothing is on a different disk, so
     nothing is more likely to be lost.

  2. A venv is not relocatable by moving it. bin/* scripts carry absolute
     shebangs, pyvenv.cfg records `home`, and the editable finder has the path
     compiled in. Relocating means rebuilding the environment, which would
     replace the one that produced these results with a differently-built one --
     strictly worse for reproducing them. The manifest freeze files
     (manifests/venv_locked_freeze.txt, venv_reference_freeze.txt) describe THIS
     environment.

  3. It was found while queue 41 was training. Editing a live run's environment
     is the one action here that can silently invalidate a result.
"""
import json
import os
import subprocess
import time
from pathlib import Path

R = Path("/root/autodl-fs/wafer-vlm")
OUT = R / "outputs/reports/environment_location.json"

rec = {
    "recorded": time.strftime("%Y-%m-%d %H:%M:%S"),
    "question": "Does everything live under /root/autodl-fs/wafer-vlm as required?",
    "answer": "Yes for all data, models, outputs, benchmarks, projects, secrets, "
              "tools and the git repository. No, literally, for the environment: "
              "venvs/wafer and src/ms-swift are symlinks into "
              "/root/autodl-tmp/wafer-vlm, so the interpreter and ms-swift source "
              "are physically on the autodl-tmp side.",
    "symlinks": {
        "venvs/wafer": {"target": "/root/autodl-tmp/wafer-vlm/venvs/wafer",
                        "size": "8.0G"},
        "src/ms-swift": {"target": "/root/autodl-tmp/wafer-vlm/src/ms-swift",
                         "size": "49M"},
    },
    "editable_mapping": {
        "file": "venvs/wafer/lib/python3.12/site-packages/"
                "__editable___ms_swift_4_6_0_dev0_finder.py",
        "MAPPING": {"swift": "/root/autodl-tmp/wafer-vlm/src/ms-swift/swift"},
    },
    "same_device": {
        "evidence": "/proc/mounts lists /dev/md0 at both /root/autodl-fs and "
                    "/root/autodl-tmp, both xfs with prjquota; free space differs "
                    "(129.1G vs 21.8G) because the quotas differ, not the device.",
        "conclusion": "no data is on a volume with a different failure mode; this "
                      "is a quota-domain difference, not a durability difference",
    },
    "action_taken": "none",
    "why_not_moved": [
        "same block device as the required root, so moving it buys no safety",
        "a venv cannot be relocated by mv (absolute shebangs, pyvenv.cfg home, "
        "compiled-in editable MAPPING); relocating means rebuilding, which would "
        "replace the environment that produced the recorded results",
        "it was discovered while queue 41 was training",
    ],
    "verified_used_paths": {
        "model": "/root/autodl-fs/wafer-vlm/models/... (from the live swift "
                 "command line of pid 169230)",
        "outputs": "/root/autodl-fs/wafer-vlm/outputs/...",
    },
    "third_path_note": "/autodl-fs/data/wafer-vlm/projects/wafer-defect-vlm/src is "
                       "a separate autofs mount but resolves to the same inode as "
                       "/root/autodl-fs/wafer-vlm/projects/wafer-defect-vlm/src, "
                       "so the editable .pth naming it is not a fourth copy.",
    "tmp_models_note": "/root/autodl-tmp/wafer-vlm/models holds an older "
                       "Qwen3.5-9B (19G) that the current runs do not read; a "
                       "candidate for reclaim, but autodl-fs has 129G free and the "
                       "tmp quota is 21.8G, so there is no pressure and it was left "
                       "in place while training runs.",
}

OUT.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"wrote {OUT}")
print("action taken: none (recorded only)")
