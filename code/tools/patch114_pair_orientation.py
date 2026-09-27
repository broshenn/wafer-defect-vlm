"""A pair entry is a flat list in lr order, and that order is an assumption worth reading.

patch113 crashed with `KeyError: 1e-05`: `matched_pairs.<name>` is not keyed by learning
rate the way `runs.<tag>` entries are. It is one dict per pair with the two runs' values
side by side in two-element lists --

    "GRPO G=4": {"lr": [1e-05, 5e-05], "kl": [1.106793, 1.043522],
                 "completion_length": [115.48, 97.92], "idle": [0.36, 0.326667], ...}

-- so position carries the meaning, and the meaning is "index 0 is the lower rate". Every
item that reads a pair's two numbers assumes that. If the tool ever wrote them in the other
order, the KL item would compare lo against hi and still print ok, because both numbers
belong to the pair; only the *comparison* in the sentence (KL falls when the rate rises,
the completion shortens) would be silently about the wrong direction.

So the orientation is read rather than assumed: `lr` must be ascending, and `kl` must equal
the two runs' `mean_kl` in that same order -- which is the one place the two halves of the
record are compared directly, and the reason `_rec_kl` keeps reading `runs` while `_rec_len`
reads `matched_pairs`.
"""
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path("/root/autodl-fs/wafer-vlm")
CHECKER = ROOT / "tools/check_quantified_claims.py"
BAK = "/tmp/check_quantified_claims.py.bak-patch114"

C_OLD = '''    _mp2 = {k: v for k, v in _mp.items() if k in _kpairs}
    # What the record says about each pair, in the shapes the sentences quote it in.
    _rec_kl = {k: (round(v[1e-05]["mean_kl"], 4), round(v[5e-05]["mean_kl"], 4))
               for k, v in _kpairs.items()}
    _rec_len = {k: (round(v[1e-05]["completion_length"], 1),
                    round(v[5e-05]["completion_length"], 1))
                for k, v in _mp2.items()}
    # Section 8 quotes the lengths exactly, so the percentage printed beside them is
    # required to follow from them; 5.2.4 quotes them to 1 dp and prints no percentage.
    _rec_len2 = {k: (round(v[1e-05]["completion_length"], 2),
                     round(v[5e-05]["completion_length"], 2),
                     round(100 * (v[5e-05]["completion_length"]
                                  / v[1e-05]["completion_length"] - 1), 1))
                 for k, v in _mp2.items()}
    _rec_idle = {k: (round(100 * v[1e-05]["idle"], 2), round(100 * v[5e-05]["idle"], 2),
                     round(v[5e-05]["idle"] - v[1e-05]["idle"], 4))
                 for k, v in _mp2.items()}
'''

C_NEW = '''    _mp2 = {k: v for k, v in _mp.items() if k in _kpairs}
    # A pair entry holds the two runs side by side in two-element lists, so position is
    # what carries the meaning: index 0 is the lower learning rate. Read, not assumed --
    # if the tool wrote them the other way, every value below would still belong to the
    # pair and the items would still print ok, while the sentence's claims about
    # *direction* (KL falls, the completion shortens) would be about the wrong way round.
    _orient = [k for k, v in _mp2.items()
               if not (isinstance(v.get("lr"), list) and len(v["lr"]) == 2
                       and v["lr"][0] < v["lr"][1])]
    cmp("every pair entry's two runs are listed at the lower rate first", [], _orient,
        why="the sentences compare the two members of each pair; that comparison only "
            "means what it says if the position means what this item checks")
    _mismatch = []
    for _k, _v in _mp2.items():
        _r = _kpairs.get(_k) or {}
        _want = [round(_r[x]["mean_kl"], 6) for x in (1e-05, 5e-05) if x in _r]
        if isinstance(_v.get("kl"), list) and len(_want) == 2:
            if [round(x, 6) for x in _v["kl"]] != _want:
                _mismatch.append(f"{_k[0]}(G={_k[1]})")
    cmp("the pair entries and the run entries agree on the KL, in the same order", [],
        _mismatch,
        why="the only place the two halves of the record are compared directly. The KL "
            "items read `runs` and the length and idle items read `matched_pairs`; if the "
            "two disagree there is no way to say which half a sentence is quoting")
    # What the record says about each pair, in the shapes the sentences quote it in.
    _rec_kl = {k: (round(v[1e-05]["mean_kl"], 4), round(v[5e-05]["mean_kl"], 4))
               for k, v in _kpairs.items()}
    _rec_len = {k: (round(v["completion_length"][0], 1), round(v["completion_length"][1], 1))
                for k, v in _mp2.items()}
    # Section 8 quotes the lengths exactly, so the percentage printed beside them is
    # required to follow from them; 5.2.4 quotes them to 1 dp and prints no percentage.
    _rec_len2 = {k: (round(v["completion_length"][0], 2),
                     round(v["completion_length"][1], 2),
                     round(100 * (v["completion_length"][1]
                                  / v["completion_length"][0] - 1), 1))
                 for k, v in _mp2.items()}
    _rec_idle = {k: (round(100 * v["idle"][0], 2), round(100 * v["idle"][1], 2),
                     round(v["idle"][1] - v["idle"][0], 4))
                 for k, v in _mp2.items()}
'''

c = CHECKER.read_text(encoding="utf-8")
shutil.copy2(CHECKER, BAK)
if c.count(C_OLD) != 1:
    sys.exit(f"the pair-value block appears {c.count(C_OLD)} times, expected 1; "
             f"nothing written")

c = c.replace(C_OLD, C_NEW, 1)
CHECKER.write_text(c, encoding="utf-8")

r = subprocess.run([sys.executable, "-m", "py_compile", str(CHECKER)],
                   capture_output=True, text=True)
if r.returncode != 0:
    shutil.copy2(BAK, CHECKER)
    sys.exit(f"the patched checker does not compile, restored:\n{r.stderr}")

print("the two halves of the record are now read in the same order, and that order is "
      "checked rather than assumed.")
print("\n--- re-running the checker ---")
r = subprocess.run([sys.executable, str(CHECKER)], capture_output=True, text=True,
                   cwd=str(ROOT))
out = r.stdout.strip().splitlines()
for ln in out:
    s_ = ln.strip()
    if ("pair" in s_ or "KL" in s_ or "completion" in s_ or "enumerates" in s_
            or "raises the share" in s_ or "lower rate first" in s_):
        print("  " + s_)
if r.stderr.strip():
    print("--- stderr ---")
    for ln in r.stderr.strip().splitlines()[-12:]:
        print("  " + ln)
print("  ...")
for ln in out[-8:]:
    print("  " + ln.strip())
print(f"\nchecker exit {r.returncode}")
