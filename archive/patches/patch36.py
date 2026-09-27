"""Make 29_grpo_train.sh write its result where it says it will.

The launcher passes the result path to the embedded Python as argv[1] and the
Python unpacks it into a variable named `result` -- then never uses it, writing
instead to a hardcoded "outputs/reports/grpo_train_result.json". patch29b fixed
the shell variable that feeds argv[1] and the parameterised name appeared to
work, so the second GSPO run was launched believing the records were separate.

It was not. The lr 1e-5 run overwrote the GRPO record, exactly as the first GSPO
run had before patch29b. That one was recoverable only because the GRPO record
had already been committed; this one likewise, and by the same luck. The fix is
one line: write to the path that was passed in.
"""
import ast
import sys
from pathlib import Path

path = Path("/root/autodl-fs/wafer-vlm/projects/wafer-defect-vlm/scripts/29_grpo_train.sh")
src = path.read_text(encoding="utf-8")

old = 'out = Path(root) / "outputs/reports/grpo_train_result.json"'
new = ('# argv[1], not a literal: the launcher computes a per-run path and passes it\n'
       '# in, so a hardcoded write here silently collides between runs. It did --\n'
       '# twice, both times overwriting the GRPO record.\n'
       'out = Path(result)')

n = src.count(old)
if n != 1:
    sys.exit(f"FAILED: {n} occurrences of the hardcoded output path")
src = src.replace(old, new)

# The write is inside a bash heredoc; check the Python it delimits still parses.
start = src.index("<<'PY'") + len("<<'PY'")
end = src.index("\nPY\n", start)
ast.parse(src[start:end])
path.write_text(src, encoding="utf-8")
print("29_grpo_train.sh: result now written to the path argv[1] carries")
