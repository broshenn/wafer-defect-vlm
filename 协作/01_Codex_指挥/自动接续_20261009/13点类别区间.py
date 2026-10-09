"""固定候选的同图配对区间：图为单位，不把字段当独立样本。"""
import importlib.util
import json
import random
from pathlib import Path

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("supplement", HERE / "13点补核.py")
s = importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
result = {}
for subset in ("frozen36", "exclude_train_same_image", "api_all_returned_and_no_train_same_image"):
    ids = s.sets[subset]
    gold = [s.gold[x] for x in ids]
    result[subset] = {}
    for other in ("Base", "qwen3.8-max-0902"):
        # 同一完整围栏诊断对所有模型施行，避免只给某个模型宽口径。
        lp = [s.a.pred_of(s.a.fence_parse(s.models["L-N3072-3407"][x])[0]) for x in ids]
        op = [s.a.pred_of(s.a.fence_parse(s.models[other][x])[0]) for x in ids]
        n = len(ids); rng = random.Random(3407); ad, fd = [], []
        for _ in range(5000):
            pick = [rng.randrange(n) for _ in range(n)]
            lg = [(gold[i], lp[i]) for i in pick]
            og = [(gold[i], op[i]) for i in pick]
            ad.append(sum(g == p for g, p in lg)/n - sum(g == p for g, p in og)/n)
            fd.append(s.a.prf(lg)[0] - s.a.prf(og)[0])
        ci = lambda v: [sorted(v)[124], sorted(v)[4874]]
        result[subset]["L_minus_" + other] = {"n": n,
            "accuracy_delta": sum(g == p for g, p in zip(gold, lp))/n - sum(g == p for g, p in zip(gold, op))/n,
            "accuracy_bootstrap95": ci(ad),
            "macro_f1_delta": s.a.prf(list(zip(gold, lp)))[0] - s.a.prf(list(zip(gold, op)))[0],
            "macro_f1_bootstrap95": ci(fd),
            "L_only_correct": sum(p == g and q != g for g, p, q in zip(gold, lp, op)),
            "other_only_correct": sum(p != g and q == g for g, p, q in zip(gold, lp, op)),
            "replicates": 5000, "seed": 3407}
out = {"scope": "开发集条件性图级配对区间；不反映模型复核错误或实际业务总体不确定性，不替原主表", "comparisons": result}
(HERE / "13点类别配对区间.json").write_text(json.dumps(out, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=2))
