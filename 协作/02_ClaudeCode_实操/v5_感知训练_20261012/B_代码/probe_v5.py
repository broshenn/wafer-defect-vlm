#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""线性探针（B 卡 11；合同 8.4，P2）。
原生基座视觉编码器输出（merger 之后的图像 token）→ 平均池化 + 4×4 空间平均池化拼接
→ PCA 到 256 维（numpy SVD）→ 用 TRAIN_POOL 原图 2000 张训练 LogisticRegression(max_iter=2000)
→ TEST 预测 coverage_band / red_mass_zones（整个集合当一个类别）/ angular_type；
报告准确率与多数类基线。输出 probe.json。
注意（DECISIONS）：视觉塔具体调用方式本机无法验证，需会话 C 在服务器上用 5 张图试跑确认。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--test-jsonl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--geom-jsonl", required=True, help="geom_all.jsonl：训练标签来源")
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--limit", type=int, default=0, help="仅用于小样本试跑")
    a = ap.parse_args()
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    def load_rows(p):
        return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

    train_rows = load_rows(a.train_jsonl)[: a.n_train]
    test_rows = load_rows(a.test_jsonl)
    if a.limit:
        train_rows = train_rows[: a.limit]
        test_rows = test_rows[: a.limit]

    proc = AutoProcessor.from_pretrained(a.model)
    model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16)
    model.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)

    def image_tokens(path):
        img = Image.open(path).convert("RGB")
        inputs = proc.image_processor(images=[img], return_tensors="pt")
        pv = inputs["pixel_values"].to(dev, dtype=torch.bfloat16)
        grid = inputs.get("image_grid_thw")
        g2 = grid.to(dev) if grid is not None else None
        with torch.no_grad():
            visual = model.visual if hasattr(model, "visual") else model.model.visual
            feats = visual(pv, grid_thw=g2) if g2 is not None else visual(pv)
        if not torch.is_tensor(feats):
            for attr in ("last_hidden_state", "hidden_states", "pooler_output"):
                v = getattr(feats, attr, None)
                if torch.is_tensor(v):
                    feats = v
                    break
            else:
                if isinstance(feats, (tuple, list)):
                    feats = feats[0]
        feats = feats.flatten(0, 1) if feats.dim() == 3 else feats  # [tokens, h]
        f = feats.float().cpu().numpy()
        mean_pool = f.mean(0)
        if grid is not None:
            thw = grid[0].tolist() if hasattr(grid[0], "tolist") else list(grid[0])
            t, hg, wg = int(thw[0]), int(thw[1]), int(thw[2])
            if hg > 0 and wg > 0 and t * hg * wg == len(f):
                gridf = f.reshape(t, hg, wg, -1).reshape(hg, wg, -1)
                # 4×4 空间平均池化
                hs = np.array_split(np.arange(hg), 4)
                ws = np.array_split(np.arange(wg), 4)
                cells = [gridf[np.ix_(hh, ww)].mean((0, 1)) for hh in hs for ww in ws]
                return np.concatenate([mean_pool] + cells)
        return np.concatenate([mean_pool] + [mean_pool] * 16)

    def extract(rows, tag):
        feats, labels = [], []
        for i, r in enumerate(rows, 1):
            f = image_tokens(r["images"][0])
            feats.append(f)
            ref = json.loads(r["reference_json"]) if "reference_json" in r else None
            if ref is None:
                labels.append(None)
            else:
                labels.append((ref["coverage_band"], json.dumps(ref["red_mass_zones"], sort_keys=True),
                               ref["angular_type"]))
            if i % 100 == 0:
                print(f"  {tag} {i}/{len(rows)}", flush=True)
        return np.stack(feats), labels

    X_train, _ = extract(train_rows, "train-feat")
    X_test, y_test = extract(test_rows, "test-feat")
    print("特征维度", X_train.shape, X_test.shape)

    # PCA 256（numpy SVD）
    mu = X_train.mean(0, keepdims=True)
    Xc = X_train - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = min(256, Vt.shape[0])
    W = Vt[:k].T
    Z_train = Xc @ W
    Z_test = (X_test - mu) @ W

    # 探针训练标签来自 geom_all.jsonl（sft_meta 只有 id/类/图；TEST 侧用 reference_json）
    geo = {json.loads(l)["sample_id"]: json.loads(l) for l in open(a.geom_jsonl, encoding="utf-8") if l.strip()}
    tr_targets = {
        "coverage_band": [geo[r["sample_id"]]["coverage_band"] for r in train_rows],
        "red_mass_zones": [json.dumps(geo[r["sample_id"]]["red_mass_zones"], sort_keys=True) for r in train_rows],
        "angular_type": [geo[r["sample_id"]]["angular_type"] for r in train_rows],
    }
    try:
        from sklearn.linear_model import LogisticRegression
        impl = "sklearn LogisticRegression(max_iter=2000)"
        make_clf = lambda: LogisticRegression(max_iter=2000)  # noqa: E731
    except Exception:
        impl = "numpy 多类 softmax（全批量梯度下降 500 步、L2=1e-3）"

        class Softmax:
            def fit(self, X, y):
                classes = sorted(set(y))
                self.classes_ = classes
                Y = np.zeros((len(y), len(classes)))
                for i, c in enumerate(y):
                    Y[i, classes.index(c)] = 1
                Xb = np.hstack([X, np.ones((len(X), 1))])
                Wm = np.zeros((Xb.shape[1], len(classes)))
                for _ in range(500):
                    logits = Xb @ Wm
                    logits -= logits.max(1, keepdims=True)
                    P = np.exp(logits)
                    P /= P.sum(1, keepdims=True)
                    Wm -= 1e-1 * (Xb.T @ (P - Y) / len(X) + 1e-3 * Wm)
                self.Wm = Wm
                return self

            def predict(self, X):
                Xb = np.hstack([X, np.ones((len(X), 1))])
                return [self.classes_[i] for i in (Xb @ self.Wm).argmax(1)]

        make_clf = lambda: Softmax()  # noqa: E731

    results = {}
    for field, labels in tr_targets.items():
        ytr_clean = [lab if lab != "unknown" else "__unknown__" for lab in labels]
        idx = {"coverage_band": 0, "red_mass_zones": 1, "angular_type": 2}[field]
        yte = [t[idx] for t in y_test]   # extract() 返回元组 (band, zones_json, angular)
        yte = [t if t != "unknown" else "__unknown__" for t in yte]
        clf = make_clf().fit(Z_train, ytr_clean)
        pred = clf.predict(Z_test)
        acc = sum(1 for p, t in zip(pred, yte) if p == t) / len(yte)
        maj = Counter(yte).most_common(1)[0]
        results[field] = {"acc": round(acc, 5), "n": len(yte),
                          "majority_class": maj[0], "majority_baseline": round(maj[1] / len(yte), 5)}
        print(f"{field}: acc={acc:.4f} 多数类基线={maj[1] / len(yte):.4f}")
    mean_acc = np.mean([v["acc"] for v in results.values()])
    out = {"impl": impl, "feature_dim_raw": int(X_train.shape[1]), "pca_dim": int(k),
           "n_train": len(train_rows), "n_test": len(test_rows),
           "fields": results, "mean_acc": round(float(mean_acc), 5)}
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "probe.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n",
                                        encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
