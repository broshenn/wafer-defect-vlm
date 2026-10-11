#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CNN 类别基线（B 卡 12；合同 8.4）。
torchvision ResNet18（weights=None，num_classes=9）；输入 224×224（448 图最近邻缩小）；
TRAIN_POOL 原图训练 20 轮，Adam 1e-3，batch 64，seed 3407；TEST 报类别准确率与 Macro-F1，
写 per_sample.jsonl（与 score_v5 同格式，便于 bootstrap）。
torchvision 不可用时改用 4 层卷积小网络并在报告写明。
用法：python cnn_v5.py --train-jsonl sft_meta_train.jsonl --test-jsonl test_main.jsonl \
      --out eval/CNN --seed 3407 [--epochs 20]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CLASSES = ['Center', 'Donut', 'Edge_Loc', 'Edge_Ring', 'Loc', 'Near_full', 'Random',
           'Scratch', 'none']


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--test-jsonl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    a = ap.parse_args()
    import numpy as np
    import torch
    import torch.nn as nn
    from PIL import Image

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev} torch={torch.__version__}")

    def load_rows(p):
        return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

    train_rows = load_rows(a.train_jsonl)
    test_rows = load_rows(a.test_jsonl)
    print(f"train {len(train_rows)} test {len(test_rows)}")

    def to_tensor(path):
        img = Image.open(path).convert("RGB").resize((224, 224), Image.NEAREST)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    try:
        import torchvision
        from torchvision.models import resnet18
        net = resnet18(weights=None, num_classes=len(CLASSES))
        impl = f"torchvision ResNet18（torchvision {torchvision.__version__}）"
    except Exception as exc:
        print(f"torchvision 不可用（{exc}），改用 4 层卷积小网络")

        class Small(nn.Module):
            def __init__(self):
                super().__init__()
                self.body = nn.Sequential(
                    nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
                    nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
                    nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),
                    nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1))
                self.head = nn.Linear(256, len(CLASSES))

            def forward(self, x):
                return self.head(self.body(x).flatten(1))

        net = Small()
        impl = "4 层卷积小网络（torchvision 不可用）"
    net = net.to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    y = {c: i for i, c in enumerate(CLASSES)}
    X_train = torch.stack([to_tensor(r["images"][0]) for r in train_rows])
    Y_train = torch.tensor([y[r["failure_type"]] for r in train_rows])
    X_test = torch.stack([to_tensor(r["images"][0]) for r in test_rows])
    for ep in range(a.epochs):
        net.train()
        perm = torch.randperm(len(X_train))
        tot = 0.0
        for i in range(0, len(perm), a.batch_size):
            idx = perm[i:i + a.batch_size]
            opt.zero_grad()
            out = net(X_train[idx].to(dev))
            loss = lossf(out, Y_train[idx].to(dev))
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        print(f"  epoch {ep + 1}/{a.epochs} loss={tot / len(X_train):.4f}", flush=True)
    net.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_test), 128):
            out = net(X_test[i:i + 128].to(dev))
            preds.extend(out.argmax(1).cpu().tolist())
    per = []
    tp = {c: 0 for c in CLASSES}
    fp = {c: 0 for c in CLASSES}
    fn = {c: 0 for c in CLASSES}
    for r, p in zip(test_rows, preds):
        truth = r["failure_type"]
        pred = CLASSES[p]
        per.append({"sample_id": r["sample_id"], "lot_name": r.get("lot_name", ""),
                    "present": True, "class_pred": pred, "class_ok": pred == truth,
                    "true_class": truth})
        if pred == truth:
            tp[truth] += 1
        else:
            fn[truth] += 1
            fp[pred] += 1
    f1s = []
    per_class = {}
    for c in CLASSES:
        p = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        rr = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        f1 = 2 * p * rr / (p + rr) if (p + rr) else 0.0
        per_class[c] = {"precision": round(p, 4), "recall": round(rr, 4), "f1": round(f1, 4),
                        "support": tp[c] + fn[c]}
        f1s.append(f1)
    acc = sum(1 for r in per if r["class_ok"]) / len(per)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "per_sample.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in per:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    metrics = {"kind": "CNN", "impl": impl, "epochs": a.epochs, "seed": a.seed,
               "class": {"acc": round(acc, 5), "macro_f1": round(sum(f1s) / len(f1s), 5),
                         "per_class": per_class}}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1) + "\n",
                                          encoding="utf-8", newline="\n")
    print(f"CNN acc={metrics['class']['acc']} macro_f1={metrics['class']['macro_f1']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
