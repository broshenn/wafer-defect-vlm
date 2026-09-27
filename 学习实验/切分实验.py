# -*- coding: utf-8 -*-
"""切分实验：hash 分桶  vs  随机数抽样

直接运行：
    python 切分实验.py

对照的真实源码：
    code/projects/wafer-defect-vlm/src/wafer_vlm/utils.py:82-93

这个脚本只做一件事：让你亲眼看到
    "随机数分法"里，一个批次去哪，取决于它【排第几】；
    "hash 分法"里，一个批次去哪，只取决于它【叫什么】。
"""

import hashlib
import json
import random
from pathlib import Path

# ============ 你可以改这三个 ============

SEED = 3407        # 改成别的数（比如 0 或 123），看整个划分怎么变
TRAIN_EDGE = 80    # 桶号 < 80  -> train
VAL_EDGE = 90      # 桶号 < 90  -> val，其余 -> test
# 想算多少批次的对照统计，往小了调跑得快

# ========================================


def bucket_of(name, seed=SEED):
    """一个批次名 -> 0~99 的桶号。

    注意：这里只用到 name 这一个输入，没读任何别的批次、不知道总数、
    不知道它排第几。这就是"纯函数"的意思。
    """
    digest = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 100


def split_of(name, seed=SEED):
    b = bucket_of(name, seed)
    if b < TRAIN_EDGE:
        return "train", b
    if b < VAL_EDGE:
        return "val", b
    return "test", b


def split_by_random_each(order, seed=SEED):
    """随机数分法：按【位置】一个个发数。"""
    rng = random.Random(seed)
    out = {}
    for name in order:
        r = rng.random()
        out[name] = "train" if r < 0.8 else ("val" if r < 0.9 else "test")
    return out


def random_sequence(n=8, seed=SEED):
    """固定 seed 时，第几次调用 -> 永远同一个数。"""
    rng = random.Random(seed)
    return [(i + 1, rng.random()) for i in range(n)]


def find_manifest():
    """找项目里的真实批次名单（找不到就跳过实验 4）。"""
    here = Path(__file__).resolve().parent
    for base in (here.parent, here.parent.parent):
        p = base / "data" / "manifest.jsonl"
        if p.is_file():
            return p
    p = Path("data/manifest.jsonl")
    return p if p.is_file() else None


def line(title):
    print()
    print("=" * 62)
    print(title)
    print("=" * 62)


def main():
    line("实验 1：hash 分桶 —— 只看批次自己的名字")
    print(f"（当前 SEED = {SEED}，阈值 {TRAIN_EDGE} / {VAL_EDGE}）\n")
    print(f"{'批次名':<24}{'桶号':>6}{'去哪':>10}")
    for name in ["lot96", "lot534", "lot48", "lot1", "lot2", "lot6",
                 "随便编的批次名"]:
        s, b = split_of(name)
        print(f"{name:<24}{b:>6}{s:>10}")
    print("\n把 SEED 改成别的再跑一次 —— 所有批次会重新洗牌。")

    line("实验 2：随机数分法 —— 看清楚数是怎么按【位置】发下去的")
    print("固定 seed，第几次调用拿到什么数，是【永远不变】的：\n")
    print(f"{'第几个调用':<12}{'拿到的数':>12}")
    for i, r in random_sequence(8):
        where = "train" if r < 0.8 else ("val" if r < 0.9 else "test")
        print(f"{i:<12}{r:>12.4f}   -> {where}")
    print("\n位置 1 永远是同一个数，位置 2 也永远是同一个数。")
    print("但【谁】站在这个位置上，是会变的 —— 看实验 3。")

    line("实验 3：同一个批次，换个顺序就换结果")
    order_a = ["lot96", "lot534", "lot48", "lot1", "lot2", "lot6"]
    order_b = ["lot48", "lot1", "lot96", "lot534", "lot2", "lot6"]  # 同样的人，换顺序

    rng_a = random.Random(SEED)
    raw_a = {name: rng_a.random() for name in order_a}
    rng_b = random.Random(SEED)
    raw_b = {name: rng_b.random() for name in order_b}

    print(f"{'批次名':<10}{'顺序A 位置':>10}{'抽到':>10}{'顺序B 位置':>11}"
          f"{'抽到':>10}{'变了吗':>9}")
    for name in order_a:
        ia, ib = order_a.index(name) + 1, order_b.index(name) + 1
        changed = "变了" if raw_a[name] != raw_b[name] else "没变"
        print(f"{name:<10}{ia:>10}{raw_a[name]:>10.4f}"
              f"{ib:>11}{raw_b[name]:>10.4f}{changed:>9}")
    print("\n同一个批次，两次抽到的数不一样 —— 因为它换了位置。")
    print("换成 hash 分法，它是多少号就永远是 train/val/test，与顺序无关。")

    manifest = find_manifest()
    if manifest is None:
        print("\n（没找到 data/manifest.jsonl，实验 4 跳过）")
        return

    line("实验 4：拿项目【真实】的批次名单来数一数")
    lots = []
    seen = set()
    with manifest.open(encoding="utf-8") as fh:
        for row in fh:
            lot = json.loads(row)["lot_name"]
            if lot not in seen:
                seen.add(lot)
                lots.append(lot)
    print(f"清单：{manifest}")
    print(f"真实批次数：{len(lots)}")

    # hash 分法：顺序无关
    hash_a = {lot: split_of(lot)[0] for lot in lots}
    hash_b = {lot: split_of(lot)[0] for lot in reversed(lots)}
    hash_changed = sum(1 for lot in lots if hash_a[lot] != hash_b[lot])

    # 随机数分法：换个顺序
    rand_a = split_by_random_each(lots)
    rand_b = split_by_random_each(list(reversed(lots)))
    rand_changed = sum(1 for lot in lots if rand_a[lot] != rand_b[lot])

    print()
    print(f"{'分法':<14}{'换顺序后换了 split 的批次数':>28}")
    print(f"{'hash':<14}{hash_changed:>28}")
    print(f"{'随机数':<14}{rand_changed:>28}"
          f"   ({rand_changed / len(lots) * 100:.1f}%)")
    print("\nhash 分法换顺序后一个都没变；随机数分法乱了一片。")
    print("而这些批次如果混进考卷，分数就会虚高，且不会报错。")

    line("想接着玩，可以改这些")
    print("1. 把 SEED 改成 0，重跑 —— 看 hash 分法里所有批次重新洗牌。")
    print("2. 把 TRAIN_EDGE 改成 60 —— 看 train 变小、test 变大。")
    print("3. 在实验 1 里加你自己的批次名，看它落到哪一份。")
    print("4. 去掉实验 3 里 order_b 的注释，自己编一个新顺序试试。")


if __name__ == "__main__":
    main()
