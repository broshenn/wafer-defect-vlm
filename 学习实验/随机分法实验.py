# -*- coding: utf-8 -*-
"""随机数分法实验：看清 random() 是怎么按【位置】发号的

运行：
    python 随机分法实验.py

要玩的东西都在下面那个「改这里」里，改完直接重跑。
"""

import random

# ==================== 改这里 ====================

SEED = 3407                                      # 固定住种子
ORDER = ["lot96", "lot534", "lot48", "lot1", "lot2", "lot6"]   # 批次名单（顺序随便调）

TRAIN_EDGE = 0.8    # 抽到的数 < 0.8   -> train
VAL_EDGE = 0.9      # 抽到的数 < 0.9   -> val，其余 -> test

# ================================================


def where(r):
    """一个 0~1 的数落在哪一份。"""
    if r < TRAIN_EDGE:
        return "train"
    if r < VAL_EDGE:
        return "val"
    return "test"


def assign(order, seed=SEED):
    """按名单顺序，一个个发号。"""
    rng = random.Random(seed)
    rows = []
    for pos, name in enumerate(order, start=1):
        r = rng.random()
        rows.append((pos, name, r, where(r)))
    return rows


def line(title):
    print()
    print("=" * 58)
    print(title)
    print("=" * 58)


def main():
    line("第 1 部分：固定 seed 时，第几次调用 -> 永远同一个数")
    print("这里【不看批次】，只看「第几个调用」：\n")
    rng = random.Random(SEED)
    print(f"{'第几个调用':<12}{'拿到的数':>12}{'落在哪':>10}")
    for i in range(1, 9):
        r = rng.random()
        print(f"{i:<12}{r:>12.4f}{where(r):>10}")
    print("\n看第 1 行和第 5 行 —— 不管跑多少次，它们永远是这个数。")

    line("第 2 部分：按当前 ORDER 分下去")
    print(f"ORDER = {ORDER}\n")
    print(f"{'位置':<6}{'批次名':<12}{'抽到':>10}{'落在哪':>10}")
    for pos, name, r, w in assign(ORDER):
        print(f"{pos:<6}{name:<12}{r:>10.4f}{w:>10}")
    print("\n注意：位置 1 拿到的数，跟第 1 部分的「第 1 个调用」完全一样。")
    print("数是发给【位置】的，不是发给【批次】的。")

    line("第 3 部分：把名单反过来，看谁变了")
    reversed_order = list(reversed(ORDER))
    print(f"ORDER  = {ORDER}")
    print(f"反过来 = {reversed_order}\n")
    a = {name: (r, w) for _, name, r, w in assign(ORDER)}
    b = {name: (r, w) for _, name, r, w in assign(reversed_order)}
    print(f"{'批次名':<12}{'原顺序':>9}{'去哪':>8}"
          f"{'反过来':>10}{'去哪':>8}{'变了吗':>9}")
    for name in ORDER:
        ra, wa = a[name]
        rb, wb = b[name]
        flag = "变了" if ra != rb else "没变"
        print(f"{name:<12}{ra:>9.4f}{wa:>8}{rb:>10.4f}{wb:>8}{flag:>9}")
    print("\n同一个批次，只是换了位置，抽到的数就变了。")
    print("（这次 6 个全变了 —— 因为反过来之后没有一个批次留在原位。）")
    print("重点看 lot534：它从 train 掉到了 val。")

    line("自己试试")
    print("1. 改 ORDER 的顺序，重跑 —— 看同一个批次落到哪儿变了。")
    print("2. 改 SEED = 0，重跑 —— 看整个序列换了一套数。")
    print("3. 改 TRAIN_EDGE = 0.5，重跑 —— 看 train 变小、val/test 变大。")
    print("4. 在 ORDER 里加一个新批次名放在【开头】，重跑 ——")
    print("   看它后面所有人的数是不是全往后挪了一位。")


if __name__ == "__main__":
    main()
