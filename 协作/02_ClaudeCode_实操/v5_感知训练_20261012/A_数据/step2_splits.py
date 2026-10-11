#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 2 步：划分与隔离检查（合同 4.2–4.4）。
输入：geom_all.jsonl + 8 个受保护文件。
输出：TRAIN_POOL.txt、SELECT200.txt、TEST.txt、TEST200.txt、ROT200.txt、SHUF100.txt（各一行一个 id，排序）。
验收：打印四个集合大小与类别分布、HOLD56/CONF120 命中数、isolation_check 通过；
      test_isolation.py 4/4 PASS（另跑）。
"""
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))
from isolation_check import check, check_subset  # noqa: E402

PROTECTED_FILES = [
    REPO / "benchmark/core.jsonl",
    REPO / "协作/02_ClaudeCode_实操/56图候选补测与最终结项_20261009/冻结清单.json",
    REPO / "协作/02_ClaudeCode_实操/收尾_RL90与报告_20261006/R0_dev18.jsonl",
    REPO / "协作/02_ClaudeCode_实操/收尾_RL90与报告_20261006/R0_val90.jsonl",
    REPO / "协作/02_ClaudeCode_实操/多字段RL_v2_CPU准备_20261009/清单_确认池.jsonl",
    REPO / "协作/02_ClaudeCode_实操/多字段RL_v2_CPU准备_20261009/清单_规则开发池.jsonl",
    REPO / "协作/01_Codex_指挥/无卡CPU整备_20261010/pools/confirmation.jsonl",
    REPO / "协作/01_Codex_指挥/无卡CPU整备_20261010/pools/rule_dev.jsonl",
]
SID_RE = re.compile(r"wafer_\d{8}_\d{3}")


def read_field_jsonl(path):
    """每行 JSON，取 sample_id 字段。"""
    ids = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            ids.add(json.loads(line)["sample_id"])
    return ids


def read_regex_json(path):
    """对整个文件文本做 sample_id 正则（用于嵌套/未知结构的 JSON）。"""
    text = Path(path).read_text(encoding="utf-8")
    return set(SID_RE.findall(text))


def main():
    geo = [json.loads(l) for l in (HERE / "geom_all.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    meta = {r["sample_id"]: r for r in geo}
    print(f"geom_all 可用图 {len(meta)}")

    # ── 1. 受保护 ID ──
    protected = {}
    all_protected = set()
    for p in PROTECTED_FILES:
        head = p.read_text(encoding="utf-8")[:300].replace("\n", " ")
        reader = read_regex_json if p.suffix == ".json" else read_field_jsonl
        ids = reader(p)
        protected[p.name] = ids
        all_protected |= ids
        print(f"[protected] {p.name}: 读到 {len(ids)} 个 ID；头300字: {head[:120]}…")
    for name, ids in protected.items():
        assert len(ids) > 0, f"{name} 读到 0 个 ID"
    print(f"受保护合计（去重）{len(all_protected)}；其中在可用图里的 "
          f"{len(all_protected & set(meta))}；落在 train 的在 TRAIN_POOL 中剔除")

    # ── 2. 四个集合 ──
    train_avail = sorted(s for s, r in meta.items() if r["split"] == "train")
    val_avail = sorted(s for s, r in meta.items() if r["split"] == "val")
    test_avail = sorted(s for s, r in meta.items() if r["split"] == "test")
    print(f"可用：train {len(train_avail)} / val {len(val_avail)} / test {len(test_avail)}")

    # TEST = test 全部可用图
    TEST = test_avail

    def strat_sample(ids, k, seed, min_per_class=5):
        """按 failure_type 分层随机抽样（random.Random(seed)），每类至少 min(5, 该类数量) 张，不足全取。"""
        groups = {}
        for s in ids:
            groups.setdefault(meta[s]["failure_type"], []).append(s)
        counts = {c: len(v) for c, v in groups.items()}
        total = sum(counts.values())
        k = min(k, total)
        floor_min = {c: min(min_per_class, counts[c]) for c in counts}
        alloc = {c: max(floor_min[c], min(counts[c], round(k * counts[c] / total))) for c in counts}
        # 调整到恰好 k
        order_hi = sorted(counts, key=lambda c: (-(alloc[c] - floor_min[c]), c))
        while sum(alloc.values()) > k:
            for c in order_hi:
                if alloc[c] > floor_min[c]:
                    alloc[c] -= 1
                    break
            else:
                raise RuntimeError("alloc 无法缩减")
        order_lo = sorted(counts, key=lambda c: (-(counts[c] - alloc[c]), c))
        while sum(alloc.values()) < k:
            for c in order_lo:
                if alloc[c] < counts[c]:
                    alloc[c] += 1
                    break
            else:
                raise RuntimeError("alloc 无法增长")
        rng = random.Random(seed)
        picked = []
        for c in sorted(counts):
            picked.extend(rng.sample(sorted(groups[c]), alloc[c]))
        assert len(picked) == k
        return sorted(picked)

    SELECT200 = strat_sample(val_avail, 200, 3407)
    TEST200 = strat_sample(TEST, 200, 3407)          # P0 评测子集
    ROT200 = TEST200                                  # 合同：ROT200 与 TEST200 同一次分层抽样
    SHUF100 = strat_sample(TEST, 100, 3407)

    # TRAIN_POOL = train 可用 - 受保护 - 与 SELECT200/TEST SHA 相同
    sel_test_shas = {meta[s]["png_sha256"] for s in set(SELECT200) | set(TEST)}
    pool = [s for s in train_avail if s not in all_protected and meta[s]["png_sha256"] not in sel_test_shas]
    TRAIN_POOL = sorted(pool)
    n_prot_train = sum(1 for s in train_avail if s in all_protected)
    n_sha_train = sum(1 for s in train_avail if s not in all_protected and meta[s]["png_sha256"] in sel_test_shas)
    print(f"TRAIN_POOL {len(TRAIN_POOL)}（train 可用 {len(train_avail)} - 受保护 {n_prot_train} - SHA 重叠 {n_sha_train}）")

    # ── 3. 隔离检查（合同 4.4）──
    r1 = check(TRAIN_POOL, set(SELECT200) | set(TEST), meta, "TRAIN_POOL", "SELECT200∪TEST")
    r2 = check(TRAIN_POOL, ROT200, meta, "TRAIN_POOL", "ROT200(基图)")
    r3 = check(TRAIN_POOL, SHUF100, meta, "TRAIN_POOL", "SHUF100(基图)")
    print("隔离检查通过:", r1, r2, r3)

    # ── 4. HOLD56 / CONF120 命中 ──
    hold56 = protected["冻结清单.json"]
    conf120 = protected["confirmation.jsonl"]
    print(f"HOLD56（冻结清单 ∩ TEST）命中 {len(hold56 & set(TEST))} / 56")
    print(f"CONF120（v3 确认池 ∩ TEST）命中 {len(conf120 & set(TEST))} / {len(conf120)}")

    # ── 5. 写出 ──
    for name, ids in [("TRAIN_POOL", TRAIN_POOL), ("SELECT200", SELECT200), ("TEST", TEST),
                      ("TEST200", TEST200), ("ROT200", ROT200), ("SHUF100", SHUF100)]:
        (HERE / f"{name}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8", newline="\n")
        dist = Counter(meta[s]["failure_type"] for s in ids)
        print(f"{name}: {len(ids)} 张 分布={dict(sorted(dist.items()))}")

    check_subset(SELECT200, val_avail, meta, "SELECT200", "val可用")
    check_subset(TEST200, TEST, meta, "TEST200", "TEST")
    check_subset(SHUF100, TEST, meta, "SHUF100", "TEST")
    return 0


if __name__ == "__main__":
    sys.exit(main())
