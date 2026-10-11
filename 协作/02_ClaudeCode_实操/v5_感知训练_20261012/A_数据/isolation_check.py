#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""隔离检查（合同 4.4）。独立脚本，第 2 步与第 6/7 步复用。

check(train_ids, eval_ids, meta)：检查 sample_id / lot_name / png_sha256 三层交集；
交集非空或任一输入为空 → raise。
meta: {sample_id: {"lot_name":..., "png_sha256":...}}
"""


class IsolationError(AssertionError):
    pass


def _lots(ids, meta):
    return {meta[i]["lot_name"] for i in ids if i in meta}


def _shas(ids, meta):
    return {meta[i]["png_sha256"] for i in ids if i in meta and meta[i].get("png_sha256")}


def check(train_ids, eval_ids, meta, name_a="TRAIN", name_b="EVAL"):
    train_ids = set(train_ids)
    eval_ids = set(eval_ids)
    if not train_ids:
        raise IsolationError(f"{name_a} 为空集合——空集交集恒为 0，必须报错")
    if not eval_ids:
        raise IsolationError(f"{name_b} 为空集合——空集交集恒为 0，必须报错")
    i_id = train_ids & eval_ids
    i_lot = _lots(train_ids, meta) & _lots(eval_ids, meta)
    i_sha = _shas(train_ids, meta) & _shas(eval_ids, meta)
    problems = []
    if i_id:
        problems.append(f"sample_id 交集 {len(i_id)}：{sorted(i_id)[:5]}")
    if i_lot:
        problems.append(f"lot_name 交集 {len(i_lot)}：{sorted(i_lot)[:5]}")
    if i_sha:
        problems.append(f"png_sha256 交集 {len(i_sha)}：{sorted(i_sha)[:3]}")
    if problems:
        raise IsolationError(f"{name_a} vs {name_b} 隔离失败：" + "；".join(problems))
    return {"id_overlap": 0, "lot_overlap": 0, "sha_overlap": 0,
            "n_a": len(train_ids), "n_b": len(eval_ids)}


def check_subset(sub_ids, pool_ids, meta, name_sub="SUBSET", name_pool="POOL"):
    """子集检查：sub 必须全部落在 pool 内（用于 rl_candidates ∈ TRAIN_POOL、派生图 ∈ TRAIN_POOL）。"""
    sub_ids = set(sub_ids)
    pool_ids = set(pool_ids)
    if not sub_ids:
        raise IsolationError(f"{name_sub} 为空集合——必须报错")
    if not pool_ids:
        raise IsolationError(f"{name_pool} 为空集合——必须报错")
    outside = sub_ids - pool_ids
    if outside:
        raise IsolationError(f"{name_sub} 有 {len(outside)} 个不在 {name_pool} 内：{sorted(outside)[:5]}")
    return {"n_sub": len(sub_ids), "outside": 0}
