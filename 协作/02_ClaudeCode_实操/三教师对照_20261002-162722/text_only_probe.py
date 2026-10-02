#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测「一条提示词文本」占多少 token —— 同题面、不带图。

用来把首次请求的 prompt_tokens 拆成「文本 + 图片」两部分，得到图片的 token 成本。
只发纯文本请求，仍受同一套凭据与守卫约束；客户端重试 0。
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_teacher_compare import (CANDIDATES, PROMPT, MAX_TOKENS_LOCKED,   # noqa: E402
                                 post_once, load_api_key)

OUT = HERE / "text_only_probe.json"


def main() -> int:
    prompt = PROMPT.read_text(encoding="utf-8")
    print(f"题面文本 {len(prompt.encode('utf-8'))} 字节 / {len(prompt)} 字符")
    out = {"prompt_bytes": len(prompt.encode("utf-8")), "prompt_chars": len(prompt),
           "note": "纯文本、不带图；用于与带图请求的 prompt_tokens 做差",
           "results": []}
    for name in ("deepseek", "qwen"):          # kimi 未开通，不发
        cand = CANDIDATES[name]
        key, src = load_api_key(cand["credential_env"])
        if not key:
            print(f"  {name}: 凭据未就绪，跳过")
            continue
        # 与带图请求**同一段文本**，只是没有图
        body = {"model": cand["model"],
                "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                "max_tokens": 64, "stream": False, "n": 1}
        r = post_once(cand["endpoint"], body, key)
        rec = {"candidate": name, "model": cand["model"], "http_status": r["http_status"],
               "elapsed_seconds": r["elapsed_seconds"], "error": r["error"],
               "raw_text": r["raw_text"][:1500]}
        try:
            o = json.loads(r["raw_text"])
            rec["usage"] = o.get("usage")
            rec["returned_model"] = o.get("model")
            rec["finish_reason"] = (o.get("choices") or [{}])[0].get("finish_reason")
        except Exception:
            rec["usage"] = None
        out["results"].append(rec)
        pt = (rec.get("usage") or {}).get("prompt_tokens")
        print(f"  {name:9s} HTTP={r['http_status']}  prompt_tokens={pt}  "
              f"returned={rec.get('returned_model')}")
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    print(f"\n写出 {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
