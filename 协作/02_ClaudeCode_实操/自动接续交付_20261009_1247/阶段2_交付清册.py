#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CPU 交付协作者 · 阶段 2：本目录**交付清册 + 敏感扫描**（只写本目录）。

1. 列出本目录全部交付物，逐个记 sha256 与字节数；
2. 计算**总封条**（按 relative_path 排序后拼接 sha256 摘要字节再取 sha256，
   口径写明，避免与「hex 字符串拼接」混用）；
3. 敏感扫描：本目录内所有文本文件，命中私有信息模式即**列出并判失败**，
   不静默跳过（本交付全程不含 .env / SSH / API Key / CLI 日志 / 权重）。
"""
from __future__ import annotations
import hashlib, json, re, sys
from pathlib import Path

D = Path(__file__).resolve().parent
SKIP_DIRS = {"__pycache__", ".git"}
TEXT_EXT = {".md", ".json", ".jsonl", ".py", ".html", ".txt", ".csv", ".mjs", ".js", ".css"}

# 不参与内容扫描的文件：**规则字面量**与**扫描自身的产物**必然自命中，
# 扫它们只会产生假警报。这些文件仍然被列进清册并计 sha256，只是不做模式匹配。
SCAN_SKIP = {
    "阶段2_交付清册.py",     # 本脚本：PATTERNS 的字面量在这里
    "阶段2_建恢复包.py",     # FORBID 文件名黑名单的字面量在这里
    "交付清册.json",         # 上一轮扫描的输出（含命中片段文本）
    "交付清册.md",           # 同上
}

PATTERNS = [
    (r"sk-[A-Za-z0-9]{16,}", "疑似 API Key（sk-…）"),
    (r"(?i)authorization\s*[:=]\s*\S+", "疑似 Authorization 头"),
    (r"(?i)\bapi[_-]?key\b\s*[:=]\s*[\"']?[A-Za-z0-9_\-]{12,}", "疑似 api_key 赋值"),
    (r"DASHSCOPE_API_KEY\s*=\s*\S+", "疑似百炼 Key 赋值"),
    (r"(?i)\bssh\s+-[ip]\b|sshpass|id_rsa|id_ed25519", "疑似 SSH 私钥/口令用法"),
    (r"(?i)password\s*[:=]\s*\S+", "疑似口令"),
    (r"C:\\\\Users\\\\[^\\\\\s\"']+", "Windows 用户目录绝对路径"),
    (r"/c/Users/[^\s\"')]+", "本机用户目录绝对路径"),
    (r"(?i)\.codex[/\\]skills|\.claude[/\\]skills", "本机技能目录路径"),
    (r"谌伦杰", "真实用户名"),
    (r"https://[^/\s]*\.aliyuncs\.com/[^\s\"')]*", "含端点的私有接入地址"),
]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    files = sorted((p for p in D.rglob("*")
                    if p.is_file() and not any(s in p.parts for s in SKIP_DIRS)),
                   key=lambda p: p.relative_to(D).as_posix())

    SELF_FILES = {"交付清册.json", "交付清册.md"}
    entries = []
    for p in files:
        rel = p.relative_to(D).as_posix()
        if p.name in SELF_FILES:
            # 本清册自身的字节随「上一轮登记了什么」变化，登记其 sha256 会导致
            # 每跑一次值就变、无法复核；故只记字节数并注明原因。
            entries.append({"relative_path": rel, "bytes": p.stat().st_size,
                            "sha256": None,
                            "说明": "本清册自身：内容随运行变化，故不登记 sha256（也不计入总封条）"})
        else:
            entries.append({"relative_path": rel, "bytes": p.stat().st_size,
                            "sha256": sha(p)})

    # 总封条**排除本清册自身**（否则每跑一次值就变，无法复核）
    sealed = [e for e in entries if e["relative_path"] not in ("交付清册.json", "交付清册.md")]
    seal = hashlib.sha256(b"".join(bytes.fromhex(e["sha256"]) for e in sealed)).hexdigest()

    # ---- 敏感扫描 ----
    hits, policy = [], []
    for p in files:
        if p.suffix.lower() not in TEXT_EXT:
            continue
        if p.name in SCAN_SKIP:
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:                                    # noqa: BLE001
            continue
        for ln, line in enumerate(txt.splitlines(), 1):
            for pat, why in PATTERNS:
                m = re.search(pat, line)
                if not m:
                    continue
                rec = {"file": p.relative_to(D).as_posix(), "line": ln,
                       "why": why, "命中": m.group(0)[:80]}
                # 政策文本豁免：封包清册里**列出**被禁止打包的文件名模式，是红线的说明，
                # 不是泄漏本身。豁免项**仍然逐条列出**，不隐藏。
                if p.name == "封包SHA256清册.json" and any(
                        f in line for f in (".env", "id_rsa", "id_ed25519", ".pem",
                                            ".key", "credentials", "token")):
                    rec["分类"] = "政策文本（列出被禁模式，非泄漏）"
                    policy.append(rec)
                else:
                    hits.append(rec)

    report = {
        "目录": str(D),
        "生成时间": "2026-10-09",
        "生成者": "CPU 交付整理协作者",
        "文件数": len(entries),
        "总字节": sum(e["bytes"] for e in entries),
        "总封条口径": "按 relative_path 排序，拼接各文件 sha256 的 **digest 字节**后取 sha256"
                    "（不是 hex 字符串拼接；两种口径结果必然不同属正常）；"
                    "**不包含本清册自身**（`交付清册.json` / `交付清册.md`），故可重复复核",
        "总封条覆盖文件数": len(sealed),
        "总封条 sha256": seal,
        "文件": entries,
        "敏感扫描": {
            "扫描模式数": len(PATTERNS),
            "扫描文件数": sum(1 for p in files
                            if p.suffix.lower() in TEXT_EXT and p.name not in SCAN_SKIP),
            "未扫描": sorted(SCAN_SKIP) + ["二进制文件（.png/.zip/.pdf，无法按文本匹配）"],
            "未扫描原因": "规则字面量 / 本扫描自身的产物必然自命中，会产生假警报；"
                        "这些文件仍计入清册与总封条",
            "命中数": len(hits),
            "命中明细": hits[:50],
            "政策文本豁免数": len(policy),
            "政策文本豁免明细": policy[:20],
            "结论": "无命中" if not hits else "**有命中，需人工处置**",
        },
        "不含": ["SSH 配置/私钥", ".env", "API Key", "CLI 日志", "虚拟环境", "模型权重"],
    }
    (D / "交付清册.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    md = ["# 交付清册（本目录）与敏感扫描", "",
          f"- 文件数 **{len(entries)}**；总字节 **{report['总字节']/1048576:.2f} MiB**",
          f"- **总封条 sha256**：`{seal}`",
          f"- 口径：{report['总封条口径']}",
          f"- 敏感扫描：模式 {len(PATTERNS)} 个，命中 **{len(hits)}** → "
          f"{report['敏感扫描']['结论']}",
          f"- 另有**政策文本豁免** {len(policy)} 条（封包清册里列出被禁文件名模式，非泄漏，逐条列在下方）",
          f"- 不含：{'、'.join(report['不含'])}", "",
          "## 逐文件", "", "| 相对路径 | 字节 | sha256(前16) |", "|---|---:|---|"]
    for e in entries:
        h = e["sha256"]
        md.append(f"| `{e['relative_path']}` | {e['bytes']} | "
                  f"{'（本清册自身，不登记）' if h is None else '`' + h[:16] + '`'} |")
    if hits:
        md += ["", "## 敏感扫描命中", "", "| 文件 | 行 | 原因 | 命中片段 |", "|---|---:|---|---|"]
        for h in hits[:50]:
            md.append(f"| `{h['file']}` | {h['line']} | {h['why']} | `{h['命中']}` |")
    if policy:
        md += ["", "## 政策文本豁免（列出被禁模式，非泄漏）", "",
               "| 文件 | 行 | 原因 | 文本 |", "|---|---:|---|---|"]
        for h in policy[:20]:
            md.append(f"| `{h['file']}` | {h['line']} | {h['why']} | `{h['命中']}` |")
    (D / "交付清册.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    sys.stdout.buffer.write((json.dumps({
        "文件数": len(entries), "总字节": report["总字节"],
        "总封条sha256": seal,
        "敏感命中数": len(hits),
        "命中示例": hits[:5],
    }, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))
    return 0 if not hits else 3


if __name__ == "__main__":
    raise SystemExit(main())
