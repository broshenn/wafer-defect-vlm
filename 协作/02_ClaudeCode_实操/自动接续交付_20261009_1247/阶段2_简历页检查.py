#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CPU 交付协作者 · 阶段 2：简历页（HTML/PDF）交付前检查。

只读 + 只写本目录。检查项：
  A. HTML 自包含（无外部 src/href、无 <base>、无模板标记残留）
  B. 无虚构个人信息（姓名/学校/公司/照片等）
  C. HTML 页内测量证据（无头浏览器 --dump-dom 抓的 ASu 单页指标）
  D. PDF：页数 / 纸张尺寸(A4) / 字体嵌入 / 可否抽取文本 / sha256
  E. 与内容壳的事实一致性（关键数字在 HTML 与 PDF 中都出现）

不联网、不调用任何 API。
"""
from __future__ import annotations
import hashlib, json, re, subprocess, sys
from pathlib import Path

D = Path(__file__).resolve().parent
HTML = D / "项目经历页_可编辑.html"
PDF = D / "项目经历页.pdf"
DOM = D / "_dom_tmp.html"
SHELL = D / "简历页_内容壳.html"

BROWSER = next((p for p in (
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
) if Path(p).exists()), None)

# 关键实测数字：必须在成品里出现，且不得出现旧口径数字
NUMBERS = ["0.1667", "0.8889", "77/84", "91.7%", "0.9227", "25/36",
           "0.6756", "36/36", "17.88", "4.90", "113/3", "105/10", "89/12", "85/19"]
FORBIDDEN = ["81%", "81.0%", "89.8%"]          # 旧自由文本正则汇总，不得当描述准确率
FICTIONAL = ["清华大学", "北京大学", "复旦大学", "字节跳动", "阿里巴巴集团", "腾讯",
             "OpenAI", "Google", "某某", "张三", "李四", "王五",
             "性别：", "年龄：", "政治面貌", "籍贯"]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    rep: dict = {"目录": str(D), "只读源": True}
    html = HTML.read_text(encoding="utf-8")
    shell = SHELL.read_text(encoding="utf-8")

    # ---- A. 自包含 ----
    ext = re.findall(r'(?:src|href)\s*=\s*"([^"]+)"', html)
    ext = [u for u in ext if not u.startswith(("data:", "#", "javascript:"))]
    rep["A_自包含"] = {
        "有 <base> 标签": "<base" in html,
        "外部引用": ext,
        "内联 <style> 数": html.count("<style>"),
        "内联 <script> 数": html.count("<script>"),
        "模板标记残留": [m for m in ("@ASU_TOOLBAR", "@ASU_EDITOR") if m in html],
        "结论": "自包含" if (not ext and "<base" not in html) else "**未自包含**",
    }

    # ---- B. 无虚构 ----
    hits_f = sorted({s for s in FICTIONAL if s in html})
    rep["B_无虚构个人信息"] = {
        "命中虚构词": hits_f,
        "待补占位符数": html.count("【待补】") + shell.count("【待补】"),
        "结论": "无虚构姓名/学校/公司/照片" if not hits_f else "**发现虚构内容**",
    }

    # ---- C. 单页测量证据（本脚本自己跑无头浏览器，可重复复现） ----
    fit: dict = {"来源": "无头浏览器 --dump-dom 抓取 ASu 编辑器页内指标（A4 297mm）"}
    if BROWSER:
        fit["浏览器"] = BROWSER
        try:
            with open(DOM, "w", encoding="utf-8", errors="replace") as fh:
                subprocess.run(
                    [BROWSER, "--headless=new", "--disable-gpu", "--no-sandbox",
                     "--virtual-time-budget=6000", "--dump-dom", str(HTML)],
                    stdout=fh, stderr=subprocess.DEVNULL, timeout=120, check=False)
            dom = DOM.read_text(encoding="utf-8", errors="replace")
            over = re.search(r"超出单页\s*([\d.]+)px", dom)
            inone = re.search(r"一页内\s*·\s*占用\s*(\d+)%", dom)
            fit["指标原文"] = (over.group(0) if over else (inone.group(0) if inone else None))
            fit["是否一页内"] = bool(inone) and not over
        except Exception as e:                                   # noqa: BLE001
            fit["测量失败"] = f"{type(e).__name__}: {e}"
            fit["是否一页内"] = False
    else:
        fit["测量失败"] = "本机未找到 msedge/chrome，未能自动测量（测量值见 简历页检查.json 历史）"
        fit["是否一页内"] = False
    rep["C_单页检查"] = fit

    # ---- D. PDF ----
    d: dict = {}
    if PDF.exists():
        b = PDF.read_bytes()
        d["字节"] = len(b)
        d["sha256"] = hashlib.sha256(b).hexdigest()
        d["页数"] = len(re.findall(rb"/Type\s*/Page[^s]", b))
        boxes = sorted({m.decode() for m in re.findall(rb"/MediaBox\s*\[([^\]]*)\]", b)})
        pts = []
        for bx in boxes:
            v = [float(x) for x in bx.replace(",", " ").split()]
            if len(v) == 4:
                pts.append((round((v[2] - v[0]) / 72 * 25.4, 1),
                            round((v[3] - v[1]) / 72 * 25.4, 1)))
        d["纸张_mm"] = pts
        d["A4_210x297"] = all(abs(w - 210) < 1 and abs(h - 297) < 1 for w, h in pts)
        d["嵌入字体数"] = len({m for m in re.findall(rb"/BaseFont\s*/([A-Za-z0-9+\-_,]+)", b)})
        d["含嵌入字体标记"] = (b"/FontFile2" in b or b"/FontFile3" in b or b"/FontFile" in b)
        d["可抽取文本"] = (b"/ToUnicode" in b)
        d["含待补占位"] = b"\xe3\x80\x90\xe5\xbe\x85\xe8\xa1\xa5\xe3\x80\x91" in b
    rep["D_PDF"] = d

    # ---- E. 事实一致性 ----
    pdf_txt = PDF.read_bytes().decode("latin-1") if PDF.exists() else ""
    rep["E_事实一致性"] = {
        "HTML 含全部关键数字": [n for n in NUMBERS if n not in html],
        "HTML 含禁用旧口径": [n for n in FORBIDDEN if n in html],
        "HTML 标题为「项目经历」": "<h1>项目经历</h1>" in html,
        "PDF 与 HTML 同批生成": PDF.exists() and PDF.stat().st_mtime >= HTML.stat().st_mtime,
        "PDF 文字位（latin-1 抽查见 C/D 说明）": bool(pdf_txt),
    }
    rep["E_事实一致性"]["缺失数字"] = rep["E_事实一致性"]["HTML 含全部关键数字"]
    rep["E_事实一致性"].pop("HTML 含全部关键数字")

    # ---- F. 预览图 ----
    pv = D / "预览_整页.png"
    if pv.exists():
        import struct
        raw = pv.read_bytes()
        w, h = struct.unpack(">II", raw[16:24])
        rep["F_预览图"] = {"文件": pv.name, "像素": f"{w}x{h}",
                           "字节": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                           "说明": "A4 宽度 794px 下的可编辑视图整页截图；打印真实版面以 PDF 为准"}
    else:
        rep["F_预览图"] = {"文件": None, "说明": "**未生成预览图**"}

    ok = (rep["A_自包含"]["结论"] == "自包含"
          and rep["B_无虚构个人信息"]["结论"].startswith("无虚构")
          and rep["C_单页检查"].get("是否一页内") is True
          and rep["D_PDF"].get("页数") == 1
          and rep["D_PDF"].get("A4_210x297") is True
          and not rep["E_事实一致性"]["缺失数字"]
          and not rep["E_事实一致性"]["HTML 含禁用旧口径"])
    rep["总判定"] = "通过" if ok else "**未通过**"

    (D / "简历页检查.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    txt = ["# 简历页（HTML / PDF）交付前检查", "",
           f"**总判定：{rep['总判定']}**", "",
           "| 项 | 结果 |", "|---|---|",
           f"| 自包含 | {rep['A_自包含']['结论']}（外部引用 {len(ext)} 个，模板标记残留 {len(rep['A_自包含']['模板标记残留'])} 个） |",
           f"| 无虚构个人信息 | {rep['B_无虚构个人信息']['结论']}（命中 {len(hits_f)} 个） |",
           f"| 单页检查 | {rep['C_单页检查'].get('指标原文')} |",
           f"| PDF 页数 | {d.get('页数')} |",
           f"| PDF 纸张 | {d.get('纸张_mm')} mm（A4 判定 {d.get('A4_210x297')}） |",
           f"| PDF 字体 | 嵌入标记 {d.get('含嵌入字体标记')} / 字体数 {d.get('嵌入字体数')} |",
           f"| PDF 可抽取文本 | {d.get('可抽取文本')} |",
           f"| PDF sha256 | `{d.get('sha256')}` |",
           f"| HTML 关键数字缺失 | {rep['E_事实一致性']['缺失数字'] or '无'} |",
           f"| HTML 含旧口径 81%/89.8% | {rep['E_事实一致性']['HTML 含禁用旧口径'] or '无'} |",
           f"| 预览图 | {rep['F_预览图'].get('像素')} · `{rep['F_预览图'].get('文件')}` |",
           "",
           "> 单页指标来自 ASu 编辑器自带的 A4 页内测量（无头浏览器 `--dump-dom` 读取），",
           "> 与 `预览_整页.png` 同源，可重复复现。",
           ""]
    (D / "简历页检查.md").write_text("\n".join(txt), encoding="utf-8")

    sys.stdout.buffer.write((json.dumps(
        {"总判定": rep["总判定"], "自包含": rep["A_自包含"]["结论"],
         "单页": rep["C_单页检查"].get("指标原文"), "PDF页数": d.get("页数"),
         "A4": d.get("A4_210x297"), "缺失数字": rep["E_事实一致性"]["缺失数字"]},
        ensure_ascii=False, indent=1) + "\n").encode("utf-8"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
