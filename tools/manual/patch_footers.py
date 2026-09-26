#!/usr/bin/env python3
"""Пост-обработка собранного руководства: формат номеров страниц в подвалах
(ROMAN для аннотации/оглавления, arabic для тела) и чистка пустых pgNumType
обложки. Запуск после add_toc_placeholders.py: python3 patch_footers.py <docx>"""
import os
import re
import shutil
import sys
import zipfile

src = sys.argv[1] if len(sys.argv) > 1 else "МУХОЛЁТ_руководство.docx"
tmp = src + ".patch"

if os.path.exists(tmp):
    shutil.rmtree(tmp)
with zipfile.ZipFile(src) as z:
    z.extractall(tmp)

doc_path = os.path.join(tmp, "word", "document.xml")
xml = open(doc_path, encoding="utf-8").read()
removed = xml.count("<w:pgNumType/>")
open(doc_path, "w", encoding="utf-8").write(xml.replace("<w:pgNumType/>", ""))

refs = re.findall(r'w:footerReference[^>]*r:id="(rId\d+)"', xml)
rels = open(os.path.join(tmp, "word", "_rels", "document.xml.rels"), encoding="utf-8").read()

def target(rid):
    m = re.search(r'Id="%s"[^>]*Target="([^"]+)"' % rid, rels)
    return m.group(1) if m else None

fmts = ["ROMAN", "arabic"]  # секция 2 — аннотация/оглавление, секция 3 — тело
seen = []
for rid in refs:
    f = target(rid)
    if f is None or f in seen:
        continue
    seen.append(f)
    fmt = fmts[len(seen) - 1]
    fp = os.path.join(tmp, "word", f)
    fx = open(fp, encoding="utf-8").read()
    fx, n = re.subn(
        r"(<w:instrText[^>]*>)\s*PAGE\s*(</w:instrText>)",
        r"\1 PAGE \\* %s \\* MERGEFORMAT \2" % fmt,
        fx,
    )
    open(fp, "w", encoding="utf-8").write(fx)
    print(f"{f} -> {fmt} ({n} поле)")

os.remove(src)
zf = zipfile.ZipFile(src, "w", zipfile.ZIP_DEFLATED)
for root, _, files in os.walk(tmp):
    for f in files:
        full = os.path.join(root, f)
        zf.write(full, os.path.relpath(full, tmp))
zf.close()
shutil.rmtree(tmp)
print(f"готово: {src} (убрано пустых pgNumType: {removed})")
