/* МУХОЛЁТ — генератор руководства (docx-js). Запуск: node generate.js */
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  ImageRun, PageBreak, Header, Footer, PageNumber, NumberFormat, SectionType,
  AlignmentType, HeadingLevel, WidthType, BorderStyle, ShadingType,
  TableLayoutType, TableOfContents, VerticalAlign,
  Math: OoxmlMath, MathRun, MathFraction, MathSuperScript, MathSubScript,
} = require("docx");
const fs = require("fs");
const path = require("path");

/* ── палитра DM-1 (Deep Cyan) и общие цвета ── */
const PAL = {
  bg: "162235", accent: "37DCF2",
  cover: { titleColor: "FFFFFF", subtitleColor: "B0B8C0", metaColor: "90989F", footerColor: "687078" },
  table: { headerBg: "1B6B7A", headerText: "FFFFFF", accentLine: "1B6B7A", innerLine: "C8DDE2", surface: "EDF3F5" },
};
const INK = "1A2430";        // основной текст
const MUTED = "5A6A78";      // вторичный текст
const FONT = "Calibri";
const SHOT_DIR = path.join(__dirname, "shots");

const allNoBorders = {
  top: { style: BorderStyle.NONE }, bottom: { style: BorderStyle.NONE },
  left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE },
  insideHorizontal: { style: BorderStyle.NONE }, insideVertical: { style: BorderStyle.NONE },
};
const noBorders = {
  top: { style: BorderStyle.NONE }, bottom: { style: BorderStyle.NONE },
  left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE },
};

/* ── расчёт обложки (по design-system) ── */
function splitTitleLines(title, charsPerLine) {
  if (title.length <= charsPerLine) return [title];
  const breakAfter = new Set([..."，。、；：！？", ..."—–·/-_ \t", ..."«»"]);
  const lines = [];
  let remaining = title;
  while (remaining.length > charsPerLine) {
    let breakAt = -1;
    for (let i = charsPerLine; i >= Math.floor(charsPerLine * 0.6); i--) {
      if (i < remaining.length && breakAfter.has(remaining[i - 1])) { breakAt = i; break; }
    }
    if (breakAt === -1) {
      const limit = Math.min(remaining.length, Math.ceil(charsPerLine * 1.3));
      for (let i = charsPerLine + 1; i < limit; i++) {
        if (breakAfter.has(remaining[i - 1])) { breakAt = i; break; }
      }
    }
    if (breakAt === -1) breakAt = charsPerLine;
    lines.push(remaining.slice(0, breakAt).trim());
    remaining = remaining.slice(breakAt).trim();
  }
  if (remaining) lines.push(remaining);
  if (lines.length > 1 && lines[lines.length - 1].length <= 2) {
    const last = lines.pop();
    lines[lines.length - 1] += last;
  }
  return lines;
}
function calcTitleLayout(title, maxWidthTwips, preferredPt = 40, minPt = 24) {
  const charWidth = (pt) => pt * 12; // кириллица уже CJK: ~0.6 em
  const charsPerLine = (pt) => Math.floor(maxWidthTwips / charWidth(pt));
  let titlePt = preferredPt;
  let lines;
  while (titlePt >= minPt) {
    const cpl = charsPerLine(titlePt);
    if (cpl < 2) { titlePt -= 2; continue; }
    lines = splitTitleLines(title, cpl);
    if (lines.length <= 3) break;
    titlePt -= 2;
  }
  if (!lines || lines.length > 3) {
    lines = splitTitleLines(title, charsPerLine(minPt));
    titlePt = minPt;
  }
  return { titlePt, titleLines: lines };
}
function calcCoverSpacing(params) {
  const { titleLineCount = 1, titlePt = 36, hasSubtitle = false, hasEnglishLabel = false,
    metaLineCount = 0, fixedHeight = 800, pageHeight = 16838, marginTop = 0, marginBottom = 0 } = params;
  const SAFETY = 1200;
  const usableHeight = pageHeight - marginTop - marginBottom - SAFETY;
  const titleHeight = titleLineCount * (titlePt * 23 + 200);
  const subtitleHeight = hasSubtitle ? (12 * 23 + 600) : 0;
  const englishLabelHeight = hasEnglishLabel ? (9 * 23 + 600) : 0;
  const metaHeight = metaLineCount * (10 * 23 + 100);
  const implicitParaHeight = 3 * 300;
  const contentHeight = titleHeight + subtitleHeight + englishLabelHeight + metaHeight + fixedHeight + implicitParaHeight;
  const safeRemaining = Math.max(usableHeight - contentHeight, 400);
  const FOOTER_MIN = 800;
  const rawTop = Math.floor(safeRemaining * 0.45);
  const rawBottom = Math.floor(safeRemaining * 0.45);
  const bottomSpacing = Math.max(rawBottom, FOOTER_MIN);
  const topSpacing = Math.max(rawTop - Math.max(0, FOOTER_MIN - rawBottom), 400);
  const midSpacing = Math.max(safeRemaining - topSpacing - bottomSpacing, 0);
  return { topSpacing, midSpacing, bottomSpacing };
}

/* ── Recipe R1: Pure Paragraph Cover ── */
function buildCoverR1(config) {
  const P = config.palette;
  const padL = 1200, padR = 800;
  const availableWidth = 11906 - padL - padR - 300;
  const { titlePt, titleLines } = calcTitleLayout(config.title, availableWidth, 40, 24);
  const titleSize = titlePt * 2;
  const spacing = calcCoverSpacing({
    titleLineCount: titleLines.length, titlePt,
    hasSubtitle: !!config.subtitle, hasEnglishLabel: !!config.englishLabel,
    metaLineCount: (config.metaLines || []).length, fixedHeight: 400,
  });
  const accentLeft = { style: BorderStyle.SINGLE, size: 8, color: P.accent, space: 12 };
  const children = [];
  children.push(new Paragraph({ spacing: { before: spacing.topSpacing } }));
  if (config.englishLabel) {
    children.push(new Paragraph({
      indent: { left: padL, right: padR }, spacing: { after: 500 },
      border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: P.accent, space: 8 } },
      children: [new TextRun({ text: config.englishLabel.split("").join("  "), size: 18, color: P.accent, font: FONT, characterSpacing: 40 })],
    }));
  }
  for (let i = 0; i < titleLines.length; i++) {
    children.push(new Paragraph({
      indent: { left: padL },
      spacing: { after: i < titleLines.length - 1 ? 100 : 300, line: Math.ceil(titlePt * 23), lineRule: "atLeast" },
      children: [new TextRun({ text: titleLines[i], size: titleSize, bold: true, color: P.cover.titleColor, font: FONT })],
    }));
  }
  if (config.subtitle) {
    children.push(new Paragraph({
      indent: { left: padL, right: padR }, spacing: { after: 800, line: 320, lineRule: "atLeast" },
      children: [new TextRun({ text: config.subtitle, size: 24, color: P.cover.subtitleColor, font: FONT })],
    }));
  }
  for (const line of (config.metaLines || [])) {
    children.push(new Paragraph({
      indent: { left: padL + 200, right: padR }, spacing: { after: 80 },
      border: { left: accentLeft },
      children: [new TextRun({ text: line, size: 22, color: P.cover.metaColor, font: FONT })],
    }));
  }
  children.push(new Paragraph({ spacing: { before: spacing.bottomSpacing } }));
  children.push(new Paragraph({
    indent: { left: padL, right: padR },
    border: { top: { style: BorderStyle.SINGLE, size: 2, color: P.accent, space: 8 } },
    spacing: { before: 200 },
    children: [
      new TextRun({ text: config.footerLeft || "", size: 16, color: P.cover.footerColor, font: FONT }),
      new TextRun({ text: "                                                                  " }),
      new TextRun({ text: config.footerRight || "", size: 16, color: P.cover.footerColor, font: FONT }),
    ],
  }));
  return [new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    layout: TableLayoutType.FIXED,
    borders: allNoBorders,
    rows: [new TableRow({
      height: { value: 16838, rule: "exact" },
      children: [new TableCell({
        shading: { type: ShadingType.CLEAR, fill: P.bg }, borders: noBorders,
        children,
      })],
    })],
  })];
}

/* ── помощники тела документа ── */
function runsFromMarkup(text, base = {}) {
  // **жирный** и `моно` внутри строки
  const runs = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`)/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) runs.push(new TextRun({ text: text.slice(last, m.index), font: FONT, size: 22, color: INK, ...base }));
    const piece = m[0];
    if (piece.startsWith("**")) runs.push(new TextRun({ text: piece.slice(2, -2), bold: true, font: FONT, size: 22, color: INK, ...base }));
    else runs.push(new TextRun({ text: piece.slice(1, -1), font: "Courier New", size: 20, color: "0F4C5C", ...base }));
    last = m.index + piece.length;
  }
  if (last < text.length) runs.push(new TextRun({ text: text.slice(last), font: FONT, size: 22, color: INK, ...base }));
  return runs;
}
function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1, spacing: { before: 400, after: 160, line: 312 },
    children: [new TextRun({ text, bold: true, size: 32, color: "0E3A46", font: FONT })],
  });
}
function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2, spacing: { before: 280, after: 120, line: 312 },
    children: [new TextRun({ text, bold: true, size: 26, color: "0E3A46", font: FONT })],
  });
}
function h3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3, spacing: { before: 220, after: 100, line: 312 },
    children: [new TextRun({ text, bold: true, size: 23, color: INK, font: FONT })],
  });
}
function p(text, opts = {}) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED, spacing: { after: 100, line: 312 },
    children: runsFromMarkup(text), ...opts,
  });
}
function bullet(text) {
  return new Paragraph({
    bullet: { level: 0 }, spacing: { after: 60, line: 312 },
    children: runsFromMarkup(text),
  });
}
function note(text) {
  return new Paragraph({
    spacing: { before: 60, after: 140, line: 312 }, indent: { left: 340 },
    border: { left: { style: BorderStyle.SINGLE, size: 8, color: PAL.table.headerBg, space: 10 } },
    children: runsFromMarkup(text, { size: 20, color: MUTED }),
  });
}
/* формулы: токены — строка | {f:[числ,знам]} | {sup:[осн,ст]} | {sub:[осн,инд]} | массив токенов */
function tok(t) {
  if (typeof t === "string") return new MathRun(t);
  if (Array.isArray(t)) return new MathRun(t.map((x) => (typeof x === "string" ? x : "")).join("")); // не используется напрямую
  if (t.f) {
    const num = Array.isArray(t.f[0]) ? t.f[0].map(tok) : [tok(t.f[0])];
    const den = Array.isArray(t.f[1]) ? t.f[1].map(tok) : [tok(t.f[1])];
    return new MathFraction({ numerator: num, denominator: den });
  }
  if (t.sup) return new MathSuperScript({ children: Array.isArray(t.sup[0]) ? t.sup[0].map(tok) : [tok(t.sup[0])], superScript: [tok(t.sup[1])] });
  if (t.sub) return new MathSubScript({ children: Array.isArray(t.sub[0]) ? t.sub[0].map(tok) : [tok(t.sub[0])], subScript: [tok(t.sub[1])] });
  return new MathRun(String(t));
}
function M(parts) { return new OoxmlMath({ children: parts.map(tok) }); }
function eq(parts, num) {
  const kids = [M(parts)];
  if (num) kids.push(new TextRun({ text: "      " + num, size: 20, color: MUTED, font: FONT }));
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 100, after: 140, line: 360, lineRule: "atLeast" }, children: kids });
}
let tblN = 0, figN = 0;
function tblCaption(text) {
  tblN += 1;
  return new Paragraph({
    keepNext: true, spacing: { before: 160, after: 80 },
    children: [new TextRun({ text: `Таблица ${tblN}. `, bold: true, size: 20, color: "0E3A46", font: FONT }),
    new TextRun({ text, size: 20, color: MUTED, font: FONT })],
  });
}
function tbl(headers, rows, widths) {
  const mk = (cell, isHead, w) => {
    // ячейка-объект { m: [...токены] } рендерится формулой OMML
    const kids = (cell && typeof cell === "object" && cell.m)
      ? [M(cell.m)]
      : runsFromMarkup(String(cell), { size: 19, color: isHead ? PAL.table.headerText : INK, bold: isHead });
    return new TableCell({
      children: [new Paragraph({ spacing: { line: 276 }, children: kids })],
      shading: isHead ? { type: ShadingType.CLEAR, fill: PAL.table.headerBg } : undefined,
      margins: { top: 60, bottom: 60, left: 110, right: 110 },
      width: w ? { size: w, type: WidthType.PERCENTAGE } : undefined,
      verticalAlign: VerticalAlign.CENTER,
    });
  };
  return new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    borders: {
      top: { style: BorderStyle.SINGLE, size: 4, color: PAL.table.accentLine },
      bottom: { style: BorderStyle.SINGLE, size: 4, color: PAL.table.accentLine },
      left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 1, color: PAL.table.innerLine },
      insideVertical: { style: BorderStyle.NONE },
    },
    rows: [
      new TableRow({ tableHeader: true, cantSplit: true, children: headers.map((t, i) => mk(t, true, widths ? widths[i] : undefined)) }),
      ...rows.map((r) => new TableRow({ cantSplit: true, children: r.map((t, i) => mk(t, false, widths ? widths[i] : undefined)) })),
    ],
  });
}
function fig(file, caption) {
  figN += 1;
  const buf = fs.readFileSync(path.join(SHOT_DIR, file));
  const W = 600, H = Math.round(600 * 1080 / 1920);
  const capText = caption.charAt(0).toUpperCase() + caption.slice(1);
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { before: 140, after: 60 }, keepNext: true,
      children: [new ImageRun({ data: buf, transformation: { width: W, height: H }, type: "png" })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { after: 160 },
      children: [new TextRun({ text: `Рис. ${figN}. `, bold: true, size: 19, color: "0E3A46", font: FONT }),
      new TextRun({ text: capText, size: 19, color: MUTED, font: FONT })],
    }),
  ];
}
function spacer(after = 120) { return new Paragraph({ spacing: { after }, children: [] }); }

module.exports = {
  PAL, INK, MUTED, FONT, allNoBorders, noBorders,
  buildCoverR1, runsFromMarkup, h1, h2, h3, p, bullet, note, eq, M, tbl, tblCaption, fig, spacer,
  docx: { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun, PageBreak, Header, Footer, PageNumber, NumberFormat, SectionType, AlignmentType, HeadingLevel, WidthType, BorderStyle, ShadingType, TableLayoutType, TableOfContents, VerticalAlign },
};
