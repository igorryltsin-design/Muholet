/* МУХОЛЁТ — сборка документа. Запуск: node generate.js */
const L = require("./lib");
const {
  Document, Packer, Paragraph, TextRun, PageBreak, Header, Footer, PageNumber,
  NumberFormat, SectionType, AlignmentType, HeadingLevel, TableOfContents, BorderStyle,
} = L.docx;
const fs = require("fs");
const path = require("path");
const C = require("./content");

const FONT = "Calibri";
const OUT = process.argv[2] || path.join(__dirname, "..", "..", "МУХОЛЁТ_руководство.docx");

function pageFooter() {
  return new Footer({
    children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "8090A0", font: FONT })],
    })],
  });
}
function pageHeader() {
  return new Header({
    children: [new Paragraph({
      alignment: AlignmentType.RIGHT,
      border: { bottom: { style: BorderStyle.SINGLE, size: 2, color: "C8DDE2", space: 4 } },
      children: [new TextRun({ text: "МУХОЛЁТ · руководство", size: 16, color: "9AA6B2", font: FONT })],
    })],
  });
}

/* титульный лист */
const cover = L.buildCoverR1({
  palette: L.PAL,
  englishLabel: "GUIDANCE TEST BENCH",
  title: "МУХОЛЁТ",
  subtitle: "Учебный стенд закона наведения «воздух–воздух»: эталонный метод пропорционального сближения против био-контура дрозофилы. Полное руководство с математикой и описанием интерфейса",
  metaLines: [
    "Режимы: МПС · МПС с компенсацией нормального ускорения цели · метод погони · метод трёх точек · МПС с переменным навигационным коэффициентом N · био-контур · био с резервным МПС",
    "Мозги: схема (279) · полный (4 439) · коннектом FlyWire FAFB (≈139 тыс нейронов)",
    "Обучение: подражание МПС · доводка по результату (CEM) · рой мух · коэволюция · дистилляция",
    "Наука: метрики v4 · N_экв · гипотезы закона · карта преимуществ · переносимость · масштаб мозга",
    "Версия стенда: 2026-09 · Документ собран из кода и живых прогонов",
  ],
  footerLeft: "Репозиторий «Наведение» · API :8091 · UI :5173",
  footerRight: "2026",
});

/* аннотация + оглавление (римская нумерация) */
const front = [
  new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { before: 200, after: 240 },
    children: [new TextRun({ text: "АННОТАЦИЯ", bold: true, size: 30, color: "0E3A46", font: FONT })],
  }),
  L.p("Руководство описывает исследовательский стенд **«МУХОЛЁТ»** — модель перехвата «воздух–воздух», в которой управляемая ракета (точка-масса в трёх измерениях) наводится либо классическим методом пропорционального сближения (МПС) и его вариантами, включая **МПС с переменным навигационным коэффициентом N**, либо **био-контуром**, воспроизводящим урезанный зрительно-моторный путь дрозофилы. Био-агент не получает истинную угловую скорость линии визирования: он видит только **фовеальную сетчатку 16×16** (плотный центр, редкая периферия до 165°) и сам извлекает пеленги, их скорости, угловой размер цели, «ломб» и фазу сближения ρ."),
  L.p("Документ содержит: модель движения и манёвров цели с **профилями скорости**; геометрию и физику фовеальной головки; выводы эталонных законов; описание всех слоёв био-контура и трёх мозгов; **метрики v4** и диагностику **N_экв**; режимы обучения с **dt-вариативной доводкой по результату**; рой, коэволюцию и дистилляцию; руководство по интерфейсу из четырёх рабочих пространств (Полёт · Мозг · Рой · Лаборатория) и справочник API. Все скриншоты сняты с работающего интерфейса; формулы набраны редактором формул Word и редактируемы."),
  new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 100, line: 312 }, children: [...L.runsFromMarkup("Для быстрого старта достаточно главы 1 и пространства «Полёт»: нажмите «Пуск» — и читайте дальше по мере погружения."), new PageBreak()] }),
  new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { before: 200, after: 300 },
    children: [new TextRun({ text: "ОГЛАВЛЕНИЕ", bold: true, size: 30, color: "0E3A46", font: FONT })],
  }),
  new TableOfContents("Оглавление", { hyperlink: true, headingStyleRange: "1-2" }),
  new Paragraph({
    spacing: { before: 200 },
    children: [new TextRun({
      text: "Примечание: оглавление построено на полях Word. После редактирования документа щёлкните по нему правой кнопкой и выберите «Обновить поле», чтобы номера страниц пересчитались.",
      italics: true, size: 18, color: "888888", font: FONT,
    })],
  }),
];

const body = [
  ...C.ch1, ...C.ch2, ...C.ch3, ...C.ch4, ...C.ch5, ...C.ch6, ...C.ch7, ...C.ch8, ...C.ch9, ...C.ch10, ...C.ch11,
  ...C.appA, ...C.appB,
];

const pgSize = { width: 11906, height: 16838 };
const pgMargin = { top: 1440, bottom: 1440, left: 1701, right: 1417 };

const doc = new Document({
  creator: "МУХОЛЁТ",
  title: "МУХОЛЁТ — руководство",
  styles: {
    default: {
      document: {
        run: { font: { ascii: FONT, hAnsi: FONT, cs: FONT, eastAsia: FONT }, size: 22, color: L.INK },
        paragraph: { spacing: { line: 312 } },
      },
      heading1: {
        run: { font: { ascii: FONT, hAnsi: FONT }, size: 32, bold: true, color: "0E3A46" },
        paragraph: { spacing: { before: 400, after: 160, line: 312 }, outlineLevel: 0 },
      },
      heading2: {
        run: { font: { ascii: FONT, hAnsi: FONT }, size: 26, bold: true, color: "0E3A46" },
        paragraph: { spacing: { before: 280, after: 120, line: 312 }, outlineLevel: 1 },
      },
      heading3: {
        run: { font: { ascii: FONT, hAnsi: FONT }, size: 23, bold: true, color: L.INK },
        paragraph: { spacing: { before: 220, after: 100, line: 312 }, outlineLevel: 2 },
      },
    },
  },
  sections: [
    { /* обложка */
      properties: { page: { size: pgSize, margin: { top: 0, bottom: 0, left: 0, right: 0 } } },
      children: cover,
    },
    { /* аннотация + оглавление: римские номера */
      properties: {
        type: SectionType.NEXT_PAGE,
        page: { size: pgSize, margin: pgMargin, pageNumbers: { start: 1, formatType: NumberFormat.UPPER_ROMAN } },
      },
      headers: { default: pageHeader() },
      footers: { default: pageFooter() },
      children: front,
    },
    { /* тело: арабские с 1 */
      properties: {
        type: SectionType.NEXT_PAGE,
        page: { size: pgSize, margin: pgMargin, pageNumbers: { start: 1, formatType: NumberFormat.DECIMAL } },
      },
      headers: { default: pageHeader() },
      footers: { default: pageFooter() },
      children: body,
    },
  ],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(OUT, buf);
  console.log("written:", OUT, buf.length, "bytes");
});
