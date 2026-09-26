# Генератор руководства МУХОЛЁТ

`МУХОЛЁТ_руководство.docx` в корне репозитория собирается автоматически
из этого каталога: текст и формулы — `content.js`, движок docx-js — `lib.js`,
сборка — `generate.js`, снимки интерфейса — `shots/`.

## Сборка

```bash
cd tools/manual
npm install                       # docx, image-size
node generate.js                  # соберёт ../../МУХОЛЁТ_руководство.docx
python3 patch_footers.py ../../МУХОЛЁТ_руководство.docx
# номера страниц в оглавлении — поля Word: скрипт ниже расставляет заглушки
DOCX_SCRIPTS=~/.zcode/cli/plugins/cache/zcode-plugins-official/document-skills/0.1.5/skills/docx/scripts
python3 "$DOCX_SCRIPTS/add_toc_placeholders.py" ../../МУХОЛЁТ_руководство.docx --auto
python3 patch_footers.py ../../МУХОЛЁТ_руководство.docx   # повторно — после заглушек
```

Порядок важен: `generate.js` → `add_toc_placeholders.py` (заглушки оглавления,
`updateFields=true`) → `patch_footers.py` (ROMAN/arabic в полях PAGE, чистка
пустых `pgNumType`).

## Обновление скриншотов

`shots/*.png` снимаются с работающего интерфейса (UI :5173, вьюпорт
1920×1080). Нумерация: 01 «Полёт» ночь, 02 день, 03 полная телеметрия,
04 «Мозг», 05 «Рой», 06–18 экраны лаборатории, 19 справка. После замены
снимка достаточно пересобрать документ — нумерация рисунков считается
автоматически.
