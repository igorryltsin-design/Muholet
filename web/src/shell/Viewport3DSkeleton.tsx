/**
 * Заставка на время загрузки 3D-чанка (three.js — самая тяжёлая часть
 * бандла, грузится асинхронно при первом показе сцены, см. lazyViews.tsx).
 * Чистый CSS: конус-«радар» разворачивается по сетке того же тона, что и
 * настоящая сцена (--scene-bg/--chart-grid) — на пустой canvas это не похоже,
 * похоже на «стенд уже что-то делает». Без JS-цикла, без веса в бандле.
 */
export function Viewport3DSkeleton() {
  return (
    <div className="viewport viewport--3d viewport--skeleton" aria-hidden="true">
      <div className="viewport-skeleton__grid" />
      <div className="viewport-skeleton__sweep" />
      <p className="viewport-skeleton__label">Загрузка 3D-сцены…</p>
    </div>
  )
}
