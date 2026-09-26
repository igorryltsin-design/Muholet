/** Баннер после демо-перехвата при первом визите (App.tsx::runDemo) — сообщает,
 * что показ уже кончился, и предлагает тур или свой пуск. Закрывается сам
 * следующим реальным пуском (setDemoBanner(false) в App.tsx) или крестиком. */
export function DemoBanner({ onTour, onClose }: { onTour: () => void; onClose: () => void }) {
  return (
    <div className="demo-banner" role="status">
      <span>Это демо-перехват · нажмите «Пуск», чтобы попробовать свой сценарий</span>
      <button type="button" className="primary" onClick={onTour}>
        Показать тур
      </button>
      <button type="button" className="ghost icon-btn" onClick={onClose} aria-label="Закрыть баннер">
        ✕
      </button>
    </div>
  )
}
