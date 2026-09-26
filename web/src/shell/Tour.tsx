import { useEffect, useState } from 'react'
import { buzz, HAPTIC } from '../haptics'

type TourStep = { id?: string; title: string; text: string }

/** 5 шагов «с чего начать»: только элементы, всегда видимые в шапке независимо
 * от ширины экрана и текущего пространства (иначе на телефоне/не-«Полёте»
 * подсветка попадала бы в закрытую выезжающую панель или вовсе не находилась).
 * У шага «параметры пуска» нет привязки — карточка просто по центру: сама
 * панель настроек то в колонке инспектора, то за выезжающей кнопкой. */
const STEPS: TourStep[] = [
  { id: 'workspaces', title: 'Рабочие пространства', text: 'Полёт, Мозг, Рой, Дуэль, Лаборатория — переключаются здесь. Одновременно виден только один режим.' },
  { id: 'launch', title: 'Пуск', text: 'Прогоняет перехват с текущими условиями. Пробел работает из любого пространства, не только по клику.' },
  { id: 'camera', title: 'Камера и геометрия', text: 'В «Полёте»: авто-камера следит за сближением сама, «геометрия» рисует треугольник перехвата, круг боевой части и цвет следа по перегрузке.' },
  { title: 'Параметры пуска', text: 'Дальность, манёвр цели, закон наведения — панель справа от сцены (на узком экране — по кнопке «Параметры пуска» над сценой).' },
  { id: 'lab-nav', title: 'Лаборатория', text: '17 экранов в пяти разделах: графики обучения, сравнения законов, карты преимуществ, экспорт данных.' },
]

/** Короткий интерактивный тур «с чего начать» — не стартует сам, только по явному
 * запросу (баннер демо-прогона / меню «Ещё»). Скрим не перехватывает клики
 * (pointer-events:none) — можно трогать реальный интерфейс поверх подсказки. */
export function Tour({ onClose }: { onClose: () => void }) {
  const [i, setI] = useState(0)
  const [rect, setRect] = useState<DOMRect | null>(null)
  const step = STEPS[i]

  useEffect(() => {
    const update = () => {
      const el = step.id ? document.querySelector(`[data-tour="${step.id}"]`) : null
      setRect(el ? el.getBoundingClientRect() : null)
    }
    update()
    window.addEventListener('resize', update)
    return () => window.removeEventListener('resize', update)
  }, [step.id])

  const next = () => {
    buzz(HAPTIC.tourStep)
    if (i >= STEPS.length - 1) onClose()
    else setI((v) => v + 1)
  }
  const prev = () => setI((v) => Math.max(0, v - 1))

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.code === 'Escape') onClose()
      else if (e.code === 'ArrowRight' || e.code === 'Enter') next()
      else if (e.code === 'ArrowLeft') prev()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [i])

  const cardTop = rect ? Math.min(window.innerHeight - 190, rect.bottom + 14) : undefined
  const cardLeft = rect ? Math.max(12, Math.min(window.innerWidth - 312, rect.left)) : undefined

  return (
    <div className="tour-scrim" role="dialog" aria-label="Короткий тур: с чего начать">
      {rect && <div className="tour-hole" style={{ top: rect.top - 6, left: rect.left - 6, width: rect.width + 12, height: rect.height + 12 }} />}
      <div className="tour-card" style={rect ? { top: cardTop, left: cardLeft } : undefined}>
        <b>{step.title}</b>
        <p>{step.text}</p>
        <div className="tour-card__foot">
          <span>
            {i + 1} / {STEPS.length}
          </span>
          {i > 0 && (
            <button type="button" className="ghost" onClick={prev}>
              Назад
            </button>
          )}
          <button type="button" className="ghost" onClick={onClose}>
            Пропустить
          </button>
          <button type="button" className="primary" onClick={next}>
            {i >= STEPS.length - 1 ? 'Готово' : 'Далее'}
          </button>
        </div>
      </div>
    </div>
  )
}
