import type { Scenario } from '../types'

/** Русские подписи и формат чисел — общие для App и оболочек пространств. */

export const ASPECT_LABEL: Record<Scenario['aspect'], string> = {
  'head-on': 'встречные',
  beam: 'пересечение',
  'tail-chase': 'вдогон',
  free: 'свободная',
}

export const MODE_LABEL: Record<Scenario['mode'], string> = {
  pn: 'ПН',
  bio: 'био',
  both: 'био + ПН',
}

export const SPEED_MODE_LABEL: Record<string, string> = {
  constant: 'пост.',
  accelerate: 'разгон',
  decelerate: 'тормож.',
  pulse: 'импульс',
  sine: 'синус',
}

/** Законы наведения по-русски. Значения обязаны совпадать с PYTHON LAW_LABEL
 *  в navedenie/glossary.py — единый источник терминов (§4). Проверка:
 *  tests/test_glossary.py::test_ts_law_labels_match_python. */
export const LAW_RU: Record<string, string> = {
  pn: 'Метод пропорциональной навигации (ПН)',
  tpn: 'Истинная пропорциональная навигация (команда по нормали к ЛВ)',
  apn: 'ПН с компенсацией нормального ускорения цели (APN)',
  pure: 'Метод погони',
  clos: 'Метод трёх точек (CLOS)',
  pn_gsn: 'ПН по измерениям сенсорного канала ГСН',
  pn_sched_oracle: 'Экспериментальная ПН с переменным навигационным коэффициентом N по точному состоянию',
  pn_sched_sensor: 'Экспериментальная ПН с переменным навигационным коэффициентом N по сенсорным измерениям',
}

export function fmt(n: number | null | undefined, d = 1) {
  if (n === null || n === undefined || !Number.isFinite(n)) return '—'
  return n.toFixed(d)
}

/** Законы уклонения цели в дуэли. Значения обязаны совпадать с PYTHON
 *  EVADER_LABEL в navedenie/glossary.py — тот же принцип единого источника,
 *  что у LAW_RU. Проверка: tests/test_duel_api.py. */
export const EVADER_RU: Record<string, string> = {
  away: 'Уклон от точки встречи',
  negpn: 'ПН наоборот (отрицательный коэффициент по ЛВ ракеты)',
  cpa_max: 'Градиент наименьшего сближения',
  brain: 'Мозг-уклонист (обучаемая схема цели)',
}

export const nCellsLabel = (b: Scenario['brain']) => (b === 'full' ? 4439 : b === 'connectome' ? 108781 : 279)

export const EVENT_RU: Record<string, string> = {
  launch: 'пуск',
  hit: 'перехват',
  lost: 'потеря захвата',
  miss_pass: 'отказ перехвата',
  fuse_expired: 'время вышло',
  error: 'ошибка',
}

/** Слои контура: подпись на диаграмме активности и подсказка. */
export const LAYERS = [
  ['Зрение', 'Зрение'],
  ['Поток', 'Поток'],
  ['Приближение', 'Приближение'],
  ['Решение', 'Решение'],
  ['Мотор', 'Выход'],
] as const

export const LAYER_TIP: Record<string, string> = {
  Зрение: 'Свет от цели в омматидиях — «видит ли муха».',
  Поток: 'T4/T5: оптический поток — куда ползёт образ.',
  Приближение: 'LPLC2: пятно растёт — цель прёт на муху.',
  Решение: 'AOTU/грибовидное тело: выработано решение о манёвре.',
  Выход: 'DN: команда дошла до рулей.',
}
