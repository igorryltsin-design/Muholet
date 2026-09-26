import type { BrainKind, GenPoint, LabData, TrainPoint } from '../types'

export type { LabData, GenPoint, TrainPoint }

/** Экраны лаборатории, сгруппированные по разделам. */
export type LabTab =
  | 'run'
  | 'summary'
  | 'train'
  | 'swarm'
  | 'compare'
  | 'neff'
  | 'rob'
  | 'law'
  | 'map'
  | 'transfer'
  | 'scaling'
  | 'formula'
  | 'params'
  | 'mc'
  | 'cz'

/** Ответ /api/science/capture-zone: сетка «дальность × перегрузка цели» + границы
 * связного блока перехвата (зона — отрезок: ближняя и дальняя граница). */
export type CaptureZoneData = {
  law: string
  mode: string
  model: string
  aspect: string
  v_m: number
  v_t: number
  t_max_s: number
  n_max_g: number
  pn_n: number
  trigger_range_m: number
  ranges_m: number[]
  target_gs: number[]
  rows: {
    n_target_g: number
    status: 'all_hit' | 'no_hit' | 'bracketed' | 'lobed'
    hit_frac: number
    monotonic_violations: number
    hit_blocks: number
    cells: {
      range_m: number
      hit: boolean
      r_min_m: number
      t_end_s: number
      time_limited: boolean
      // вероятностный режим (n_runs > 1): доля серии, ДИ Уилсона, медиана CPA
      p_hit?: number
      p_hit_ci95?: [number, number]
      median_cpa_m?: number
      deterministic_cell?: boolean
    }[]
    r_last_hit_m?: number
    r_first_miss_m?: number
    boundary_m?: number
    boundary_bracket_m?: [number, number]
    boundary_time_limited?: boolean
    inner_boundary_m?: number
    inner_boundary_bracket_m?: [number, number]
  }[]
  /** задан и >1 — вероятностный режим: серии Monte-Carlo на ячейку */
  n_runs?: number
  seed_start?: number
  refine_skipped?: boolean
  series_deterministic?: boolean
  seconds: number
}

/** Ответ /api/science/monte-carlo: сводка рассеивания + ряд прогонов. */
export type MonteCarloData = {
  n_runs: number
  seed_start: number
  model: string
  trigger_range_m: number
  p_hit: number
  p_hit_ci95: [number, number]
  r_min_mean_m: number
  r_min_std_m: number | null
  r_min_median_m: number | null
  r_min_p90_m: number | null
  r_95_m: number | null
  cep50_m: number | null
  r_min_miss_mean_m: number | null
  r_min_miss_std_m: number | null
  h0_mean_m: number | null
  deterministic: boolean
  seconds: number
  per_run: { seed: number; r_min_m: number; hit: boolean; reason: string; h0_m: number | null }[]
}

export const LAB_GROUPS: { title: string; items: [LabTab, string][] }[] = [
  { title: 'Обзор', items: [['run', 'Прогон'], ['summary', 'Сводка']] },
  { title: 'Обучение', items: [['train', 'Обучение'], ['swarm', 'Рой']] },
  { title: 'Анализ', items: [['compare', 'Сравнение'], ['neff', 'N_экв'], ['rob', 'Устойчивость к возмущениям'], ['mc', 'Разброс'], ['cz', 'Зона перехвата'], ['law', 'Закон']] },
  { title: 'Исследования', items: [['map', 'Карта'], ['transfer', 'Переносимость'], ['scaling', 'Масштаб мозга']] },
  { title: 'Модель', items: [['formula', 'Формула'], ['params', 'Параметры']] },
]

/** Строка сравнения: мозг или закон наведения с честными метриками прогона. */
export type CmpRow = {
  kind: string
  label: string
  n_cells: number
  trained: boolean | null
  miss_m: number
  hit: boolean
  n_peak: number
  lock_frac: number
  t_guide?: number | null
  ref_dev_m?: number | null
  ref_nrms?: number | null
  t_end?: number | null
  t_ref?: number | null
  n_int?: number
}

export type MapRow = {
  maneuver: string
  law: string
  law_id: string
  miss_pn: number
  hit_pn: boolean
  miss_fly: number
  hit_fly: boolean
  advantage: number
  n_int_pn?: number
  n_int_fly?: number
  miss_fly_min?: number
  miss_fly_max?: number
  hit_fly_share?: number
  repeats?: number
}
export type FaultsData = { rows: { fraction: number; miss: number; worst: number; hit_rate: number }[] }
export type LadderData = { fly_miss_before: number; fly_miss_after: number; generations: number; restored?: string }
export type AblationData = {
  kind: string
  base: { miss: number; hit_rate: number; ref_dev: number | null; t_guide: number | null }
  rows: { zone: string; channels: number; miss: number; hit_rate: number; ref_dev: number | null; t_guide: number | null }[]
}
export type MapData = { rows: MapRow[]; brain: string; n_cells: number }
export type DistillData = {
  epochs: number
  teacher_kind?: string
  teacher: { miss: number; hit_rate: number }
  distilled: { miss: number; hit_rate: number; ref_dev: number }
  scratch: { miss: number; hit_rate: number; ref_dev: number }
}
export type CoevData = {
  gen: number
  best: { miss_fly: number; maneuver: string; n_target: number; range_m: number; off_axis_m: number }
  history: { gen: number; miss_fly: number; maneuver: string; median_miss?: number; retrained?: boolean; fly_before?: number; fly_after?: number; n_target?: number; range_m?: number; off_axis_m?: number }[]
}
export type TransferRow = { train: string; train_miss_med: number; tests: Record<string, number> }
export type TransferData = {
  kind: string
  episodes: number
  maneuvers: string[]
  test_maneuvers: string[]
  rows: TransferRow[]
  seconds: number
}
export type ScalingRow = {
  size: number
  params: number
  n_cells: number
  miss_before: number
  miss_after: number
  hit_rate_after: number
  ref_dev_after: number
  train_miss_med: number
}
export type ScalingData = { kind: string; episodes: number; rows: ScalingRow[]; seconds: number }
export type { BrainKind }
