export type GuideMode = 'pn' | 'bio' | 'both'
export type Law = 'pn' | 'tpn' | 'apn' | 'pure' | 'clos' | 'pn_gsn' | 'pn_sched_oracle' | 'pn_sched_sensor'
export type Aspect = 'head-on' | 'beam' | 'tail-chase' | 'free'
export type Maneuver = 'straight' | 'turn' | 'weave' | 'weave_var' | 'break' | 'scissors' | 'dive' | 'combo'
export type BrainKind = 'stub' | 'full' | 'connectome'
/** закон уклонения цели в дуэли; подписи — labels.EVADER_RU (зеркало EVADER_LABEL) */
export type EvaderLaw = 'away' | 'negpn' | 'cpa_max' | 'brain'
/** профиль скорости цели внутри эпизода */
export type TargetSpeedMode = 'constant' | 'accelerate' | 'decelerate' | 'pulse' | 'sine'

/** версия схемы признаков контура (v2: 10 признаков, + theta и rho) */
export const FEATURE_SCHEMA_VERSION = 2
/** версия модели стенда */
export const MODEL_VERSION = '2.0.0'

export type Scenario = {
  aspect: Aspect
  v_m: number
  v_t: number
  range_m: number
  off_axis_m: number
  /** «Свободная расстановка» (aspect='free'): цель в метрах от точки пуска
   *  (X — вперёд, Y — бок, Z — абсолютная высота), курсы — азимут от +X против
   *  часовой к +Y и подъём над горизонтом, град. range_m в этой ветке не участвует. */
  free_tx: number
  free_ty: number
  free_talt: number
  free_mhdg: number
  free_mclimb: number
  free_thdg: number
  free_tclimb: number
  n_max: number
  n_target: number
  maneuver: Maneuver
  pn_n: number
  /** МПС с переменным N: N = clip(N0 + k_rho·rho, N_min, N_max) — отдельный контрольный закон */
  pn_sched_n0: number
  pn_sched_k_rho: number
  pn_sched_n_min: number
  pn_sched_n_max: number
  /** профиль скорости цели (продольное ускорение меняет модуль, не вираж) */
  target_speed_mode: TargetSpeedMode
  target_longitudinal_g: number
  target_speed_min: number
  target_speed_max: number
  target_speed_period_s: number
  target_speed_phase: number
  mode: GuideMode
  circuit_gain: number
  tau_s: number
  /** лаг рулевого привода (1-е апериодическое звено на исполнении команды), с; 0 — мгновенно */
  tau_act_s: number
  fov_deg: number
  /** физическое поле зрения БИО: фовеальная сетчатка (плотный центр, редкая периферия) */
  bio_fov_deg: number
  seeker_delay_s: number
  t_max: number
  kill_radius_m: number
  alt_m: number
  /** шумы измерителей */
  noise_az_deg: number
  noise_range_m: number
  lock_drop_p: number
  seeker_jitter_s: number
  /** постоянная карта «умерших» омматидиев + независимый дропаут на кадр */
  retina_death_p: number
  retina_dropout_p: number
  /** закон наведения для режима pn */
  law: Law
  brain: BrainKind
  /** дуэль: цель уходит реактивными законами (evader_law), а не «по нотам» */
  duel: boolean
  evader_law: EvaderLaw
  /** боевая жизнь ракеты в дуэли, с: не сбила за это время —win у цели */
  fuse_life_s: number
  /** модель движения ракеты: кинематическая (совм.) или трёхстепенная физическая (только сервер) */
  model: ModelKind
  /** трёхстепенная физика (только при model='point_mass_3dof'); имена и дефолты
   *  зеркалят Python ScenarioIn — сверка тестом tests/test_phys_fields.py */
  phys_atmos: boolean
  phys_cx_wave: number
  phys_mach_kr: number
  phys_mach_band: number
  phys_cn_max: number
  phys_wn_act: number
  phys_zeta_act: number
  phys_tau_a_s: number
}

/** Модель движения центра масс. `point_mass_3dof` недоступна в локальном окне. */
export type ModelKind = 'kinematic_legacy' | 'point_mass_3dof'

/** Режим камеры сцены: авто-следование, свободная (фиксированная) или вдогон ракете. */
export type CamMode = 'auto' | 'free' | 'chase'

/** Регион мозга в карте активности: кусок канонической склейки нейронов. */
export type BrainRegion = { name: string; start: number; n: number; mean: number }

/** Снимок контура цели-уклониста (дуэль, evader_law='brain'): компактная копия
 *  circuit-снапшота без картинки сетчатки; обучаемая схема цели (фаза 3). */
export type EvaderSnap = {
  kind: string
  t4: number[]
  lplc2: number
  dn: { pitch: number; yaw: number }
  layers: Record<string, number>
  weights: string
  n_cells: number
  n_neurons: number
  regions: BrainRegion[]
  act_b64: string
  lock: boolean
}

/** Муха роя: мозг дрозофилы (bio) или постоянный МПС (pn). */
export type FlyGenome = {
  kind: 'bio' | 'pn'
  /** Развёрнутые в строку веса 2×8 — только для bio. */
  w: number[]
  gain: number
  pn_n: number
}

export type FlyResult = {
  fly: FlyGenome
  fitness: number
  miss_m: number
  hit: boolean
  n_int: number
  traj_m: number[][]
  traj_t: number[][]
  /** телеметрия для анимации панелей (глаз/мозг/рули) */
  tel?: TelPoint[]
}

/** Точка телеметрии полёта мухи: пеленг, захват, команды, перегрузка. */
export type TelPoint = {
  t: number
  az: number
  el: number
  lock: boolean
  size: number
  sizeDot: number
  azDot: number
  elDot: number
  pitch: number
  yaw: number
  nReq: number
  miss: number
  rng: number
  v: number[]
  /** фаза сближения (если передана сервером/локальным движком) */
  theta?: number
  rho?: number
  tauContact?: number
}

export type SwarmGenResponse = {
  results: FlyResult[]
  next_population: FlyGenome[]
  stats: { best: number; avg: number; best_idx: number; worst?: number; hit_rate?: number; diversity?: number; canon_best?: number }
}

export type Frame = {
  t: number
  missile: number[]
  target: number[]
  missile_v: number[]
  target_v: number[]
  a_cmd: number[]
  a_pn: number[]
  n_req: number
  n_lim: number
  range_m: number
  v_c: number
  omega_los: number
  az: number
  el: number
  lock: boolean
  miss: number
  ghost?: number[] | null
  seeker: { image: number[][]; size: number; az_dot: number; el_dot: number }
  circuit: {
    kind: string
    photo: number[][]
    t4: number[]
    dn: { pitch: number; yaw: number }
    lplc2: number
    layers: Record<string, number>
    weights: string
    n_cells: number
    /** карта реальной активности: регионы + проекция всех нейронов в корзины */
    n_neurons: number
    regions: BrainRegion[]
    act_b64: string
  }
  layers: Record<string, number>
  event: string | null
  /** фаза сближения (декодирована из изображения) */
  theta: number
  theta_dot: number
  rho: number
  tau_contact: number
  /** диагностика N_экв (истинная геометрия — постфактум, не для управления) */
  n_eff: number | null
  n_eff_valid: boolean
  n_eff_reason: string | null
  n_eff_yaw: number | null
  n_eff_pitch: number | null
  sat: boolean
  /** DEPRECATED alias = tRadial (R/Vc, с) */
  tgo: number | null
  /** радиальная оценка оставшегося времени R/Vc, с */
  tRadial?: number | null
  t_radial?: number | null
  /** время до ближайшего сближения −r·Vотн/|Vотн|², с */
  tCpa?: number | null
  t_cpa?: number | null
  speed_mode: TargetSpeedMode
  target_speed: number
  /** снимок контура цели-уклониста (дуэль, evader_law='brain'); null — цель не мозг */
  evader?: EvaderSnap | null
}

/** ─── лаборатория: метрики и история экспериментов ─── */

/** Итог прогона: честные метрики одного полёта. */
/** Метрики прогона, версия 4 (R_min/усилия/захват + h_cv/h₀/R_end + диагностика N_экв — см. справку). */
export type RunMetrics = {
  metricsVersion: 4
  /** непрерывный минимум геометрического расстояния, м */
  miss: number
  cpaM: number
  /** радиус срабатывания БЧ, м */
  triggerRangeM: number
  hit: boolean
  /** время наведения: интерполированное t входа в сферу БЧ; null — не перехватил */
  tGuide: number | null
  tEnd: number
  nPeak: number
  /** среднее нормальное ускорение, g */
  nMean: number
  /** control effort: ∫|a_cmd|/g dt, g·с */
  nInt: number
  lockFrac: number
  /** среднее отклонение от траектории призрака-МПС (эталона), м */
  refDev: number | null
  /** нормированное СКО рассогласования (к начальной дальности) */
  refNrms: number | null
  /** h_cv: прогноз промаха при неизменных скоростях на последнем шаге, м (legacy-имя terminalZem) */
  terminalZem?: number | null
  hCvM?: number | null
  /** h₀: промах при снятой команде (интегрирование модели), м */
  h0M?: number | null
  /** R_end: дальность на последнем кадре, м */
  endRangeM?: number | null
  /** η «угол встречи»: угол между векторами скоростей ракеты и цели в момент
   * наибольшего сближения, град. (180° — лоб, 0° — догон) */
  impactAngleM?: number | null
  /** ── диагностика N_экв (только валидные отсчёты) ── */
  nEffMedian: number | null
  nEffQ25: number | null
  nEffQ75: number | null
  nEffMin: number | null
  nEffMax: number | null
  /** доля шагов наведения с валидным N_экв */
  nEffValidFrac: number
  /** доля кадров с насыщением команды по n_max */
  satFrac: number
  /** корреляция N_экв с rho / t_cpa (если выборка достаточна) */
  corrNEffRho: number | null
  corrNEffTgo: number | null
  nEffCount: number
}

/** Точка диагностики N_экв одного кадра — для лабораторных графиков адаптивности. */
export type NeffPoint = {
  t: number
  tgo: number | null
  rho: number
  vc: number
  nEff: number | null
  valid: boolean
  sat: boolean
  speedMode: TargetSpeedMode
}

export type TrainPoint = {
  ep: number
  miss: number
  hit: boolean
  tGuide: number | null
  refDev: number
  /** веса DN после эпизода, развёрнутые 2×8 */
  w: number[]
  /** номер сессии обучения — для вертикальных разделителей на графиках */
  session?: number
  /** «цена» манёвра и качество наведения по эпизоду */
  nAvg?: number
  nPeak?: number
  lockFrac?: number
  nrms?: number
}

export type GenPoint = {
  gen: number
  best: number
  avg: number
  worst?: number
  /** средний разброс весов биомух (разнообразие популяции) */
  diversity?: number
  /** доля перехватов в поколении, 0…1 */
  hitRate?: number
  /** лучший промах на эталонном трио (валидационные поколения) */
  champ?: number
  /** поколение посчитано локальной демо-моделью (не серверной научной) */
  local?: boolean
}

export type FramePoint = { t: number; zem: number; dev: number | null; rng: number }

/** ─── дуэль: матрица «Ринг» (ответ /api/duel) ─── */

export type DuelCell = {
  row: string
  col: string
  /** вердикт ячейки: кто взял — ракета или цель */
  win: 'missile' | 'evader'
  hit_rate: number
  t_survived: number
  cpa_m: number
  n_int: number
  n_peak: number
  fuse_expired: boolean
}

export type DuelMatrix = {
  missiles: string[]
  evaders: string[]
  repeats: number
  scenario: { aspect: Aspect; fuse_life_s: number; duel: boolean; n_target: number }
  cells: DuelCell[]
}

export type RunPoint = {
  label: string
  at: number
  miss: number
  hit: boolean
  tGuide: number | null
  refDev: number | null
  nPeak: number
  lockFrac: number
  /** η — угол встречи в момент наибольшего сближения, град.; null — нет данных */
  eta?: number | null
}

export type LabData = {
  train: TrainPoint[]
  gen: GenPoint[]
  /** последний прогон: ZEM и отклонение от эталона по кадрам */
  zem: FramePoint[]
  runs: RunPoint[]
  /** диагностика N_экв последнего прогона (графики адаптивности) */
  neff: NeffPoint[]
  /** траектории последнего прогона, план X-Y: [дальность, бок] */
  traj?: {
    missile: number[][]
    ghost: number[][]
    target: number[][]
  } | null
  /** сводка последнего обучения до/после (каноническое трио) */
  summary?: {
    missBefore?: number
    missAfter?: number
    tGuideAfter?: number | null
    refDevBefore?: number
    refDevAfter?: number
    hitRateAfter?: number
  } | null
}

export function emptyLab(): LabData {
  return { train: [], gen: [], zem: [], runs: [], neff: [], traj: null, summary: null }
}

export const DEFAULT_SCENARIO: Scenario = {
  aspect: 'head-on',
  v_m: 780,
  v_t: 260,
  range_m: 8000,
  off_axis_m: 420,
  free_tx: 8000,
  free_ty: 0,
  free_talt: 4000,
  free_mhdg: 0,
  free_mclimb: 0,
  free_thdg: 180,
  free_tclimb: 0,
  n_max: 30,
  n_target: 0,
  maneuver: 'straight',
  pn_n: 4,
  pn_sched_n0: 3,
  pn_sched_k_rho: 0.8,
  pn_sched_n_min: 2,
  pn_sched_n_max: 6,
  target_speed_mode: 'constant',
  target_longitudinal_g: 1,
  target_speed_min: 120,
  target_speed_max: 520,
  target_speed_period_s: 6,
  target_speed_phase: 0,
  mode: 'pn',
  circuit_gain: 1.15,
  tau_s: 0.025,
  tau_act_s: 0,
  fov_deg: 14,
  seeker_delay_s: 0.02,
  t_max: 28,
  kill_radius_m: 45,
  alt_m: 4000,
  noise_az_deg: 0,
  noise_range_m: 0,
  lock_drop_p: 0,
  seeker_jitter_s: 0,
  law: 'pn',
  brain: 'stub',
  duel: false,
  evader_law: 'away',
  fuse_life_s: 30,
  model: 'kinematic_legacy',
  phys_atmos: false,
  phys_cx_wave: 0,
  phys_mach_kr: 1.0,
  phys_mach_band: 0.1,
  phys_cn_max: 0,
  phys_wn_act: 0,
  phys_zeta_act: 1.0,
  phys_tau_a_s: 0.05,
  bio_fov_deg: 165,
  retina_death_p: 0,
  retina_dropout_p: 0,
}
