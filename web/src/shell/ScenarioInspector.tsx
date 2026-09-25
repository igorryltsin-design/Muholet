import { Accordion } from '../ui'
import { SeekerView } from '../SeekerView'
import { AdvancedScenarioSettings } from './AdvancedScenarioSettings'
import { freePatchFromPreset } from './FreeGeometryEditor'
import { LAW_RU } from './labels'
import type { Maneuver, Scenario } from '../types'

/**
 * Контекстный инспектор экрана «Полёт»: один столбец аккордеон-секций вместо
 * стены равноправных карточек. Редкие параметры — в закрытой секции «Дополнительно».
 */
export function ScenarioInspector({
  sc,
  set,
  onManeuver,
  busy,
  frame,
  day,
}: {
  sc: Scenario
  set: <K extends keyof Scenario>(key: K, value: Scenario[K]) => void
  /** Выбор манёвра с авто-включением перегрузки цели (нулевая перегрузка делает манёвр пустым). */
  onManeuver: (m: Maneuver) => void
  busy: boolean
  frame: import('../types').Frame | null
  day: boolean
}) {
  return (
    <div className="inspector__sections">
      <Accordion id="seeker" title="Кадр головки" defaultOpen={false} extra={frame?.lock ? 'захват' : `${((frame?.seeker.size ?? 0) * 100).toFixed(1)}%`}>
        <SeekerView frame={frame} day={day} bioFovDeg={sc.bio_fov_deg} />
        <p className="lab-hint" style={{ padding: '6px 0 2px' }}>
          Фовеальная сетчатка 16×16: плотный центр, редкая периферия. Светлое пятно — цель, рамка — захват.
        </p>
      </Accordion>

      <Accordion id="scenario" title="Сценарий" hint="Стартовая геометрия перехвата.">
        <div className="fields">
          <label data-tip="Встречные — цель летит навстречу; пересечение — поперёк курса; вдогон — убегает. «Свободная расстановка» — ручной сценарий: крупная панель «Расстановка» поверх сцены, цель тянется на плане, профиле и прямо в 3D (Shift — высота), курсы — за кончики стрелок; расстановки сохраняются пресетами.">
            Схема сближения
            <select
              value={sc.aspect}
              onChange={(e) => {
                const a = e.target.value as Scenario['aspect']
                set('aspect', a)
                if (a === 'free') for (const [k, v] of Object.entries(freePatchFromPreset(sc))) set(k as keyof Scenario, v as never)
              }}
            >
              <option value="head-on">встречные курсы</option>
              <option value="beam">пересечение курсов</option>
              <option value="tail-chase">вдогон</option>
              <option value="free">свободная расстановка</option>
            </select>
          </label>
          {sc.aspect === 'free' && (
            <p className="lab-hint" style={{ gridColumn: '1 / -1', padding: 0 }}>
              Ручная геометрия — в панели «Расстановка» поверх сцены (кнопка «расстановка» в углу).
            </p>
          )}
          <label data-tip="Прямолинейно, вираж, змейка — или сложные: переменная змейка, резкий вираж, «ножницы», горка/пике, всё сразу. Манёвр работает при «перегрузке цели» > 0.">
            Манёвр цели
            <select value={sc.maneuver} onChange={(e) => onManeuver(e.target.value as Maneuver)}>
              <option value="straight">прямолинейно</option>
              <option value="turn">вираж</option>
              <option value="weave">змейка</option>
              <option value="weave_var">перем. змейка</option>
              <option value="break">резкий вираж</option>
              <option value="scissors">ножницы</option>
              <option value="dive">горка/пике</option>
              <option value="combo">всё сразу</option>
            </select>
          </label>
          <label data-tip={sc.aspect === 'free' ? 'В свободной расстановке дальность задаёт сама расстановка — поле показывает наклонную дальность «пуск—цель».' : 'Дистанция пуска до цели.'}>
            Дальность, км
            <input
              type="number"
              value={sc.aspect === 'free' ? Math.hypot(sc.free_tx, sc.free_ty, sc.free_talt - sc.alt_m) / 1000 : sc.range_m / 1000}
              disabled={sc.aspect === 'free'}
              onChange={(e) => set('range_m', Number(e.target.value) * 1000)}
            />
          </label>
          <label data-tip="Подлёт ближе этого расстояния — срабатывание боевой части и «перехват».">
            Радиус срабатывания БЧ, м
            <input type="number" value={sc.kill_radius_m} onChange={(e) => set('kill_radius_m', Number(e.target.value))} />
          </label>
        </div>
      </Accordion>

      <Accordion id="guidance" title="Наведение" hint="Кто ведёт ракету и по какому закону." extra={MODE_SHORT[sc.mode]}>
        <div className="fields">
          <label data-tip="ПН — классика (гасим вращение линии визирования). Био — контур дрозофилы, видит только кадр 16×16. Смешанный — био, при потере захвата включает ПН.">
            Способ наведения
            <select value={sc.mode} onChange={(e) => set('mode', e.target.value as Scenario['mode'])}>
              <option value="pn">пропорциональное</option>
              <option value="bio">по коннектому</option>
              <option value="both">смешанный</option>
            </select>
          </label>
          <label data-tip="Мозг режима «био»: схема, полный или коннектом. Обучение и файлы мозга — в пространстве «Мозг».">
            Мозг
            <select value={sc.brain} onChange={(e) => set('brain', e.target.value as Scenario['brain'])} disabled={busy}>
              <option value="stub">схема (279 клеток)</option>
              <option value="full">полный (4 439 клеток)</option>
              <option value="connectome">коннектом-модель (~109 тыс)</option>
            </select>
          </label>
          <label data-tip="Эталонные законы (ПН, ПН с компенсацией ускорения цели, метод погони, метод трёх точек) видят точную геометрию — это сравнение «с подсказкой». «ПН по измерениям ГСН» — сенсорная: команда только из декодированных углов кадра.">
            Закон наведения
            <select value={sc.law} onChange={(e) => set('law', e.target.value as Scenario['law'])} disabled={sc.mode !== 'pn'}>
              <option value="pn">Метод пропорциональной навигации (ПН) — базовый</option>
              <option value="tpn">Истинная ПН — команда по нормали к линии визирования (эталон)</option>
              <option value="apn">ПН с компенсацией нормального ускорения цели (эталон)</option>
              <option value="pure">Метод погони (эталон)</option>
              <option value="clos">Метод трёх точек (эталон)</option>
              <option value="pn_gsn">ПН по измерениям сенсорного канала ГСН (сенсорная)</option>
              <option value="pn_sched_oracle">ПН с переменным навигационным коэффициентом N (эталон, по истинному t_cpa)</option>
              <option value="pn_sched_sensor">ПН с переменным навигационным коэффициентом N (сенсорная, по ρ)</option>
            </select>
          </label>
          <label data-tip="Кинематическая модель — постоянный модуль скорости (обратная совместимость, работает и в локальном окне). Трёхстепенная модель движения центра масс добавляет тягу, сопротивление, тяжесть и инерцию исполнительного контура; считается только на сервере (Python — авторитет).">
            Модель движения
            <select value={sc.model} onChange={(e) => set('model', e.target.value as Scenario['model'])}>
              <option value="kinematic_legacy">Кинематическая (постоянная скорость)</option>
              <option value="point_mass_3dof">Трёхстепенная (тяга/сопротивление/тяжесть) — сервер</option>
            </select>
          </label>
          <label data-tip="Навигационный коэффициент N (безразмерный): a = N·(ω_ЛВ × V). Больше — резче наведение и выше перегрузка. Обычно 3…5.">
            Навигационный коэффициент N
            <input type="number" step={0.5} value={sc.pn_n} onChange={(e) => set('pn_n', Number(e.target.value))} />
          </label>
          {sc.law === 'pn_sched_oracle' || sc.law === 'pn_sched_sensor' ? (
            <label data-tip="Параметры переменного навигационного коэффициента: N = clamp(N₀ + k_ρ·ρ, N_min, N_max).">
              N₀ · k_ρ · N_min · N_max
              <span className="pair">
                <input type="number" step={0.5} value={sc.pn_sched_n0} onChange={(e) => set('pn_sched_n0', Number(e.target.value))} />
                <input type="number" step={0.1} value={sc.pn_sched_k_rho} onChange={(e) => set('pn_sched_k_rho', Number(e.target.value))} />
                <input type="number" step={0.5} value={sc.pn_sched_n_min} onChange={(e) => set('pn_sched_n_min', Number(e.target.value))} />
                <input type="number" step={0.5} value={sc.pn_sched_n_max} onChange={(e) => set('pn_sched_n_max', Number(e.target.value))} />
              </span>
            </label>
          ) : null}
          <label data-tip="Общее усиление биоконтура: насколько сильно выход отклоняет рули.">
            Коэффициент усиления контура
            <input type="number" step={0.05} value={sc.circuit_gain} onChange={(e) => set('circuit_gain', Number(e.target.value))} />
          </label>
          <label data-tip="τ нейронов: меньше — реакция резче, больше — контур инертнее.">
            Постоянная времени, с
            <input type="number" step={0.005} value={sc.tau_s} onChange={(e) => set('tau_s', Number(e.target.value))} />
          </label>
        </div>
        <p className="lab-hint" style={{ padding: '8px 0 2px' }}>Текущий закон: {LAW_RU[sc.law] ?? sc.law}</p>
      </Accordion>

      <Accordion id="missile" title="Ракета и цель" hint="Скорости, перегрузки и профиль скорости цели.">
        <div className="fields">
          <label data-tip="Скорость ракеты после пуска.">
            Скорость ракеты, м/с
            <input type="number" value={sc.v_m} onChange={(e) => set('v_m', Number(e.target.value))} />
          </label>
          <label data-tip="Фактический начальный модуль скорости цели — одинаково для всех схем сближения.">
            Скорость цели, м/с
            <input type="number" value={sc.v_t} onChange={(e) => set('v_t', Number(e.target.value))} />
          </label>
          <label data-tip="Предел нормального ускорения ракеты, g (поворот вектора скорости).">
            Норм. ускорение ракеты
            <input type="number" value={sc.n_max} onChange={(e) => set('n_max', Number(e.target.value))} />
          </label>
          <label data-tip="Лаг рулевого привода: первое апериодическое звено между командой закона и фактическим ускорением (τ·a_исп′ + a_исп = a_ком). Типично 0.02…0.2 с; 0 — идеальное исполнение (как раньше). Лаг увеличивает промах по [Гусев] гл. 5; в трёхстепенной модели для этого есть отдельный параметр τ_a.">
            Лаг привода, с
            <input type="number" step={0.01} min={0} value={sc.tau_act_s} onChange={(e) => set('tau_act_s', Math.max(0, Number(e.target.value)))} />
          </label>
          <label data-tip="Нормальное ускорение уклонения цели, g. Ноль — цель не маневрирует, и выбор манёвра ничего не меняет.">
            Норм. ускорение цели
            <input type="number" value={sc.n_target} onChange={(e) => set('n_target', Number(e.target.value))} />
          </label>
          <label data-tip="Профиль скорости цели внутри полёта: constant — модуль сохраняется; разгон/торможение; импульс; синусоида.">
            Профиль скорости цели
            <select value={sc.target_speed_mode} onChange={(e) => set('target_speed_mode', e.target.value as Scenario['target_speed_mode'])}>
              <option value="constant">постоянная</option>
              <option value="accelerate">разгон</option>
              <option value="decelerate">торможение</option>
              <option value="pulse">импульс</option>
              <option value="sine">синусоида</option>
            </select>
          </label>
          {sc.target_speed_mode !== 'constant' && (
            <>
              <label data-tip="Продольное ускорение цели (разгон/торможение), g. Учебные пределы: 0.5–1.5 g.">
                Продольное ускорение, g
                <input type="number" step={0.1} min={0.2} max={2} value={sc.target_longitudinal_g} onChange={(e) => set('target_longitudinal_g', Number(e.target.value))} />
              </label>
              <label data-tip="Нижняя/верхняя граница модуля скорости цели, м/с.">
                Границы скорости, м/с
                <span className="pair">
                  <input type="number" value={sc.target_speed_min} onChange={(e) => set('target_speed_min', Number(e.target.value))} />
                  <input type="number" value={sc.target_speed_max} onChange={(e) => set('target_speed_max', Number(e.target.value))} />
                </span>
              </label>
              {(sc.target_speed_mode === 'sine' || sc.target_speed_mode === 'pulse') && (
                <>
                  <label data-tip="Период синусоиды (окно импульса = 15% периода), с.">
                    Период профиля, с
                    <input type="number" step={0.5} value={sc.target_speed_period_s} onChange={(e) => set('target_speed_period_s', Number(e.target.value))} />
                  </label>
                  <label data-tip="Сдвиг старта профиля по времени, с (детерминирован).">
                    Фаза, с
                    <input type="number" step={0.5} value={sc.target_speed_phase} onChange={(e) => set('target_speed_phase', Number(e.target.value))} />
                  </label>
                </>
              )}
            </>
          )}
        </div>
      </Accordion>

      <Accordion id="noises" title="Шумы и отказы" hint="Насколько честно (плохо) видит головка." extra={sc.noise_az_deg > 0 || sc.lock_drop_p > 0 ? 'включены' : 'идеально'}>
        <div className="fields">
          <label data-tip="σ шума пеленга по азимуту и углу места, град. Ноль — идеальный измеритель.">
            Шум пеленга, °
            <input type="number" step={0.5} value={sc.noise_az_deg} onChange={(e) => set('noise_az_deg', Number(e.target.value))} />
          </label>
          <label data-tip="σ шума дальности до цели, м.">
            Шум дальности, м
            <input type="number" step={10} value={sc.noise_range_m} onChange={(e) => set('noise_range_m', Number(e.target.value))} />
          </label>
          <label data-tip="Вероятность пропустить захват на кадр, 0…1. Проверяет живучесть наведения при срывах головки.">
            Пропуски захвата
            <input type="number" step={0.05} min={0} max={1} value={sc.lock_drop_p} onChange={(e) => set('lock_drop_p', Math.max(0, Math.min(1, Number(e.target.value))))} />
          </label>
          <label data-tip="Независимый дропаут рецепторов на кадр (в дополнение к постоянной карте отказов).">
            Дропаут сетчатки
            <input type="number" step={0.05} min={0} max={1} value={sc.retina_dropout_p} onChange={(e) => set('retina_dropout_p', Math.max(0, Math.min(1, Number(e.target.value))))} />
          </label>
        </div>
      </Accordion>

      <Accordion id="advanced" title="Дополнительно" defaultOpen={false}>
        <AdvancedScenarioSettings sc={sc} set={set} />
      </Accordion>
    </div>
  )
}

const MODE_SHORT: Record<Scenario['mode'], string> = { pn: 'ПН', bio: 'био', both: 'смешанный' }
