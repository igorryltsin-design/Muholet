import type { Scenario } from '../types'

/**
 * «Дополнительно» — редкие параметры контура и зрения: задержки, джиттер,
 * размеры полей зрения; при трёхстепенной модели — блок phys-ручек
 * (атмосфера, волновой кризис, Cn_max, привод). По умолчанию секция закрыта.
 */
export function AdvancedScenarioSettings({ sc, set }: { sc: Scenario; set: <K extends keyof Scenario>(key: K, value: Scenario[K]) => void }) {
  return (
    <div className="fields">
      <label data-tip="Общее узкое поле эталонного ПН и центральное разрешение БИО (фовея): около этого угла цель видна точно.">
        Узкий кадр (эталон), град
        <input type="number" value={sc.fov_deg} onChange={(e) => set('fov_deg', Number(e.target.value))} />
      </label>
      <label data-tip="Физическое поле зрения БИО: фовеальная сетчатка 16×16 — плотный центр, редкая периферия до ~165°.">
        Поле зрения БИО, град
        <input type="number" value={sc.bio_fov_deg} onChange={(e) => set('bio_fov_deg', Number(e.target.value))} />
      </label>
      <label data-tip="Задержка данных головки: реальная ГСН не мгновенна, контур видит прошлое.">
        Запаздывание головки, с
        <input type="number" step={0.005} value={sc.seeker_delay_s} onChange={(e) => set('seeker_delay_s', Number(e.target.value))} />
      </label>
      <label data-tip="Случайные дополнительные задержки головки, с: контур иногда видит более старый кадр.">
        Разброс задержки, с
        <input type="number" step={0.01} value={sc.seeker_jitter_s} onChange={(e) => set('seeker_jitter_s', Number(e.target.value))} />
      </label>
      {sc.model === 'point_mass_3dof' && (
        <div className="fields" style={{ borderTop: '1px solid var(--border, #333)', marginTop: 8, paddingTop: 8 }}>
          <label data-tip="Плотность воздуха: постоянная phys_rho_air или ρ(H) по стандартной атмосфере ICAO/US1976 (0…20 км, линейные слои). Атмосфера включает скоростной напор q̄=ρV²/2 по реальной высоте — без неё волновой кризис и граница динамического полёта считаются на одной высоте.">
            Плотность воздуха
            <select value={sc.phys_atmos ? 'atmos' : 'const'} onChange={(e) => set('phys_atmos', e.target.value === 'atmos')}>
              <option value="const">постоянная</option>
              <option value="atmos">атмосфера ICAO ρ(H)</option>
            </select>
          </label>
          <label data-tip="Прирост ΔCx волнового кризиса на сверхзвуковом плато: Cx(M) = Cx₀ + ΔCx·S(M), S — гладкий перегиб около M_кр. 0 — сопротивление постоянно (доминирует трение). Учебный порядок: 0.1–0.3 для осиковых тел (Гусев гл. 3; облик — А.С. Погожев).">
            ΔCx волнового кризиса
            <input type="number" min={0} max={1} step={0.01} value={sc.phys_cx_wave} onChange={(e) => set('phys_cx_wave', Number(e.target.value))} />
          </label>
          <label data-tip="Число Маха перегиба волнового роста M_кр (критическое). Типовое 0.9–1.1.">
            M_кр кризиса
            <input type="number" min={0.5} max={2} step={0.05} value={sc.phys_mach_kr} onChange={(e) => set('phys_mach_kr', Number(e.target.value))} />
          </label>
          <label data-tip="Ширина околозвукового перехода δM в S-кривой кризиса (крутизна = 1/δM). 0.05 — почти ступенька, 0.3 — пологий рост.">
            Ширина δM
            <input type="number" min={0.02} max={0.6} step={0.01} value={sc.phys_mach_band} onChange={(e) => set('phys_mach_band', Number(e.target.value))} />
          </label>
          <label data-tip="Граница динамического полёта: располагаемая перегрузка n_расп = min(n_max, q̄·S·Cn_max/(m·g)) из скоростного напора. 0 — прежний константный предел. Малому Cn_max ракета «недостижима» на большой высоте/малой скорости.">
            Cn_max (0 — константа)
            <input type="number" min={0} max={10} step={0.1} value={sc.phys_cn_max} onChange={(e) => set('phys_cn_max', Number(e.target.value))} />
          </label>
          <label data-tip="Постоянная времени рулевого привода τ_a, с: 1-е апериодическое звено τ·ȧ_акт + a_акт = a_ком. 0 — мгновенное исполнение.">
            Лаг привода τ_a, с
            <input type="number" min={0} max={1} step={0.005} value={sc.phys_tau_a_s} onChange={(e) => set('phys_tau_a_s', Number(e.target.value))} />
          </label>
          <label data-tip="Собственная частота привода 2-го порядка ω_n, рад/с: ä + 2ζω_n·ȧ + ω_n²·a = ω_n²·a_ком. 0 — остаётся 1-е звено (лаг выше). Реальные рулевые приводы 20–60 рад/с.">
            ω_n привода, рад/с
            <input type="number" min={0} max={200} step={1} value={sc.phys_wn_act} onChange={(e) => set('phys_wn_act', Number(e.target.value))} />
          </label>
          <label data-tip="Демпфирование привода ζ: <1 звено колеблет команду (перерегулирование), ≥1 — апериодический переход.">
            Демпфирование ζ
            <input type="number" min={0} max={3} step={0.05} value={sc.phys_zeta_act} onChange={(e) => set('phys_zeta_act', Number(e.target.value))} />
          </label>
        </div>
      )}
    </div>
  )
}
