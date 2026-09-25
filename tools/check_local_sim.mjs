// Сверка траекторий цели JS-фолбэка (localSim) с эталоном python.
// Использование: node tools/check_local_sim.mjs /tmp/ref.json /tmp/ls.mjs
// ls.mjs — бандл localSim.ts (npx esbuild src/localSim.ts --bundle --format=esm).
import { readFileSync } from 'node:fs'
import { pathToFileURL } from 'node:url'

const [refPath, lsPath] = process.argv.slice(2)
const ref = JSON.parse(readFileSync(refPath, 'utf8'))
const mod = await import(pathToFileURL(lsPath).href)

const base = {
  aspect: 'head-on',
  v_m: 780,
  v_t: 240,
  range_m: 6000,
  off_axis_m: 200,
  n_max: 30,
  n_target: 3,
  alt_m: 4000,
  t_max: 5,
  dt: 0.02,
  mode: 'pn',
  pn_n: 4,
  law: 'pn',
  circuit_gain: 1.15,
  tau_s: 0.025,
  fov_deg: 14,
  seeker_delay_s: 0,
  kill_radius_m: 45,
  noise_az_deg: 0,
  noise_range_m: 0,
  lock_drop_p: 0,
  seeker_jitter_s: 0,
  brain: 'stub',
}

let fails = 0

// ракета (МПС) в свободной расстановке с виражем: кадры localRun против collect
{
  const refData = ref.missile
  if (refData) {
    const sc = { ...base, ...refData.scenario }
    let maxErr = 0
    let i = 0
    // последний кадр — терминальная дубль-рамка engine (пост-шаговое состояние);
    // localRun её не эмитит — сверяем строгую пред-шаговую часть
    for (const fr of mod.localRun(sc)) {
      if (i >= refData.pos.length - 1) break
      const q = refData.pos[i]
      maxErr = Math.max(maxErr, Math.abs(fr.missile[0] - q[0]), Math.abs(fr.missile[1] - q[1]), Math.abs(fr.missile[2] - q[2]))
      i += 1
    }
    const status = maxErr < 1e-6 && refData.pos.length - i <= 2 ? 'ok' : 'FAIL'
    console.log(`missile(free,turn): кадров ${i}, max|Δ| = ${maxErr.toExponential(2)} [${status}]`)
    if (status === 'FAIL' || i === 0) fails += 1
  }
}

for (const [maneuver, refData] of Object.entries(ref)) {
  if (maneuver === 'missile') continue
  const sc = { ...base, maneuver }
  const frames = []
  for (const fr of mod.localRun(sc)) {
    frames.push(fr)
    if (frames.length >= refData.pos.length * 10) break
  }
  const pts = frames.filter((_, i) => i % 10 === 0).map((f) => f.target)
  if (pts.length !== refData.pos.length) {
    console.error(`${maneuver}: точек ${pts.length} != ${refData.pos.length}`)
    fails += 1
    continue
  }
  let maxErr = 0
  pts.forEach((p, i) => {
    const q = refData.pos[i]
    maxErr = Math.max(maxErr, Math.abs(p[0] - q[0]), Math.abs(p[1] - q[1]), Math.abs(p[2] - q[2]))
  })
  const status = maxErr < 1e-6 ? 'ok' : 'FAIL'
  console.log(`${maneuver}: max|Δ| = ${maxErr.toExponential(2)} [${status}]`)
  if (status === 'FAIL') fails += 1
}

if (fails) {
  console.error(`PARITY FAIL: ${fails}`)
  process.exit(1)
}
console.log('parity ok')
