/** Обучаемый выход контура в окне, если сервер молчит. Схема признаков v2 (10 шт.).
 *
 * Миграция: старые веса 2×8 (без theta/rho) расширяются нулями на новых
 * столбцах — выученная политика не меняется до дообучения. */

import { FEATURE_SCHEMA_VERSION, type BrainKind } from './types'

export const FEAT = 10
const LEGACY_FEAT = 8
/** старые столбцы v1 (az, el, ωβ, ωλ, loom, Φx, Φy, захват) → позиции в v2 */
const LEGACY_COLUMN_MAP = [0, 1, 2, 3, 5, 7, 8, 9]
/** новые столбцы v2: 4 = theta, 6 = rho */
export const IDX_THETA = 4
export const IDX_RHO = 6
/** нормировка фазовых признаков (зеркало navedenie/circuit.py) */
export const K_THETA = 4.0
export const K_RHO = 0.4

function migrateW(matrix: number[][]): number[][] {
  const width = matrix[0]?.length ?? 0
  if (width === FEAT) return matrix.map((row) => row.slice())
  if (width !== LEGACY_FEAT) return matrix.map((row) => row.slice(0, FEAT).concat(Array(Math.max(0, FEAT - (row.length ?? 0))).fill(0)))
  return matrix.map((row) => {
    const out = Array(FEAT).fill(0)
    LEGACY_COLUMN_MAP.forEach((newIdx, oldIdx) => {
      out[newIdx] = row[oldIdx] ?? 0
    })
    return out
  })
}

function defaultW(): number[][] {
  const w = Array.from({ length: 2 }, () => Array(FEAT).fill(0))
  w[0][1] = 1.8
  w[0][3] = 0.35
  w[0][5] = 0.25
  w[0][8] = 0.4
  w[1][0] = 1.8
  w[1][2] = 0.35
  w[1][5] = 0.25
  w[1][7] = 0.4
  return w
}

let W = defaultW()
let trained = false
let kind: BrainKind = 'stub'
export const featureSchemaVersion = FEATURE_SCHEMA_VERSION

export function brainKind() {
  return kind
}
export function setBrainKind(next: BrainKind) {
  kind = next
}
export function isTrained() {
  return trained
}
export function nCells() {
  return kind === 'full' ? 4439 : kind === 'connectome' ? 138700 : 279
}
export function getW() {
  return W
}

export function features(
  az: number,
  el: number,
  azDot: number,
  elDot: number,
  theta: number,
  loom: number,
  rho: number,
  flowX: number,
  flowY: number,
  lock: boolean,
): number[] {
  const L = lock ? 1 : 0
  const tanh = Math.tanh
  return [
    tanh(az * 3 * L),
    tanh(el * 3 * L),
    tanh(azDot * 0.4 * L),
    tanh(elDot * 0.4 * L),
    tanh(theta * K_THETA * L),
    tanh(loom),
    tanh(rho * K_RHO * L),
    tanh(flowX * 4),
    tanh(flowY * 4),
    L,
  ]
}

export function readout(feat: number[]): { pitch: number; yaw: number } {
  const pitch = Math.tanh(W[0].reduce((s, v, i) => s + v * (feat[i] ?? 0), 0))
  const yaw = Math.tanh(W[1].reduce((s, v, i) => s + v * (feat[i] ?? 0), 0))
  return { pitch, yaw }
}

export function sgd(feat: number[], teacher: number[], lr = 0.04) {
  const pred = readout(feat)
  const err = [teacher[0] - pred.pitch, teacher[1] - pred.yaw]
  for (let i = 0; i < FEAT; i += 1) {
    W[0][i] = Math.max(-4, Math.min(4, W[0][i] + lr * err[0] * (feat[i] ?? 0)))
    W[1][i] = Math.max(-4, Math.min(4, W[1][i] + lr * err[1] * (feat[i] ?? 0)))
  }
}

export function markTrained() {
  trained = true
}

export function applyServerW(matrix: number[][] | undefined) {
  if (!matrix || matrix.length !== 2) return
  W = migrateW(matrix)
  trained = true
}
