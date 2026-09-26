import { lazy, Suspense, type ComponentProps } from 'react'
import { Viewport3DSkeleton } from './shell/Viewport3DSkeleton'

/**
 * Ленивые 3D-вьюхи: three.js — самая тяжёлая часть бандла, поэтому она живёт
 * в отдельных асинхронных чанках и грузится при первом показе сцены, а не
 * блокирует старт приложения. Фолбэк занимает ту же геометрию, что канвас
 * (абсолютное заполнение .scene-card), — без прыжков раскладки.
 */

const EngagementViewImpl = lazy(() => import('./EngagementView').then((m) => ({ default: m.EngagementView })))

export function EngagementView(props: ComponentProps<typeof EngagementViewImpl>) {
  return (
    <Suspense fallback={<Viewport3DSkeleton />}>
      <EngagementViewImpl {...props} />
    </Suspense>
  )
}

const Brain3DViewImpl = lazy(() => import('./Brain3DView').then((m) => ({ default: m.Brain3DView })))

export function Brain3DView(props: ComponentProps<typeof Brain3DViewImpl>) {
  return (
    <Suspense fallback={<Viewport3DSkeleton />}>
      <Brain3DViewImpl {...props} />
    </Suspense>
  )
}

export type { Playback } from './EngagementView'
