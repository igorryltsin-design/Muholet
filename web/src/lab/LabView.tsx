import { LAB_GROUPS, type LabTab } from './types'
import type { CmpRow } from './types'
import { RunScreen, SummaryScreen } from './screens/OverviewScreens'
import { TrainScreen, SwarmLabScreen } from './screens/TrainingScreens'
import { CompareScreen, CzScreen, LawScreen, McScreen, NeffScreen, RobScreen } from './screens/AnalysisScreens'
import { MapScreen, ScalingScreen, TransferScreen } from './screens/ResearchScreens'
import { FormulaScreen, ParamsScreen } from './screens/ModelScreens'
import type { AblationData, CaptureZoneData, CoevData, DistillData, FaultsData, LadderData, MapData, MonteCarloData, ScalingData, TransferData } from './types'
import type { BrainKind, LabData } from '../types'

export type { LabTab, AblationData, CaptureZoneData, CoevData, DistillData, FaultsData, LadderData, MapData, MonteCarloData, ScalingData, TransferData, CmpRow }

/**
 * Лаборатория — самостоятельное рабочее пространство: слева двухуровневая навигация
 * «раздел → экран», справа активный экран по единому шаблону (LabScreen).
 * Все вычисления и обработчики остаются в App — сюда приходят только данные и колбэки.
 */
export function LabView({
  data,
  tab,
  onTab,
  onClose,
  brain,
  compare,
  compareLoading,
  onRunCompare,
  onOverlay,
  overlayBusy,
  ablation,
  ablationBusy,
  onRunAblation,
  mapData,
  mapBusy,
  onRunMap,
  mapRepeats,
  onMapRepeats,
  mapRetina,
  onMapRetina,
  mapColor,
  onMapColor,
  faults,
  faultsBusy,
  onRunFaults,
  onCoevTrain,
  coevTraining,
  coevLadder,
  distill,
  distillBusy,
  onRunDistill,
  onApplyDistill,
  distillTeacher,
  onDistillTeacher,
  onDistillSave,
  onDistillLoad,
  distillSaved,
  transfer,
  transferBusy,
  transferKind,
  onTransferKind,
  onRunTransfer,
  scaling,
  scalingBusy,
  scalingKind,
  onScalingKind,
  onRunScaling,
  coev,
  coevBusy,
  onCoevStart,
  onCoevStep,
  onCoevReset,
  swarmExport,
  onExportSwarmCsv,
  championToBrain,
  expList,
  onRefreshExperiments,
  onSaveExperimentServer,
  onLoadExperimentServer,
  onDeleteExperimentServer,
  onClearLab,
  onMatlabExport,
  onTrajExport,
  onExportNeff,
  rob,
  robBusy,
  onRunRobustness,
  mc,
  mcBusy,
  mcRuns,
  onMcRuns,
  onRunMc,
  cz,
  czBusy,
  czGmax,
  onCzGmax,
  czRuns,
  onCzRuns,
  onRunCz,
  tune,
  onTune,
  onRebuild,
  trainCfg,
  onTrainCfg,
  training,
  trainProgress,
  onTrain,
  onTrainStop,
  smooth,
  showFact,
  onLabCfg,
  egg,
  onEgg,
  soundOn,
  onSound,
  voiceKind,
  onVoiceKind,
  humorOn,
  onHumor,
}: {
  data: LabData
  tab: LabTab
  onTab: (t: LabTab) => void
  onClose: () => void
  brain: BrainKind
  compare: { rows?: CmpRow[]; error?: string } | null
  compareLoading: boolean
  onRunCompare: () => void
  onOverlay: () => void
  overlayBusy: boolean
  ablation: AblationData | null
  ablationBusy: boolean
  onRunAblation: () => void
  mapData: MapData | null
  mapBusy: boolean
  onRunMap: () => void
  mapRepeats: number
  onMapRepeats: (v: number) => void
  mapRetina: number
  onMapRetina: (v: number) => void
  mapColor: 'miss' | 'energy'
  onMapColor: (v: 'miss' | 'energy') => void
  faults: FaultsData | null
  faultsBusy: boolean
  onRunFaults: () => void
  onCoevTrain: () => void
  coevTraining: boolean
  coevLadder: LadderData | null
  distill: DistillData | null
  distillBusy: boolean
  onRunDistill: () => void
  onApplyDistill: () => void
  distillTeacher: 'connectome' | 'full' | 'stub'
  onDistillTeacher: (t: 'connectome' | 'full' | 'stub') => void
  onDistillSave: () => void
  onDistillLoad: () => void
  distillSaved: boolean
  transfer: TransferData | null
  transferBusy: boolean
  transferKind: BrainKind
  onTransferKind: (k: BrainKind) => void
  onRunTransfer: () => void
  scaling: ScalingData | null
  scalingBusy: boolean
  scalingKind: 'full' | 'connectome'
  onScalingKind: (k: 'full' | 'connectome') => void
  onRunScaling: () => void
  coev: CoevData | null
  coevBusy: boolean
  onCoevStart: () => void
  onCoevStep: () => void
  onCoevReset: () => void
  swarmExport: () => void
  onExportSwarmCsv: () => void
  championToBrain: () => void
  expList: { name: string; saved_at: string; size: number }[]
  onRefreshExperiments: () => void
  onSaveExperimentServer: () => void
  onLoadExperimentServer: (name: string) => void
  onDeleteExperimentServer: (name: string) => void
  onClearLab: () => void
  onMatlabExport: () => void
  onTrajExport: () => void
  onExportNeff: () => void
  rob: { levels: number[]; series: { kind: string; label: string; miss: number[]; nrms: (number | null)[] }[] } | null
  robBusy: boolean
  onRunRobustness: () => void
  mc: MonteCarloData | null
  mcBusy: boolean
  mcRuns: number
  onMcRuns: (n: number) => void
  onRunMc: () => void
  cz: CaptureZoneData | null
  czBusy: boolean
  czGmax: number
  onCzGmax: (g: number) => void
  czRuns: number
  onCzRuns: (n: number) => void
  onRunCz: () => void
  tune: { pool: number; channels: number }
  onTune: (patch: { pool?: number; channels?: number }) => void
  onRebuild: (pool: number, channels: number) => void
  trainCfg: { mode: 'scratch' | 'finetune' | 'result'; lr: number; episodes: number }
  onTrainCfg: (patch: Partial<{ mode: 'scratch' | 'finetune' | 'result'; lr: number; episodes: number }>) => void
  training: boolean
  trainProgress: { ep: number; total: number } | null
  onTrain: () => void
  onTrainStop: () => void
  smooth: number
  showFact: boolean
  onLabCfg: (patch: Partial<{ smooth: number; showFact: boolean }>) => void
  egg: boolean
  onEgg: () => void
  soundOn: boolean
  onSound: (v: boolean) => void
  voiceKind: 'male' | 'female'
  onVoiceKind: (v: 'male' | 'female') => void
  humorOn: boolean
  onHumor: (v: boolean) => void
}) {
  return (
    <div className="labws">
      <nav className="lab-nav" aria-label="Разделы лаборатории">
        {LAB_GROUPS.map((group) => (
          <div key={group.title} className="lab-nav__group">
            <span>{group.title}</span>
            {group.items.map(([key, label]) => (
              <button key={key} type="button" className={tab === key ? 'on' : ''} aria-current={tab === key ? 'page' : undefined} onClick={() => onTab(key)}>
                {label}
              </button>
            ))}
          </div>
        ))}
        <button type="button" className="lab-nav__back" onClick={onClose}>
          ← На сцену
        </button>
        <button type="button" className="lab-nav__clear" onClick={onClearLab} data-tip="Очистить всю историю лаборатории: прогоны, обучение, рой, графики. Действие необратимо.">
          Очистить историю
        </button>
      </nav>
      {tab === 'run' && <RunScreen data={data} onMatlabExport={onMatlabExport} onTrajExport={onTrajExport} />}
      {tab === 'summary' && <SummaryScreen data={data} />}
      {tab === 'train' && (
        <TrainScreen data={data} training={training} trainProgress={trainProgress} onTrain={onTrain} onTrainStop={onTrainStop} smooth={smooth} showFact={showFact} />
      )}
      {tab === 'swarm' && (
        <SwarmLabScreen
          data={data}
          expList={expList}
          onRefreshExperiments={onRefreshExperiments}
          onLoadExperimentServer={onLoadExperimentServer}
          onDeleteExperimentServer={onDeleteExperimentServer}
          onSaveExperimentServer={onSaveExperimentServer}
          onSwarmExport={swarmExport}
          onExportSwarmCsv={onExportSwarmCsv}
          championToBrain={championToBrain}
        />
      )}
      {tab === 'compare' && (
        <CompareScreen
          compare={compare}
          compareLoading={compareLoading}
          onRunCompare={onRunCompare}
          onOverlay={onOverlay}
          overlayBusy={overlayBusy}
          ablation={ablation}
          ablationBusy={ablationBusy}
          onRunAblation={onRunAblation}
          faults={faults}
          faultsBusy={faultsBusy}
          onRunFaults={onRunFaults}
          distill={distill}
          distillBusy={distillBusy}
          onRunDistill={onRunDistill}
          onApplyDistill={onApplyDistill}
          distillTeacher={distillTeacher}
          onDistillTeacher={onDistillTeacher}
          onDistillSave={onDistillSave}
          onDistillLoad={onDistillLoad}
          distillSaved={distillSaved}
        />
      )}
      {tab === 'neff' && <NeffScreen data={data} onExport={onExportNeff} />}
      {tab === 'rob' && <RobScreen rob={rob} robBusy={robBusy} onRunRobustness={onRunRobustness} />}
      {tab === 'mc' && <McScreen mc={mc} mcBusy={mcBusy} mcRuns={mcRuns} onMcRuns={onMcRuns} onRunMc={onRunMc} />}
      {tab === 'cz' && <CzScreen cz={cz} czBusy={czBusy} czGmax={czGmax} onCzGmax={onCzGmax} czRuns={czRuns} onCzRuns={onCzRuns} onRunCz={onRunCz} />}
      {tab === 'law' && <LawScreen brain={brain} />}
      {tab === 'map' && (
        <MapScreen
          mapData={mapData}
          mapBusy={mapBusy}
          onRunMap={onRunMap}
          mapRepeats={mapRepeats}
          onMapRepeats={onMapRepeats}
          mapRetina={mapRetina}
          onMapRetina={onMapRetina}
          mapColor={mapColor}
          onMapColor={onMapColor}
          coev={coev}
          coevBusy={coevBusy}
          onCoevStart={onCoevStart}
          onCoevStep={onCoevStep}
          onCoevReset={onCoevReset}
          onCoevTrain={onCoevTrain}
          coevTraining={coevTraining}
          coevLadder={coevLadder}
        />
      )}
      {tab === 'transfer' && <TransferScreen transfer={transfer} transferBusy={transferBusy} transferKind={transferKind} onTransferKind={onTransferKind} onRunTransfer={onRunTransfer} />}
      {tab === 'scaling' && <ScalingScreen scaling={scaling} scalingBusy={scalingBusy} scalingKind={scalingKind} onScalingKind={onScalingKind} onRunScaling={onRunScaling} />}
      {tab === 'formula' && <FormulaScreen />}
      {tab === 'params' && (
        <ParamsScreen
          tune={tune}
          onTune={onTune}
          onRebuild={onRebuild}
          trainCfg={trainCfg}
          onTrainCfg={onTrainCfg}
          egg={egg}
          onEgg={onEgg}
          soundOn={soundOn}
          onSound={onSound}
          voiceKind={voiceKind}
          onVoiceKind={onVoiceKind}
          humorOn={humorOn}
          onHumor={onHumor}
          smooth={smooth}
          showFact={showFact}
          onLabCfg={onLabCfg}
        />
      )}
    </div>
  )
}
