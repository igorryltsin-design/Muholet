#!/usr/bin/env python3
"""Сборка НАСТОЯЩЕЙ проводки FlyWire (FAFB v783) → data/circuit_v1.npz.

Вход (data/raw/):
  proofread_connections_783.feather   — агрегированные пары pre→post, FlyWire v783
                                        (Dorkenwald et al., Nature 2024; CC BY-NC-SA)
  proofread_root_ids_783.npy          — список proofread-нейронов
  Supplemental_file1_neuron_annotations.tsv — типировки (flyconnectome/flywire_annotations)

Схема отображения на модель (каркас регионов фиксирован, см. navedenie.circuit):
  каждому модельному нейрону региона назначен реальный нейрон (детерминированно,
  по сортированному root_id; при нехватке реальных нейроны повторяются —
  «расширение до масштаба модели», документировано в мете);
  веса блоков — РЕАЛЬНЫЕ синаптические счётчики (со знаком нейромедиатора:
  ацетилхолин ≥ GABA → «+», иначе «−»), агрегированные на детерминированные
  группы базиса. Скрытые блоки влияют на карту активности мозга и счётчик
  синапсов; выход на рули (W_fx, W_dn) остаётся обучаемым.

Запуск: .venv/bin/python tools/extract_subgraph.py
Результат: data/circuit_v1.npz (+ data/circuit_v1_meta.json) — его подхватывает
navedenie.circuit.ConnectomeCircuit при старте.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT_NPZ = ROOT / "data" / "circuit_v1.npz"
OUT_META = ROOT / "data" / "circuit_v1_meta.json"

SOURCE = "FlyWire FAFB v783 (proofread connections, Dorkenwald et al., Nature 2024)"
LICENSE = "CC BY-NC-SA 4.0 (учебное некоммерческое использование)"
DATA_VERSION = "783"
BASIS = 192
BASIS_T45 = slice(0, 64)     # диапазоны групп мишеней на базисе
BASIS_LOB = slice(64, 128)
BASIS_CXMB = slice(128, 192)

# каркас регионов модели: имя → размер (совместим с navedenie.circuit.CONNECTOME_REGIONS)
MODEL_SIZES: dict[str, int] = {
    "ламина": 1600,
    "медулла": 75000,
    "T4/T5": 6400,
    "лобула": 44000,
    "LPLC2": 1500,
    "центральный комплекс": 2000,
    "грибовидное тело": 8000,
    "DN": 200,
}
REGION_ORDER = list(MODEL_SIZES)


def _log(msg: str) -> None:
    print(msg, flush=True)


def load_annotations() -> "pd.DataFrame":
    import pandas as pd

    df = pd.read_csv(
        RAW / "Supplemental_file1_neuron_annotations.tsv",
        sep="\t",
        usecols=["root_id", "super_class", "cell_class", "cell_sub_class", "cell_type", "pos_x", "pos_y", "pos_z"],
        dtype={"root_id": "string"},  # 16-значные ID не представимы в float64 без потерь
        low_memory=False,
    )
    df = df[df.root_id.notna()]
    df["root_id"] = df.root_id.astype(str).astype(np.int64)
    return df


def select_populations(ann: "pd.DataFrame") -> dict[str, np.ndarray]:
    """Реальные root_id по регионам — отсортированные, без повторов."""
    ct = ann.cell_type.fillna("")
    cc = ann.cell_class.fillna("")
    pops = {
        "ламина": ann.cell_sub_class.eq("L1-5"),
        "медулла": cc.isin(["ME", "ME>LO", "ME>LOP", "ME>LO.LOP", "LA>ME", "ME>LA"]),
        "T4/T5": ct.str.startswith(("T4", "T5")) & ann.super_class.eq("optic"),
        "лобула": cc.isin(["LO", "LO>LOP", "LOP>ME.LO"]) | ct.str.startswith(("LC", "LPLC")),
        "LPLC2": ct.eq("LPLC2"),
        "центральный комплекс": cc.eq("CX"),
        "грибовидное тело": cc.eq("Kenyon_Cell"),
        "DN": ann.super_class.eq("descending"),
    }
    out: dict[str, np.ndarray] = {}
    for name, mask in pops.items():
        ids = np.sort(ann.loc[mask.fillna(False).to_numpy(), "root_id"].dropna().astype(np.int64).unique())
        out[name] = ids
        _log(f"  {name}: реальных нейронов {len(ids):,} → позиций модели {MODEL_SIZES[name]:,}")
    return out


def assign_positions(real_ids: np.ndarray, size: int) -> tuple[np.ndarray, dict[int, int], np.ndarray]:
    """Детерминированное назначение реальных нейронов на позиции модели.

    Все реальные нейроны популяции распределяются по позициям циклически
    (позиция = «микроколонка» из нескольких нейронов, их синапсы суммируются).
    Возвращает (real_id на позиции, словарь real_id→позиция, массив групп позиций не нужен).
    """
    pos_of = np.arange(len(real_ids), dtype=np.int64) % size
    mapping = {int(r): int(p) for r, p in zip(real_ids, pos_of)}
    if len(real_ids) >= size:
        idx = np.linspace(0, len(real_ids) - 1, size).astype(np.int64)
    else:
        idx = np.arange(size) % len(real_ids)
    return real_ids[idx], mapping, pos_of


def load_connections(real_all: np.ndarray) -> "pd.DataFrame":
    import pyarrow.feather as feather
    import pandas as pd

    _log("читаю proofread_connections_783.feather …")
    t = feather.read_table(
        str(RAW / f"proofread_connections_{DATA_VERSION}.feather"),
        columns=["pre_pt_root_id", "post_pt_root_id", "syn_count", "ach_avg", "gaba_avg"],
    )
    df = t.to_pandas()
    mask = df.pre_pt_root_id.isin(real_all) & df.post_pt_root_id.isin(real_all)
    df = df[mask].copy()
    # знак веса по нейромедиатору: ацетилхолин — возбуждение, GABA — торможение
    df["sign"] = np.where(df.ach_avg.fillna(0) >= df.gaba_avg.fillna(0), 1.0, -1.0)
    df["w"] = df.syn_count.astype(np.float64) * df.sign
    _log(f"  пар с участием выбранных нейронов: {len(df):,}; синапсов: {int(df.syn_count.sum()):,}")
    return df[["pre_pt_root_id", "post_pt_root_id", "w"]]


def group_of(real_ids: np.ndarray, lo: int, span: int) -> np.ndarray:
    """Детерминированная группа базиса для реального нейрона в диапазоне [lo, lo+span)."""
    return (real_ids % np.int64(span)).astype(np.int64) + np.int64(lo)


def normalize(w: np.ndarray, scale: float = 0.5) -> np.ndarray:
    """Нормировка на 95-перцентиль |веса| с сохранением знака."""
    a = np.abs(w).astype(np.float64).ravel()
    pos = a[a > 0]
    p95 = float(np.percentile(pos, 95)) if pos.size else 0.0
    if p95 <= 0:
        return np.zeros_like(w)
    return (w.astype(np.float64) / p95 * scale).astype(np.float32)


def normalize_median(w: np.ndarray, scale: float = 0.6, clip: float = 2.0) -> np.ndarray:
    """Нормировка на медиану положительных весов (для ретинотопических слоёв,
    где входная сила распределена тяжело: типичный нейрон должен быть заметен)."""
    a = np.abs(w).astype(np.float64).ravel()
    pos = a[a > 0]
    med = float(np.median(pos)) if pos.size else 0.0
    if med <= 0:
        return np.zeros_like(w)
    return np.clip(w.astype(np.float64) / med * scale, -clip, clip).astype(np.float32)


def main() -> int:
    try:
        import pandas as pd  # noqa: F401
    except ImportError:
        print("нужны pandas и pyarrow: .venv/bin/pip install pandas pyarrow", file=sys.stderr)
        return 1
    missing = [f for f in (
        f"proofread_connections_{DATA_VERSION}.feather",
        "proofread_root_ids_783.npy",
        "Supplemental_file1_neuron_annotations.tsv",
    ) if not (RAW / f).exists()]
    if missing:
        print("нет файлов в data/raw/:", ", ".join(missing))
        print("скачайте таблицы FlyWire v783 (см. NOTICE) и повторите")
        return 1

    _log("1/6 аннотации и популяции")
    ann = load_annotations()
    pops = select_populations(ann)

    pos_map: dict[str, np.ndarray] = {}
    pos_index: dict[str, dict[int, int]] = {}
    for name in REGION_ORDER:
        on_grid, mapping, _ = assign_positions(pops[name], MODEL_SIZES[name])
        pos_map[name] = on_grid
        pos_index[name] = mapping

    real_all = np.unique(np.concatenate([pops[n] for n in REGION_ORDER]))
    con = load_connections(real_all)

    meta_pairs: dict[str, dict] = {}

    def pair_edges(src: str, dst: str) -> "pd.DataFrame":
        """Рёбра src→dst в модельных позициях; синапсы микроколонок просуммированы."""
        e = con[con.pre_pt_root_id.isin(pops[src]) & con.post_pt_root_id.isin(pops[dst])].copy()
        if len(e) == 0:
            meta_pairs[f"{src}→{dst}"] = {"edges": 0, "synapses": 0.0, "model_pairs": 0}
            return pd.DataFrame(columns=["pin", "pout", "w"])
        e["pin"] = e.pre_pt_root_id.map(pos_index[src]).astype("Int64")
        e["pout"] = e.post_pt_root_id.map(pos_index[dst]).astype("Int64")
        e = e.dropna(subset=["pin", "pout"])
        g = e.groupby(["pin", "pout"]).w.sum().reset_index()
        g["pin"] = g.pin.astype(np.int64)
        g["pout"] = g.pout.astype(np.int64)
        meta_pairs[f"{src}→{dst}"] = {
            "edges": int(len(e)),
            "synapses": float(e.w.abs().sum()),
            "model_pairs": int(len(g)),
        }
        return g

    def assignment(region: str, group: np.ndarray, width: int = BASIS) -> np.ndarray:
        m = np.zeros((MODEL_SIZES[region], width), dtype=np.float32)
        m[np.arange(MODEL_SIZES[region]), group] = 1.0
        return m

    _log("2/6 ламина → медулла (через базис)")
    g_lame = pair_edges("ламина", "медулла")
    me_group = group_of(pos_map["медулла"], 0, BASIS)
    w_ba = np.zeros((BASIS, MODEL_SIZES["ламина"]), dtype=np.float64)
    if len(g_lame):
        np.add.at(w_ba, (me_group[g_lame.pout.to_numpy()], g_lame.pin.to_numpy()), g_lame.w.to_numpy())
    W_bA = normalize(w_ba)
    W_med = assignment("медулла", me_group)

    _log("3/6 медулла → T4/T5, лобула, ЦК, ГТ (общий базис, диапазоны групп мишеней)")
    t45_group = group_of(pos_map["T4/T5"], BASIS_T45.start, BASIS_T45.stop - BASIS_T45.start)
    lob_group = group_of(pos_map["лобула"], BASIS_LOB.start, BASIS_LOB.stop - BASIS_LOB.start)
    cx_group = group_of(pos_map["центральный комплекс"], BASIS_CXMB.start, 32)
    mb_group = group_of(pos_map["грибовидное тело"], BASIS_CXMB.start + 32, 32)
    targets = {
        "T4/T5": t45_group,
        "лобула": lob_group,
        "центральный комплекс": cx_group,
        "грибовидное тело": mb_group,
    }
    w_bb = np.zeros((BASIS, MODEL_SIZES["медулла"]), dtype=np.float64)
    for tname, tgroup in targets.items():
        e = pair_edges("медулла", tname)
        if len(e):
            np.add.at(w_bb, (tgroup[e.pout.to_numpy()], e.pin.to_numpy()), e.w.to_numpy())
    W_bB = normalize(w_bb)
    W_t4 = assignment("T4/T5", t45_group)
    W_lob = assignment("лобула", lob_group)
    W_c = assignment("центральный комплекс", cx_group)
    W_mb = assignment("грибовидное тело", mb_group)

    _log("4/6 T4/T5 → LPLC2 и лобула")
    lp_group = group_of(pos_map["LPLC2"], 0, 64)
    g_t4lp = pair_edges("T4/T5", "LPLC2")
    w_bc = np.zeros((64, MODEL_SIZES["T4/T5"]), dtype=np.float64)
    if len(g_t4lp):
        np.add.at(w_bc, (lp_group[g_t4lp.pout.to_numpy()], g_t4lp.pin.to_numpy()), g_t4lp.w.to_numpy())
    W_bC = normalize(w_bc)
    W_lp = assignment("LPLC2", lp_group, width=64)

    g_t4lob = pair_edges("T4/T5", "лобула")
    w_lob2 = np.zeros((MODEL_SIZES["лобула"], 64), dtype=np.float64)
    if len(g_t4lob):
        np.add.at(w_lob2, (g_t4lob.pout.to_numpy(), t45_group[g_t4lob.pin.to_numpy()]), g_t4lob.w.to_numpy())
    W_lob2 = normalize(w_lob2)

    _log("5/6 ламина: ретинотопическая привязка 8×8 по координатам сомы")
    # координаты сомы — анатомические; берём две самые «пространственные» оси
    # (для ламины это плоскость слоя) и квантуем по РАНГАМ — сетка заполняется
    # равномерно, порядок вдоль оси (ретинотопия) сохраняется
    real_pos_all = ann[ann.cell_sub_class.eq("L1-5")].dropna(subset=["pos_x", "pos_y", "pos_z"])
    axes = np.vstack([
        real_pos_all.pos_x.to_numpy(),
        real_pos_all.pos_y.to_numpy(),
        real_pos_all.pos_z.to_numpy(),
    ])
    spread = axes.std(axis=1)
    axes = axes[np.argsort(spread)[::-1][:2]]  # две оси с наибольшим разбросом
    rank8 = np.zeros((2, axes.shape[1]), dtype=np.int64)
    for a in range(2):
        order = np.argsort(np.argsort(axes[a], kind="stable"))
        rank8[a] = np.clip(order / max(len(axes[a]) - 1, 1) * 8, 0, 7).astype(np.int64)
    rp = real_pos_all.set_index("root_id")
    ranks = pd.DataFrame({"u": rank8[0], "v": rank8[1]}, index=real_pos_all.root_id)
    ru = ranks.reindex(pos_map["ламина"]).u.fillna(0).astype(np.int64).to_numpy()
    rv = ranks.reindex(pos_map["ламина"]).v.fillna(0).astype(np.int64).to_numpy()
    chin = con[con.post_pt_root_id.isin(pops["ламина"])].copy()
    chin["p"] = chin.post_pt_root_id.map(pos_index["ламина"]).astype("Int64")
    chin = chin.dropna(subset=["p"]).groupby("p").w.sum()
    w_lam = np.zeros((MODEL_SIZES["ламина"], 64), dtype=np.float64)
    indeg = np.zeros(MODEL_SIZES["ламина"], dtype=np.float64)
    indeg[chin.index.astype(np.int64).to_numpy()] = chin.to_numpy()
    np.add.at(w_lam, (np.arange(MODEL_SIZES["ламина"]), ru * 8 + rv), indeg)
    W_lam = normalize_median(w_lam, scale=0.6)

    _log("6/6 сборка circuit_v1.npz")
    blocks = {
        "W_lam": W_lam, "W_bA": W_bA, "W_med": W_med, "W_bB": W_bB,
        "W_t4": W_t4, "W_bC": W_bC, "W_lob": W_lob, "W_lob2": W_lob2,
        "W_lp": W_lp, "W_c": W_c, "W_mb": W_mb,
    }
    h = hashlib.sha256()
    for bname in blocks:
        h.update(np.ascontiguousarray(blocks[bname].astype(np.float16)).tobytes())
    n_syn_real = float(sum(p["synapses"] for p in meta_pairs.values()))
    meta = {
        "source": SOURCE,
        "license": LICENSE,
        "data_version": DATA_VERSION,
        "basis": BASIS,
        "regions": [[n, MODEL_SIZES[n]] for n in REGION_ORDER],
        "n_synapses": n_syn_real,
        "pairs": meta_pairs,
        "populations": {n: int(len(pops[n])) for n in REGION_ORDER},
        "note": "веса = реальные синаптические счётчики FlyWire (знак по медиатору), "
                "агрегированные на детерминированные группы базиса; позиции модели "
                "заполнены реальными нейронами (при нехватке — циклическое расширение)",
    }
    np.savez_compressed(
        OUT_NPZ,
        **{k: v.astype(np.float16) for k, v in blocks.items()},
        meta=np.array(json.dumps(meta, ensure_ascii=False)),
    )
    OUT_META.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    _log(f"готово: {OUT_NPZ.name} ({OUT_NPZ.stat().st_size / 1e6:.1f} МБ), "
         f"реальных синапсов в проводке: {n_syn_real:,.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
