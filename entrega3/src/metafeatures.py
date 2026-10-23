from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

from . import config as cfg


def _stratified_subsample(y: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    classes, inv, counts = np.unique(y, return_inverse=True, return_counts=True)
    take = np.maximum(1, np.floor(counts * (n / len(y)))).astype(int)
    idx = [rng.choice(np.flatnonzero(inv == c), size=min(t, counts[c]), replace=False) for c, t in enumerate(take)]
    out = np.concatenate(idx)
    rng.shuffle(out)
    return out


def prepare_xy(name: str, max_rows: int | None = cfg.PYMFE_MAX_ROWS, seed: int = cfg.SEED):
    """(X padronizado por dataset, y) de `data/datasets/<name>.npz`; subamostra estratificada se passar de `max_rows`."""
    with np.load(cfg.DATASETS_DIR / f"{name}.npz") as f:
        X, y = f["X"].astype(np.float64), f["y"]
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    X = np.nan_to_num((X - X.mean(axis=0)) / sd)
    if max_rows is not None and len(X) > max_rows:
        sel = _stratified_subsample(y, max_rows, np.random.default_rng(seed))
        X, y = X[sel], y[sel]
    return X, y


def _pymfe_one(name: str, seed: int):
    """Retorna (dataset, {meta-característica: valor}, erro, segundos)."""
    try:
        warnings.filterwarnings("ignore")
        from pymfe.mfe import MFE

        t0 = time.perf_counter()
        X, y = prepare_xy(name, seed=seed)
        mfe = MFE(random_state=seed)
        mfe.fit(X, y)
        names, vals = mfe.extract(suppress_warnings=True)
        return name, dict(zip(names, vals)), None, time.perf_counter() - t0
    except Exception as e:  # noqa: BLE001 - um dataset problemático não derruba o lote
        return name, None, repr(e), np.nan


def extract_pymfe(names: list[str], n_jobs: int = -2) -> tuple[pd.DataFrame, dict]:
    """Extrai o PyMFE de todos os datasets (em paralelo). Retorna (DataFrame dataset x meta-característica, tempos)."""
    from joblib import Parallel, delayed

    t0 = time.perf_counter()
    results = Parallel(n_jobs=n_jobs)(delayed(_pymfe_one)(n, cfg.SEED) for n in names)
    wall = time.perf_counter() - t0
    rows = {n: feats for n, feats, err, _ in results if feats is not None}
    failed = [(n, err) for n, feats, err, _ in results if feats is None]
    if failed:
        print(f"[PyMFE] {len(failed)} dataset(s) falharam e entram com NaN: {failed[:5]}")
    df = pd.DataFrame.from_dict(rows, orient="index").reindex(names).apply(pd.to_numeric, errors="coerce")
    df.index.name = "ds_n"
    return df, {"parede_s": wall, "serial_s": float(np.nansum([r[3] for r in results])), "falhas": len(failed)}


def load_or_extract_pymfe(names: list[str], force: bool = False, n_jobs: int = -2) -> pd.DataFrame:
    """Lê `representations/PyMFE.csv` (ou extrai e grava, se não existir ou `force`)."""
    cfg.make_dirs()
    if cfg.PYMFE_CSV.exists() and not force:
        return pd.read_csv(cfg.PYMFE_CSV, index_col=0).loc[names]
    df, tempos = extract_pymfe(names, n_jobs=n_jobs)
    df.to_csv(cfg.PYMFE_CSV)
    print(f"[PyMFE] extraído em {tempos['parede_s']:.0f} s de parede ({tempos['serial_s']:.0f} s serial)")
    return df
