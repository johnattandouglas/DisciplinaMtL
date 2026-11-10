"""Dados e divisão comuns da Atividade 4 (HPO). Só usa numpy, para rodar em qualquer ambiente (Windows ou WSL).

Os 120 datasets (X, y) vêm de entrega3/data/datasets/*.npz. O zip original do ISMLL não traz a divisão
treino/validação/teste usada pelos autores do Dataset2Vec (só features.dat, labels.dat, metadataset.dat e
metafeatures.dat), então a divisão é nossa: hold-out estratificado 75/25 (semente 0), igual para todos os métodos.
Treino = 75% (o AutoML e as MLPs só veem essa parte); teste = 25% (métrica: acurácia).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HPO = Path(__file__).resolve().parent
E3 = HPO.parents[1] / "entrega3"
DATASETS = E3 / "data" / "datasets"
SPLITS = HPO / "splits"
RES = HPO / "resultados"
LOGS = HPO / "logs"
SEMENTE = 0
FRAC_TESTE = 0.25


def nomes() -> list[str]:
    return sorted(p.stem for p in DATASETS.glob("*.npz"))


def _divisao(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estratificada por classe; classes com 1 exemplo ficam no treino. Garante >= 1 exemplo de teste."""
    rng = np.random.default_rng(SEMENTE)
    te = []
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        n = int(round(FRAC_TESTE * len(idx))) if len(idx) > 1 else 0
        te.extend(idx[:n])
    te = np.array(sorted(te), dtype=np.int64)
    if len(te) == 0:
        te = np.array([rng.integers(len(y))])
    tr = np.setdiff1d(np.arange(len(y)), te)
    return tr, te


def carregar(nome: str):
    """(Xtr, ytr, Xte, yte); grava a divisão em splits/<nome>.npz na primeira chamada."""
    with np.load(DATASETS / f"{nome}.npz") as f:
        X, y = f["X"].astype(np.float32), f["y"].astype(np.int64)
    arq = SPLITS / f"{nome}.npz"
    if arq.exists():
        with np.load(arq) as f:
            tr, te = f["treino"], f["teste"]
    else:
        SPLITS.mkdir(parents=True, exist_ok=True)
        tr, te = _divisao(y)
        np.savez_compressed(arq, treino=tr, teste=te)
    return X[tr], y[tr], X[te], y[te]


def ja_feito(arq: Path, chave: str) -> bool:
    if not arq.exists():
        return False
    with open(arq, encoding="utf-8") as f:
        return any(json.loads(l).get("chave") == chave for l in f if l.strip())


def gravar(arq: Path, reg: dict) -> None:
    arq.parent.mkdir(parents=True, exist_ok=True)
    with open(arq, "a", encoding="utf-8") as f:
        f.write(json.dumps(reg, ensure_ascii=False) + "\n")
