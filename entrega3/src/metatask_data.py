"""Carregamento da meta-task de warm-start de HPO do paper Dataset2Vec (Jomaa et al., 2021).

Dados (mesmos do pacote `ismlldataset` e do repositório hadijomaa/dataset2vec):
  * 120 datasets de classificação do UCI (pré-processados por Fernández-Delgado et al.), com features/rótulos;
  * "UCI surrogate dataset": 3456 configurações de rede feed-forward (grid completo) avaliadas em cada dataset
    -> acurácia de treino/validação/teste na época 50.

ATENÇÃO: cada zip traz o grid de 3456 configs em uma ORDEM DIFERENTE por dataset. Aqui as linhas são canonizadas
(ordenadas pelas colunas de config) para que o índice `i` designe a mesma configuração em todos os datasets — requisito
para transferir configs de um dataset para outro. Alguns zips trazem linhas REPETIDAS da mesma config (ex.:
acute-inflammation tem 3463 linhas). Nesses casos considerei a primeira ocorrência, para sobrar 3456 configs por dataset.
"""
from __future__ import annotations

import io
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as cfg

# id do servidor ISMLL -> nome do dataset (cópia de `ismlldataset/zipfiles2.py::_map`)
ISMLL_ID2NAME = {
    0: "oocytes_trisopterus_states_5b", 1: "pittsburg-bridges-SPAN", 2: "statlog-heart",
    3: "molec-biol-promoter", 4: "yeast", 5: "monks-3", 6: "titanic", 7: "synthetic-control",
    8: "ionosphere", 9: "pittsburg-bridges-T-OR-D", 10: "breast-tissue", 11: "ecoli",
    12: "oocytes_merluccius_nucleus_4d", 13: "plant-margin", 14: "conn-bench-vowel-deterding",
    15: "optical", 16: "magic", 17: "miniboone", 18: "heart-switzerland", 19: "breast-cancer-wisc-prog",
    20: "ringnorm", 21: "lung-cancer", 22: "steel-plates", 23: "plant-shape", 24: "echocardiogram",
    25: "lymphography", 26: "energy-y2", 27: "musk-1", 28: "plant-texture",
    29: "statlog-australian-credit", 30: "vertebral-column-2clases", 31: "abalone", 32: "blood",
    33: "credit-approval", 34: "molec-biol-splice", 35: "wine-quality-white", 36: "bank", 37: "car",
    38: "low-res-spect", 39: "horse-colic", 40: "hill-valley", 41: "statlog-shuttle", 42: "hayes-roth",
    43: "cardiotocography-3clases", 44: "breast-cancer-wisc", 45: "adult", 46: "glass", 47: "fertility",
    48: "mammographic", 49: "statlog-german-credit", 50: "oocytes_merluccius_states_2f",
    51: "congressional-voting", 52: "soybean", 53: "planning", 54: "pittsburg-bridges-MATERIAL",
    55: "statlog-vehicle", 56: "zoo", 57: "arrhythmia", 58: "lenses", 59: "ozone", 60: "seeds",
    61: "cylinder-bands", 62: "wine", 63: "tic-tac-toe", 64: "acute-nephritis", 65: "connect-4",
    66: "pima", 67: "statlog-image", 68: "chess-krvkp", 69: "musk-2", 70: "waveform", 71: "flags",
    72: "wall-following", 73: "pendigits", 74: "iris", 75: "cardiotocography-10clases",
    76: "statlog-landsat", 77: "twonorm", 78: "heart-cleveland", 79: "primary-tumor",
    80: "oocytes_trisopterus_nucleus_2f", 81: "post-operative", 82: "spect", 83: "acute-inflammation",
    84: "chess-krvk", 85: "dermatology", 86: "libras", 87: "mushroom", 88: "parkinsons",
    89: "waveform-noise", 90: "heart-hungarian", 91: "heart-va", 92: "audiology-std",
    93: "haberman-survival", 94: "energy-y1", 95: "page-blocks", 96: "conn-bench-sonar-mines-rocks",
    97: "semeion", 98: "hepatitis", 99: "contrac", 100: "led-display", 101: "breast-cancer-wisc-diag",
    102: "vertebral-column-3clases", 103: "ilpd-indian-liver", 104: "monks-1", 105: "image-segmentation",
    106: "pittsburg-bridges-TYPE", 107: "thyroid", 108: "nursery", 109: "wine-quality-red",
    110: "breast-cancer", 111: "letter", 112: "pittsburg-bridges-REL-L", 113: "monks-2", 114: "balloons",
    115: "spectf", 116: "balance-scale", 117: "teaching", 118: "spambase", 119: "annealing",
}

_META_NPZ = cfg.DATA_DIR / "meta_task.npz"
_GRID_CSV = cfg.DATA_DIR / "config_grid.csv"


# ---------------------------------------------------------------------------
# Download / parsing
# ---------------------------------------------------------------------------
def _download(url: str, retries: int = 3, timeout: int = 180) -> bytes:
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return resp.read()
        except Exception as e:  # noqa: BLE001 - qualquer falha de rede -> retry
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Falha ao baixar {url}: {last!r}")


def _parse_zip(data: bytes) -> dict:
    """Extrai de um zip do ISMLL: X, y e as respostas (train/valid/test acc @ época 50) com o grid."""
    resp_cols = {p: f"{p}_acc_{cfg.RESPONSE_IDX}" for p in ("trn", "vld", "tst")}
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        X = pd.read_csv(z.open("features.dat"), header=None).to_numpy(dtype=np.float64)
        y_raw = pd.read_csv(z.open("labels.dat"), header=None)[0].to_numpy()
        meta = pd.read_csv(z.open("metadataset.dat"), usecols=cfg.CONFIG_COLS + list(resp_cols.values()))

    n_rows = len(meta)
    meta = meta.drop_duplicates(cfg.CONFIG_COLS, keep="first")          # 1 linha por config
    meta = meta.sort_values(cfg.CONFIG_COLS, kind="stable").reset_index(drop=True)   # ordem canônica do grid
    _, y = np.unique(y_raw, return_inverse=True)   # rótulos -> 0..C-1
    out = {"X": X.astype(np.float32), "y": y.astype(np.int64), "grid": meta[cfg.CONFIG_COLS],
           "n_replicas": n_rows - len(meta)}
    for p, col in resp_cols.items():
        out[p] = np.round(meta[col].to_numpy(dtype=np.float64), cfg.RESPONSE_DECIMALS).astype(np.float32)
    return out


def _fetch_one(dataset_id: int) -> tuple[int, dict]:
    return dataset_id, _parse_zip(_download(f"{cfg.ISMLL_SERVER}{dataset_id}.zip"))


def download_all(n_workers: int = 4) -> None:
    """Baixa os 120 zips do servidor do ISMLL (~1 GB) e grava data/datasets/*.npz, data/meta_task.npz e o grid."""
    from tqdm.auto import tqdm

    cfg.make_dirs()
    ids = sorted(ISMLL_ID2NAME)
    parsed: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for dataset_id, item in tqdm(pool.map(_fetch_one, ids), total=len(ids), desc="Baixando meta-task (ISMLL)"):
            parsed[dataset_id] = item

    names = sorted(ISMLL_ID2NAME.values())          # ordem alfabética por nome
    id_of = {v: k for k, v in ISMLL_ID2NAME.items()}
    ref_grid = parsed[id_of[names[0]]]["grid"].astype(str).to_numpy()
    bad = [n for n in names if not np.array_equal(parsed[id_of[n]]["grid"].astype(str).to_numpy(), ref_grid)]
    if bad:
        raise RuntimeError(f"Grid de configs difere do de referência em: {bad}")
    for name in names:
        item = parsed[id_of[name]]
        if item["n_replicas"]:
            print(f"[AVISO] '{name}': {item['n_replicas']} linha(s) repetida(s) de config descartada(s).")
        np.savez_compressed(cfg.DATASETS_DIR / f"{name}.npz", X=item["X"], y=item["y"])

    parsed[id_of[names[0]]]["grid"].to_csv(_GRID_CSV, index=False)
    np.savez_compressed(
        _META_NPZ,
        names=np.array(names),
        ids=np.array([id_of[n] for n in names]),
        Y_train=np.stack([parsed[id_of[n]]["trn"] for n in names]),
        Y_valid=np.stack([parsed[id_of[n]]["vld"] for n in names]),
        Y_test=np.stack([parsed[id_of[n]]["tst"] for n in names]),
    )


# ---------------------------------------------------------------------------
# Container da meta-task
# ---------------------------------------------------------------------------
@dataclass
class MetaTask:
    names: list[str]               # datasets em ordem alfabética (índice = posição nesta lista)
    ids: np.ndarray                # id no servidor ISMLL
    Y_train: np.ndarray            # (n_ds, n_cfg) acurácia de treino @ época 50
    Y_valid: np.ndarray            # (n_ds, n_cfg) acurácia de validação @ época 50  -> resposta otimizada
    Y_test: np.ndarray             # (n_ds, n_cfg) acurácia de teste @ época 50
    configs: pd.DataFrame          # (n_cfg, 7) grid canônico

    @property
    def n_ds(self) -> int:
        return len(self.names)

    @property
    def n_cfg(self) -> int:
        return self.Y_valid.shape[1]

    def load_xy(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """(X float32 [n, d], y int64 [n]) do dataset."""
        with np.load(cfg.DATASETS_DIR / f"{name}.npz") as f:
            return f["X"], f["y"]

    def summary(self) -> pd.DataFrame:
        rows = []
        for n in self.names:
            X, y = self.load_xy(n)
            counts = np.bincount(y)
            rows.append({"dataset": n, "n_instancias": X.shape[0], "n_features": X.shape[1],
                         "n_classes": len(counts), "maioria_%": 100 * counts.max() / counts.sum()})
        return pd.DataFrame(rows)


def meta_task_is_downloaded() -> bool:
    return all(p.exists() for p in (_META_NPZ, _GRID_CSV)) and \
        all((cfg.DATASETS_DIR / f"{n}.npz").exists() for n in ISMLL_ID2NAME.values())


def load_meta_task() -> MetaTask:
    """Lê a meta-task de `data/` (NÃO baixa nada: use `download_all()` antes, se faltar)."""
    if not meta_task_is_downloaded():
        raise FileNotFoundError(f"Meta-task não encontrada em {cfg.DATA_DIR}. Rode `download_all()` (notebook 01).")
    with np.load(_META_NPZ) as f:
        return MetaTask(names=[str(n) for n in f["names"]], ids=f["ids"], Y_train=f["Y_train"],
                        Y_valid=f["Y_valid"], Y_test=f["Y_test"], configs=pd.read_csv(_GRID_CSV))
