"""Carregamento da meta-task do paper Dataset2Vec (Jomaa et al.): warm-start de HPO.

Dados (mesmos do pacote `ismlldataset` e do repositório hadijomaa/dataset2vec):
  * 120 datasets de classificação do UCI (pré-processados), com features/rótulos;
  * "UCI surrogate dataset": 3456 configurações de rede feed-forward (grid completo) avaliadas
    em cada dataset -> acurácia de treino/validação/teste na época 50;
  * 5 splits train/valid/test (80/16/24 datasets) usados para treinar/avaliar o D2V;
  * meta-features MF1 (Wistuba et al., 2016), MF2 (Feurer et al., 2015) e D2V (por split), dos autores.

ATENÇÃO: cada zip traz o grid de 3456 configs em uma ORDEM DIFERENTE por dataset. Aqui as linhas são
canonizadas (ordenadas pelas colunas de config) para que o índice `i` designe a mesma configuração em
todos os datasets — requisito para transferir configs de um dataset vizinho para o dataset alvo.
Alguns zips também trazem linhas REPETIDAS da mesma config (re-execuções com resultados diferentes; ex.:
acute-inflammation tem 3463 linhas): fica a primeira ocorrência no arquivo, para sobrar 3456 configs por dataset.

Download e leitura são separados: `download_all()` (chamado por `01_baixar_meta_task.py`) grava tudo em `data/`;
`load_meta_task()` e `load_reference_metafeatures()` só leem esse cache e falham com instrução clara se faltar.
"""
from __future__ import annotations

import io
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

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
_SPLITS_CSV = cfg.DATA_DIR / "splits.csv"


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

    # Alguns zips (ex.: acute-inflammation: 3463 linhas) trazem re-execuções da mesma config com respostas
    # diferentes; mantém a PRIMEIRA ocorrência (ordem das linhas no arquivo) -> exatamente 1 linha por config.
    n_rows = len(meta)
    meta = meta.drop_duplicates(cfg.CONFIG_COLS, keep="first")
    # Canoniza a ordem do grid (o zip entrega as 3456 configs embaralhadas, em ordem distinta por dataset)
    meta = meta.sort_values(cfg.CONFIG_COLS, kind="stable").reset_index(drop=True)
    _, y = np.unique(y_raw, return_inverse=True)   # rótulos -> 0..C-1
    out = {"X": X.astype(np.float32), "y": y.astype(np.int64), "grid": meta[cfg.CONFIG_COLS],
           "n_replicas": n_rows - len(meta)}
    for p, col in resp_cols.items():
        out[p] = np.round(meta[col].to_numpy(dtype=np.float64), cfg.RESPONSE_DECIMALS).astype(np.float32)
    return out


def _fetch_one(dataset_id: int) -> tuple[int, dict]:
    return dataset_id, _parse_zip(_download(f"{cfg.ISMLL_SERVER}{dataset_id}.zip"))


def _download_meta_task(n_workers: int = 4) -> None:
    from tqdm.auto import tqdm

    cfg.make_dirs()
    ids = sorted(ISMLL_ID2NAME)
    parsed: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for dataset_id, item in tqdm(pool.map(_fetch_one, ids), total=len(ids), desc="Baixando meta-task (ISMLL)"):
            parsed[dataset_id] = item

    # Ordem alfabética por nome (mesma convenção do notebook da meta-task de IR)
    names = sorted(ISMLL_ID2NAME.values())
    id_of = {v: k for k, v in ISMLL_ID2NAME.items()}

    ref_grid = parsed[id_of[names[0]]]["grid"].astype(str).to_numpy()
    bad = [n for n in names if not np.array_equal(parsed[id_of[n]]["grid"].astype(str).to_numpy(), ref_grid)]
    if bad:
        raise RuntimeError(f"Grid de configs difere do de referência (após canonizar a ordem e descartar linhas repetidas) em: {bad}")
    for name in names:
        item = parsed[id_of[name]]
        if item["n_replicas"]:
            print(f"[AVISO] '{name}': {item['n_replicas']} linha(s) repetida(s) de config descartada(s) (mantida a primeira).")
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


def _download_splits(names: list[str]) -> None:
    """Baixa os splits oficiais do D2V (train/valid/test = 80/16/24 por split). Fallback: 5 folds aleatórios."""
    try:
        _SPLITS_CSV.write_bytes(_download(cfg.SPLITS_URL))
    except Exception as e:  # noqa: BLE001
        print(f"[AVISO] não foi possível baixar os splits oficiais ({e!r}); usando 5 folds aleatórios (seed={cfg.SEED}).")
        rng = np.random.default_rng(cfg.SEED)
        perm = list(rng.permutation(names))
        cols = {}
        for k in range(cfg.N_SPLITS):
            test = perm[k * 24:(k + 1) * 24]
            rest = [n for n in perm if n not in test]
            cols[f"test-{k}"] = pd.Series(test)
            cols[f"valid-{k}"] = pd.Series(rest[:16])
            cols[f"train-{k}"] = pd.Series(rest[16:])
        pd.DataFrame(cols).to_csv(_SPLITS_CSV)


def _load_splits(names: list[str]) -> pd.DataFrame:
    splits = pd.read_csv(_SPLITS_CSV, index_col=0)
    allowed = set(names)
    for c in splits.columns:
        bad = set(splits[c].dropna()) - allowed
        if bad:
            raise RuntimeError(f"Splits referenciam datasets desconhecidos em {c}: {sorted(bad)[:5]}")
    return splits


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
    splits: pd.DataFrame           # colunas train-k / valid-k / test-k (nomes de datasets)
    t_read_s: float = field(default=0.0, repr=False)       # custo de ler o cache (lido pelo registro de custo)

    @property
    def n_ds(self) -> int:
        return len(self.names)

    @property
    def n_cfg(self) -> int:
        return self.Y_valid.shape[1]

    @property
    def name_to_idx(self) -> dict[str, int]:
        return {n: i for i, n in enumerate(self.names)}

    def load_xy(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """(X float32 [n, d], y int64 [n]) do dataset — features como vieram do repositório do D2V."""
        with np.load(cfg.DATASETS_DIR / f"{name}.npz") as f:
            return f["X"], f["y"]

    def split_indices(self, split: int) -> dict[str, list[int]]:
        idx = self.name_to_idx
        return {role: [idx[n] for n in self.splits[f"{role}-{split}"].dropna()] for role in ("train", "valid", "test")}

    def test_split_of(self) -> np.ndarray:
        """Para cada dataset, o split k em que ele é de teste (os 5 conjuntos de teste particionam os 120)."""
        out = np.full(self.n_ds, -1, dtype=int)
        for k in range(cfg.N_SPLITS):
            for i in self.split_indices(k)["test"]:
                out[i] = k
        if (out < 0).any():
            raise RuntimeError("Há datasets que não pertencem a nenhum conjunto de teste dos splits.")
        return out

    def summary(self) -> pd.DataFrame:
        rows = []
        test_split = self.test_split_of()
        for i, n in enumerate(self.names):
            X, y = self.load_xy(n)
            counts = np.bincount(y)
            yv = self.Y_valid[i]
            rows.append({
                "dataset": n, "n_instancias": X.shape[0], "n_features": X.shape[1], "n_classes": len(counts),
                "maioria_%": 100 * counts.max() / counts.sum(),
                "acc_valid_max": yv.max(), "acc_valid_media": yv.mean(), "acc_valid_min": yv.min(),
                "n_valores_distintos": len(np.unique(yv)), "split_teste": int(test_split[i]),
            })
        return pd.DataFrame(rows)


def _zips_are_downloaded() -> bool:
    return all(p.exists() for p in (_META_NPZ, _GRID_CSV)) and \
        all((cfg.DATASETS_DIR / f"{n}.npz").exists() for n in ISMLL_ID2NAME.values())


def meta_task_is_downloaded() -> bool:
    return _zips_are_downloaded() and _SPLITS_CSV.exists()


def reference_metafeatures_are_downloaded() -> bool:
    return all((cfg.REF_MF_DIR / f"{key}.csv").exists() for key in cfg.REF_MF_URLS)


def download_all(force: bool = False, n_workers: int = 4) -> dict[str, float]:
    """Baixa para `data/` tudo que a meta-task usa (chamado por `01_baixar_meta_task.py`; o notebook só LÊ o cache).

    Partes: os 120 zips do ISMLL (datasets + respostas das 3456 configs), os splits oficiais e as meta-features
    de referência do paper (MF1, MF2, D2V por split). O que já está em `data/` é pulado, a menos que `force`.
    Retorna os segundos gastos por parte (só as que de fato foram baixadas): meta_task, splits, MF1, MF2, D2V.
    """
    cfg.make_dirs()
    t: dict[str, float] = {}

    if force or not _zips_are_downloaded():
        t0 = time.perf_counter()
        _download_meta_task(n_workers=n_workers)
        t["meta_task"] = time.perf_counter() - t0
    if force or not _SPLITS_CSV.exists():
        t0 = time.perf_counter()
        _download_splits(sorted(ISMLL_ID2NAME.values()))
        t["splits"] = time.perf_counter() - t0

    for key, url in cfg.REF_MF_URLS.items():
        path = cfg.REF_MF_DIR / f"{key}.csv"
        if force or not path.exists():
            t0 = time.perf_counter()
            path.write_bytes(_download(url))
            who = "D2V" if key.startswith("D2V_split") else key
            t[who] = t.get(who, 0.0) + time.perf_counter() - t0
    return t


def load_meta_task() -> MetaTask:
    """Lê a meta-task de `data/` (NÃO baixa nada: rode `01_baixar_meta_task.py` antes)."""
    if not meta_task_is_downloaded():
        raise FileNotFoundError(
            f"Meta-task não encontrada em {cfg.DATA_DIR}. Rode primeiro `python 01_baixar_meta_task.py` "
            "(baixa ~1 GB do servidor do ISMLL).")
    t0 = time.perf_counter()
    with np.load(_META_NPZ) as f:
        names = [str(n) for n in f["names"]]
        mt = MetaTask(
            names=names, ids=f["ids"], Y_train=f["Y_train"], Y_valid=f["Y_valid"], Y_test=f["Y_test"],
            configs=pd.read_csv(_GRID_CSV), splits=_load_splits(names),
        )
    mt.t_read_s = time.perf_counter() - t0
    return mt


# ---------------------------------------------------------------------------
# Meta-features de referência do paper (MF1, MF2, D2V) — arquivos dos próprios autores
# ---------------------------------------------------------------------------
def load_reference_metafeatures(names: list[str]) -> dict:
    """Retorna {"MF1": DataFrame, "MF2": DataFrame, "D2V": {split: DataFrame}} indexados por dataset
    (na ordem de `names`). D2V é específico de cada split: o extrator do split k nunca viu os datasets de teste-k.
    Só lê o cache de `data/` (baixado por `01_baixar_meta_task.py`)."""
    if not reference_metafeatures_are_downloaded():
        raise FileNotFoundError(
            f"Meta-features de referência (MF1/MF2/D2V) não encontradas em {cfg.REF_MF_DIR}. "
            "Rode primeiro `python 01_baixar_meta_task.py`.")
    out: dict = {"D2V": {}}
    t = {m: {"leitura": 0.0} for m in ("MF1", "MF2", "D2V")}   # segundos de leitura, para o registro de custo
    for key in cfg.REF_MF_URLS:
        path = cfg.REF_MF_DIR / f"{key}.csv"
        who = "D2V" if key.startswith("D2V_split") else key
        t0 = time.perf_counter()
        df = pd.read_csv(path, index_col=0).loc[names]
        t[who]["leitura"] += time.perf_counter() - t0
        if key.startswith("D2V_split"):
            out["D2V"][int(key.removeprefix("D2V_split"))] = df
        else:
            out[key] = df
    out["_tempos"] = t
    return out
