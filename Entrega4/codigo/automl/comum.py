"""Partes comuns aos scripts de automl/: lista dos 215 datasets da IR, leitura de um dataset, divisões treino/teste e medição
de recursos (tempo de parede, tempo de CPU, pico de memória).

Divisões (gravadas uma vez em automl/divisoes/ e lidas por todos os scripts, para que pares, recomendações e AutoML usem os
MESMOS índices):
  * holdout (padrão): train_test_split(test_size=0.3, random_state=42, shuffle=True), o mesmo do executor do Meta-IR
    (github.com/JusciAvelino/MetaIR-Framework, meta_ir/executor.py), que é como os autores avaliam uma recomendação num
    dataset novo. Um único treino/teste por dataset.
  * rkf: as 20 partições do protocolo que gerou a matriz (RepeatedKFold(n_splits=10, n_repeats=2, random_state=42) do
    notebook Meta_IR.ipynb). Só para validar a reprodução da matriz; caro demais para o AutoML.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent                    # .../DoutoradoIR/automl
PROJETO = AQUI.parent
DATASETS_DIR = PROJETO / "src" / "datasets"
META_REF = PROJETO / "src" / "META-IR_ECoL.csv"           # define os 215 datasets da IR (mesma regra de Our-CE-Align/ocl/ir.py)
F1_ORIG = PROJETO / "meta_ir" / "base" / "F1_score.csv"
SERA_ORIG = PROJETO / "meta_ir" / "base" / "SERA.csv"
IR_IGNORE = {"meta", "maximal_torque"}
DIVISOES_DIR = AQUI / "divisoes"
RESULTADOS_DIR = AQUI / "resultados"
SEMENTE = 42

MODELOS = ["BG", "DT", "ML", "RF", "SV", "XG"]             # códigos das colunas da matriz (ML = MLP, SV = SVR)
ESTRATEGIAS = ["NO", "GN", "RO", "RU", "SG", "SM", "WC"]   # SM = SmoteR, SG = SMOGN, WC = WERCS, NO = sem balanceamento
PARES = [f"{m}.{s}" for s in ESTRATEGIAS for m in MODELOS]  # ordem das colunas de F1_score.csv


def nomes_ir() -> list[str]:
    meta = pd.read_csv(META_REF, usecols=["ds_n"])
    nomes = sorted(set(meta["ds_n"].astype(str)) - IR_IGNORE)
    return nomes


def ler_dataset(nome: str) -> tuple[np.ndarray, np.ndarray]:
    """Como no notebook original: alvo = primeira coluna, atributos = o resto, sem pré-processamento."""
    df = pd.read_csv(DATASETS_DIR / f"{nome}.csv")
    y = df.iloc[:, 0].to_numpy(dtype=float)
    X = df.iloc[:, 1:].to_numpy(dtype=float)
    return X, y


# ---------------------------------------------------------------------------
# Divisões
# ---------------------------------------------------------------------------
def _gerar(n: int, protocolo: str) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import RepeatedKFold, train_test_split
    idx = np.arange(n)
    if protocolo == "holdout":
        tr, te = train_test_split(idx, test_size=0.3, random_state=SEMENTE, shuffle=True)
        return [(np.sort(tr), np.sort(te))]
    if protocolo == "rkf":
        rkf = RepeatedKFold(n_splits=10, n_repeats=2, random_state=SEMENTE)
        return [(tr, te) for tr, te in rkf.split(idx)]
    raise ValueError(protocolo)


def divisoes(nome: str, n: int, protocolo: str = "holdout") -> list[tuple[np.ndarray, np.ndarray]]:
    """Lê as partições gravadas; grava se ainda não existem (um .npz por protocolo e dataset, nunca sobrescrito)."""
    pasta = DIVISOES_DIR / protocolo
    arq = pasta / f"{nome}.npz"
    if arq.exists():
        z = np.load(arq)
        k = len([c for c in z.files if c.startswith("treino")])
        parts = [(z[f"treino{i}"], z[f"teste{i}"]) for i in range(k)]
        if sum(len(t) for t in parts[0]) != n:
            raise RuntimeError(f"{arq}: divisão gravada para outro tamanho de dataset")
        return parts
    parts = _gerar(n, protocolo)
    pasta.mkdir(parents=True, exist_ok=True)
    tmp = arq.with_name(arq.stem + f".tmp{os.getpid()}.npz")
    np.savez(tmp, **{f"treino{i}": tr for i, (tr, _) in enumerate(parts)},
             **{f"teste{i}": te for i, (_, te) in enumerate(parts)})
    if arq.exists():                                       # outro processo gravou antes: fica a primeira versão
        tmp.unlink()
    else:
        os.replace(tmp, arq)
    return divisoes(nome, n, protocolo)


def ordem_fila(nomes: list[str]) -> list[str]:
    """Maior dataset primeiro (tamanho do CSV): com workers que pegam o próximo livre, isso equilibra a carga (LPT)."""
    return sorted(nomes, key=lambda n: -os.path.getsize(DATASETS_DIR / f"{n}.csv"))


def reivindicar(pasta: Path, nome: str) -> bool:
    """Trava atômica (mkdir) para vários workers dividirem a mesma lista sem repetir dataset. A fila apaga as travas
    (pasta/.travas) no começo de cada etapa, então uma execução interrompida é retomada sem problema."""
    travas = pasta / ".travas"
    travas.mkdir(parents=True, exist_ok=True)
    try:
        os.mkdir(travas / nome)
        return True
    except FileExistsError:
        return False


# ---------------------------------------------------------------------------
# Recursos
# ---------------------------------------------------------------------------
def limitar_threads(n: int) -> None:
    """Fixa o número de threads das bibliotecas numéricas (tem de rodar antes de importar numpy/sklearn no processo filho;
    a fila também exporta essas variáveis)."""
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = str(n)


class Medidor:
    """Mede tempo de parede, tempo de CPU (processo + filhos) e pico de memória (RSS do processo + filhos, amostrado a cada
    0,2 s), além do número máximo de threads e de processos filhos vistos. Uso: `with Medidor() as m: ...; m.resumo()`."""

    def __init__(self, intervalo: float = 0.2):
        import psutil
        self.p = psutil.Process()
        self.intervalo = intervalo
        self.pico_mb = 0.0
        self.max_threads = 0
        self.max_filhos = 0
        self._parar = threading.Event()

    def _cpu(self) -> float:
        t = self.p.cpu_times()
        total = t.user + t.system
        for c in self._filhos():
            try:
                ct = c.cpu_times()
                total += ct.user + ct.system
            except Exception:
                pass
        return total

    def _filhos(self):
        try:
            return self.p.children(recursive=True)
        except Exception:
            return []

    def _amostrar(self):
        while not self._parar.is_set():
            try:
                filhos = self._filhos()
                rss = self.p.memory_info().rss
                for c in filhos:
                    try:
                        rss += c.memory_info().rss
                    except Exception:
                        pass
                self.pico_mb = max(self.pico_mb, rss / 2 ** 20)
                self.max_threads = max(self.max_threads, self.p.num_threads())
                self.max_filhos = max(self.max_filhos, len(filhos))
            except Exception:
                pass
            self._parar.wait(self.intervalo)

    def __enter__(self):
        self.cpu0 = self._cpu()
        self.t0 = time.perf_counter()
        self._th = threading.Thread(target=self._amostrar, daemon=True)
        self._th.start()
        return self

    def __exit__(self, *exc):
        self.cpu1 = self._cpu()            # antes de parar: filhos que já morreram não entram (limitação conhecida do psutil)
        self.t1 = time.perf_counter()
        self._parar.set()
        self._th.join()
        return False

    def resumo(self) -> dict:
        parede = self.t1 - self.t0
        cpu = self.cpu1 - self.cpu0
        return {"parede_s": parede, "cpu_s": cpu, "cpu_por_parede": cpu / parede if parede > 0 else np.nan,
                "ram_pico_mb": self.pico_mb, "max_threads": self.max_threads, "max_filhos": self.max_filhos}
