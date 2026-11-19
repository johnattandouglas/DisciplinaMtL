"""Custo online da meta-aprendizagem que faltava (Atividade 4): extrair o PyMFE do dataset alvo, exatamente como na
Atividade 3 (entrega3/src/metafeatures.py, MFE padrão, subamostra estratificada), em série, e consultar o meta-modelo
(1-NN no espaço PyMFE padronizado). Grava resultados/custo_mtl_online.csv e imprime o resumo.
Uso (com o Python da entrega3): python medir_custo_mtl.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HPO = Path(__file__).resolve().parent
E3 = HPO.parents[1] / "entrega3"
sys.path.insert(0, str(E3))
from src import metafeatures as M  # noqa: E402

import comum  # noqa: E402

nomes = comum.nomes()
X = pd.read_csv(E3 / "representations" / "PyMFE.csv", index_col=0).loc[nomes]
Z = ((X - X.mean()) / X.std().replace(0, 1)).fillna(0).values
linhas = []
for i, n in enumerate(nomes):
    _, feats, err, seg = M._pymfe_one(n, 0)
    t0 = time.perf_counter()
    d = np.linalg.norm(np.delete(Z, i, axis=0) - Z[i], axis=1)   # leave-one-dataset-out
    int(np.argmin(d))
    linhas.append({"dataset": n, "seg_pymfe": seg, "seg_consulta": time.perf_counter() - t0, "erro": err})
    print(f"{i + 1:3d}/120 {n:35s} {seg:8.2f} s", flush=True)
df = pd.DataFrame(linhas)
df.to_csv(comum.RES / "custo_mtl_online.csv", index=False)
s = df.seg_pymfe
print(f"\nPyMFE por dataset: mediana {s.median():.2f} s, média {s.mean():.2f} s, máximo {s.max():.1f} s "
      f"({df.loc[s.idxmax(), 'dataset']}), total {s.sum() / 60:.1f} min, falhas {df.erro.notna().sum()}")
print(f"Consulta 1-NN: mediana {df.seg_consulta.median() * 1e3:.3f} ms")
