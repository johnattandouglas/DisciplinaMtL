"""Para cada dataset alvo, as configurações de MLP que cada abordagem da Atividade 3 escolhe (lidas de entrega3/results/curvas).

Abordagens (rótulos do report3):
  padrao        Configuração padrão (fixa)
  media         Melhor configuração média: rank 1 no rank médio dos outros 119 (1º ponto do SMBO-Independent)
  rec_k1/3/5    Recomendação por meta-aprendizagem (1 tiro): 1ª config do warm-start SMBO-Similarity (PyMFE, k vizinhos)
  <busca>_T     Incumbente após T=100 avaliações (semente 0) de: Random Search, SMBO-Cold, SMBO-Similarity k=1/3/5,
                SMBO-Independent
  oraculo       Melhor das 3.456 configs na validação do meta-dataset (D2V)
A escolha usa a matriz do D2V (leave-one-dataset-out, como na Atividade 3); a avaliação (mlp_gpu.py) é nossa.
Rodar com o Python da entrega3:  python selecionar_configs.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HPO = Path(__file__).resolve().parent
E3 = HPO.parents[1] / "entrega3"
sys.path.insert(0, str(E3))
from src.metatask_data import load_meta_task  # noqa: E402

BUSCAS = {"random_search": "Random Search", "smbo_cold": "SMBO-Cold", "smbo_ws_PyMFE_k1": "SMBO-Similarity (k=1)",
          "smbo_ws_PyMFE_k3": "SMBO-Similarity (k=3)", "smbo_ws_PyMFE_k5": "SMBO-Similarity (k=5)",
          "smbo_ws_avgrank": "SMBO-Independent"}


def main():
    meta = load_meta_task()
    L = 1 - meta.Y_valid.astype(float)
    cur = {k: np.load(E3 / "results" / "curvas" / f"{k}.npz", allow_pickle=True)
           for k in ["default", *BUSCAS]}
    for k, d in cur.items():
        assert list(d["names"]) == meta.names, k
    linhas = []
    for i, n in enumerate(meta.names):
        def add(chave, rotulo, c, seg=np.nan):
            linhas.append({"dataset": n, "abordagem": chave, "rotulo": rotulo, "config": int(c),
                           "perda_d2v": L[i, c], "seg_busca": seg})
        add("padrao", "Configuração padrão", cur["default"]["evaluated"][i, 0, 0])
        add("media", "Melhor configuração média", cur["smbo_ws_avgrank"]["inits"][i, 0, 0])
        for k in (1, 3, 5):
            add(f"rec_k{k}", f"Recomendação PyMFE (k={k})", cur[f"smbo_ws_PyMFE_k{k}"]["inits"][i, 0, 0])
        for k, rot in BUSCAS.items():
            ev = cur[k]["evaluated"][i, 0]
            ev = ev[ev >= 0]
            add(f"{k}_T", f"{rot}, T=100", ev[np.argmin(L[i, ev])], cur[k]["seconds"][i] / 5)
        add("oraculo", "Oráculo (D2V)", np.argmin(L[i]))
    df = pd.DataFrame(linhas)
    df = df.join(meta.configs, on="config")
    out = HPO / "resultados" / "configs_por_abordagem.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(out, len(df), "linhas;", df.groupby("dataset").config.nunique().sum(), "pares (dataset, config) distintos")


if __name__ == "__main__":
    main()
