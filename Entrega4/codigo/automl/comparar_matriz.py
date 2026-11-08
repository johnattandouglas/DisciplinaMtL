"""Compara o F1 (e o SERA) dos 42 pares re-rodados com a matriz original (meta_ir/base/F1_score.csv e SERA.csv).

Para cada limiar de evento (1,0 e 0,8) mostra: correlação de Pearson e de Spearman em todas as células (dataset x par);
Spearman mediano entre os 42 pares dentro de cada dataset; fração de datasets em que o melhor par re-rodado é o melhor da
matriz (e em que ele está entre os 3 melhores dela); e a correlação do F1 médio por par (o que define a recomendação constante).

Uso: automl/envs/pares/Scripts/python.exe automl/comparar_matriz.py [--protocolo holdout|rkf]
Saída: impressa e em automl/resultados/comparacao_matriz_<protocolo>.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comum as C  # noqa: E402


def carregar(protocolo: str) -> pd.DataFrame:
    arqs = sorted((C.RESULTADOS_DIR / "pares" / protocolo).glob("*.csv"))
    if not arqs:
        sys.exit(f"nenhum resultado em resultados/pares/{protocolo}")
    return pd.concat([pd.read_csv(a) for a in arqs], ignore_index=True)


def comparar(novo: pd.DataFrame, orig: pd.DataFrame, nome: str, maior_melhor: bool = True) -> list[str]:
    ds = [d for d in novo.index if d in orig.index]
    a = novo.loc[ds, C.PARES].to_numpy(float)
    b = orig.loc[ds, C.PARES].to_numpy(float)
    ok = np.isfinite(a) & np.isfinite(b)
    L = [f"[{nome}] {len(ds)} datasets, {ok.sum()} células"]
    L.append(f"  Pearson (todas as células) = {pearsonr(a[ok], b[ok])[0]:.3f}; Spearman = {spearmanr(a[ok], b[ok])[0]:.3f}")
    rho = []
    for i in range(len(ds)):
        m = ok[i]
        if m.sum() > 2 and np.std(a[i, m]) > 0 and np.std(b[i, m]) > 0:
            rho.append(spearmanr(a[i, m], b[i, m])[0])
    if rho:
        L.append(f"  Spearman dentro do dataset (42 pares): mediana = {np.median(rho):.3f}, "
                 f"quartis = {np.quantile(rho, 0.25):.3f} a {np.quantile(rho, 0.75):.3f} ({len(rho)} datasets sem empate total)")
    sinal = 1 if maior_melhor else -1
    best_a = np.nanargmax(sinal * a, axis=1)
    rank_b = np.argsort(np.argsort(-sinal * np.nan_to_num(b, nan=-sinal * np.inf), axis=1), axis=1)
    top1 = np.mean([np.isclose(b[i, best_a[i]], np.nanmax(sinal * b[i]) * sinal) for i in range(len(ds))])
    top3 = np.mean([rank_b[i, best_a[i]] < 3 for i in range(len(ds))])
    L.append(f"  melhor par re-rodado = melhor da matriz em {top1:.0%} dos datasets; entre os 3 melhores em {top3:.0%}")
    ma, mb = np.nanmean(a, axis=0), np.nanmean(b, axis=0)
    L.append(f"  média por par (42 valores): Pearson = {pearsonr(ma, mb)[0]:.3f}, Spearman = {spearmanr(ma, mb)[0]:.3f}; "
             f"melhor par médio: re-rodado {C.PARES[int(np.argmax(sinal * ma))]}, matriz {C.PARES[int(np.argmax(sinal * mb))]}")
    L.append(f"  média geral: re-rodado {np.nanmean(a):.4f}, matriz {np.nanmean(b):.4f}")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocolo", choices=["holdout", "rkf"], default="holdout")
    a = ap.parse_args()
    df = carregar(a.protocolo)
    media = df.groupby(["dataset", "par"])[["f1_thr1", "f1_thr08", "sera"]].mean()
    f1o = pd.read_csv(C.F1_ORIG).set_index("index")
    serao = pd.read_csv(C.SERA_ORIG).set_index("index")
    L = [f"protocolo {a.protocolo}: {df.dataset.nunique()} datasets, {df.particao.nunique()} partição(ões) por dataset"]
    for col in ["f1_thr1", "f1_thr08"]:
        L += comparar(media[col].unstack()[C.PARES], f1o, f"F1, limiar de evento {col[6:]}")
    L += comparar(media["sera"].unstack()[C.PARES], serao, "SERA (menor = melhor)", maior_melhor=False)
    fal = df.groupby("estrategia").bal_falhou.mean()
    L.append("balanceamento que falhou (fração das partições): " + ", ".join(f"{k} {v:.0%}" for k, v in fal.items()))
    txt = "\n".join(L)
    print(txt)
    (C.RESULTADOS_DIR / f"comparacao_matriz_{a.protocolo}.txt").write_text(txt + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
