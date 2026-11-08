"""Recomendações de meta-aprendizado avaliadas no F1 re-rodado do holdout (automl/resultados/pares/holdout).

Por dataset alvo i (limiar de evento 1,0, o que reproduz a matriz):
  * constante: o par de maior F1 médio na MATRIZ ORIGINAL sobre os outros 214 datasets;
  * k-NN (k = 1 e 11): vizinhos de i entre os outros 214 no embedding da partição (fold) em que i é de teste, distância
    euclidiana após min-max (topk.reps_ir / knn_fixo.py do relatorio_neurips); escore do par = média do F1 ORIGINAL dos k
    vizinhos; empates: valor esperado sobre os empatados. A representação de cada k é a de maior F1 médio com esse k na matriz
    original (relatorio_neurips/cache/knn_fixo_ir.csv: k = 1 -> ECoL, k = 11 -> TabPFN);
  * warm-start: análogo da Prática 3 (SMBO-Similarity, k = 1) no espaço de 42 pares. Não há GP útil sobre 42 pares
    categóricos; a busca avalia, por validação interna, os B primeiros pares do ranking do vizinho mais próximo (1-NN da
    representação do k = 1) e fica com o melhor na validação. Validação = holdout 70/30 (semente 42) DENTRO do treino, F1 com
    a relevância do alvo do treino. Orçamento B = 5 (padrão, 12% dos 42) e também 3 e 10 (colunas ws_B3, ws_B10). A Prática 3
    usa T = 100 de 3456 configurações (2,9%); com 42 pares isso daria 1 avaliação, que é o próprio 1-NN;
  * oráculo: o melhor dos 42 pares re-rodados.

Etapas (dois ambientes, porque os embeddings vêm do .venv do projeto e o balanceamento do ambiente automl/envs/pares):
  1. .venv/Scripts/python.exe automl/avaliar_mtl.py --vizinhos   -> resultados/mtl/vizinhos_ir.npz (só leitura do projeto)
  2. automl/envs/pares/Scripts/python.exe automl/avaliar_mtl.py --nucleos 2  -> validação interna (cache em
     resultados/mtl/validacao/<ds>.csv), resultados/mtl/mtl_por_dataset.csv e resultados/mtl/resumo.txt
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
PROJETO = AQUI.parent
SAIDA = AQUI / "resultados" / "mtl"
VIZ = SAIDA / "vizinhos_ir.npz"
REPS = {1: "ECoL", 11: "TabPFN"}
BS = (3, 5, 10)
B_MAIN = 5


def calcular_vizinhos():
    """Roda no .venv do projeto: ordem dos vizinhos (entre os outros) de cada alvo, por representação."""
    import numpy as np
    sys.path.insert(0, str(PROJETO / "relatorio_neurips"))
    os.chdir(PROJETO / "relatorio_neurips")
    import topk as T
    names, Y, reps = T.reps_ir()
    out = {"names": np.array(names), "Y": np.asarray(Y, float)}
    for rep in set(REPS.values()):
        mat_of, cache = reps[rep][0], {}
        ordem = np.empty((len(names), len(names) - 1), dtype=int)
        for i in range(len(names)):
            M = mat_of(i)
            if id(M) not in cache:
                cache[id(M)] = T.minmax(np.asarray(M, float))
            d = ((cache[id(M)] - cache[id(M)][i]) ** 2).sum(1)
            d[i] = np.inf
            ordem[i] = np.argsort(d, kind="stable")[:-1]
        out[f"ordem_{rep}"] = ordem
    SAIDA.mkdir(parents=True, exist_ok=True)
    np.savez(VIZ, **out)
    print("vizinhos gravados:", VIZ, {k: v.shape for k, v in out.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vizinhos", action="store_true")
    ap.add_argument("--nucleos", type=int, default=2)
    a = ap.parse_args()
    if a.vizinhos:
        return calcular_vizinhos()
    os.environ.setdefault("TQDM_DISABLE", "1")
    import numpy as np
    import pandas as pd
    sys.path.insert(0, str(AQUI))
    import comum as C
    import ir_metrica as M
    import rodar_pares as RP

    z = np.load(VIZ)
    nomes = [str(n) for n in z["names"]]
    pos = {n: i for i, n in enumerate(nomes)}
    orig = pd.read_csv(C.F1_ORIG).set_index("index").loc[nomes, C.PARES].to_numpy(float)
    arqs = sorted((C.RESULTADOS_DIR / "pares" / "holdout").glob("*.csv"))
    (SAIDA / "validacao").mkdir(parents=True, exist_ok=True)

    def valor(escore, y):
        best = np.flatnonzero(escore >= escore.max() - 1e-12)
        return float(y[best].mean()), C.PARES[best[0]]

    linhas = []
    for arq in arqs:
        ds = arq.stem
        if ds not in pos:
            continue
        i = pos[ds]
        r = pd.read_csv(arq).set_index("par").loc[C.PARES]
        y = r["f1_thr1"].to_numpy(float)
        outros = np.delete(np.arange(len(nomes)), i)
        lin = {"dataset": ds}
        lin["constante"], lin["par_constante"] = valor(orig[outros].mean(0), y)
        for k, rep in REPS.items():
            viz = z[f"ordem_{rep}"][i][:k]
            lin[f"knn{k}_{rep}"], lin[f"par_knn{k}"] = valor(orig[viz].mean(0), y)
        # warm-start: ranking do 1-NN, validação interna dos B primeiros
        viz1 = z[f"ordem_{REPS[1]}"][i][0]
        ranking = [C.PARES[c] for c in np.argsort(-orig[viz1], kind="stable")[:max(BS)]]
        cache = SAIDA / "validacao" / f"{ds}.csv"
        if cache.exists():
            val = pd.read_csv(cache).set_index("par")["f1_val"]
        else:
            X, yy = C.ler_dataset(ds)
            tr, _ = C.divisoes(ds, len(yy), "holdout")[0]
            from sklearn.model_selection import train_test_split
            itr, iva = train_test_split(np.arange(len(tr)), test_size=0.3, random_state=C.SEMENTE, shuffle=True)
            rel_tr = M.relevancia(yy[tr])
            val = pd.Series(RP.validar_pares(X[tr][itr], yy[tr][itr], X[tr][iva], yy[tr][iva], ranking, rel_tr,
                                             a.nucleos), name="f1_val")
            val.rename_axis("par").reset_index().to_csv(cache, index=False)
        for B in BS:
            cand = ranking[:B]
            esc = max(cand, key=lambda p: (val[p], -cand.index(p)))
            lin[f"ws_B{B}"] = float(r.loc[esc, "f1_thr1"])
            if B == B_MAIN:
                lin["par_ws"] = esc
        lin["oraculo"] = float(np.nanmax(y))
        lin["par_oraculo"] = C.PARES[int(np.nanargmax(y))]
        lin["t_42pares_s"] = float(r["t_par_s"].sum())
        lin["t_par_recomendado_knn1_s"] = float(r.loc[lin["par_knn1"], "t_par_s"])
        linhas.append(lin)
    df = pd.DataFrame(linhas)
    df.to_csv(SAIDA / "mtl_por_dataset.csv", index=False)
    cols = ["constante", f"knn1_{REPS[1]}", f"knn11_{REPS[11]}"] + [f"ws_B{B}" for B in BS] + ["oraculo"]
    L = [f"{len(df)} datasets (F1 re-rodado no holdout, limiar 1,0)"]
    for c in cols:
        L.append(f"  {c:16s} média {df[c].mean():.4f}  mediana {df[c].median():.4f}")
    from scipy.stats import wilcoxon
    if len(df) >= 10:
        for c in cols[1:-1]:
            d = df[c] - df["constante"]
            p = wilcoxon(d).pvalue if (d != 0).any() else 1.0
            L.append(f"  {c} vs constante: dif. média {d.mean():+.4f}, V/E/D {(d > 0).sum()}/{(d == 0).sum()}/{(d < 0).sum()}, "
                     f"Wilcoxon p = {p:.3g} (sem correção)")
    else:
        L.append("  testes omitidos: menos de 10 datasets")
    txt = "\n".join(L)
    print(txt)
    (SAIDA / "resumo.txt").write_text(txt + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
