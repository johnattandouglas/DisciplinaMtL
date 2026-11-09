"""Figuras do IJCNN (autor, 08/11): (1) distribuição das "classes" de cada meta-tarefa, mostrando a popularidade;
(2) os datasets num plano de meta-features feitas à mão (PyMFE), coloridos pelo rótulo que a CE ensina, mostrando que a
similaridade não agrupa os rótulos.

(1) Para cada configuração c, a fração de datasets em que c está entre as κ melhores (κ = 20, 10, 1), ordenada; linha
tracejada = κ/C, o valor de uma escolha uniforme. Também a fração de datasets cujo rótulo (valor dominante entre as κ
melhores) coincide com o rótulo mais frequente, por componente.
(2) PCA (2 componentes) das meta-features do PyMFE após min-max (NaN -> média da coluna); cor = rótulo do componente com
mais valores distintos entre os datasets (IR: o modelo). kDN no plano e no espaço completo vão para o texto.

Uso: python ijcnn_fig_dist.py -> DisciplinaMtL/Entrega4/figuras/fig_pop.pdf, fig_mfe.pdf e ijcnn_fig_dist.txt
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

import avaliar as A  # noqa: E402
import hipoteses as H  # noqa: E402
import kdn as KD  # noqa: E402
import topk as T  # noqa: E402

OUT = A.HERE.parent.parent / "DisciplinaMtL" / "Entrega4" / "figuras"
TITLE = {"hpo": "HPO (120 datasets, 3,456 configurations)", "pipes": "PIPES (105 datasets, 8,873 pipelines)",
         "ir": "IR (215 datasets, 42 pairs)"}
COMP = {"ir": ["model", "strategy"]}


def topk_share(Y, higher, kappa):
    Yb = Y if higher else -Y
    top = np.argsort(-Yb, axis=1, kind="stable")[:, :kappa]
    cnt = np.zeros(Y.shape[1])
    for row in top:
        cnt[row] += 1
    return cnt / len(Y)


def main():
    plt.rcParams.update({"font.size": 7, "font.family": "serif"})
    L = [__doc__.strip().splitlines()[0], ""]
    f1, ax1 = plt.subplots(1, 3, figsize=(7.1, 2.0))
    f2, ax2 = plt.subplots(1, 3, figsize=(7.1, 2.3))
    for c, task in enumerate(("hpo", "pipes", "ir")):
        loader = T.TASKS[task][0]
        names, Y, reps = loader()
        Y = np.asarray(Y, dtype=float)
        higher = task != "hpo"
        kappa = H.K[task]
        s = np.sort(topk_share(Y, higher, kappa))[::-1]
        unif = kappa / Y.shape[1]
        top50 = s[:50]
        ax = ax1[c]
        ax.bar(np.arange(1, len(top50) + 1), top50, color="#1f5f99", width=0.8)
        ax.axhline(unif, color="black", linestyle="--", linewidth=0.8)
        ax.set_yscale("log")
        ax.set_xlabel("configurations ranked by popularity")
        ax.set_title(TITLE[task], fontsize=7)
        if c == 0:
            ax.set_ylabel(r"share of datasets with $c$ in top-$\kappa$")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        Q, lab = KD.labels(task, names)
        maj = [float((lab[:, g] == np.bincount(lab[:, g]).argmax()).mean()) for g in range(lab.shape[1])]
        L.append(f"== {task.upper()} κ={kappa} C={Y.shape[1]}: uniforme {unif:.4f}; mais popular {s[0]:.3f} "
                 f"({s[0] / unif:.0f}x); top-5 {', '.join(f'{x:.3f}' for x in s[:5])}; "
                 f"configurações em nenhum top-κ: {(s == 0).mean():.3f}")
        L.append(f"   rótulo majoritário por componente (fração de datasets): {', '.join(f'{x:.2f}' for x in maj)}")
        # (2) PyMFE
        M = np.asarray(reps["PyMFE"][0](0), dtype=float)
        M = np.where(np.isfinite(M), M, np.nan)
        cm = np.nanmean(M, 0)
        M = np.where(np.isnan(M), cm, M)
        M = M[:, np.nanstd(M, 0) > 0]
        Z = T.minmax(M)
        P = PCA(2, random_state=0).fit_transform(Z)
        g = int(np.argmax([len(np.unique(lab[:, j])) for j in range(lab.shape[1])])) if task != "ir" else 0
        y = lab[:, g]
        vals, cnts = np.unique(y, return_counts=True)
        order = vals[np.argsort(-cnts)]
        cmap = plt.get_cmap("tab10")
        ax = ax2[c]
        for r, v in enumerate(order):
            m = y == v
            ax.scatter(P[m, 0], P[m, 1], s=6, color=cmap(r % 10), alpha=0.8, linewidths=0,
                       label=f"value {r + 1}" if task != "ir" else None)
        ax.set_title(TITLE[task].split(" (")[0] + (" (color: best model)" if task == "ir" else f" (color: best value of component {g + 1})"), fontsize=7)
        ax.set_xlabel("PC 1 of PyMFE meta-features")
        if c == 0:
            ax.set_ylabel("PC 2")
        ax.set_xticks([]); ax.set_yticks([])
        allp = np.arange(len(y))
        kd2 = np.mean([float((y[KD.neighbors(P, i, allp)[:5]] != y[i]).mean()) for i in allp])
        kdf = np.mean([float((y[KD.neighbors(Z, i, allp)[:5]] != y[i]).mean()) for i in allp])
        rnd = np.mean([float((y[allp != i] != y[i]).mean()) for i in allp])
        var = PCA(2).fit(Z).explained_variance_ratio_.sum()
        L.append(f"   PCA do PyMFE: variância explicada {var:.2f}; componente colorido {g + 1} com {len(vals)} valores "
                 f"(maior classe {cnts.max() / len(y):.2f}); kDN(5) no plano {kd2:.3f}, no espaço completo {kdf:.3f}, acaso {rnd:.3f}")
    for f, nome in ((f1, "fig_pop.pdf"), (f2, "fig_mfe.pdf")):
        f.tight_layout()
        f.savefig(OUT / nome, bbox_inches="tight")
        f.savefig((OUT / nome).with_suffix(".png"), dpi=160, bbox_inches="tight")
    (A.HERE / "ijcnn_fig_dist.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
