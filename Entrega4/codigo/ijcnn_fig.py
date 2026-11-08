"""Figura e testes do artigo IJCNN (só as sete representações existentes; sem o método do main).

1. Figura: valor top-1 do k-NN contra k nas três meta-tarefas: faixa das sete representações, vizinhos ao acaso, oráculo(k),
   oráculo q e constante (mesmas definições de metrica_dificuldade.py). -> ../paper_ijcnn/fig_k.pdf
2. Teste pareado que faltava no METRIC.txt (item 5): para cada dataset, o excesso de kDN (k=5) sobre o acaso quando ele é
   de ajuste (média sobre as partições em que é de treino ou validação, vizinhos entre os de ajuste) contra quando é de
   teste (vizinhos entre todos); Wilcoxon pareado por dataset, Holm nas sete representações.
3. Valores com k = 1, 11 e 30 por representação, contra o acaso (Wilcoxon, Holm nas 7). -> ijcnn_numeros.txt
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import avaliar as A  # noqa: E402
import destilacao as D  # noqa: E402
import kdn as KD  # noqa: E402
import knn_fixo as K  # noqa: E402
import topk as T  # noqa: E402

REFS = list(A.BASE_HPO)
KS = [1, 3, 5, 7, 11, 15, 21, 30, 45, 60]
NDRAW = 20
OUT = A.HERE.parent / "paper_ijcnn"
TITLE = {"hpo": "HPO (ADTM, lower is better)", "pipes": "PIPES (accuracy)", "ir": "IR (F1)"}


def curves(task):
    loader, higher, _, _ = T.TASKS[task]
    names, Y, reps = loader()
    Y = np.asarray(Y, dtype=float)
    Ys = Y if task == "ir" else K.rownorm(Y)
    base = Y if higher else 1.0 - Y                            # métrica do paper (HPO: ADTM)
    G = T.transfer_topk(Y, 1)                                  # maior = melhor fonte
    n = len(names)
    Q, lab = KD.labels(task, names)
    allp = np.arange(n)
    oth = [np.delete(allp, i) for i in range(n)]
    rng = np.random.default_rng(0)
    val = {"constant": np.array([K.value_of_score(Ys[oth[i]].mean(0), base[i]) for i in range(n)])}
    val["random"] = np.array([[np.mean([K.value_of_score(Ys[j], base[i]) for j in oth[i]]) if k == 1 else
                               np.mean([K.value_of_score(Ys[rng.choice(oth[i], k, replace=False)].mean(0), base[i])
                                        for _ in range(NDRAW)]) for i in range(n)] for k in KS])
    val["oracle"] = np.array([[K.value_of_score(Ys[oth[i][np.argsort(-G[i, oth[i]], kind="stable")[:k]]].mean(0), base[i])
                               for i in range(n)] for k in KS])
    spaces = {r: reps[r][0] for r in REFS}
    spaces["oracle q"] = lambda i: Q
    for name, mat_of in spaces.items():
        cache, v = {}, np.empty((len(KS), n))
        for i in range(n):
            M = mat_of(i)
            if id(M) not in cache:
                cache[id(M)] = Q if name == "oracle q" else T.minmax(np.asarray(M, dtype=float))
            o = KD.neighbors(cache[id(M)], i, allp)
            for r, k in enumerate(KS):
                v[r, i] = K.value_of_score(Ys[o[:k]].mean(0), base[i])
        val[name] = v
    return val, higher, (reps, lab, names)


def kdn_train_test(task, reps, lab, names):
    n = len(names)
    allp = np.arange(n)
    dis = (lab[:, None, :] != lab[None, :, :]).mean(2)
    np.fill_diagonal(dis, np.nan)
    rk = np.nanmean(dis, 1)
    _, _, _, _, part = D.task_data(task)
    part = np.asarray(part)
    res = {}
    for r in REFS:
        tr_sum, tr_cnt, te = np.zeros(n), np.zeros(n), np.full(n, np.nan)
        for f in sorted(set(part.tolist())):
            tei, tri = np.flatnonzero(part == f), np.flatnonzero(part != f)
            Z = T.minmax(np.asarray(reps[r][0](int(tei[0])), dtype=float))
            for i in tri:
                tr_sum[i] += KD.kdn_from_order(lab, i, KD.neighbors(Z, i, tri), 5) - np.nanmean(dis[i, tri])
                tr_cnt[i] += 1
            for i in tei:
                te[i] = KD.kdn_from_order(lab, i, KD.neighbors(Z, i, allp), 5) - rk[i]
        res[r] = (tr_sum / tr_cnt, te)
    return res


def main():
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "font.family": "serif"})
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.1))
    L = [__doc__.strip().splitlines()[0], "", "k: " + " ".join(map(str, KS))]
    for ax, task in zip(axes, ("hpo", "pipes", "ir")):
        val, higher, (reps, lab, names) = curves(task)
        s = 1 if higher else -1
        L.append(f"== {task.upper()} constante {val['constant'].mean():.4f}")
        for name in ["oracle", "oracle q", "random"] + REFS:
            L.append(f"  {name:9s} " + " ".join(f"{x:.4f}" for x in val[name].mean(1)))
        for k in (1, 11, 30):
            r = KS.index(k)
            ds = [s * (val[x][r] - val["random"][r]) for x in REFS]
            hp = multipletests([wilcoxon(d).pvalue for d in ds], method="holm")[1]
            L.append(f"  k={k} contra o acaso (Holm 7): " + " | ".join(f"{x} {d.mean():+.4f} ({h:.2g})" for x, d, h in zip(REFS, ds, hp)))
            ds = [s * (val[x][r] - val["constant"]) for x in REFS]
            hp = multipletests([wilcoxon(d).pvalue for d in ds], method="holm")[1]
            L.append(f"  k={k} contra a constante (Holm 7): " + " | ".join(f"{x} {d.mean():+.4f} ({h:.2g})" for x, d, h in zip(REFS, ds, hp)))
        tt = kdn_train_test(task, reps, lab, names)
        ds = [a - b for a, b in tt.values()]                   # negativo: mais organizado no ajuste que no teste
        hp = multipletests([wilcoxon(d).pvalue for d in ds], method="holm")[1]
        L.append("  kDN excesso ajuste - teste (Wilcoxon por dataset, Holm 7): " + " | ".join(
            f"{x} {np.mean(a):+.3f}/{np.mean(b):+.3f} d={d.mean():+.3f} ({h:.2g})" for x, (a, b), d, h in zip(tt, tt.values(), ds, hp)))
        R = np.array([val[x].mean(1) for x in REFS])
        ax.fill_between(KS, R.min(0), R.max(0), color="#9a9890", alpha=0.35, linewidth=0, label="seven representations (range)")
        ax.plot(KS, val["random"].mean(1), color="#444", linestyle=":", marker="s", markersize=2, linewidth=1.0, label="random neighbors")
        ax.plot(KS, val["oracle"].mean(1), color="#1f5f99", linestyle="-.", linewidth=1.0, label="oracle neighbors")
        ax.plot(KS, val["oracle q"].mean(1), color="#c0392b", marker="o", markersize=2, linewidth=1.0, label="oracle $q$")
        ax.axhline(val["constant"].mean(), color="black", linestyle="--", linewidth=0.8, label="constant")
        ax.set_xscale("log")
        ax.set_xticks([1, 3, 11, 30, 60])
        ax.set_xticklabels(["1", "3", "11", "30", "60"])
        ax.minorticks_off()
        ax.set_xlabel("number of neighbors $k$")
        ax.set_title(TITLE[task], fontsize=7)
        ax.grid(True, color="#e5e5e5", linewidth=0.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        L.append("")
    h, lab_ = axes[0].get_legend_handles_labels()
    fig.legend(h, lab_, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.07), fontsize=6.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_k.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_k.png", dpi=200, bbox_inches="tight")
    (A.HERE / "ijcnn_numeros.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
