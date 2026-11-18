"""O que o Auto-sklearn escolheu (Atividade 4): família do membro de maior peso por orçamento, presença e peso da MLP
nos ensembles e comparação com a melhor MLP reexecutada (oráculo observado). Saídas: resultados/escolhas_resumo.txt,
figuras/escolhas_familias.(pdf|png). Uso: python analisar_escolhas.py"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import comum  # noqa: E402

FAM = {"random_forest": "Random forest", "extra_trees": "Extra trees", "gradient_boosting": "Boosting",
       "adaboost": "Boosting", "mlp": "MLP", "dummy": "Dummy", "lda": "Linear", "qda": "Linear",
       "passive_aggressive": "Linear", "sgd": "Linear", "liblinear_svc": "Linear", "libsvm_svc": "SVM",
       "k_nearest_neighbors": "k-NN", "decision_tree": "Árvore", "bernoulli_nb": "Naive Bayes",
       "gaussian_nb": "Naive Bayes", "multinomial_nb": "Naive Bayes"}
ORDEM = ["Random forest", "Extra trees", "Boosting", "MLP", "Linear", "SVM", "k-NN", "Árvore", "Naive Bayes", "Dummy"]

mlp = pd.read_json(comum.RES / "mlp.jsonl", lines=True)
oraculo = mlp.groupby("dataset").acc_teste.max()
out = []
linhas = {}
for s, rot in (("ml", "com MtL"), ("noml", "sem MtL")):
    d = pd.read_json(comum.RES / f"automl_autosklearn_{s}.jsonl", lines=True)
    d["acc"] = pd.to_numeric(d.acc_teste, errors="coerce")
    d["fam"] = d.melhor.astype(str).str.split("+").str[0].map(FAM).fillna("Outro")
    for o, g in d.groupby("orcamento"):
        comp = g.composicao.fillna("")
        peso_mlp = comp.map(lambda c: sum(float(x.split(":")[0]) for x in c.split(";") if x and x.split(":")[1].startswith("mlp+")))
        presente = comp.str.contains(r"mlp\+")
        a = g.set_index("dataset").acc
        dif = (a - oraculo.reindex(a.index)).dropna()
        topo_mlp = g.fam.eq("MLP").values
        out.append(f"[{rot} {o}s] topo: {g.fam.value_counts().to_dict()}")
        out.append(f"   MLP presente em {presente.sum()}/{len(g)}; peso médio {peso_mlp.mean():.3f} "
                   f"(quando presente {peso_mlp[presente].mean():.3f})")
        out.append(f"   ASKL - melhor MLP: média {dif.mean():+.4f}; ASKL vence em {(dif > 0).sum()}, empata {(dif == 0).sum()}, "
                   f"perde em {(dif < 0).sum()}")
        out.append(f"   acc média quando o topo é MLP: {g.acc[topo_mlp].mean():.4f} (n={topo_mlp.sum()}); "
                   f"quando é random forest: {g.acc[g.fam.eq('Random forest').values].mean():.4f}")
        linhas[(rot, o)] = g.fam.value_counts()
txt = "\n".join(out)
print(txt)
(comum.RES / "escolhas_resumo.txt").write_text(txt, encoding="utf-8")

tab = pd.DataFrame(linhas).fillna(0).reindex([f for f in ORDEM if f in pd.DataFrame(linhas).index])
fig, ax = plt.subplots(figsize=(6.4, 3.0))
cores = plt.cm.tab10(np.arange(len(tab)))
base = np.zeros(tab.shape[1])
rot = [f"{c[0]}\n{c[1]} s" for c in tab.columns]
for (fam, row), cor in zip(tab.iterrows(), cores):
    ax.bar(range(tab.shape[1]), row.values, bottom=base, label=fam, color=cor, edgecolor="white", linewidth=0.5)
    base += row.values
ax.set_xticks(range(tab.shape[1]), rot, fontsize=7)
ax.set_ylabel("datasets (de 120)")
ax.legend(fontsize=7, ncol=1, bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(comum.HPO / "figuras" / f"escolhas_familias.{ext}", dpi=150)
