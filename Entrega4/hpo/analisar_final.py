"""Análises finais da Atividade 4 (HPO), com o conjunto mínimo de métodos do enunciado.

Métodos (todos avaliados na mesma partição 75/25, semente 0, acurácia de teste):
  Média      melhor configuração média (rank médio nos outros 119), padrão sem meta-características (item 1)
  Rec. k=1   recomendação PyMFE k=1 da Atividade 3, leave-one-dataset-out (item 2)
  SMBO-Sim   SMBO-Similarity k=1, T=100 (item 3)
  ASKL+/-    Auto-sklearn 0.15.0 com e sem meta-aprendizagem, em cada orçamento (item 4)
  Oráculo    melhor acurácia entre as configurações de MLP reexecutadas no dataset (item 5); é um limite inferior do
             oráculo verdadeiro, porque não reexecutamos as 3.456 configurações. Fica fora do Friedman.
Saídas: resultados/final_*.csv, resultados/final_resumo.txt, figuras/final_cd_<orc>s.(pdf|png), figuras/final_custo.(pdf|png)
Uso: python analisar_final.py
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import friedmanchisquare, rankdata, studentized_range, wilcoxon, spearmanr  # noqa: E402
import comum  # noqa: E402

RES, FIG = comum.RES, comum.HPO / "figuras"
MTL = {"media": "Média", "rec_k1": "Rec. k=1", "smbo_ws_PyMFE_k1_T": "SMBO-Sim"}
out = []
p = out.append

sel = pd.read_csv(RES / "configs_por_abordagem.csv")
mlp = pd.read_json(RES / "mlp.jsonl", lines=True)
x = sel.merge(mlp[["dataset", "config", "acc_teste", "seg_treino"]], on=["dataset", "config"])
p(f"Correlação perda da matriz D2V x perda reexecutada: Spearman {spearmanr(x.perda_d2v, 1 - x.acc_teste)[0]:.3f} "
  f"({len(x)} pares método-dataset)")
mt = x[x.abordagem.isin(MTL)].assign(metodo=lambda d: d.abordagem.map(MTL))
acc_mtl = mt.pivot_table(index="dataset", columns="metodo", values="acc_teste")
oraculo = mlp.groupby("dataset").acc_teste.max().rename("Oráculo")
custo_mtl = (mt.assign(c=mt.seg_treino + mt.seg_busca.fillna(0))
             .pivot_table(index="dataset", columns="metodo", values="c"))

au = []
for s, rot in (("ml", "ASKL+"), ("noml", "ASKL-")):
    d = pd.read_json(RES / f"automl_autosklearn_{s}.jsonl", lines=True)
    d["metodo"] = rot
    au.append(d)
au = pd.concat(au)
au["acc_teste"] = pd.to_numeric(au.acc_teste, errors="coerce")
au["custo"] = au.seg_ajuste + au.seg_predicao.fillna(0)
ov = pd.read_csv(RES / "sobreposicao_openml.csv").set_index("dataset").na_base


def q_nemenyi(k, n):
    return studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * n))


def cd_plot(rk, cd, titulo, arq):
    r = rk.sort_values()
    fig, ax = plt.subplots(figsize=(6, 0.35 * len(r) + 1.1))
    ax.hlines(range(len(r)), 1, r.values, color="0.85")
    ax.plot(r.values, range(len(r)), "o", color="C0")
    for i, v in enumerate(r.values):
        ax.text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=7)
    ax.set_yticks(range(len(r)), r.index)
    ax.invert_yaxis()
    ax.errorbar(r.values[0], -0.8, xerr=[[0], [cd]], color="C3", capsize=4)
    ax.text(r.values[0], -1.15, f"CD = {cd:.2f}", color="C3", fontsize=7)
    ax.set_xlabel("rank médio (1 = melhor)")
    ax.set_title(titulo, fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{arq}.{ext}", dpi=150)
    plt.close(fig)


linhas, custo_rows = [], []
for orc in sorted(au.orcamento.unique()):
    a = au[au.orcamento == orc].pivot_table(index="dataset", columns="metodo", values="acc_teste")
    if len(a) < 100:
        p(f"\n[{orc} s] só {len(a)} datasets; pulado")
        continue
    t = acc_mtl.join(a, how="inner").dropna()
    melhor = t.join(oraculo).max(axis=1)          # melhor observado em cada dataset (inclui o oráculo)
    rk = pd.DataFrame(rankdata(-t.values, axis=1), index=t.index, columns=t.columns).mean()
    st, pf = friedmanchisquare(*[t[c] for c in t.columns])
    cd = q_nemenyi(t.shape[1], len(t))
    cd_plot(rk, cd, f"Orçamento do Auto-sklearn {orc} s, n = {len(t)} (Friedman p = {pf:.1e})", f"final_cd_{orc}s")
    p(f"\n[{orc} s] n={len(t)} Friedman chi2={st:.1f} p={pf:.2e} CD={cd:.2f}")
    for c in t.columns:
        perda = melhor - t[c]
        linhas.append({"orcamento": orc, "metodo": c, "n": len(t), "acc_media": t[c].mean(),
                       "perda_media_p_melhor": perda.mean(), "perda_mediana": perda.median(),
                       "rank_medio": rk[c], "vitorias_entre_metodos": int((t[c] == t.max(axis=1)).sum())})
        p(f"  {c:9s} acc={t[c].mean():.4f} perda={perda.mean():.4f} rank={rk[c]:.2f}")
    p(f"  Oráculo   acc={oraculo.reindex(t.index).mean():.4f} perda={(melhor - oraculo.reindex(t.index)).mean():.4f}")
    # pares de interesse: ASKL+ contra cada braço de MtL (Wilcoxon, Holm na família de 3)
    ps = []
    for c in MTL.values():
        d = (t["ASKL+"] - t[c])
        ps.append((c, d.mean(), (d > 0).sum(), (d < 0).sum(), wilcoxon(d[d != 0]).pvalue))
    m = len(ps)
    ordem = sorted(range(m), key=lambda i: ps[i][4])
    holm, mx = [None] * m, 0
    for j, i in enumerate(ordem):
        mx = max(mx, min(1, (m - j) * ps[i][4]))
        holm[i] = mx
    for (c, md, v, dr, pv), ph in zip(ps, holm):
        p(f"  ASKL+ - {c:8s} dif={md:+.4f} vit={v} der={dr} p={pv:.2g} pHolm={ph:.2g}")
    # sobreposição dentro do orçamento
    d = (t["ASKL+"] - t["ASKL-"])
    for g, nome in ((True, "na base"), (False, "fora")):
        z = d[ov.reindex(d.index).fillna(False) == g]
        p(f"  ASKL+ - ASKL- {nome}: n={len(z)} dif={z.mean():+.4f} vit={(z>0).sum()} der={(z<0).sum()} "
          f"p={wilcoxon(z[z != 0]).pvalue:.2g}")
    # custo online
    for c in a.columns:
        cc = au[(au.orcamento == orc) & (au.metodo == c)].set_index("dataset").custo
        custo_rows.append({"metodo": f"{c} {orc}s", "acc": t[c].mean(), "custo_mediano_s": cc.median(),
                           "custo_total_h": cc.sum() / 3600})
pd.DataFrame(linhas).to_csv(RES / "final_agregado.csv", index=False)

for c in MTL.values():
    custo_rows.append({"metodo": c, "acc": acc_mtl[c].mean(), "custo_mediano_s": custo_mtl[c].median(),
                       "custo_total_h": custo_mtl[c].sum() / 3600})
cu = pd.DataFrame(custo_rows)
cu.to_csv(RES / "final_custo.csv", index=False)
p("\nCusto online (mediana por dataset, s) e acurácia média:")
for _, r in cu.iterrows():
    p(f"  {r.metodo:12s} custo={r.custo_mediano_s:7.1f} s  total={r.custo_total_h:5.2f} h  acc={r.acc:.4f}")
off = json.load(open(RES / "b_offline.json"))
p(f"Custo offline estimado da matriz (3.456 configs x 120 datasets na nossa GPU): {off['offline_matriz_gpu_h_estimada']:.1f} h")

fig, ax = plt.subplots(figsize=(6, 3.8))
for _, r in cu.iterrows():
    cor = "C0" if r.metodo.startswith("ASKL+") else "C1" if r.metodo.startswith("ASKL-") else "C2"
    ax.plot(r.custo_mediano_s, r.acc, "o", color=cor)
    ax.annotate(r.metodo, (r.custo_mediano_s, r.acc), fontsize=7, xytext=(3, 2), textcoords="offset points")
ax.set_xscale("log")
ax.set_xlabel("custo online mediano por dataset (s, escala log)")
ax.set_ylabel("acurácia média de teste")
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(FIG / f"final_custo.{ext}", dpi=150)

# o que o AutoML escolheu x recomendação (concordância de família de algoritmo)
d = au[(au.metodo == "ASKL+")].copy()
d["alg"] = d.melhor.astype(str).str.split("+").str[0]
p("\nMembro de maior peso do ASKL+ por orçamento: " + str({o: g.alg.value_counts().head(4).to_dict()
                                                          for o, g in d.groupby("orcamento")}))
p("Fração em que o membro de maior peso é MLP (a família recomendada pela MtL): " +
  str(d.groupby("orcamento").alg.apply(lambda s: round((s == "mlp").mean(), 3)).to_dict()))
txt = "\n".join(out)
print(txt)
(RES / "final_resumo.txt").write_text(txt, encoding="utf-8")
