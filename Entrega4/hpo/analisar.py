"""Análises da Atividade 4 (HPO): lê resultados/*.jsonl e configs_por_abordagem.csv e grava tabelas em resultados/ e
figuras em figuras/. Roda com o Python da entrega3 (pandas, scipy, matplotlib). Usa só datasets completos em todos os
métodos comparados em cada análise.

(a) acurácia de teste por dataset e agregada; perda para o melhor observado; Friedman + Nemenyi (diagrama CD) por orçamento
(b) desempenho x custo: tempo de cada método; custo online e offline da meta-aprendizagem; amortização
(c) o que o AutoML escolheu; relação do ganho do AutoML com meta-features, distância ao vizinho e perda do meta-modelo
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import friedmanchisquare, rankdata, spearmanr, studentized_range  # noqa: E402

HPO = Path(__file__).resolve().parent
RES, FIG = HPO / "resultados", HPO / "figuras"
E3 = HPO.parents[1] / "entrega3"
FIG.mkdir(exist_ok=True)
PRINCIPAL = ["autosklearn_ml", "autosklearn_noml"]
SUPLEMENTAR = ["autogluon", "flaml"]
ROT_AUTOML = {"autosklearn_ml": "Auto-sklearn (com MtL)", "autosklearn_noml": "Auto-sklearn (sem MtL)",
              "autogluon": "AutoGluon", "flaml": "FLAML", "tpot": "TPOT"}


def carregar():
    sel = pd.read_csv(RES / "configs_por_abordagem.csv")
    mlp = pd.read_json(RES / "mlp.jsonl", lines=True)
    mtl = sel.merge(mlp[["dataset", "config", "acc_teste", "seg_treino"]], on=["dataset", "config"], how="inner")
    mtl["metodo"] = mtl["rotulo"]
    au = []
    for s in ROT_AUTOML:
        f = RES / f"automl_{s}.jsonl"
        if f.exists():
            d = pd.read_json(f, lines=True)
            # falhas ficam, com a acurácia da classe majoritária (gravada pelo rodar_automl_hpo.py)
            d["metodo"] = ROT_AUTOML[s] + " " + d.orcamento.astype(str) + "s"
            au.append(d)
    au = pd.concat(au) if au else pd.DataFrame()
    return sel, mtl, au


def nemenyi_cd(k, n, alpha=0.05):
    q = studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    return q * np.sqrt(k * (k + 1) / (6 * n))


def diagrama_cd(ranks, cd, titulo, arq):
    r = ranks.sort_values()
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(r) + 1.2))
    ax.hlines(range(len(r)), 1, r.values, color="0.8")
    ax.plot(r.values, range(len(r)), "o", color="C0")
    ax.set_yticks(range(len(r)), r.index)
    ax.invert_yaxis()
    ax.errorbar(r.values[0], -0.8, xerr=[[0], [cd]], color="C3", capsize=4)
    ax.text(r.values[0], -1.1, f"CD = {cd:.2f}", color="C3", fontsize=8)
    ax.set_xlabel("rank médio (1 = melhor)")
    ax.set_title(titulo, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / f"{arq}.pdf")
    fig.savefig(FIG / f"{arq}.png", dpi=150)
    plt.close(fig)


def analise_a(mtl, au, sistemas, sufixo):
    linhas = []
    tab = mtl.pivot_table(index="dataset", columns="metodo", values="acc_teste")
    tabs = {}
    for orc in sorted(au.orcamento.unique()) if len(au) else []:
        a = au[(au.orcamento == orc) & au.sistema.isin(sistemas)].pivot_table(index="dataset", columns="metodo",
                                                                               values="acc_teste")
        t = tab.join(a, how="inner").dropna()
        if len(t) < 3:
            continue
        tabs[orc] = t
        rk = pd.DataFrame(rankdata(-t.values, axis=1), index=t.index, columns=t.columns).mean()
        stat, p = friedmanchisquare(*[t[c] for c in t.columns])
        cd = nemenyi_cd(t.shape[1], len(t))
        diagrama_cd(rk, cd, f"Orçamento {orc}s, n = {len(t)} datasets (Friedman p = {p:.1e})", f"cd_{sufixo}_{orc}s")
        melhor = t.max(axis=1)
        for c in t.columns:
            linhas.append({"orcamento": orc, "metodo": c, "n": len(t), "acc_media": t[c].mean(),
                           "acc_mediana": t[c].median(), "perda_p_melhor_media": (melhor - t[c]).mean(),
                           "vitorias": int((t[c] == melhor).sum()), "rank_medio": rk[c], "friedman_p": p, "cd": cd})
        t.to_csv(RES / f"a_por_dataset_{sufixo}_{orc}s.csv")
    out = pd.DataFrame(linhas)
    out.to_csv(RES / f"a_agregado_{sufixo}.csv", index=False)
    return out, tabs


def analise_b(sel, mtl, au):
    """Custo: MLP (treino da config recomendada, GPU) + busca/recomendação (tempo da Atividade 3, por alvo) +
    PyMFE do alvo (online) ; offline = PyMFE dos 119 + matriz de 3.456 x 119 treinos (estimada pelo tempo médio de MLP)."""
    rows = []
    seg_mlp = mtl.groupby("dataset").seg_treino.mean()
    for (ds, met), g in mtl.groupby(["dataset", "metodo"]):
        rows.append({"dataset": ds, "metodo": met, "acc": g.acc_teste.iloc[0],
                     "custo_online_s": g.seg_treino.iloc[0] + np.nan_to_num(g.seg_busca.iloc[0])})
    for _, r in au.iterrows():
        rows.append({"dataset": r.dataset, "metodo": r.metodo, "acc": r.acc_teste,
                     "custo_online_s": r.seg_ajuste + r.get("seg_predicao", 0)})
    c = pd.DataFrame(rows)
    resumo = c.groupby("metodo").agg(acc=("acc", "mean"), custo_mediano_s=("custo_online_s", "median"),
                                     custo_total_h=("custo_online_s", lambda s: s.sum() / 3600)).reset_index()
    resumo.to_csv(RES / "b_custo.csv", index=False)
    offline_h = 3456 * seg_mlp.sum() / 3600
    json.dump({"offline_matriz_gpu_h_estimada": offline_h, "seg_mlp_medio_por_dataset": seg_mlp.mean(),
               "obs": "matriz K x n re-treinada na nossa GPU, extrapolada do tempo medido das configs selecionadas"},
              open(RES / "b_offline.json", "w"), indent=1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for _, r in resumo.iterrows():
        ax.plot(r.custo_mediano_s, r.acc, "o")
        ax.annotate(r.metodo, (r.custo_mediano_s, r.acc), fontsize=6)
    ax.set_xscale("log")
    ax.set_xlabel("custo online mediano por dataset (s)")
    ax.set_ylabel("acurácia média de teste")
    fig.tight_layout()
    fig.savefig(FIG / "b_desempenho_custo.pdf")
    fig.savefig(FIG / "b_desempenho_custo.png", dpi=150)
    plt.close(fig)
    return resumo


def analise_c(sel, mtl, au):
    if not len(au):
        return
    esc = au.groupby(["sistema", "orcamento"]).agg(
        n=("dataset", "size"), ensemble=("ensemble", lambda s: s.astype(float).mean()) if "ensemble" in au else
        ("dataset", "size"), melhor_mais_comum=("melhor", lambda s: s.astype(str).str.split("+").str[0].mode().iat[0]))
    esc.to_csv(RES / "c_escolhas.csv")
    # ganho do AutoML (Auto-sklearn com MtL, maior orçamento) sobre a recomendação PyMFE k=1
    rec = mtl[mtl.abordagem == "rec_k1"].set_index("dataset").acc_teste
    a = au[au.sistema == "autosklearn_ml"]
    if not len(a):
        return
    a = a[a.orcamento == a.orcamento.max()].set_index("dataset").acc_teste
    ganho = (a - rec).dropna().rename("ganho")
    viz = pd.read_csv(E3 / "results" / "tables" / "vizinhos_pymfe.csv")
    mf = pd.read_csv(E3 / "representations" / "PyMFE.csv").set_index("ds_n")
    perda_rec = sel[sel.abordagem == "rec_k1"].set_index("dataset").perda_d2v
    df = pd.DataFrame({"ganho": ganho}).join(perda_rec.rename("perda_meta_modelo"))
    for c in viz.columns:
        if "dist" in c.lower():
            df = df.join(viz.set_index(viz.columns[0])[c])
    for c in ["nr_inst", "nr_attr", "nr_class", "class_ent"]:
        if c in mf:
            df = df.join(mf[c])
    cor = [{"variavel": c, "spearman": spearmanr(df.ganho, df[c], nan_policy="omit")[0],
            "p": spearmanr(df.ganho, df[c], nan_policy="omit")[1]} for c in df.columns if c != "ganho"]
    pd.DataFrame(cor).to_csv(RES / "c_correlacoes_ganho.csv", index=False)
    df.to_csv(RES / "c_ganho_por_dataset.csv")


def main():
    sel, mtl, au = carregar()
    analise_a(mtl, au, PRINCIPAL, "principal")
    analise_a(mtl, au, PRINCIPAL + SUPLEMENTAR, "todos")
    analise_b(sel, mtl, au)
    analise_c(sel, mtl, au)
    print("ok:", sorted(p.name for p in RES.glob("[abc]_*")))


if __name__ == "__main__":
    sys.exit(main())
