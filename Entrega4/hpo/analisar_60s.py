"""Análise da campanha de 60 s do Auto-sklearn (com x sem meta-aprendizagem) e da sobreposição com a base do OpenML.
Grava resultados/autosklearn_60s_resumo.txt. Uso: python analisar_60s.py"""
import json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import comum

def ler(s):
    r = [json.loads(l) for l in open(comum.RES / f"automl_autosklearn_{s}.jsonl", encoding="utf-8")]
    return pd.DataFrame([x for x in r if x["orcamento"] == 60]).set_index("dataset")

ml, no = ler("ml"), ler("noml")
ov = pd.read_csv(comum.RES / "sobreposicao_openml.csv").set_index("dataset").na_base
out = []
p = out.append
for n, d in (("com", ml), ("sem", no)):
    a = pd.to_numeric(d.acc_teste, errors="coerce")
    p(f"[{n}] n={len(d)} sem_result={a.isna().sum()} dummy={(d.melhor=='dummy').sum()} "
      f"({', '.join(d.index[d.melhor=='dummy'])}) acc_media={a.mean():.4f} seg_ajuste_soma={d.seg_ajuste.sum():.1f} "
      f"media={d.seg_ajuste.mean():.1f} pred_soma={d.seg_predicao.sum():.1f} aval_media={d.n_avaliados.mean():.1f} "
      f"ens_media={d.n_modelos_ensemble.mean():.1f}")
    comp = d.composicao.fillna("")
    for alg in ("random_forest", "mlp"):
        tem = comp.str.contains(alg + r"\+")
        peso = comp.map(lambda c: sum(float(x.split(":")[0]) for x in c.split(";") if x and x.split(":")[1].startswith(alg + "+")))
        p(f"   {alg}: presente {tem.sum()}/{len(d)}, peso medio {peso.mean():.2f}, maior peso em {(d.melhor.str.startswith(alg+'+')).sum()}")
    p("   melhores mais comuns: " + str(d.melhor.str.split("+").str[0].value_counts().head(5).to_dict()))
    p("   preproc mais comuns: " + str(d.melhor.str.split("+").str[1].value_counts().head(5).to_dict()))
a, b = pd.to_numeric(ml.acc_teste), pd.to_numeric(no.acc_teste)
dd = (a - b).dropna()
w = wilcoxon(dd[dd != 0])
nz = dd[dd != 0]
rb = (np.sum(np.sign(nz) * pd.Series(np.abs(nz)).rank().values)) / pd.Series(np.abs(nz)).rank().sum()
p(f"pares={len(dd)} vit_com={(dd>0).sum()} vit_sem={(dd<0).sum()} empates={(dd==0).sum()} "
  f"W={w.statistic:.1f} p={w.pvalue:.3f} mediana_dif={dd.median():+.4f} media_dif={dd.mean():+.4f} rank_biserial={rb:.2f}")
for g, nome in ((True, "na base OpenML"), (False, "fora da base")):
    x = dd[ov.reindex(dd.index).fillna(False) == g]
    t = wilcoxon(x[x != 0]).pvalue if (x != 0).sum() > 5 else float("nan")
    p(f"  {nome}: n={len(x)} media_dif={x.mean():+.4f} vit_com={(x>0).sum()} vit_sem={(x<0).sum()} p={t:.3f}")
p("maiores ganhos com MtL: " + str(dd.sort_values().tail(5).round(3).to_dict()))
p("maiores ganhos sem MtL: " + str(dd.sort_values().head(5).round(3).to_dict()))
txt = "\n".join(out); print(txt)
(comum.RES / "autosklearn_60s_resumo.txt").write_text(txt, encoding="utf-8")
