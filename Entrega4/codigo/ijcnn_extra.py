"""Quatro análises extras do artigo IJCNN "Popularity, Not Similarity" (só CPU; só as sete representações existentes).

Protocolo comum (o de ijcnn_fig.py e metrica_dificuldade.py): cada alvo usa o embedding da partição em que é de teste;
distância euclidiana após min-max por coluna; escore de uma configuração c = média de ỹ_{j,c} (linha de Y normalizada por
min-max; IR: F1 bruto) sobre os vizinhos; recomenda a de maior escore (top-1, empates pelo valor esperado). Constante =
k-NN com todos os outros datasets; acaso(k) = k datasets sorteados (20 sorteios; k=1 exato). Os valores de acaso, constante,
oráculo e do k-NN simples vêm de ijcnn_fig.curves (mesmos sorteios). Valores no texto na métrica do paper (HPO: ADTM, menor
melhor); diferenças como vantagem na escala G (maior melhor; HPO: G = 1 - ADTM), positiva = melhor.
Testes: Wilcoxon pareado por dataset alvo, Holm nas sete representações (salvo indicação). "não significativo": p >= 0,05.
V/D = alvos em que a representação ganha/perde (o Wilcoxon usa os postos: a média e o p podem apontar lados opostos).

1. Outros meta-modelos.
   (a) k-NN ponderado pela distância: escore = sum_j w_j ỹ_{j,c} / sum_j w_j, w_j = 1/(d_ij + 1e-6), k em {1,3,5,11,21,30}.
       Mesmos vizinhos do k-NN simples. Comparado com o acaso(k) e com a constante.
   (b) Floresta de regressão (RandomForestRegressor, semente fixa, multi-saída): prevê ỹ_{i,c} de todas as configurações a
       partir da representação e recomenda o argmax. Para cada partição f, treina nos datasets que não são de teste em f
       (treino + validação de f: 96 no HPO, 84 no PIPES, 172 no IR), com o embedding da partição f, e prevê os de teste de f.
       A linha do alvo em Y nunca entra no treino. Referências: a constante do paper, a constante só com os datasets de
       treino da floresta, e a mesma floresta com as linhas da representação permutadas entre os datasets de treino
       (popularidade sem similaridade, o análogo do acaso; média de NPERM permutações).
2. Simulação. Y = m + a_i + b_c + e_ic, com a_i ~ N(0, 1), b_c ~ N(0, 1 - s), e = sqrt(s) U V^T / sqrt(r) (posto r = 3;
   U_i, V_c ~ N(0, I_r)). s = var(e) / (var(b) + var(e)) é a parte de interação na variância das colunas (a_i some na
   normalização por linha). n em {50, 100, 200} datasets, C em {42, 1000} configurações, s em {0; 0,1; 0,25; 0,5; 0,75};
   NSIM réplicas por célula. Em cada réplica: H(k) como no paper (escala ỹ) e uma representação sem oráculo Z_i = U_i +
   N(0, tau^2 I), tau = 1 (versão ruidosa dos fatores verdadeiros de interação); k* = k de maior valor médio dessa
   representação. Para situar as meta-tarefas reais, o mesmo s calculado na decomposição de dois fatores de ỹ.
3. k escolhido sem o teste. Em cada partição f, para cada representação, k em {1,3,5,7,11,15,21,30} é escolhido pelo valor
   médio nos datasets que não são de teste em f, cada um como consulta com vizinhos entre os demais deles (leave-one-out;
   empate: menor k). Os alvos de teste de f usam esse k (vizinhos entre todos os outros, como no paper). Comparado com o
   acaso e a constante com o mesmo k. Variante: só os datasets de validação de f como consultas (vizinhos entre treino +
   validação), a regra que o README adota para não favorecer as representações treinadas nos datasets de treino.
4. Oráculo residual. Como o oráculo q, mas sobre o resíduo ỹ_{i,c} - média_j ỹ_{j,c} (desempenho menos popularidade):
   r_i = fração de cada valor de cada componente entre as κ configurações de maior resíduo (κ = 20, 10, 1; mesma função
   soft_targets do q). k-NN com k = 1 e 11 nesse espaço. Usa os resultados do alvo: é um limite superior, como o oráculo q.

Conferência: o k-NN simples recalculado aqui deve repetir ijcnn_fig.curves por alvo e as médias de ijcnn_numeros.txt e
folga_k.txt.

Uso: python ijcnn_extra.py  -> ijcnn_extra.txt (e cache/ijcnn_extra_*.csv por dataset)
"""
from __future__ import annotations

import re
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestRegressor
from statsmodels.stats.multitest import multipletests

import avaliar as A
import avaliar_pipes as P
import destilacao as D
import hipoteses as H
import ijcnn_fig as I
import kdn as KD
import knn_fixo as K
import topk as T
from fig_tempo_meta import percentile
from ocl.data import soft_targets  # noqa: E402

REFS = list(A.BASE_HPO)
KS_W = [1, 3, 5, 11, 21, 30]                 # k-NN ponderado e H(k) da simulação
KS_N = [1, 3, 5, 7, 11, 15, 21, 30]          # grade do k aninhado e do k* da simulação
EPS = 1e-6
SEED = 0
RF_TREES = 100
RF_MAXF = "sqrt"
NPERM = 3
NJOBS = 8
NSIM = 10
SIM_N = [50, 100, 200]
SIM_C = [42, 1000]
SIM_S = [0.0, 0.1, 0.25, 0.5, 0.75]
SIM_R = 3
SIM_TAU = 1.0
OUT = A.HERE / "ijcnn_extra.txt"


# ---------------------------------------------------------------------------
# utilidades
# ---------------------------------------------------------------------------
def pv(d):
    d = np.asarray(d, dtype=float)
    return 1.0 if np.allclose(d, 0) else float(wilcoxon(d).pvalue)


def fp(h):
    return f"p={h:.2g}" + ("" if h < 0.05 else ", não significativo")


def family(ds, labs):
    """Linha com a vantagem média e o p de Holm de cada comparação da família."""
    hp = multipletests([pv(d) for d in ds], method="holm")[1]
    vd = [f"{int((np.asarray(d) > 1e-12).sum())}/{int((np.asarray(d) < -1e-12).sum())}" for d in ds]
    return " | ".join(f"{x} {np.mean(d):+.4f} V/D {v} ({fp(h)})" for x, d, v, h in zip(labs, ds, vd, hp)), hp


def show(g, higher):
    """Valor na métrica do paper a partir da escala G."""
    return g if higher else 1.0 - g


def knn_scores(Z, i, pool, Ys, ks, weighted):
    """Escores das configurações para o alvo i com os k vizinhos (entre pool, sem i) de cada k."""
    d2 = ((Z[pool] - Z[i]) ** 2).sum(1)
    d2[pool == i] = np.inf
    o = np.argsort(d2, kind="stable")                         # a mesma ordem de kdn.neighbors
    out = []
    for k in ks:
        nb = o[:k]
        if weighted:
            w = 1.0 / (np.sqrt(d2[nb]) + EPS)
            out.append(w @ Ys[pool[nb]] / w.sum())
        else:
            out.append(Ys[pool[nb]].mean(0))
    return out


# ---------------------------------------------------------------------------
# dados de cada meta-tarefa
# ---------------------------------------------------------------------------
def setup(task):
    val, higher, (reps, _, names) = I.curves(task)            # acaso, constante, oráculo e k-NN simples (paper)
    _, Y, _ = T.TASKS[task][0]()
    Y = np.asarray(Y, dtype=float)
    Ys = Y if task == "ir" else K.rownorm(Y)
    G = Y                                                      # escala G: no HPO, Y já é 1 - ADTM
    gval = {x: (v if higher else 1.0 - v) for x, v in val.items()}
    part = np.asarray(D.task_data(task)[4])
    n = len(names)
    folds = sorted(set(part.tolist()))
    Zf = {r: {f: T.minmax(np.asarray(reps[r][0](int(np.flatnonzero(part == f)[0])), dtype=float)) for f in folds}
          for r in REFS}
    valid = valid_sets(task, names, part)
    return dict(task=task, names=names, n=n, Y=Y, Ys=Ys, G=G, higher=higher, gval=gval, part=part, folds=folds,
                Zf=Zf, valid=valid)


def valid_sets(task, names, part):
    """Datasets de validação de cada partição (HPO: splits oficiais; PIPES e IR: a regra de folds/ir_folds)."""
    n = len(names)
    if task == "hpo":
        data = A.load_hpo_data(with_arrays=False)
        assert list(data.names) == list(names)
        out = {f: np.sort(np.asarray(data.splits[f]["valid"], dtype=int)) for f in data.splits}
    elif task == "pipes":
        sys.path.insert(0, str(P.PIPES))
        from src.metatask_data import folds
        out = {f: va for f, (_, _, _, va) in enumerate(folds(n))}
    else:
        from run_ir_kfold import ir_folds
        fl = ir_folds(n)
        if not all(np.array_equal(te, np.flatnonzero(part == f)) for f, (_, te, _, _) in enumerate(fl)):
            return None
        out = {f: va for f, (_, _, _, va) in enumerate(fl)}
    for f, va in out.items():                                  # validação nunca contém teste
        assert not np.isin(va, np.flatnonzero(part == f)).any()
    return out


# ---------------------------------------------------------------------------
# conferência com ijcnn_numeros.txt e folga_k.txt
# ---------------------------------------------------------------------------
def parse_curves(path):
    """{tarefa: {nome: médias por k}} das linhas de curvas de ijcnn_fig/fig_folga."""
    out, cur = {}, None
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"== (\w+) constante ([\d.]+)", line)
        if m:
            cur = m.group(1).lower()
            out[cur] = {"constant": np.array([float(m.group(2))])}
            continue
        parts = line.split()
        if cur and len(parts) >= 11 and all(re.fullmatch(r"[\d.]+", x) for x in parts[-10:]) and line.startswith("  "):
            out[cur][" ".join(parts[:-10])] = np.array([float(x) for x in parts[-10:]])
    return out


def check(S, L):
    """k-NN simples recalculado contra ijcnn_fig.curves (por alvo) e contra os arquivos de texto (médias, 4 casas)."""
    task, n, Ys, G, gval, part, Zf = S["task"], S["n"], S["Ys"], S["G"], S["gval"], S["part"], S["Zf"]
    allp = np.arange(n)
    worst = 0.0
    for r in REFS:
        mine = np.array([[K.value_of_score(s, G[i]) for s in knn_scores(Zf[r][part[i]], i, allp, Ys, I.KS, False)]
                         for i in range(n)]).T
        worst = max(worst, float(np.abs(mine - gval[r]).max()))
    msg = [f"  k-NN simples recalculado contra ijcnn_fig.curves, por alvo: maior diferença {worst:.2e}"]
    for fname in ("ijcnn_numeros.txt", "folga_k.txt"):
        ref = parse_curves(A.HERE / fname).get(task, {})
        diffs = []
        for name, v in ref.items():
            key = name if name in gval else None
            if key is None:
                continue
            mine = (gval[key] if S["higher"] else 1.0 - gval[key]).mean(-1)
            mine = np.atleast_1d(mine)
            diffs.append((name, float(np.abs(np.round(mine, 4) - v).max())))
        bad = [f"{a} {b:.4f}" for a, b in diffs if b > 1.5e-4]
        msg.append(f"  médias contra {fname}: {len(diffs)} linhas conferidas, "
                   + ("todas iguais na 4a casa" if not bad else "DIFERENÇAS: " + "; ".join(bad)))
    L += msg
    return worst


# ---------------------------------------------------------------------------
# 1a. k-NN ponderado
# ---------------------------------------------------------------------------
def weighted(S, L):
    task, n, Ys, G, gval, part, Zf, hi = S["task"], S["n"], S["Ys"], S["G"], S["gval"], S["part"], S["Zf"], S["higher"]
    allp = np.arange(n)
    W = {r: np.array([[K.value_of_score(s, G[i]) for s in knn_scores(Zf[r][part[i]], i, allp, Ys, KS_W, True)]
                      for i in range(n)]).T for r in REFS}
    cons = gval["constant"]
    L.append(f"  (a) k-NN ponderado; constante {show(cons.mean(), hi):.4f}")
    L.append("      k            " + " ".join(f"{k:>7d}" for k in KS_W))
    L.append("      acaso        " + " ".join(f"{show(gval['random'][I.KS.index(k)].mean(), hi):7.4f}" for k in KS_W))
    for r in REFS:
        L.append(f"      {r:12s} " + " ".join(f"{show(W[r][j].mean(), hi):7.4f}" for j in range(len(KS_W))))
    summ = {}
    for j, k in enumerate(KS_W):
        rnd = gval["random"][I.KS.index(k)]
        s1, h1 = family([W[r][j] - rnd for r in REFS], REFS)
        s2, h2 = family([W[r][j] - cons for r in REFS], REFS)
        s3, _ = family([W[r][j] - gval[r][I.KS.index(k)] for r in REFS], REFS)
        L.append(f"      k={k} contra o acaso (Holm 7): {s1}")
        L.append(f"      k={k} contra a constante (Holm 7): {s2}")
        L.append(f"      k={k} ponderado menos simples (Holm 7): {s3}")
        summ[k] = (int((h1 < 0.05).sum()), int(sum(h < 0.05 and np.mean(W[r][j] - cons) > 0 for r, h in zip(REFS, h2))),
                   int(sum(h < 0.05 and np.mean(W[r][j] - cons) < 0 for r, h in zip(REFS, h2))))
    L.append("      resumo (k: n.º acima do acaso | acima da constante | abaixo da constante, Holm p<0,05): " + "; ".join(
        f"k={k}: {a}|{b}|{c}" for k, (a, b, c) in summ.items()))
    pd.DataFrame({(r, k): W[r][j] for r in REFS for j, k in enumerate(KS_W)}, index=S["names"]).to_csv(
        A.CACHE / f"ijcnn_extra_wknn_{task}.csv")


# ---------------------------------------------------------------------------
# 1b. floresta de regressão
# ---------------------------------------------------------------------------
def rf_fit_predict(Xtr, Ytr, Xte, seed, perm_seed=None):
    if perm_seed is not None:                                  # popularidade sem similaridade: Z desligado de Y
        Xtr = Xtr[np.random.default_rng(perm_seed).permutation(len(Xtr))]
    rf = RandomForestRegressor(n_estimators=RF_TREES, max_features=RF_MAXF, random_state=seed, n_jobs=1)
    rf.fit(Xtr, Ytr)
    return rf.predict(Xte)


def forest(S, L):
    task, n, Ys, G, gval, part, Zf, hi = S["task"], S["n"], S["Ys"], S["G"], S["gval"], S["part"], S["Zf"], S["higher"]
    jobs = []
    for r in REFS:
        for f in S["folds"]:
            tr, te = np.flatnonzero(part != f), np.flatnonzero(part == f)
            assert not np.isin(te, tr).any()
            Z = Zf[r][f]
            jobs.append((r, f, None, Z[tr], Z[te]))
            for p in range(NPERM):
                jobs.append((r, f, p, Z[tr], Z[te]))
    preds = Parallel(n_jobs=NJOBS, prefer="threads")(delayed(rf_fit_predict)(Xtr, Ys[part != f], Xte, SEED,
                                                           None if p is None else 1000 * f + p)
                                   for (r, f, p, Xtr, Xte) in jobs)
    RF = {r: np.empty(n) for r in REFS}
    PERM = {r: np.zeros(n) for r in REFS}
    for (r, f, p, _, _), pr in zip(jobs, preds):
        te = np.flatnonzero(part == f)
        v = np.array([K.value_of_score(pr[a], G[i]) for a, i in enumerate(te)])
        if p is None:
            RF[r][te] = v
        else:
            PERM[r][te] += v / NPERM
    cons = gval["constant"]
    cons_tr = np.empty(n)                                      # constante só com os datasets de treino da floresta
    for f in S["folds"]:
        tr, te = np.flatnonzero(part != f), np.flatnonzero(part == f)
        s = Ys[tr].mean(0)
        cons_tr[te] = [K.value_of_score(s, G[i]) for i in te]
    ntr = int((part != S["folds"][0]).sum())
    L.append(f"  (b) floresta de regressão ({RF_TREES} árvores, max_features={RF_MAXF}, semente {SEED}; treino = "
             f"{ntr} datasets que não são de teste na partição do alvo)")
    L.append(f"      constante (todos os outros) {show(cons.mean(), hi):.4f} | constante (só treino da floresta) "
             f"{show(cons_tr.mean(), hi):.4f}")
    L.append("      " + " | ".join(f"{r} {show(RF[r].mean(), hi):.4f} (permutada {show(PERM[r].mean(), hi):.4f})"
                                   for r in REFS))
    s1, _ = family([RF[r] - PERM[r] for r in REFS], REFS)
    s2, h2 = family([RF[r] - cons for r in REFS], REFS)
    s3, _ = family([RF[r] - cons_tr for r in REFS], REFS)
    s4, _ = family([PERM[r] - cons for r in REFS], REFS)
    L.append(f"      floresta contra a permutada (Holm 7): {s1}")
    L.append(f"      floresta contra a constante (Holm 7): {s2}")
    L.append(f"      floresta contra a constante do treino (Holm 7): {s3}")
    L.append(f"      permutada contra a constante (Holm 7): {s4}")
    pd.DataFrame({**{(r, "rf"): RF[r] for r in REFS}, **{(r, "perm"): PERM[r] for r in REFS}},
                 index=S["names"]).to_csv(A.CACHE / f"ijcnn_extra_rf_{task}.csv")


# ---------------------------------------------------------------------------
# 2. simulação
# ---------------------------------------------------------------------------
def sim_one(n, C, s, seed):
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 1, (n, 1))
    b = rng.normal(0, np.sqrt(1 - s), (1, C))
    U, V = rng.normal(size=(n, SIM_R)), rng.normal(size=(C, SIM_R))
    Y = a + b + np.sqrt(s) * U @ V.T / np.sqrt(SIM_R)
    Ys = K.rownorm(Y)
    G = Ys[:, Ys.argmax(1)]                                    # G[i, j]: valor em i da melhor de j
    Z = U + SIM_TAU * rng.normal(size=U.shape)
    allp = np.arange(n)
    nk = len(KS_N)
    cons, rnd, orc, rep = np.empty(n), np.empty((nk, n)), np.empty((nk, n)), np.empty((nk, n))
    for i in range(n):
        oth = np.delete(allp, i)
        y = Ys[i]
        cons[i] = K.value_of_score(Ys[oth].mean(0), y)
        oo = oth[np.argsort(-G[i, oth], kind="stable")]
        d2 = ((Z[oth] - Z[i]) ** 2).sum(1)
        oz = oth[np.argsort(d2, kind="stable")]
        for j, k in enumerate(KS_N):
            rnd[j, i] = (G[i, oth].mean() if k == 1 else
                         np.mean([K.value_of_score(Ys[rng.choice(oth, k, replace=False)].mean(0), y) for _ in range(20)]))
            orc[j, i] = K.value_of_score(Ys[oo[:k]].mean(0), y)
            rep[j, i] = K.value_of_score(Ys[oz[:k]].mean(0), y)
    c, r, o, z = cons.mean(), rnd.mean(1), orc.mean(1), rep.mean(1)
    pop = np.maximum(r, c)
    den = o[0] - r[0]                                          # s = 0: linhas iguais, H indefinido
    Hk = (o - pop) / den if den > 1e-9 else np.full(len(KS_N), np.nan)
    ks = int(np.argmax(z))                                     # primeiro máximo = menor k
    room = o[ks] - pop[ks]
    return dict(H=Hk, kstar=KS_N[ks], gain=z[ks] - c, use=(z[ks] - pop[ks]) / room if room > 1e-9 else np.nan)


def interaction_share(Ys):
    m = Ys.mean()
    a = Ys.mean(1, keepdims=True) - m
    b = Ys.mean(0, keepdims=True) - m
    e = Ys - m - a - b
    return float(e.var() / (b.var() + e.var()))


def simulation(L, real):
    cells = [(n, C, s) for C in SIM_C for n in SIM_N for s in SIM_S]
    jobs = [(n, C, s, 10_000 * ci + rep) for ci, (n, C, s) in enumerate(cells) for rep in range(NSIM)]
    try:                                                       # loky falha às vezes no Windows (WinError 6)
        res = Parallel(n_jobs=NJOBS)(delayed(sim_one)(*j) for j in jobs)
    except Exception as err:  # noqa: BLE001
        print(f"loky falhou ({type(err).__name__}); simulação em série", flush=True)
        res = [sim_one(*j) for j in jobs]
    L.append(f"  s das meta-tarefas reais (mesma decomposição em ỹ; inclui ruído sem estrutura): " + " | ".join(
        f"{t.upper()} {v:.2f}" for t, v in real.items()))
    hk = [KS_N.index(k) for k in KS_W]
    L.append("  C     n    s   " + " ".join(f"{'H(' + str(k) + ')':>12s}" for k in KS_W)
             + "   k* mediana [mín-máx]   ganho sobre a constante em k*   aproveitamento em k*")
    for ci, (n, C, s) in enumerate(cells):
        rr = res[ci * NSIM:(ci + 1) * NSIM]
        Hm = np.array([x["H"] for x in rr])
        ks = np.array([x["kstar"] for x in rr])
        gn = np.array([x["gain"] for x in rr])
        us = np.array([x["use"] for x in rr])
        hcell = " ".join(("   indefinido" if np.isnan(Hm[:, j]).all() else
                          f"{np.nanmean(Hm[:, j]):6.2f}±{np.nanstd(Hm[:, j], ddof=1):4.2f}") for j in hk)
        ucell = "indefinido" if np.isnan(us).all() else f"{np.nanmean(us):+.2f}±{np.nanstd(us, ddof=1):.2f}"
        L.append(f"  {C:<5d} {n:<4d} {s:<4.2f} " + hcell + f"   {int(np.median(ks)):>3d} [{ks.min()}-{ks.max()}]"
                 + f"          {gn.mean():+.4f}±{gn.std(ddof=1):.4f}          {ucell}")
    return res


# ---------------------------------------------------------------------------
# 3. k aninhado
# ---------------------------------------------------------------------------
def choose_k(Z, queries, pool, Ys, G):
    v = np.array([[K.value_of_score(s, G[i]) for s in knn_scores(Z, i, pool, Ys, KS_N, False)] for i in queries])
    m = v.mean(0)
    return KS_N[int(np.argmax(m))]                             # primeiro máximo = menor k


def nested(S, L):
    task, n, Ys, G, gval, part, Zf, hi = S["task"], S["n"], S["Ys"], S["G"], S["gval"], S["part"], S["Zf"], S["higher"]
    modes = [("LOO nos datasets de ajuste (treino + validação)", None)]
    if S["valid"] is not None:
        modes.append(("só a validação como consulta", S["valid"]))
    cons = gval["constant"]
    for title, valid in modes:
        chosen = {}
        jobs = []
        for r in REFS:
            for f in S["folds"]:
                fit = np.flatnonzero(part != f)
                q = fit if valid is None else valid[f]
                jobs.append((r, f, q, fit))
        ks = [choose_k(Zf[r][f], q, fit, Ys, G) for r, f, q, fit in jobs]     # barato: em série
        for (r, f, _, _), k in zip(jobs, ks):
            chosen[(r, f)] = k
        V, R = {}, {}
        for r in REFS:
            V[r], R[r] = np.empty(n), np.empty(n)
            for i in range(n):
                j = I.KS.index(chosen[(r, int(part[i]))])
                V[r][i], R[r][i] = gval[r][j, i], gval["random"][j, i]
        L.append(f"  k escolhido ({title}); colunas = partições {S['folds']}")
        for r in REFS:
            L.append(f"      {r:8s} " + " ".join(f"{chosen[(r, f)]:>3d}" for f in S["folds"])
                     + f"   teste {show(V[r].mean(), hi):.4f} | acaso(mesmo k) {show(R[r].mean(), hi):.4f} | "
                     f"k=11 {show(gval[r][I.KS.index(11)].mean(), hi):.4f}")
        L.append(f"      constante {show(cons.mean(), hi):.4f}")
        s1, _ = family([V[r] - R[r] for r in REFS], REFS)
        s2, _ = family([V[r] - cons for r in REFS], REFS)
        s3, _ = family([V[r] - gval[r][I.KS.index(11)] for r in REFS], REFS)
        L.append(f"      contra o acaso com o mesmo k (Holm 7): {s1}")
        L.append(f"      contra a constante (Holm 7): {s2}")
        L.append(f"      k aninhado menos k=11 (Holm 7): {s3}")


# ---------------------------------------------------------------------------
# 4. oráculo residual
# ---------------------------------------------------------------------------
def residual(S, L, fam_rows):
    task, n, Ys, G, gval, hi = S["task"], S["n"], S["Ys"], S["G"], S["gval"], S["higher"]
    n2, codes, sizes, _ = H.task_info(task)
    assert list(n2) == list(S["names"]) or set(n2) == set(S["names"])
    Rm = Ys - Ys.mean(0, keepdims=True)                        # desempenho menos popularidade
    Qr = np.stack([soft_targets(Rm[i], codes, sizes, H.K[task]) for i in range(n)])
    allp = np.arange(n)
    vals = {k: np.empty(n) for k in (1, 11)}
    nn1 = np.empty(n, dtype=int)
    for i in range(n):
        o = KD.neighbors(Qr, i, allp)
        nn1[i] = o[0]
        for k in (1, 11):
            vals[k][i] = K.value_of_score(Ys[o[:k]].mean(0), G[i])
    pc = percentile(-T.transfer_topk(S["Y"], 1), nn1)
    cons = gval["constant"]
    for k in (1, 11):
        j = I.KS.index(k)
        fam_rows.append((task, k, vals[k] - cons, vals[k] - gval["oracle q"][j], vals[k] - gval["random"][j]))
        L.append(f"  k={k}: oráculo residual {show(vals[k].mean(), hi):.4f} | oráculo q {show(gval['oracle q'][j].mean(), hi):.4f}"
                 f" | constante {show(cons.mean(), hi):.4f} | acaso {show(gval['random'][j].mean(), hi):.4f} | "
                 f"oráculo {show(gval['oracle'][j].mean(), hi):.4f}")
    L.append(f"  1-NN residual: percentil do vizinho {pc.mean():.1f} (acaso = 50)")
    pd.DataFrame({f"res_k{k}": v for k, v in vals.items()}, index=S["names"]).to_csv(A.CACHE / f"ijcnn_extra_res_{task}.csv")


# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    L = [__doc__.strip(), "", "RESULTADOS", ""]
    secs = {x: [] for x in ("conf", "meta", "sim", "nest", "res")}
    real, fam_rows = {}, []
    for task in ("hpo", "pipes", "ir"):
        ts = time.time()
        S = setup(task)
        real[task] = interaction_share(S["Ys"])
        head = f"== {task.upper()} ({S['n']} datasets; {'maior' if S['higher'] else 'menor'} é melhor)"
        for key in ("conf", "meta", "nest", "res"):
            secs[key].append(head)
        check(S, secs["conf"])
        weighted(S, secs["meta"])
        forest(S, secs["meta"])
        nested(S, secs["nest"])
        residual(S, secs["res"], fam_rows)
        print(f"{task} pronto em {time.time() - ts:.0f} s", flush=True)
    # testes do oráculo residual: Holm nas 6 comparações (3 tarefas x k = 1, 11) de cada família
    for idx, name in ((2, "contra a constante"), (3, "contra o oráculo q"), (4, "contra o acaso(k)")):
        labs = [f"{t.upper()} k={k}" for t, k, *_ in fam_rows]
        s, _ = family([row[idx] for row in fam_rows], labs)
        secs["res"].append(f"  residual {name} (Holm nas 6): {s}")
    ts = time.time()
    simulation(secs["sim"], real)
    print(f"simulação pronta em {time.time() - ts:.0f} s", flush=True)
    L += ["0. CONFERÊNCIA DO k-NN SIMPLES"] + secs["conf"] + [""]
    L += ["1. OUTROS META-MODELOS"] + secs["meta"] + [""]
    L += [f"2. SIMULAÇÃO ({NSIM} réplicas por célula; média ± desvio entre réplicas; tau = {SIM_TAU}, posto {SIM_R})"]
    L += secs["sim"] + [""]
    L += ["3. k ESCOLHIDO SEM O TESTE"] + secs["nest"] + [""]
    L += ["4. ORÁCULO RESIDUAL (usa os resultados do alvo: limite superior, como o oráculo q)"] + secs["res"] + [""]
    L.append(f"Tempo total: {time.time() - t0:.0f} s")
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[L.index("RESULTADOS"):]))


if __name__ == "__main__":
    main()
