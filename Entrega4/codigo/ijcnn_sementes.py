"""Robustez à semente e curvas de aprendizado dos três baselines treinados (MetaBu, Dida, TRIO) no artigo IJCNN.

A. Sementes 42 (a rodada oficial), 1042, 2042, 3042 e 4042, mesmas partições. Por semente: valor top-1 do 1-NN e do 11-NN,
   kDN (k=5) no teste e o excesso de kDN sobre o acaso nos datasets de ajuste e nos de teste (definições do ijcnn_fig.py).
   Média ± desvio sobre as sementes. Testes de robustez, com o valor médio sobre as sementes em cada dataset:
   (i) k-NN (k = 1 e 11) contra vizinhos ao acaso e contra a constante: Wilcoxon pareado por dataset, Holm nas sete
       representações (as quatro fixas ficam com o valor da semente 42, o único que existe);
   (ii) excesso de kDN no ajuste menos no teste (por dataset, médias sobre as sementes): Wilcoxon, Holm nas sete.
   Também o p de Holm de cada semente sozinha, para contar em quantas a conclusão se mantém.
B. Curvas de aprendizado (semente 42): N = 20, 40, 60 e todos os datasets de treino de cada partição (subamostra fixa, a
   validação não muda). Ajuste = treino sorteado + validação (o conjunto de vizinhos do excesso de ajuste). Por N: excesso de
   kDN no ajuste (e só no treino sorteado), no teste, a diferença com Wilcoxon por dataset, e o 11-NN no teste. Tendência:
   Wilcoxon pareado por dataset entre N = 20 e N = todos (diferença ajuste - teste; excesso no teste; 11-NN), Holm nas três
   representações de cada meta-tarefa.

Entradas: embeddings oficiais (semente 42, N = todos) e os de baselines_externos/sementes/s{S}/ e curvas/n{N}/
({hpo/<rep>_split{k}.csv, pipes/<rep>/fold{f}.npz, ir/<rep>/fold{f}.npz}). Rodadas que faltam aparecem como pendentes.
Saídas: ijcnn_sementes.txt e cache/ijcnn_sementes_resumo.csv. Uso: python ijcnn_sementes.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

import avaliar as A
import destilacao as D
import ijcnn_fig as F
import kdn as KD
import knn_fixo as K
import topk as T

REFS = list(A.BASE_HPO)
NOVOS = ["MetaBu", "Dida", "TRIO"]
SEMENTES = (42, 1042, 2042, 3042, 4042)
NS = (20, 40, 60, None)                                    # None = todos os datasets de treino (rodada oficial)
BASE = A.PROJECT / "baselines_externos"
KK = (1, 11)


def subamostra(train, f, n):
    """A mesma regra dos scripts rodar_*.py: os N primeiros de uma permutação fixa (semente [42, partição])."""
    train = np.asarray(train)
    if n is None or n >= len(train):
        return train
    return np.sort(train[np.random.default_rng([42, f]).permutation(len(train))[:n]])


def particoes(task):
    """{partição: (treino, validação, teste)} na ordem dos arquivos de embedding."""
    if task == "hpo":
        d = A.load_hpo_data(with_arrays=False)
        return {k: (np.asarray(s["train"]), np.asarray(s["valid"]), np.asarray(s["test"])) for k, s in d.splits.items()}
    if task == "pipes":
        sys.path.insert(0, str(A.PROJECT / "NEW-meta-task-PIPES"))
        from src.metatask_data import folds
        names, _ = D.P.load(raw=True)
        return {f: (tr, va, te) for f, (_, te, tr, va) in enumerate(folds(len(names)))}
    from run_ir_kfold import ir_folds
    return {f: (tr, va, te) for f, (_, te, tr, va) in enumerate(ir_folds(len(A.tasks.load_ir().names)))}


def contexto(task):
    """Tudo o que não depende do embedding: alvo, rótulos do kDN, acaso e constante (o mesmo cálculo do ijcnn_fig.curves)."""
    loader, higher, _, _ = T.TASKS[task]
    names, Y, reps = loader()
    Y = np.asarray(Y, dtype=float)
    Ys = Y if task == "ir" else K.rownorm(Y)
    base = Y if higher else 1.0 - Y
    n = len(names)
    _, lab = KD.labels(task, names)
    oth = [np.delete(np.arange(n), i) for i in range(n)]
    rng = np.random.default_rng(0)
    const = np.array([K.value_of_score(Ys[oth[i]].mean(0), base[i]) for i in range(n)])
    rand = np.array([[np.mean([K.value_of_score(Ys[j], base[i]) for j in oth[i]]) if k == 1 else
                      np.mean([K.value_of_score(Ys[rng.choice(oth[i], k, replace=False)].mean(0), base[i])
                               for _ in range(F.NDRAW)]) for i in range(n)] for k in F.KS])
    dis = (lab[:, None, :] != lab[None, :, :]).mean(2)
    np.fill_diagonal(dis, np.nan)
    _, _, _, _, part = D.task_data(task)
    return dict(task=task, names=list(names), n=n, Ys=Ys, base=base, lab=lab, dis=dis, rk=np.nanmean(dis, 1),
                higher=higher, const=const, rand={k: rand[F.KS.index(k)] for k in KK}, reps=reps,
                part=np.asarray(part), parts=particoes(task))


def carregar(ctx, rep, raiz):
    """mat_of(i) do embedding de rep gravado em raiz (None = oficial); None se faltar algum arquivo."""
    if raiz is None:
        return ctx["reps"][rep][0]
    part = ctx["part"]
    if ctx["task"] == "hpo":
        got = A.load_rep(raiz / "hpo", rep, ctx["names"], part)
        if got is None:
            return None
        mats = {k: A.minmax_normalize(df) for k, df in got[0].items()}
        return lambda i: mats[int(part[i])]
    files = [raiz / ctx["task"] / rep / f"fold{f}.npz" for f in range(5)]
    if not all(p.exists() for p in files):
        return None
    run = {f: np.load(p)["Z"] for f, p in enumerate(files)}
    return lambda i: run[int(part[i])]


def medir(ctx, mat_of, n_treino=None):
    """Por dataset: valor do 1-NN e do 11-NN, kDN (k=5) no teste, excesso de kDN no ajuste (média sobre as partições em que
    o dataset é de ajuste; vizinhos entre os de ajuste), o mesmo só nos de treino sorteados, e excesso no teste."""
    n, lab, dis, rk, part = ctx["n"], ctx["lab"], ctx["dis"], ctx["rk"], ctx["part"]
    allp = np.arange(n)
    v = {k: np.empty(n) for k in KK}
    kdn5 = np.empty(n)
    cache = {}
    for i in range(n):
        M = mat_of(i)
        if id(M) not in cache:
            cache[id(M)] = T.minmax(np.asarray(M, dtype=float))
        o = KD.neighbors(cache[id(M)], i, allp)
        for k in KK:
            v[k][i] = K.value_of_score(ctx["Ys"][o[:k]].mean(0), ctx["base"][i])
        kdn5[i] = KD.kdn_from_order(lab, i, o, 5)
    s_aj, c_aj, s_tr, c_tr, te = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.full(n, np.nan)
    for f, (train, valid, test) in ctx["parts"].items():
        tei = np.flatnonzero(part == f)
        assert set(tei.tolist()) == set(np.asarray(test).tolist())
        sub = subamostra(train, f, n_treino)
        pool = np.sort(np.r_[sub, valid]).astype(int)
        Z = T.minmax(np.asarray(mat_of(int(tei[0])), dtype=float))
        for i in pool:
            x = KD.kdn_from_order(lab, i, KD.neighbors(Z, i, pool), 5) - np.nanmean(dis[i, pool])
            s_aj[i] += x
            c_aj[i] += 1
            if i in set(sub.tolist()):
                s_tr[i] += x
                c_tr[i] += 1
        for i in tei:
            te[i] = kdn5[i] - rk[i]
    with np.errstate(invalid="ignore", divide="ignore"):
        return dict(v1=v[1], v11=v[11], kdn=kdn5, aj=s_aj / c_aj, tr=s_tr / c_tr, te=te)


def holm_wilcoxon(ds):
    ps = [1.0 if np.allclose(d, 0) else float(wilcoxon(d).pvalue) for d in ds]
    return multipletests(ps, method="holm")[1]


def pf(p):
    return f"{p:.2g}" + ("" if p < 0.05 else ", não significativo")


def ms(x, nd=4):
    x = np.asarray(x, dtype=float)
    return f"{x.mean():.{nd}f} ± {x.std(ddof=1):.{nd}f}" if len(x) > 1 else f"{x.mean():.{nd}f} (1 semente)"


def parte_a(ctx, L, resumo):
    task, s = ctx["task"], (1 if ctx["higher"] else -1)
    fixos = {r: medir(ctx, carregar(ctx, r, None)) for r in REFS if r not in NOVOS}
    por = {r: {} for r in NOVOS}
    for r in NOVOS:
        for S in SEMENTES:
            m = carregar(ctx, r, None if S == 42 else BASE / "sementes" / f"s{S}")
            if m is not None:
                por[r][S] = medir(ctx, m)
    L.append(f"== {task.upper()} ({ctx['n']} datasets) | constante {ctx['const'].mean():.4f} | acaso 1-NN "
             f"{ctx['rand'][1].mean():.4f}, 11-NN {ctx['rand'][11].mean():.4f} | kDN do acaso {ctx['rk'].mean():.3f}")
    for r in NOVOS:
        got = sorted(por[r])
        falta = [S for S in SEMENTES if S not in por[r]]
        L.append(f"  {r}: sementes {got}" + (f" (pendentes: {falta})" if falta else ""))
        q = {key: [por[r][S][key].mean() if key != "aj" else np.nanmean(por[r][S][key]) for S in got]
             for key in ("v1", "v11", "kdn", "aj", "te")}
        L.append(f"    1-NN {ms(q['v1'])} | 11-NN {ms(q['v11'])} | kDN {ms(q['kdn'], 3)} | excesso de kDN ajuste "
                 f"{ms(q['aj'], 3)}, teste {ms(q['te'], 3)}")
        L.append("    por semente (1-NN, 11-NN, kDN, ajuste/teste): " + " | ".join(
            f"s{S} {q['v1'][j]:.4f} {q['v11'][j]:.4f} {q['kdn'][j]:.3f} {q['aj'][j]:+.3f}/{q['te'][j]:+.3f}"
            for j, S in enumerate(got)))
        for j, S in enumerate(got):
            resumo.append(dict(tarefa=task, rep=r, semente=S, **{key: q[key][j] for key in q}))
    # médias sobre as sementes por dataset (as quatro fixas: a única rodada)
    media = {r: fixos[r] for r in fixos}
    for r in NOVOS:
        if por[r]:
            media[r] = {key: np.nanmean([por[r][S][key] for S in por[r]], axis=0) for key in ("v1", "v11", "aj", "te")}
    ordem = [r for r in REFS if r in media]
    for k in KK:
        key = f"v{k}"
        for nome, ref in (("o acaso", ctx["rand"][k]), ("a constante", ctx["const"])):
            ds = [s * (media[r][key] - ref) for r in ordem]
            hp = holm_wilcoxon(ds)
            L.append(f"  (i) {k}-NN contra {nome}, média das sementes por dataset (Wilcoxon, Holm 7): " + " | ".join(
                f"{r} {d.mean():+.4f} ({pf(h)})" for r, d, h in zip(ordem, ds, hp) if r in NOVOS))
            # semente a semente (as quatro fixas entram iguais em todas)
            cont = []
            for r in NOVOS:
                hs = []
                for S in por[r]:
                    ds_s = [s * ((por[r][S] if x == r else media[x] if x not in NOVOS else
                                  por[x].get(42, media[x]))[key] - ref) for x in ordem]
                    hs.append(holm_wilcoxon(ds_s)[ordem.index(r)])
                cont.append(f"{r} {sum(h < 0.05 for h in hs)}/{len(hs)} (p Holm {min(hs):.2g} a {max(hs):.2g})")
            L.append("      sementes com p Holm < 0,05: " + " | ".join(cont))
    ds = []
    for r in ordem:
        a, b = media[r]["aj"], media[r]["te"]
        ok = ~np.isnan(a) & ~np.isnan(b)
        ds.append((a - b)[ok])
    hp = holm_wilcoxon(ds)
    L.append("  (ii) excesso de kDN ajuste - teste, médias das sementes por dataset (Wilcoxon, Holm 7): " + " | ".join(
        f"{r} {np.nanmean(media[r]['aj']):+.3f}/{np.nanmean(media[r]['te']):+.3f} d={d.mean():+.3f} ({pf(h)})"
        for r, d, h in zip(ordem, ds, hp)))
    cont = []
    for r in NOVOS:
        hs = []
        for S in por[r]:
            ds_s = [(por[r][S] if x == r else media[x] if x not in NOVOS else por[x].get(42, media[x]))
                    for x in ordem]
            hs.append(holm_wilcoxon([m["aj"] - m["te"] for m in ds_s])[ordem.index(r)])
        dd = [np.nanmean(por[r][S]["aj"]) - np.nanmean(por[r][S]["te"]) for S in por[r]]
        cont.append(f"{r} d {ms(dd, 3)}, {sum(h < 0.05 for h in hs)}/{len(hs)} sementes com p Holm < 0,05 "
                    f"({min(hs):.2g} a {max(hs):.2g})")
    L.append("      por semente: " + " | ".join(cont))
    L.append("")


def parte_b(ctx, L, resumo):
    task = ctx["task"]
    L.append(f"== {task.upper()} | treino por partição: {len(next(iter(ctx['parts'].values()))[0])}, validação "
             f"{len(next(iter(ctx['parts'].values()))[1])}")
    tend = {}
    for r in NOVOS:
        res = {}
        for N in NS:
            m = carregar(ctx, r, None if N is None else BASE / "curvas" / f"n{N}")
            if m is not None:
                res[N] = medir(ctx, m, N)
        if not res:
            L.append(f"  {r}: pendente")
            continue
        for N, x in res.items():
            ok = ~np.isnan(x["aj"])
            d = x["aj"][ok] - x["te"][ok]
            L.append(f"  {r} N={'todos' if N is None else N}: excesso ajuste {np.nanmean(x['aj']):+.3f} (só treino "
                     f"{np.nanmean(x['tr']):+.3f}), teste {np.nanmean(x['te']):+.3f}, ajuste - teste {d.mean():+.3f} "
                     f"(Wilcoxon por dataset, n={ok.sum()}, p={pf(wilcoxon(d).pvalue)}) | 11-NN teste {x['v11'].mean():.4f}"
                     f" | kDN {x['kdn'].mean():.3f}")
            resumo.append(dict(tarefa=task, rep=r, semente=42, n_treino=N if N else "todos", v1=x["v1"].mean(),
                               v11=x["v11"].mean(), kdn=x["kdn"].mean(), aj=np.nanmean(x["aj"]), te=np.nanmean(x["te"])))
        if 20 in res and None in res:
            a, b = res[20], res[None]
            g20, gall = a["aj"] - a["te"], b["aj"] - b["te"]
            ok = ~np.isnan(g20) & ~np.isnan(gall)
            tend[r] = [gall[ok] - g20[ok], b["te"] - a["te"], (b["v11"] - a["v11"]) * (1 if ctx["higher"] else -1)]
    if tend:
        for j, nome in enumerate(("ajuste - teste (todos menos N=20; positivo = diferença menor)",
                                  "excesso no teste (todos menos N=20; positivo = pior)",
                                  "11-NN no teste (todos menos N=20; positivo = melhor)")):
            rs = list(tend)
            hp = holm_wilcoxon([tend[r][j] for r in rs])
            L.append(f"  tendência, {nome} (Wilcoxon por dataset, Holm 3): " + " | ".join(
                f"{r} {tend[r][j].mean():+.4f} ({pf(h)})" for r, h in zip(rs, hp)))
    L.append("")


def main():
    L = [__doc__.strip().splitlines()[0], ""]
    resumo = []
    ctxs = {t: contexto(t) for t in ("hpo", "pipes", "ir")}
    L.append("A. Sementes (média ± desvio entre sementes; valores: HPO = ADTM, menor é melhor; PIPES = acurácia; IR = F1)")
    L.append("")
    for t, ctx in ctxs.items():
        parte_a(ctx, L, resumo)
    L.append("B. Curvas de aprendizado (semente 42; N = datasets de treino por partição)")
    L.append("")
    for t, ctx in ctxs.items():
        parte_b(ctx, L, resumo)
    pd.DataFrame(resumo).to_csv(A.CACHE / "ijcnn_sementes_resumo.csv", index=False)
    (A.HERE / "ijcnn_sementes.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
