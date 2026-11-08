"""Por que é difícil gerar representações de datasets para meta-learning? (autor, 07/11)  -> METRIC.txt

Base de um artigo de análise SEM o nosso método: só as sete representações existentes (TabPFN, Dataset2Vec, PyMFE, ECoL,
MetaBu, Dida, TRIO), nas três meta-tarefas do paper (HPO, PIPES, IR), com o mesmo meta-modelo de vizinhança. Só CPU.
Seções: (1) curvas em k de popularidade e similaridade; (2) folga e aproveitamento; (3) kDN; (4) o que uma representação
deveria codificar (oráculo q); (5) o índice DR da primeira versão. Lê cache/knn_fixo_{tarefa}.csv só para conferência.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

import avaliar as A
import destilacao as D
import knn_fixo as K
import kdn as KD
import topk as T
from fig_tempo_meta import percentile

REFS = list(A.BASE_HPO)                       # as sete representações existentes; o Our fica de fora
KS = [1, 3, 5, 11, 21, 30, 60]
NDRAW = 20
NTRAIN = {"hpo": 80, "pipes": 67, "ir": 138}

TEXTO = r"""POR QUE É DIFÍCIL GERAR REPRESENTAÇÕES DE DATASETS PARA META-LEARNING?
(gerado por relatorio_neurips/metrica_dificuldade.py; só representações existentes, sem o método proposto no main)

PERGUNTA. Uma representação de datasets serve a um meta-modelo de vizinhança: ela escolhe, para cada dataset novo, os
datasets antigos cujas melhores configurações serão reaproveitadas. Medimos quanto as sete representações existentes
(pré-treinada: TabPFN; feitas à mão: PyMFE, ECoL; aprendidas: Dataset2Vec, MetaBu, Dida, TRIO) acrescentam a uma
recomendação que não usa representação nenhuma, e por quê acrescentam pouco.

PROTOCOLO. Meta-tarefas: HPO (120 datasets, 3.456 configurações; ADTM), PIPES (105 datasets, 8.873 pipelines; acurácia),
IR (215 datasets, 42 pares modelo-estratégia; F1). Cada alvo usa o embedding da partição em que é de teste; distância
euclidiana após min-max; vizinhos entre todos os outros datasets. Meta-modelo k-NN: escore da configuração c = média de
ỹ_{j,c} (linha de Y normalizada por min-max; IR: F1 bruto) nos k vizinhos; recomenda a de maior escore (top-1; empates
pelo valor esperado). Escala usada abaixo: G = valor no alvo, maior = melhor (no HPO, G = 1 - ADTM).

AS QUATRO RECOMENDAÇÕES DE REFERÊNCIA (dependem só de Y):
  constante  : k-NN com todos os outros datasets (k = n-1). Popularidade pura.
  acaso(k)   : k-NN com k datasets sorteados (média de 20 sorteios; k=1: valor esperado exato). Popularidade sem similaridade.
  oráculo(k) : k-NN com os k datasets que melhor transferem para o alvo (maior G[i,j], a melhor configuração de j avaliada
               em i). O que uma representação perfeita escolheria. Limite superior, não alcançável.
  oráculo q  : k-NN no espaço de q_i (a fração de cada valor de cada componente entre as κ melhores configurações do
               dataset, κ = 20, 10, 1). O que uma representação que codificasse q sem erro alcançaria.

MEDIDAS.
  Folga(k)        = [oráculo(k) - max(acaso(k), constante)] / [oráculo(1) - acaso(1)]   (médias sobre os alvos)
                    Quanto uma representação perfeita ganharia sobre a popularidade com aquele k, em unidades do ganho do
                    1-NN perfeito sobre o acaso. Só de Y: caracteriza a meta-tarefa.
  Aproveitamento  = [rep(k) - max(acaso(k), constante)] / [oráculo(k) - max(acaso(k), constante)]
                    Fração da folga que a representação alcança. Negativo: abaixo da popularidade.
  kDN (Smith, Martinez e Giraud-Carrier, 2014) = fração dos k vizinhos com rótulo diferente do alvo, com o rótulo de cada
                    componente = valor dominante entre as κ melhores (argmax de q_{i,h}), média sobre componentes; k = 5.
Testes: Wilcoxon pareado por dataset; Holm nas sete representações.
"""


LEITURA = r"""
LEITURA (números de 07/11/2026; conferir se o script for rodado de novo)

1. A popularidade explica quase todo o resultado. Com k vizinhos ao acaso, sem representação nenhuma, o k-NN sobe junto
   com as representações: com k=11, acaso 0,826 / 0,772 / 0,290 (HPO em 1-ADTM, PIPES, IR), contra 0,816 a 0,853,
   0,778 a 0,785 e 0,287 a 0,301 das sete representações. No PIPES e no IR nenhuma das sete supera a constante com k=11
   (aproveitamento negativo ou nulo; Holm p>=0,1, exceto TRIO abaixo dela no IR, p=0,025).

2. O espaço para uma representação depende da meta-tarefa e de k (folga). Com k=1 a folga é exatamente o E da primeira
   versão (0,85, 0,23, 0,35). Com k=11: HPO 0,32, IR 0,31, PIPES 0,05. No PIPES nem uma representação perfeita ganharia
   da popularidade ao fazer média (oráculo 0,801 contra constante 0,794): a meta-tarefa quase não deixa espaço. No HPO e
   no IR o espaço existe e não é aproveitado: com k=11, só ECoL (0,37 da folga; Holm p=0,02) e Dida (0,52; p=0,008)
   superam o acaso no HPO, e nenhuma representação passa a constante no IR.

3. O que uma representação deveria codificar existe e é simples. Usar q (a distribuição dos valores de cada componente
   entre as κ melhores configurações) como representação escolhe vizinhos no percentil 34, 26 e 4 (acaso = 50) e, com
   k=11, chega a 0,334 de F1 no IR, contra 0,300 da constante. No PIPES nem q supera a constante (0,795 contra 0,794).

4. As representações quase não organizam as vizinhanças pelo que importa (kDN, k=5). A redução sobre o acaso é de
   0,003 a 0,032 no HPO (acaso 0,572; oráculo q 0,356), 0,013 a 0,047 no PIPES (0,666; 0,512) e 0,010 a 0,122 no IR
   (0,789; 0,093). As feitas à mão e a pré-treinada reduzem tanto quanto as aprendidas.

5. As representações aprendidas perdem no teste o que organizam no treino (HPO). kDN menos o acaso, nos datasets de ajuste
   contra os de teste de cada partição: MetaBu -0,037 contra -0,011, Dida -0,030 contra -0,009, Dataset2Vec -0,010 contra
   -0,003. As que não treinam são o controle e não mostram essa queda (PyMFE -0,023/-0,025, ECoL -0,029/-0,032, TabPFN
   -0,026/-0,029). No PIPES e no IR a queda não aparece. Diferenças não testadas: falta um teste pareado por dataset.

HIPÓTESES PARA O ARTIGO (não testadas)
  - Benchmarks de meta-learning com folga pequena (como o PIPES) não conseguem separar representações; a folga deveria ser
    reportada antes de comparar representações.
  - O gargalo das representações aprendidas é a generalização a datasets novos (item 5), não a falta de informação útil
    (item 3).

FALTA PARA UM ARTIGO (IJCNN)
  - Mais meta-tarefas (6 a 8) para validar a folga como medida de dificuldade; com três, tudo acima é descrição.
  - Teste pareado do item 5 e repetição com outras sementes das representações aprendidas.
  - k escolhido na validação, ou a curva inteira; aqui k=11 foi escolhido olhando o teste.
"""


def run(task, L):
    loader, higher, _, _ = T.TASKS[task]
    names, Y, reps = loader()
    Y = np.asarray(Y, dtype=float)
    Ys = Y if task == "ir" else K.rownorm(Y)
    base = Y if higher else K.rownorm(Y)                       # G: maior melhor (HPO: ỹ = 1 - ADTM)
    G = T.transfer_topk(Y, 1)                                  # G[i, j]: valor da melhor de j em i (maior melhor)
    n = len(names)
    Q, lab = KD.labels(task, names)
    allp = np.arange(n)
    rng = np.random.default_rng(0)
    others = [np.delete(allp, i) for i in range(n)]

    cons = np.array([K.value_of_score(Ys[others[i]].mean(0), base[i]) for i in range(n)])
    val = {}                                                   # nome -> (len(KS), n)
    val["acaso"] = np.array([[np.mean([K.value_of_score(Ys[j], base[i]) for j in others[i]]) if k == 1 else
                              np.mean([K.value_of_score(Ys[rng.choice(others[i], k, replace=False)].mean(0), base[i])
                                       for _ in range(NDRAW)]) for i in range(n)] for k in KS])
    val["oráculo"] = np.array([[K.value_of_score(Ys[others[i][np.argsort(-G[i, others[i]], kind="stable")[:k]]].mean(0), base[i])
                                for i in range(n)] for k in KS])
    spaces = {r: reps[r][0] for r in REFS}
    spaces["oráculo q"] = lambda i: Q
    for name, mat_of in spaces.items():
        cache, v = {}, np.empty((len(KS), n))
        for i in range(n):
            M = mat_of(i)
            if id(M) not in cache:
                cache[id(M)] = Q if name == "oráculo q" else T.minmax(np.asarray(M, dtype=float))
            o = KD.neighbors(cache[id(M)], i, allp)
            for r, k in enumerate(KS):
                v[r, i] = K.value_of_score(Ys[o[:k]].mean(0), base[i])
        val[name] = v

    # ---- (1) curvas
    L.append(f"=== {task.upper()} ({n} datasets; N de treino = {NTRAIN[task]}); escala G (maior melhor)")
    L.append(f"  constante {cons.mean():.4f}")
    L.append("  k            " + " ".join(f"{k:>7d}" for k in KS))
    for name in ["oráculo", "oráculo q", "acaso"] + REFS:
        L.append(f"  {name:12s} " + " ".join(f"{x:7.4f}" for x in val[name].mean(1)))
    # ---- (2) folga e aproveitamento
    rnd, orc = val["acaso"].mean(1), val["oráculo"].mean(1)
    gap1 = orc[0] - rnd[0]
    pop = np.maximum(rnd, cons.mean())
    folga = (orc - pop) / gap1
    L.append("  folga        " + " ".join(f"{x:7.3f}" for x in folga))
    for r_i, k in enumerate(KS):
        if k not in (1, 11):
            continue
        basev = val["acaso"][r_i] if rnd[r_i] >= cons.mean() else cons
        bname = "acaso" if rnd[r_i] >= cons.mean() else "constante"
        ds = [val[r][r_i] - basev for r in REFS]
        hp = multipletests([wilcoxon(d).pvalue for d in ds], method="holm")[1]
        den = orc[r_i] - pop[r_i]
        L.append(f"  k={k}: aproveitamento sobre {bname} (Holm nas 7): " + " | ".join(
            f"{r} {d.mean() / den:+.3f} ({h:.2g})" for r, d, h in zip(REFS, ds, hp)))
    # ---- (3) kDN, k = 5
    dis = (lab[:, None, :] != lab[None, :, :]).mean(2)
    np.fill_diagonal(dis, np.nan)
    rk = np.nanmean(dis, 1)
    _, _, _, _, part = D.task_data(task)
    part = np.asarray(part)
    kd, gaps = {}, {}
    for name, mat_of in spaces.items():
        cache, x = {}, np.empty(n)
        for i in range(n):
            M = mat_of(i)
            if id(M) not in cache:
                cache[id(M)] = Q if name == "oráculo q" else T.minmax(np.asarray(M, dtype=float))
            x[i] = KD.kdn_from_order(lab, i, KD.neighbors(cache[id(M)], i, allp), 5)
        kd[name] = x
        if name == "oráculo q":
            continue
        tr_ex, te_ex = [], []                                  # excesso sobre o acaso: ajuste x teste da partição
        for f in sorted(set(part.tolist())):
            te, tr = np.flatnonzero(part == f), np.flatnonzero(part != f)
            Z = T.minmax(np.asarray(mat_of(int(te[0])), dtype=float))
            tr_ex += [KD.kdn_from_order(lab, i, KD.neighbors(Z, i, tr), 5) - np.nanmean(dis[i, tr]) for i in tr]
            te_ex += [KD.kdn_from_order(lab, i, KD.neighbors(Z, i, allp), 5) - rk[i] for i in te]
        gaps[name] = (np.mean(tr_ex), np.mean(te_ex))
    ds = [rk - kd[r] for r in REFS]
    hp = multipletests([wilcoxon(d).pvalue for d in ds], method="holm")[1]
    L.append(f"  kDN (k=5): acaso {rk.mean():.3f} | oráculo q {kd['oráculo q'].mean():.3f} | " + " | ".join(
        f"{r} {kd[r].mean():.3f}" for r in REFS))
    L.append("    redução sobre o acaso (Holm nas 7): " + " | ".join(
        f"{r} {d.mean():+.3f} ({h:.2g})" for r, d, h in zip(REFS, ds, hp)))
    L.append("    kDN menos o acaso, datasets de ajuste / de teste de cada partição: " + " | ".join(
        f"{r} {a:+.3f}/{b:+.3f}" for r, (a, b) in gaps.items()))
    # ---- (4) oráculo q no 1-NN: percentil do vizinho
    R = -G
    nnq = np.array([KD.neighbors(Q, i, allp)[0] for i in range(n)])
    pq = percentile(R, nnq)
    L.append(f"  oráculo q, 1-NN: percentil do vizinho {pq.mean():.1f} (acaso = 50)")
    L.append("")
    return dict(folga=folga, cons=cons.mean(), rnd=rnd, orc=orc)


def dr_index(task):
    """Índice DR da primeira versão (E * R * I / log2 N), mantido para comparação."""
    names, Y, _ = T.TASKS[task][0]()
    G = T.transfer_topk(np.asarray(Y, dtype=float), 1).copy()
    np.fill_diagonal(G, np.nan)
    df = pd.read_csv(A.CACHE / f"knn_fixo_{task}.csv", header=[0, 1], index_col=0)
    higher = T.TASKS[task][1]
    cons = df[("constante", "0")].to_numpy()
    cons = cons if higher else 1.0 - cons
    rand, orac = np.nanmean(G, 1), np.nanmax(G, 1)
    gap = orac - rand
    E = (orac - cons).mean() / gap.mean()
    good = np.array([np.mean(G[i][~np.isnan(G[i])] >= orac[i] - 0.1 * gap[i]) for i in range(len(G))])
    Rr = 1 - good[gap > 1e-12].mean()
    m = np.nanmean(G)
    a = np.nanmean(G, 1, keepdims=True) - m
    b = np.nanmean(G, 0, keepdims=True) - m
    e = G - m - a - b
    I = np.nanvar(e) / (np.nanvar(b) + np.nanvar(e))
    return E, Rr, I, E * Rr * I / np.log2(NTRAIN[task])


def main():
    L = [TEXTO, "RESULTADOS", ""]
    out = {}
    for task in ("hpo", "pipes", "ir"):
        out[task] = run(task, L)
    L.append("RESUMO DA FOLGA (maior = mais espaço para uma representação sobre a popularidade)")
    L.append("         " + " ".join(f"k={k:<5d}" for k in KS))
    for t in out:
        L.append(f"  {t.upper():6s} " + " ".join(f"{x:7.3f}" for x in out[t]["folga"]))
    L.append("")
    L.append("ÍNDICE DR DA PRIMEIRA VERSÃO (E * R * I / log2 N; substituído pela folga, que tem interpretação direta)")
    for t in out:
        E, Rr, I, DR = dr_index(t)
        L.append(f"  {t.upper():6s} E {E:.3f} | R {Rr:.3f} | I {I:.3f} | DR {DR:.4f}")
    L.append(LEITURA)
    (A.HERE / "METRIC.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[L.index("RESULTADOS"):]))


if __name__ == "__main__":
    main()
