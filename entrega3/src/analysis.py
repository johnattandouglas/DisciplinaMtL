"""Análises da Seção 2.1: curvas de perda, AUC, Friedman + diferença crítica (Nemenyi), papel das meta-características
e falhas do warm-start.

Convenções: as curvas de cada método têm forma [n_ds, S, T] (perda normalizada da melhor configuração até a i-ésima
avaliação; MENOR = melhor). Para os testes, as sementes são agregadas pela média, por dataset, e os datasets são os blocos.
"""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, spearmanr, studentized_range, wilcoxon
from statsmodels.stats.multitest import multipletests

from . import config as cfg
from . import search

ALPHA = 0.05

# ---------------------------------------------------------------------------
# Cores: fixas por método em todas as figuras (paleta categórica validada; baselines sem busca em tinta neutra)
# ---------------------------------------------------------------------------
WS_COLOR = "#2a78d6"                           # SMBO-Similarity (método 4); k distingue pelo traço
FIXED_STYLE = {
    "smbo_ws_avgrank": {"color": "#008300", "ls": "-", "lw": 2.0},
    "smbo_cold": {"color": "#0b0b0b", "ls": "-", "lw": 1.6},
    "random_search": {"color": "#52514e", "ls": "--", "lw": 1.4},
    "default": {"color": "#898781", "ls": ":", "lw": 1.6},
}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

# Nomes dos métodos nas FIGURAS (em inglês); tabelas e notebooks usam os nomes de `search.method_label`
FIG_NAMES = {"Configuração padrão": "Default configuration"}


def fig_name(label: str) -> str:
    """Nome do método como aparece nas figuras (em inglês)."""
    return FIG_NAMES.get(label, label)


def style_of(key: str) -> dict:
    if key in FIXED_STYLE:
        return FIXED_STYLE[key]
    k = key.rsplit("_k", 1)[1]
    return {"color": WS_COLOR, "ls": {"1": "-", "3": "--", "5": ":"}.get(k, "-"), "lw": 2.0}


def setup_matplotlib() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 220, "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "legend.frameon": False, "axes.titlecolor": INK, "text.color": INK,
    })


# ---------------------------------------------------------------------------
# Tabelas básicas
# ---------------------------------------------------------------------------
def dataset_table(meta, L: np.ndarray, default_idx: int) -> pd.DataFrame:
    """Uma linha por dataset: tamanho, classes, L*, pior perda, fração de configurações no ótimo, resolução da perda e a
    perda normalizada da configuração padrão."""
    summ = meta.summary()
    R = search.normalized_regret(L)
    return pd.DataFrame({
        "n_instancias": summ["n_instancias"].to_numpy(), "n_features": summ["n_features"].to_numpy(),
        "n_classes": summ["n_classes"].to_numpy(), "L*": L.min(1), "L pior": L.max(1),
        "frac. configs no ótimo": (L == L.min(1, keepdims=True)).mean(1),
        "valores distintos de perda": [len(np.unique(r)) for r in L],
        "perda norm. do padrão": R[:, default_idx],
    }, index=pd.Index(meta.names, name="dataset"))



def load_curves(ctx: dict, keys: list[str]) -> dict[str, np.ndarray]:
    """{método: curvas [n_ds, S, T]} lidas de results/curvas/ (falha se algum método não foi rodado)."""
    out, missing = {}, []
    for k in keys:
        res = search.load_method(k, ctx)
        if res is None:
            missing.append(k)
        else:
            out[k] = res["curves"].astype(np.float64)
    if missing:
        raise FileNotFoundError(f"Métodos sem curvas em {cfg.CURVES_DIR}: {missing}. Rode `python 03_rodar_buscas.py`.")
    return out


def per_dataset(curves: np.ndarray) -> np.ndarray:
    """[n_ds, T]: média sobre as sementes."""
    return curves.mean(axis=1)


def auc(curves: np.ndarray) -> np.ndarray:
    """Área sob a curva normalizada (média da perda normalizada sobre i = 1..T), por dataset: [n_ds] em [0, 1]."""
    return per_dataset(curves).mean(axis=1)


def values_at(curves_by_key: dict, names: list[str], at) -> pd.DataFrame:
    """(dataset x método): perda normalizada em t=`at` (1-indexado) ou a AUC (`at='AUC'`), média sobre sementes."""
    data = {}
    for k, c in curves_by_key.items():
        data[search.method_label(k)] = auc(c) if at == "AUC" else per_dataset(c)[:, at - 1]
    return pd.DataFrame(data, index=names)


def summary_table(curves_by_key: dict, names: list[str], checkpoints=cfg.CHECKPOINTS) -> pd.DataFrame:
    """(método x orçamento): perda normalizada média (datasets e sementes) em cada t e AUC, mais o rank médio da AUC."""
    cols = {f"t={t}": values_at(curves_by_key, names, t).mean() for t in checkpoints}
    auc_df = values_at(curves_by_key, names, "AUC")
    cols["AUC"] = auc_df.mean()
    cols["rank médio (AUC)"] = rank_frame(auc_df).mean()
    return pd.DataFrame(cols)


def rank_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Rank por dataset (1 = melhor = menor perda); empates recebem o rank médio."""
    return data.rank(axis=1, method="average", ascending=True)


def average_rank_per_trial(curves_by_key: dict, names: list[str]) -> pd.DataFrame:
    """(t x método): rank médio entre os métodos a cada avaliação (métrica do paper do D2V)."""
    T = next(iter(curves_by_key.values())).shape[-1]
    return pd.DataFrame({t: rank_frame(values_at(curves_by_key, names, t)).mean() for t in range(1, T + 1)}).T


# ---------------------------------------------------------------------------
# Testes: Friedman + Nemenyi (diferença crítica) e Wilcoxon pareado com Holm
# ---------------------------------------------------------------------------
def friedman(data: pd.DataFrame) -> tuple[float, float]:
    stat, p = friedmanchisquare(*[data[c].to_numpy() for c in data.columns])
    return float(stat), float(p)


def nemenyi_cd(k: int, n: int, alpha: float = ALPHA) -> float:
    """Diferença crítica de Nemenyi (Demšar, 2006): q_alpha * sqrt(k(k+1) / (6N))."""
    q = studentized_range.ppf(1 - alpha, k, np.inf) / math.sqrt(2)
    return float(q * math.sqrt(k * (k + 1) / (6.0 * n)))


def friedman_nemenyi(data: pd.DataFrame, alpha: float = ALPHA) -> dict:
    stat, p = friedman(data)
    ranks = rank_frame(data).mean().sort_values()
    return {"friedman_stat": stat, "friedman_p": p, "cd": nemenyi_cd(data.shape[1], len(data), alpha),
            "ranks": ranks, "N": len(data), "k": data.shape[1]}


def wilcoxon_p(a, b) -> float:
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return 1.0 if np.allclose(d, 0, atol=1e-12) else float(wilcoxon(d).pvalue)


def paired_tests(data: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """Para cada par (A, B): diferença média B - A (> 0 => A melhor), vitórias/empates/derrotas de A, Wilcoxon e Holm."""
    rows = []
    for a, b in pairs:
        x, y = data[a].to_numpy(float), data[b].to_numpy(float)
        d = y - x
        rows.append({"A": a, "B": b, "média A": x.mean(), "média B": y.mean(), "B − A": d.mean(),
                     "A melhor": int((d > 1e-12).sum()), "empates": int((np.abs(d) <= 1e-12).sum()),
                     "A pior": int((d < -1e-12).sum()), "p (Wilcoxon)": wilcoxon_p(x, y)})
    df = pd.DataFrame(rows)
    if len(df):
        df["p (Holm)"] = multipletests(df["p (Wilcoxon)"], method="holm")[1]
        df["signif."] = df["p (Holm)"] < ALPHA
    return df


def plot_cd(res: dict, ax, title: str = "", label_map: dict | None = None):
    """Diagrama de diferença crítica (estilo Demšar/Orange): eixo de rank médio (1 = melhor, à esquerda), barras grossas
    ligando métodos cuja diferença de rank médio não excede a CD de Nemenyi."""
    ranks = res["ranks"]
    names = [(label_map or FIG_NAMES).get(n, n) for n in ranks.index]
    r = ranks.to_numpy()
    k, cd = len(r), res["cd"]
    lo, hi = math.floor(r.min()), math.ceil(r.max())
    hi = max(hi, lo + 1)
    half = math.ceil(k / 2)
    text_w = (hi - lo) * 0.8           # espaço para os nomes dentro do painel (não invadem o painel vizinho)
    ax.set_xlim(lo - text_w, hi + text_w)
    ax.axis("off")
    ax.grid(False)
    ax.plot([lo, hi], [0, 0], color=INK, lw=0.9)
    for v in range(lo, hi + 1):
        ax.plot([v, v], [0, -0.1], color=INK, lw=0.9)
        ax.text(v, -0.17, str(v), ha="center", va="bottom", fontsize=8, color=INK2)
    ax.plot([lo, lo + cd], [-0.55, -0.55], color=INK, lw=1.2)
    for x in (lo, lo + cd):
        ax.plot([x, x], [-0.6, -0.5], color=INK, lw=1.2)
    ax.text(lo + cd / 2, -0.62, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
    # cliques (grupos sem diferença significativa)
    cliques, last_j = [], -1
    for i in range(k):
        j = int(np.max(np.flatnonzero(r - r[i] <= cd)))
        if j > i and j > last_j:
            cliques.append((i, j))
            last_j = j
    for c, (i, j) in enumerate(cliques):
        y = 0.18 + 0.11 * c
        ax.plot([r[i] - 0.03, r[j] + 0.03], [y, y], color=INK, lw=2.6, solid_capstyle="round")
    base = 0.3 + 0.11 * max(len(cliques), 1)
    ax.set_ylim(base + 0.32 * (max(half, k - half) - 1) + 0.2, -0.75)
    for i in range(k):
        if i < half:
            y = base + 0.32 * i
            ax.plot([r[i], r[i], lo - 0.05], [0, y, y], color=INK2, lw=0.8)
            ax.text(lo - 0.1, y, f"{names[i]} ({r[i]:.2f})", ha="right", va="center", fontsize=8)
        else:
            y = base + 0.32 * (k - 1 - i)
            ax.plot([r[i], r[i], hi + 0.05], [0, y, y], color=INK2, lw=0.8)
            ax.text(hi + 0.1, y, f"({r[i]:.2f}) {names[i]}", ha="left", va="center", fontsize=8)
    if title:
        ax.set_title(title, fontsize=9, loc="left", color=INK)
    return ax


# ---------------------------------------------------------------------------
# Fase inicial (só o desenho do warm-start, sem SMBO): desenhos para qualquer k, oráculo e vizinho aleatório
# ---------------------------------------------------------------------------
def config_orders(L: np.ndarray, n_seeds: int) -> np.ndarray:
    """[S, n_ds, K]: configurações de cada dataset em ordem crescente de perda, empates sorteados por semente."""
    out = np.zeros((n_seeds,) + L.shape, dtype=np.int64)
    for s in range(n_seeds):
        rng = np.random.default_rng([cfg.SEED, 7919, s])
        for j in range(L.shape[0]):
            out[s, j] = search.sorted_configs(L[j], rng)
    return out


def _design_from_orders(orders_s: np.ndarray, neighbors: np.ndarray, k: int, n_init: int) -> np.ndarray:
    chosen, seen, r = [], set(), 0
    while len(chosen) < n_init:
        for j in neighbors[:k]:
            c = int(orders_s[j, r])
            if c not in seen:
                chosen.append(c)
                seen.add(c)
                if len(chosen) == n_init:
                    break
        r += 1
    return np.asarray(chosen)


def init_regret_vs_k(ctx: dict, orders: np.ndarray, ks=range(1, 21)) -> pd.DataFrame:
    """(dataset x k): perda normalizada após os n_init pontos do warm-start por similaridade com k vizinhos."""
    R, n_init, nb = ctx["R"], ctx["n_init"], ctx["neighbors"]
    out = np.zeros((R.shape[0], len(ks)))
    for t in range(R.shape[0]):
        for c, k in enumerate(ks):
            out[t, c] = np.mean([R[t, _design_from_orders(orders[s], nb[t], k, n_init)].min()
                                 for s in range(orders.shape[0])])
    return pd.DataFrame(out, index=ctx["names"], columns=list(ks))


def transfer_matrix(ctx: dict, orders: np.ndarray) -> np.ndarray:
    """M[i, j] = perda normalizada em i depois de avaliar as n_init melhores configurações de j (média sobre sementes).
    Diagonal = NaN. É o "valor de usar j como único vizinho" de i: oráculo = min_j, vizinho aleatório = média_j."""
    R, n_init = ctx["R"], ctx["n_init"]
    M = np.zeros((R.shape[0], R.shape[0]))
    for s in range(orders.shape[0]):
        top = orders[s][:, :n_init]                       # [n_ds, n_init]
        M += R[:, top].min(axis=2)                        # [n_i, n_j]
    M /= orders.shape[0]
    np.fill_diagonal(M, np.nan)
    return M


def neighbor_quality(ctx: dict, M: np.ndarray) -> pd.DataFrame:
    """Vizinho escolhido pelo PyMFE (k=1): perda após o warm-start, percentil desse vizinho entre os 119 candidatos
    (0 = o melhor vizinho possível, 50 = um vizinho qualquer) e correlação de Spearman, por alvo, entre a distância no
    espaço PyMFE e a perda transferida M[i, ·] (> 0: mais perto => configurações mais úteis). Inclui as referências
    oráculo (melhor vizinho possível) e vizinho aleatório (média sobre os 119)."""
    n = M.shape[0]
    nb, dist = ctx["neighbors"], ctx["distances"]
    chosen, pct, rho = [], [], []
    for i in range(n):
        cand = np.delete(M[i], i)
        v = M[i, nb[i, 0]]
        chosen.append(v)
        pct.append(100.0 * ((cand < v).sum() + 0.5 * ((cand == v).sum() - 1)) / (len(cand) - 1))
        m = M[i, nb[i]]
        rho.append(np.nan if np.ptp(m) == 0 else spearmanr(dist[i], m).statistic)   # alvo trivial: todo vizinho serve
    rho = np.asarray(rho)
    rows = {cfg.REP: {"perda após n_init (vizinho escolhido)": np.mean(chosen),
                      "percentil do vizinho (0 = melhor)": np.mean(pct),
                      "ρ(distância, perda transferida) médio": np.nanmean(rho),
                      "p (Wilcoxon ρ ≠ 0)": float(wilcoxon(rho[~np.isnan(rho)]).pvalue)}}
    rows["Oráculo (melhor vizinho)"] = {"perda após n_init (vizinho escolhido)": np.nanmin(M, axis=1).mean(),
                                        "percentil do vizinho (0 = melhor)": 0.0}
    rows["Vizinho aleatório"] = {"perda após n_init (vizinho escolhido)": np.nanmean(M, axis=1).mean(),
                                 "percentil do vizinho (0 = melhor)": 50.0}
    return pd.DataFrame(rows).T


def landscape_similarity(L: np.ndarray) -> np.ndarray:
    """ρ de Spearman entre as perdas das K configurações de cada par de datasets (similaridade "verdadeira")."""
    from scipy.stats import rankdata

    Rk = rankdata(L, axis=1)
    Rk = (Rk - Rk.mean(axis=1, keepdims=True)) / Rk.std(axis=1, keepdims=True)
    S = Rk @ Rk.T / L.shape[1]
    np.fill_diagonal(S, np.nan)
    return S
