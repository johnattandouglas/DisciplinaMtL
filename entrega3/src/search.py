"""Etapa 2 — buscas simuladas sobre a matriz de perdas (métodos 1 a 5 do enunciado).

Avaliar uma configuração = consultar uma célula da matriz de perdas L (n_ds x K; perda = 1 - acurácia de validação na
época 50). Toda busca registra a perda NORMALIZADA da melhor configuração encontrada até a i-ésima avaliação:

    r_i(d) = (min_{j<=i} L[d, θ_j] - L*_d) / (L^pior_d - L*_d)        (L*_d, L^pior_d = mínimo e máximo da linha d)

A média de r_i sobre datasets é o ADTM (average distance to the minimum) usado no paper do Dataset2Vec.

Métodos (os três SMBO usam o MESMO substituto — GP Matérn 3/2 com ARD — e a MESMA aquisição — EI):
  1. default            configuração padrão (fixa, sem busca)
  2. random_search      permutação aleatória das K configurações (sem reposição)
  3. smbo_cold          n_init configurações aleatórias + SMBO
  4. smbo_ws_<rep>_k<k> n_init configurações iniciais dos k datasets mais próximos de d_t no espaço da representação
                        <rep>, em rodízio (a melhor de cada vizinho, depois a 2ª melhor, ...). Com k=1 é o protocolo do
                        paper do D2V (as n_init melhores do vizinho mais próximo); com k >= n_init, o de Feurer et al.
                        (2014) (a melhor configuração de cada um dos n_init vizinhos mais próximos)
  5. smbo_ws_avgrank    as n_init configurações de melhor rank médio sobre os datasets de treino (não olha para d_t)

Protocolo leave-one-dataset-out: para o alvo d_t, a linha de d_t sai da matriz de perdas usada para montar o warm-start
(vizinhos e ranks médios vêm só dos outros 119) e a preparação das meta-características (descarte de colunas, imputação e
min-max) é ajustada só nos outros 119. As representações aprendidas (D2V, Our-Trained) usam o extrator do split em que
d_t é de teste: ele nunca viu d_t, nem no treino nem na seleção de época.

Empates (muitos datasets têm várias configurações com a mesma acurácia de validação) são desfeitos por sorteio com a
semente da execução, tanto na ordenação das configurações de um vizinho quanto no argmax da aquisição.
"""
from __future__ import annotations

import json
import time
import warnings
import zlib

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from . import config as cfg


# ---------------------------------------------------------------------------
# Matriz de perdas, espaço de configurações e configuração padrão
# ---------------------------------------------------------------------------
def loss_matrix(meta) -> np.ndarray:
    """(n_ds x K): perda = 1 - acurácia de validação na época 50."""
    return 1.0 - meta.Y_valid.astype(np.float64)


def normalized_regret(L: np.ndarray) -> np.ndarray:
    """(L - L*) / (L^pior - L*) por dataset (linha); 0 = ótimo do grid, 1 = pior configuração do grid."""
    lo, hi = L.min(axis=1, keepdims=True), L.max(axis=1, keepdims=True)
    return (L - lo) / np.where(hi - lo > 0, hi - lo, 1.0)


def encode_config_space(configs: pd.DataFrame) -> np.ndarray:
    """One-hot das categóricas + min-max das numéricas (15 dimensões) — entrada do GP."""
    X = pd.get_dummies(configs[cfg.CONFIG_COLS]).astype(float).to_numpy()
    lo, span = X.min(axis=0), X.max(axis=0) - X.min(axis=0)
    span[span == 0] = 1.0
    return (X - lo) / span


def default_config_index(configs: pd.DataFrame, default: dict = cfg.DEFAULT_CONFIG) -> int:
    mask = np.ones(len(configs), dtype=bool)
    for col, val in default.items():
        mask &= (configs[col].astype(str) == str(val)).to_numpy()
    idx = np.flatnonzero(mask)
    if len(idx) != 1:
        raise ValueError(f"A configuração padrão {default} casa com {len(idx)} linhas do grid.")
    return int(idx[0])


def config_rank_matrix(L: np.ndarray) -> np.ndarray:
    """Rank de cada configuração dentro de cada dataset (1 = menor perda; empates = rank médio)."""
    return rankdata(L, axis=1, method="average")


# ---------------------------------------------------------------------------
# Representações (meta-características) e vizinhos, sob leave-one-dataset-out
# ---------------------------------------------------------------------------
def load_representations(names: list[str], reps=cfg.REP_ALL) -> dict:
    """{rep: matriz crua (n_ds x d)} ou {rep: {split: matriz crua}} para as representações por split."""
    out = {}
    for rep in reps:
        if rep in cfg.REP_BY_SPLIT:
            out[rep] = {}
            for k in range(cfg.N_SPLITS):
                path = cfg.REPR_DIR / (cfg.REP_BY_SPLIT[rep].format(k=k) + ".csv")
                df = pd.read_csv(path, index_col=0).drop(columns="role", errors="ignore").loc[names]
                out[rep][k] = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        else:
            df = pd.read_csv(cfg.REPR_DIR / f"{rep}.csv", index_col=0).loc[names]
            out[rep] = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    return out


def lodo_normalize(F: np.ndarray, target: int, max_missing: float = 0.05) -> np.ndarray:
    """Prepara as meta-características SEM olhar para o alvo: descarta colunas com > `max_missing` de NaN/inf nos
    outros datasets, imputa a mediana deles, descarta colunas constantes neles e aplica o min-max ajustado neles."""
    F = np.where(np.isfinite(F), F, np.nan)
    train = np.ones(F.shape[0], dtype=bool)
    train[target] = False
    F = F[:, np.isnan(F[train]).mean(axis=0) <= max_missing]
    med = np.nanmedian(F[train], axis=0)
    F = np.where(np.isnan(F), med, F)
    lo, hi = F[train].min(axis=0), F[train].max(axis=0)
    keep = hi - lo > 1e-12
    return (F[:, keep] - lo[keep]) / (hi[keep] - lo[keep])


def neighbor_table(raw, test_split_of: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Para cada alvo: os outros datasets em ordem crescente de distância euclidiana, as distâncias e a dimensão usada.

    `raw` é uma matriz (representação fixa) ou {split: matriz} (o alvo usa a do split em que ele é de teste).
    Retorna (ordem [n, n-1], distâncias [n, n-1], dimensão [n])."""
    n = len(test_split_of)
    order = np.zeros((n, n - 1), dtype=np.int64)
    dist = np.zeros((n, n - 1))
    dims = np.zeros(n, dtype=np.int64)
    for t in range(n):
        F = raw[int(test_split_of[t])] if isinstance(raw, dict) else raw
        Z = lodo_normalize(F, t)
        d = np.linalg.norm(Z - Z[t], axis=1)
        d[t] = np.inf
        o = np.argsort(d, kind="stable")[: n - 1]
        order[t], dist[t], dims[t] = o, d[o], Z.shape[1]
    return order, dist, dims


# ---------------------------------------------------------------------------
# Desenhos iniciais (warm-start)
# ---------------------------------------------------------------------------
def sorted_configs(loss_row: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Índices das configurações em ordem crescente de perda; empates desfeitos por sorteio."""
    return np.lexsort((rng.random(len(loss_row)), loss_row))


def knn_design(L: np.ndarray, neighbors: np.ndarray, k: int, n_init: int, rng: np.random.Generator) -> np.ndarray:
    """Rodízio entre os k vizinhos mais próximos: a melhor configuração de cada um (em ordem de distância), depois a
    2ª melhor de cada um, ... pulando repetidas, até completar n_init."""
    orders = [sorted_configs(L[j], rng) for j in neighbors[:k]]
    chosen, seen, r = [], set(), 0
    while len(chosen) < n_init:
        for o in orders:
            c = int(o[r])
            if c not in seen:
                chosen.append(c)
                seen.add(c)
                if len(chosen) == n_init:
                    break
        r += 1
    return np.asarray(chosen, dtype=np.int64)


def avgrank_design(ranks: np.ndarray, target: int, n_init: int, rng: np.random.Generator) -> np.ndarray:
    """As n_init configurações de menor rank médio nos datasets de treino (todos menos o alvo)."""
    mean_rank = (ranks.sum(axis=0) - ranks[target]) / (ranks.shape[0] - 1)
    return sorted_configs(mean_rank, rng)[:n_init]


# ---------------------------------------------------------------------------
# SMBO: GP (Matérn 3/2, ARD) + Expected Improvement, sobre as K configurações (sem reposição)
# ---------------------------------------------------------------------------
def _gp_predict(Xo, yo, Xall, state):
    """GP com hiperparâmetros reotimizados a cada iteração (partindo da solução anterior). Retorna (mu, sd, kernel)."""
    if state is None:
        kernel = (ConstantKernel(1.0, (1e-2, 1e2))
                  * Matern(length_scale=np.ones(Xo.shape[1]), length_scale_bounds=(1e-2, 1e2), nu=1.5)
                  + WhiteKernel(1e-2, (1e-6, 1.0)))
    else:
        kernel = state
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit(Xo, yo)
        mu, sd = gp.predict(Xall, return_std=True)
    # variância preditiva da função latente (sem o ruído de observação), como no código oficial do D2V
    y_std = float(np.ravel(getattr(gp, "_y_train_std", 1.0))[0])
    noise = float(gp.kernel_.k2.noise_level) * y_std ** 2
    sd = np.sqrt(np.maximum(sd ** 2 - noise, 1e-12))
    return mu, sd, gp.kernel_


def expected_improvement(mu: np.ndarray, sd: np.ndarray, best: float) -> np.ndarray:
    """EI para MINIMIZAÇÃO da perda."""
    sd = np.maximum(sd, 1e-9)
    imp = best - mu
    z = imp / sd
    return imp * norm.cdf(z) + sd * norm.pdf(z)


def run_smbo(Xenc: np.ndarray, loss_row: np.ndarray, init_idx: np.ndarray, T: int,
             rng: np.random.Generator) -> np.ndarray:
    """Avalia `init_idx` (em ordem) e segue com GP+EI até T avaliações. Para antes se o ótimo do grid já foi achado
    (a curva de melhor-até-agora não muda mais; isso só economiza tempo). Retorna as configurações avaliadas, em ordem."""
    observed = [int(i) for i in init_idx[:T]]
    seen = np.zeros(len(loss_row), dtype=bool)
    seen[observed] = True
    best_possible = float(loss_row.min())
    state = None
    while len(observed) < T and loss_row[observed].min() > best_possible + 1e-12:
        mu, sd, state = _gp_predict(Xenc[observed], loss_row[observed], Xenc, state)
        ei = expected_improvement(mu, sd, float(loss_row[observed].min()))
        ei[seen] = -np.inf
        cand = np.flatnonzero(ei >= ei.max() - 1e-12)     # empates (inclusive EI ~ 0) -> sorteio
        nxt = int(rng.choice(cand))
        observed.append(nxt)
        seen[nxt] = True
    return np.asarray(observed, dtype=np.int64)


def incumbent_curve(r_row: np.ndarray, evaluated: np.ndarray, T: int) -> np.ndarray:
    """Perda normalizada da melhor configuração até a i-ésima avaliação (i = 1..T)."""
    inc = np.minimum.accumulate(r_row[evaluated])
    if len(inc) < T:
        inc = np.concatenate([inc, np.full(T - len(inc), inc[-1])])
    return inc[:T]


# ---------------------------------------------------------------------------
# Especificação dos métodos
# ---------------------------------------------------------------------------
def method_key(kind: str, rep: str | None = None, k: int | None = None) -> str:
    return f"smbo_ws_{rep}_k{k}" if kind == "ws_knn" else kind


def method_label(key: str) -> str:
    fixed = {"default": "Padrão", "random_search": "Random Search", "smbo_cold": "SMBO frio",
             "smbo_ws_avgrank": "WS independente"}
    if key in fixed:
        return fixed[key]
    rep, k = key.removeprefix("smbo_ws_").rsplit("_k", 1)
    return f"WS-{rep} (k={k})"


def all_method_keys(reps=cfg.REP_ALL, ks=cfg.K_NEIGHBORS) -> list[str]:
    return (["default", "random_search", "smbo_cold", "smbo_ws_avgrank"]
            + [method_key("ws_knn", r, k) for r in reps for k in ks])


def _rng(key: str, target: int, seed: int) -> np.random.Generator:
    return np.random.default_rng([cfg.SEED, seed, target, zlib.crc32(key.encode())])


def initial_design(key: str, target: int, seed: int, ctx: dict) -> np.ndarray | None:
    """Pontos iniciais do método `key` para o alvo (None para os métodos sem SMBO)."""
    rng = _rng(key, target, seed)
    n_init, K = ctx["n_init"], ctx["L"].shape[1]
    if key == "smbo_cold":
        return rng.choice(K, size=n_init, replace=False)
    if key == "smbo_ws_avgrank":
        return avgrank_design(ctx["ranks"], target, n_init, rng)
    if key.startswith("smbo_ws_"):
        rep, k = key.removeprefix("smbo_ws_").rsplit("_k", 1)
        return knn_design(ctx["L"], ctx["neighbors"][rep][target], int(k), n_init, rng)
    return None


def run_target(key: str, target: int, loss_row: np.ndarray, r_row: np.ndarray, Xenc: np.ndarray,
               inits: np.ndarray | None, T: int, n_seeds: int, default_idx: int) -> dict:
    """Roda o método `key` no alvo `target` para todas as sementes. `inits` = desenhos iniciais [S, n_init] (só SMBO).
    Retorna curvas [S, T], avaliadas [S, T] (-1 = não avaliada porque o ótimo já tinha sido achado) e segundos."""
    curves = np.zeros((n_seeds, T), dtype=np.float32)
    evaluated = np.full((n_seeds, T), -1, dtype=np.int16)
    t0 = time.perf_counter()
    for s in range(n_seeds):
        if key == "default":
            ev = np.array([default_idx])
        elif key == "random_search":
            ev = _rng(key, target, s).permutation(len(loss_row))[:T]
        else:
            ev = run_smbo(Xenc, loss_row, inits[s], T, _rng(key + "/smbo", target, s))
        curves[s] = incumbent_curve(r_row, ev, T)
        evaluated[s, : len(ev)] = ev
    return {"curves": curves, "evaluated": evaluated, "seconds": time.perf_counter() - t0}


# ---------------------------------------------------------------------------
# Contexto + execução em lote com cache em results/curvas/<método>.npz
# ---------------------------------------------------------------------------
def build_context(meta, reps=cfg.REP_ALL, T: int = cfg.T_BUDGET, n_init: int = cfg.N_INIT,
                  n_seeds: int = cfg.N_SEEDS) -> dict:
    L = loss_matrix(meta)
    test_split_of = meta.test_split_of()
    raw = load_representations(meta.names, reps)
    neighbors, distances, dims = {}, {}, {}
    for rep in reps:
        neighbors[rep], distances[rep], dims[rep] = neighbor_table(raw[rep], test_split_of)
    return {"names": list(meta.names), "L": L, "R": normalized_regret(L), "ranks": config_rank_matrix(L),
            "Xenc": encode_config_space(meta.configs), "default_idx": default_config_index(meta.configs),
            "test_split_of": test_split_of, "neighbors": neighbors, "distances": distances, "dims": dims,
            "T": T, "n_init": n_init, "n_seeds": n_seeds}


def _curves_path(key: str):
    return cfg.CURVES_DIR / f"{key}.npz"


def _ctx_signature(ctx: dict) -> dict:
    return {"T": ctx["T"], "n_init": ctx["n_init"], "n_seeds": ctx["n_seeds"], "n_ds": len(ctx["names"]),
            "default_idx": ctx["default_idx"]}


def load_method(key: str, ctx: dict) -> dict | None:
    path = _curves_path(key)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as f:
        info = json.loads(str(f["info"]))
        if info.get("signature") != _ctx_signature(ctx) or list(f["names"]) != ctx["names"]:
            return None
        return {"curves": f["curves"], "evaluated": f["evaluated"], "inits": f["inits"],
                "seconds": f["seconds"], "info": info}


def run_method(key: str, ctx: dict, n_jobs: int = -2, force: bool = False, verbose: bool = True) -> dict:
    """Roda (ou carrega do cache) o método em todos os alvos e sementes. Curvas: [n_ds, S, T]."""
    if not force:
        cached = load_method(key, ctx)
        if cached is not None:
            return cached
    from joblib import Parallel, delayed

    cfg.make_dirs()
    t_wall = time.perf_counter()
    n, S, T = len(ctx["names"]), ctx["n_seeds"], ctx["T"]
    common = {"Xenc": ctx["Xenc"], "T": T, "n_seeds": S, "default_idx": ctx["default_idx"]}
    if key in ("default", "random_search"):
        inits = np.full((n, S, ctx["n_init"]), -1, dtype=np.int16)
        rows = [run_target(key, t, ctx["L"][t], ctx["R"][t], inits=None, **common) for t in range(n)]
    else:
        # desenhos iniciais no processo principal (baratos); os workers só recebem o que o alvo precisa
        inits = np.stack([[initial_design(key, t, s, ctx) for s in range(S)] for t in range(n)]).astype(np.int16)
        rows = Parallel(n_jobs=n_jobs)(
            delayed(run_target)(key, t, ctx["L"][t], ctx["R"][t], inits=inits[t], **common) for t in range(n))
    t_wall = time.perf_counter() - t_wall
    out = {"curves": np.stack([r["curves"] for r in rows]), "evaluated": np.stack([r["evaluated"] for r in rows]),
           "inits": inits, "seconds": np.array([r["seconds"] for r in rows])}
    info = {"key": key, "label": method_label(key), "signature": _ctx_signature(ctx), "wall_s": t_wall,
            "serial_s": float(out["seconds"].sum())}
    np.savez_compressed(_curves_path(key), names=np.array(ctx["names"]), info=json.dumps(info), **out)
    out["info"] = info
    if verbose:
        print(f"[{method_label(key)}] {t_wall:.0f}s (serial {info['serial_s']:.0f}s) | "
              f"perda normalizada média em t={ctx['n_init']}: {out['curves'][:, :, ctx['n_init'] - 1].mean():.4f} | "
              f"t={ctx['T']}: {out['curves'][:, :, -1].mean():.4f}")
    return out


def run_all(ctx: dict, keys: list[str] | None = None, n_jobs: int = -2, force: bool = False) -> dict:
    keys = keys or all_method_keys(list(ctx["neighbors"]))
    return {k: run_method(k, ctx, n_jobs=n_jobs, force=force) for k in keys}
