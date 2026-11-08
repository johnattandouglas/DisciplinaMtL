"""Métricas de regressão desbalanceada usadas na matriz da IR (Avelino et al., 2025), portadas para numpy.

A matriz `meta_ir/base/F1_score.csv` saiu do notebook `Meta_IR.ipynb` (github.com/JusciAvelino/Meta-IR), método `scores`:

    ph   = uba::phi.control(y)                # y = alvo do dataset INTEIRO (não só do treino)
    ls   = uba::loss.control(y)
    F1   = uba::util(y_pred, y_test, ph, ls, uba::util.control(umetric="Fm", beta=1, event_thr=0.8))
    SERA = IRon::sera(y_test, y_pred, phi.trues=uba::phi(y_test, ph))

O R não tem o `uba` instalado nesta máquina; este módulo reproduz o código C do pacote (github.com/rpribeiro/uba, arquivos
src/phi.c, src/pchip.c, src/bump.c, src/util.c, src/utilMetrics.c e R/phi.R, R/loss.R, R/util.R):
  * phi.control (método "extremes", coef = 1,5): pontos de controle tirados do boxplot.stats do R (dobradiças de Tukey,
    não quantis), interpolados por pchip com derivadas nulas e extrapolação constante;
  * loss.control: tloss = 3 * sd(|média(y) - y|) * sqrt(log(n)/n) (Cherkassky e Ma, 2002);
  * utilidade U = phi(y) * benefício - phi_conjunto * custo, com as "bumps" do uba e p = 0,5;
  * precisão e revocação ponderadas por utilidade; sem evento previsto (ou real), o uba devolve DELTA = 1e-5, por isso a matriz
    tem tantos valores perto de 2e-5.

Atenção ao limiar de evento. O notebook passa `event_thr=0.8` pelo rpy2 a `util.control(util.parms, ...)`. Como o nome do
argumento no R é `event.thr` e ele chega pelo `...`, o rpy2 não traduz o nome e o R o descarta; vale então o padrão
`event.thr = 1`. Por isso `f1_uba` aceita os dois limiares, e `rodar_pares.py` grava os dois: a correlação com a matriz diz
qual deles a reproduz.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DELTA = 1e-5          # uba: valor que evita o empate nulo entre P e R quando não há eventos
P_JPHI = 0.5          # uba: util.setup(p = 0.5)


# ---------------------------------------------------------------------------
# Relevância phi (uba: phi.extremes + pchip)
# ---------------------------------------------------------------------------
def _fivenum(x: np.ndarray) -> np.ndarray:
    """fivenum do R (dobradiças de Tukey)."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    n4 = np.floor((n + 3) / 2) / 2
    d = np.array([1, n4, (n + 1) / 2, n + 1 - n4, n]) - 1      # índices base 0
    return 0.5 * (x[np.floor(d).astype(int)] + x[np.ceil(d).astype(int)])


def _boxplot_stats(y: np.ndarray, coef: float = 1.5):
    """boxplot.stats do R: (stats[5], valores fora das cercas)."""
    stats = _fivenum(y)
    iqr = stats[3] - stats[1]
    out = (y < stats[1] - coef * iqr) | (y > stats[3] + coef * iqr)
    if coef > 0 and out.any():
        stats[0], stats[4] = y[~out].min(), y[~out].max()
    return stats, y[out]


@dataclass
class Relevancia:
    """Pontos de controle (x, phi) do uba e parâmetros do spline e das bumps; calculado uma vez por dataset."""
    x: np.ndarray
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    d: np.ndarray
    tloss: float
    bleft: np.ndarray
    bmax: np.ndarray
    bloss: np.ndarray

    def phi(self, v) -> np.ndarray:
        """pchip_val com extrapolação linear (aqui constante, porque as derivadas nos extremos são nulas)."""
        v = np.asarray(v, dtype=float)
        n = len(self.x)
        i = np.searchsorted(self.x, v, side="right")          # findInterval do R
        out = np.empty_like(v)
        ext = (i == 0) | (i == n)
        ie = np.where(i == n, n - 1, 0)
        out[ext] = self.a[ie[ext]] + self.b[ie[ext]] * (v[ext] - self.x[ie[ext]])
        k = i[~ext] - 1
        s = v[~ext] - self.x[k]
        out[~ext] = self.a[k] + s * (self.b[k] + s * (self.c[k] + s * self.d[k]))
        return out


def _pchip(x, a, m):
    """pchip_set + pchip_slope_monoFC do uba (Fritsch e Carlson)."""
    n = len(x)
    m = np.array(m, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.diff(x)
        delta = np.diff(a) / h
    for k in range(n - 1):
        sk = delta[k]
        if sk == 0:
            m[k] = m[k + 1] = 0.0
            continue
        if not np.isfinite(sk):            # pontos de controle repetidos: no C, alpha = beta = 0 ou NaN e nada muda
            continue
        alpha, beta = m[k] / sk, m[k + 1] / sk
        if m[k] != 0 and alpha < 0:
            m[k] = -m[k]
            alpha = m[k] / sk
        if m[k + 1] != 0 and beta < 0:
            m[k + 1] = -m[k + 1]
            beta = m[k + 1] / sk
        a2b3, ab23 = 2 * alpha + beta - 3, alpha + 2 * beta - 3
        if a2b3 > 0 and ab23 > 0 and alpha * (a2b3 + ab23) < a2b3 * a2b3:
            tau = 3 * sk / np.sqrt(alpha ** 2 + beta ** 2)
            m[k], m[k + 1] = tau * alpha, tau * beta
    c, d = np.zeros(n), np.zeros(n)
    with np.errstate(divide="ignore", invalid="ignore"):
        c[:-1] = (3 * delta - 2 * m[:-1] - m[1:]) / h
        d[:-1] = (m[:-1] - 2 * delta + m[1:]) / (h * h)
    return m, c, d


def _bumps(x, a, b, tloss):
    """bumps_set do uba (src/bump.c), linha a linha."""
    nB = len(x)
    bleft = np.full(nB + 1, np.nan)
    bmax = np.full(nB + 1, np.nan)
    bloss = np.full(nB + 1, np.nan)
    bleft[0], bmax[0], bloss[0] = -np.inf, -np.inf, np.inf
    crit = [i for i in range(nB) if abs(b[i]) == 0]
    j = len(crit)
    sum_b, nb, n, in_bump = x[crit[0]], 1, 0, True
    i = 0
    while i < j - 1:
        d1 = a[crit[i + 1]] - a[crit[i]]
        if d1 == 0:
            sum_b += x[crit[i + 1]]
            nb += 1
        else:
            if d1 < 0 and in_bump:
                bmax[n] = sum_b / nb
                if np.isfinite(bmax[n]) and np.isfinite(bleft[n]):
                    bloss[n] = abs(bmax[n] - bleft[n])
                in_bump = False
            elif d1 > 0 and (not in_bump or n == 0):
                n += 1
                bleft[n] = sum_b / nb
                if np.isfinite(bmax[n - 1]) and np.isfinite(bleft[n]):
                    delta = abs(bmax[n - 1] - bleft[n])
                    bloss[n - 1] = 2 * delta if delta < bloss[n - 1] else 2 * bloss[n - 1]
                in_bump = True
            sum_b, nb = x[crit[i + 1]], 1
        i += 1
    if n > 0:
        if in_bump:
            bmax[n] = sum_b / nb
            if np.isfinite(bmax[n]) and np.isfinite(bleft[n]):
                bloss[n] = 2 * abs(bmax[n] - bleft[n])
        else:
            n += 1
            bleft[n] = sum_b / nb
            bmax[n] = np.inf
            if np.isfinite(bmax[n - 1]) and np.isfinite(bleft[n]):
                delta = abs(bmax[n - 1] - bleft[n])
                bloss[n - 1] = 2 * delta if delta < bloss[n - 1] else 2 * bloss[n - 1]
        if not np.isfinite(bmax[0]):
            bloss[0] = bloss[1]
        if not np.isfinite(bmax[n]):
            bloss[n] = bloss[n - 1]
    else:
        bloss[0] = tloss
    n += 1
    return bleft[:n], bmax[:n], bloss[:n]


def relevancia(y, coef: float = 1.5) -> Relevancia:
    """phi.control(y) + loss.control(y) do uba (método "extremes", extr.type = "both")."""
    y = np.asarray(y, dtype=float)
    stats, out = _boxplot_stats(y, coef)
    pts = []
    pts.append((stats[0], 1.0) if np.any(out < stats[0]) else (y.min(), 0.0))
    pts.append((stats[2], 0.0))
    pts.append((stats[4], 1.0) if np.any(out > stats[4]) else (y.max(), 0.0))
    x = np.array([p[0] for p in pts])
    a = np.array([p[1] for p in pts])
    b, c, d = _pchip(x, a, np.zeros(len(x)))
    n = len(y)
    tl = np.abs(y.mean() - y)
    tloss = 3 * tl.std(ddof=1) * np.sqrt(np.log(n) / n) if n > 1 else 0.0
    bleft, bmax, bloss = _bumps(x, a, b, tloss)
    return Relevancia(x, a, b, c, d, float(tloss), bleft, bmax, bloss)


# ---------------------------------------------------------------------------
# Utilidade e F1 (uba: util.c, utilMetrics.c)
# ---------------------------------------------------------------------------
def utilidade(y, ypred, rel: Relevancia) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(u, phi(y), phi(ypred)) por exemplo, como util_core + util_value + benefcost_lin."""
    y = np.asarray(y, dtype=float)
    ypred = np.asarray(ypred, dtype=float)
    py, pp = rel.phi(y), rel.phi(ypred)
    nb = len(rel.bleft)
    if nb > 1:
        i = np.searchsorted(rel.bleft, y, side="right")
    else:
        i = np.ones(len(y), dtype=int)
    i = np.where(i > 0, i - 1, i)
    baixo = ypred <= y
    lossb = np.full(len(y), np.inf)
    lossc = np.full(len(y), np.inf)
    for k in range(len(y)):
        ik = i[k]
        if baixo[k]:
            if ik > 0 and np.isfinite(rel.bleft[ik]):
                lossb[k] = abs(y[k] - rel.bleft[ik])
            if ik > 0 and np.isfinite(rel.bmax[ik - 1]):
                lossc[k] = abs(y[k] - rel.bmax[ik - 1])
        else:
            if ik + 1 < nb and np.isfinite(rel.bleft[ik + 1]):
                lossb[k] = abs(y[k] - rel.bleft[ik + 1])
            if ik + 1 < nb and np.isfinite(rel.bmax[ik + 1]):
                lossc[k] = abs(y[k] - rel.bmax[ik + 1])
    bl = rel.bloss[i]
    lb = np.where(lossb < bl, lossb, bl)
    lc = np.where(lossc < bl, lossc, bl)
    loss = np.abs(y - ypred)
    with np.errstate(divide="ignore", invalid="ignore"):
        benef = np.where((lb == 0) | (loss > lb), 0.0, 1 - loss / lb)
        custo = np.where((lc == 0) | (loss > lc), 1.0, loss / lc)
    jphi = P_JPHI * py + (1 - P_JPHI) * pp
    return py * benef - jphi * custo, py, pp


def precisao_revocacao(y, ypred, rel: Relevancia, event_thr: float = 1.0) -> tuple[float, float]:
    u, py, pp = utilidade(y, ypred, rel)
    mp = pp >= event_thr
    prec = DELTA if not mp.any() else float(np.abs(1 + u[mp]).sum() / np.abs(1 + pp[mp]).sum())
    mr = py >= event_thr
    rec = DELTA if not mr.any() else float(np.abs(1 + u[mr]).sum() / np.abs(1 + py[mr]).sum())
    return prec, rec


def f1_uba(y, ypred, rel: Relevancia, event_thr: float = 1.0, beta: float = 1.0) -> float:
    """F1 baseado em utilidade (uba, umetric = "Fm"). `rel` vem de `relevancia(y_todo)`, como no notebook original."""
    p, r = precisao_revocacao(y, ypred, rel, event_thr)
    if p == 0 or r == 0:
        return 0.0
    return float((beta ** 2 + 1) * p * r / (beta ** 2 * p + r))


def sera(y, ypred, rel: Relevancia, step: float = 0.001) -> float:
    """SERA do IRon (área sob a curva do erro quadrático dos exemplos com phi >= t, t em [0, 1])."""
    y = np.asarray(y, dtype=float)
    ypred = np.asarray(ypred, dtype=float)
    ph = rel.phi(y)
    e2 = (y - ypred) ** 2
    th = np.arange(0, 1 + step, step)
    ordem = np.argsort(ph)
    ph_s, e_s = ph[ordem], e2[ordem]
    cum = np.concatenate([np.cumsum(e_s[::-1])[::-1], [0.0]])   # soma dos erros com phi >= ph_s[j]
    idx = np.searchsorted(ph_s, th, side="left")
    err = cum[idx]
    return float(np.sum(step * (err[:-1] + err[1:]) / 2))


def n_eventos(y, rel: Relevancia, event_thr: float) -> int:
    return int((rel.phi(y) >= event_thr).sum())
