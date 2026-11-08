"""Roda os 42 pares (6 modelos x 7 estratégias de balanceamento) da IR na divisão treino/teste fixa de cada dataset.

Reproduz o método `repeatedKfold` + `balance` + `scores` do notebook Meta_IR.ipynb (Avelino et al., 2025;
github.com/JusciAvelino/Meta-IR), que gerou meta_ir/base/F1_score.csv:
  * modelos com hiperparâmetros padrão: BaggingRegressor, DecisionTreeRegressor, MLPRegressor, RandomForestRegressor, SVR,
    XGBRegressor(verbosity=0). Diferença: aqui os modelos estocásticos recebem random_state=42 (no original não tinham
    semente) e n_jobs = núcleos do worker;
  * balanceamento só no treino, parâmetros padrão e rel_thres=0.8: GN, RO, RU, SMT (ImbalancedLearningRegression),
    SG (smogn.smoter com rel_xtrm_type='high') e WC (resreg.wercs com pdf_relevance). Se a estratégia levanta ValueError, o
    original treina sem balancear; aqui também, e a falha fica registrada (coluna `bal_falhou`);
  * o GridSearchCV do original tem grade vazia: o modelo final é o padrão reajustado no treino; por isso ele foi omitido;
  * F1 do uba (umetric "Fm") e SERA do IRon, com a relevância calculada no alvo do dataset INTEIRO (como no original). O F1 sai
    com os dois limiares de evento (1,0 e 0,8; ver ir_metrica.py).

Uso (Git Bash, PYTHONIOENCODING=utf-8):
    automl/envs/pares/Scripts/python.exe automl/rodar_pares.py --datasets a1 rabe_265 --nucleos 2
    automl/envs/pares/Scripts/python.exe automl/rodar_pares.py --todos --nucleos 2
    (com --todos, vários processos iguais dividem a lista: cada um trava e roda o próximo dataset livre, maior primeiro)
Saída: automl/resultados/pares/<protocolo>/<dataset>.csv (uma linha por partição e par); datasets prontos são pulados.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
import warnings
from pathlib import Path


def _nucleos_cli() -> int:
    for i, a in enumerate(sys.argv):
        if a == "--nucleos" and i + 1 < len(sys.argv):
            return int(sys.argv[i + 1])
    return 1


# threads das bibliotecas numéricas fixadas antes de importar numpy
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, str(_nucleos_cli()))
os.environ.setdefault("TQDM_DISABLE", "1")        # smogn e ImbalancedLearningRegression enchem o log de barras de progresso

import random  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import psutil  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comum as C  # noqa: E402
import ir_metrica as M  # noqa: E402

warnings.filterwarnings("ignore")


def modelo(cod: str, nucleos: int):
    from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor
    from xgboost import XGBRegressor
    s = C.SEMENTE
    return {"BG": lambda: BaggingRegressor(random_state=s, n_jobs=nucleos),
            "DT": lambda: DecisionTreeRegressor(random_state=s),
            "ML": lambda: MLPRegressor(random_state=s),
            "RF": lambda: RandomForestRegressor(random_state=s, n_jobs=nucleos),
            "SV": lambda: SVR(),
            "XG": lambda: XGBRegressor(verbosity=0, random_state=s, n_jobs=nucleos)}[cod]()


def balancear(train: pd.DataFrame, estrategia: str) -> pd.DataFrame:
    """Meta_IR.balance, com a semente global fixada antes (as bibliotecas usam np.random/random sem semente própria)."""
    np.random.seed(C.SEMENTE)
    random.seed(C.SEMENTE)
    y = train.columns[0]
    if estrategia == "NO":
        return train
    import ImbalancedLearningRegression as iblr
    if estrategia == "GN":
        return iblr.gn(data=train, y=y, rel_thres=0.8)
    if estrategia == "RO":
        return iblr.ro(data=train, y=y, rel_thres=0.8)
    if estrategia == "RU":
        return iblr.random_under(data=train, y=y, rel_thres=0.8)
    if estrategia == "SM":
        return iblr.smote(data=train, y=y, rel_thres=0.8)
    if estrategia == "SG":
        import smogn
        train = train.dropna()
        train = smogn.smoter(data=train, y=y, rel_xtrm_type="high", rel_thres=0.8)
        return train.dropna()
    if estrategia == "WC":
        import resreg
        X_train = train.drop(columns=[y])
        y_train = train[y]
        rel = resreg.pdf_relevance(y_train)
        Xw, yw = resreg.wercs(X_train, y_train, rel)
        out = pd.DataFrame(np.column_stack((yw, Xw)))
        out.columns = [str(c) for c in out.columns]
        return out
    raise ValueError(estrategia)


def rodar_particao(X, y, tr, te, rel, nucleos):
    """Os 42 pares numa partição. Devolve uma lista de dicts (um por par)."""
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
    base = pd.DataFrame(np.column_stack((ytr, Xtr)))
    base.columns = [str(c) for c in base.columns]       # nomes em texto, como no CSV relido do notebook de 2024
    proc = psutil.Process()
    linhas = []
    for est in C.ESTRATEGIAS:
        t0 = time.perf_counter()
        c0 = sum(proc.cpu_times()[:2])
        falhou, msg = False, ""
        try:
            bal = balancear(base.copy(), est)
        except Exception as e:                           # o original só pegava ValueError; o resto quebraria a execução
            falhou, msg, bal = True, f"{type(e).__name__}: {str(e)[:120]}", base
        t_bal = time.perf_counter() - t0
        cpu_bal = sum(proc.cpu_times()[:2]) - c0
        Xb = bal.iloc[:, 1:].to_numpy(dtype=float)
        yb = bal.iloc[:, 0].to_numpy(dtype=float)
        for cod in C.MODELOS:
            t1 = time.perf_counter()
            c1 = sum(proc.cpu_times()[:2])
            erro = ""
            try:
                m = modelo(cod, nucleos)
                m.fit(Xb, yb)
                t_fit = time.perf_counter() - t1
                p = m.predict(Xte)
            except Exception as e:
                erro = f"{type(e).__name__}: {str(e)[:120]}"
                t_fit = time.perf_counter() - t1
                p = np.full(len(yte), np.nan)
            t_tot = time.perf_counter() - t1
            cpu = sum(proc.cpu_times()[:2]) - c1
            ok = np.isfinite(p).all()
            linhas.append({
                "par": f"{cod}.{est}", "modelo": cod, "estrategia": est,
                "f1_thr1": M.f1_uba(yte, p, rel, 1.0) if ok else np.nan,
                "f1_thr08": M.f1_uba(yte, p, rel, 0.8) if ok else np.nan,
                "sera": M.sera(yte, p, rel) if ok else np.nan,
                "n_treino": len(tr), "n_treino_bal": len(yb), "n_teste": len(te),
                "eventos_teste_thr1": M.n_eventos(yte, rel, 1.0), "eventos_teste_thr08": M.n_eventos(yte, rel, 0.8),
                "bal_falhou": falhou, "bal_msg": msg, "erro": erro,
                "t_bal_s": t_bal, "cpu_bal_s": cpu_bal, "t_fit_s": t_fit, "t_pred_s": t_tot - t_fit, "t_par_s": t_bal + t_tot,
                "cpu_par_s": cpu_bal + cpu})
    return linhas


def validar_pares(Xtr, ytr, Xva, yva, pares, rel, nucleos) -> dict:
    """F1 (limiar 1,0) de validação de uma lista de pares 'MO.ES' (usado pelo warm-start de avaliar_mtl.py)."""
    base = pd.DataFrame(np.column_stack((ytr, Xtr)))
    base.columns = [str(c) for c in base.columns]
    bal_cache, out = {}, {}
    for par in pares:
        cod, est = par.split(".")
        if est not in bal_cache:
            try:
                bal_cache[est] = balancear(base.copy(), est)
            except Exception:
                bal_cache[est] = base
        bal = bal_cache[est]
        try:
            p = modelo(cod, nucleos).fit(bal.iloc[:, 1:].to_numpy(float), bal.iloc[:, 0].to_numpy(float)).predict(Xva)
            out[par] = M.f1_uba(yva, p, rel, 1.0)
        except Exception:
            out[par] = -1.0
    return out


def rodar_dataset(nome: str, protocolo: str, nucleos: int, saida: Path) -> None:
    X, y = C.ler_dataset(nome)
    rel = M.relevancia(y)                                  # alvo inteiro, como phi.control(y) no original
    parts = C.divisoes(nome, len(y), protocolo)
    linhas = []
    t0 = time.perf_counter()
    for k, (tr, te) in enumerate(parts):
        for d in rodar_particao(X, y, tr, te, rel, nucleos):
            d.update({"dataset": nome, "particao": k})
            linhas.append(d)
    df = pd.DataFrame(linhas)
    df["nucleos"] = nucleos
    df["ram_pico_proc_mb"] = psutil.Process().memory_info().peak_wset / 2 ** 20 if os.name == "nt" else np.nan
    cols = ["dataset", "particao", "par", "modelo", "estrategia"]
    df = df[cols + [c for c in df.columns if c not in cols]]
    tmp = saida.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, saida)
    nf = df["bal_falhou"].groupby(df["estrategia"]).any()
    print(f"{nome}: n={len(y)} p={X.shape[1]} partições={len(parts)} {time.perf_counter() - t0:.1f}s "
          f"F1(thr1) médio={df.f1_thr1.mean():.3f} melhor={df.groupby('par').f1_thr1.mean().idxmax()} "
          f"bal_falhou={list(nf[nf].index)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--todos", action="store_true")
    ap.add_argument("--protocolo", choices=["holdout", "rkf"], default="holdout")
    ap.add_argument("--nucleos", type=int, default=1, help="threads por modelo (RF, BG, XGB) e das bibliotecas numéricas")
    a = ap.parse_args()
    nomes = C.ordem_fila(C.nomes_ir()) if a.todos or not a.datasets else a.datasets
    out = C.RESULTADOS_DIR / "pares" / a.protocolo
    out.mkdir(parents=True, exist_ok=True)
    for nome in nomes:                     # vários workers podem rodar este laço: cada um pega o próximo dataset livre
        saida = out / f"{nome}.csv"
        if saida.exists() or (a.todos and not C.reivindicar(out, nome)):
            continue
        try:
            rodar_dataset(nome, a.protocolo, a.nucleos, saida)
        except Exception:
            print(f"{nome}: FALHOU\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    main()
