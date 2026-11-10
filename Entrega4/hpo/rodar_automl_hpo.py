"""Roda um sistema de AutoML nos 120 datasets da meta-task de HPO, na divisão de comum.py (treino 75% / teste 25%),
com acurácia como métrica, para uma lista de orçamentos (segundos). Retomável: uma linha por (sistema, dataset, orçamento)
em resultados/automl_<sistema>.jsonl. O AutoML só vê o treino; o teste é usado uma vez, para a acurácia final.

Sistemas:
  autosklearn_ml    Auto-sklearn 0.15.0, meta-aprendizagem ligada (25 configs iniciais por meta-features, padrão)  [WSL, CPU]
  autosklearn_noml  idem, initial_configurations_via_metalearning=0                                             [WSL, CPU]
  autogluon         AutoGluon 1.6.3, presets medium_quality, num_gpus=1 (XGBoost e CatBoost na GPU; o torch do ambiente
                    é CPU, então a rede do AutoGluon roda na CPU; LightGBM do pip é só CPU)
  flaml             FLAML 2.8.0, XGBoost na GPU (device=cuda); LightGBM, RF e ExtraTrees na CPU
  tpot              TPOT (substituto do Auto-sklearn, só se o WSL falhar; não respeita orçamentos curtos)

Uso:  python rodar_automl_hpo.py SISTEMA --orcamentos 30,120 [--datasets a,b] [--nucleos 4]
"""
import argparse
import json
import platform
import shutil
import sys
import time
import traceback

import numpy as np

import comum


def ajustar(sistema, Xtr, ytr, orc, nucleos, pasta):
    """Devolve (modelo com .predict, descrição dict)."""
    if sistema.startswith("autosklearn"):
        import autosklearn.classification as asc
        from autosklearn.metrics import accuracy
        shutil.rmtree(pasta, ignore_errors=True)
        kw = {} if sistema == "autosklearn_ml" else {"initial_configurations_via_metalearning": 0}
        m = asc.AutoSklearnClassifier(time_left_for_this_task=orc, per_run_time_limit=max(orc // 3, 10),
                                      n_jobs=nucleos, seed=comum.SEMENTE, metric=accuracy, memory_limit=4096,
                                      tmp_folder=str(pasta), delete_tmp_folder_after_terminate=True, **kw)
        m.fit(Xtr, ytr)
        ens = m.get_models_with_weights()
        desc = {"n_modelos_ensemble": len(ens),
                "composicao": ";".join(f"{w:.2f}:{_ask_desc(p)}" for w, p in ens),
                "n_avaliados": len(m.cv_results_["params"]),
                "melhor": _ask_desc(ens[0][1]) if ens else ""}
        return m, desc
    if sistema == "autogluon":
        import pandas as pd
        from autogluon.tabular import TabularPredictor
        shutil.rmtree(pasta, ignore_errors=True)
        df = pd.DataFrame(Xtr, columns=[f"x{j}" for j in range(Xtr.shape[1])])
        df["y"] = ytr
        p = TabularPredictor(label="y", eval_metric="accuracy", path=str(pasta), verbosity=0)
        # só modelos que rodam na GPU neste ambiente (pedido do autor): XGBoost e CatBoost; o ensemble ponderado é barato
        gpu = {"ag_args_fit": {"num_gpus": 1}}
        p.fit(df, time_limit=orc, presets="medium_quality", num_cpus=nucleos, num_gpus=1,
              hyperparameters={"XGB": [gpu], "CAT": [gpu]})
        best = p.model_best
        lb = p.leaderboard(silent=True)
        usados = [x for x in p._trainer.get_minimum_model_set(best) if x != best]

        class M:
            def predict(self, X):
                return np.asarray(p.predict(pd.DataFrame(X, columns=df.columns[:-1])))
        return M(), {"melhor": best, "ensemble": "Ensemble" in best, "composicao": ";".join(usados),
                     "n_avaliados": len(lb), "modelos": ";".join(lb.model)}
    if sistema == "flaml":
        from flaml import AutoML
        a = AutoML()
        a.fit(Xtr, ytr, task="classification", metric="accuracy", time_budget=orc, n_jobs=nucleos,
              seed=comum.SEMENTE, verbose=0, estimator_list=["xgboost"],   # só GPU (pedido do autor); catboost não está no ambiente
              custom_hp={"xgboost": {"device": {"domain": "cuda"}, "tree_method": {"domain": "hist"}}})
        return a, {"melhor": a.best_estimator, "ensemble": False, "config": json.dumps(a.best_config, default=str),
                   "n_avaliados": len(a.config_history)}
    if sistema == "tpot":
        from tpot import TPOTClassifier
        t = TPOTClassifier(max_time_mins=orc / 60, scoring="accuracy", n_jobs=nucleos, random_state=comum.SEMENTE,
                           verbosity=0)
        t.fit(Xtr, ytr)
        return t, {"melhor": str(t.fitted_pipeline_)}
    raise ValueError(sistema)


def _ask_desc(p):
    c = getattr(p, "config", None)
    if c is not None:
        return f"{c.get('classifier:__choice__')}+{c.get('feature_preprocessor:__choice__')}"
    try:
        s = p.steps
        return "|".join(f"{n}={type(getattr(o, 'choice', o)).__name__}" for n, o in s)
    except Exception:  # noqa: BLE001
        return type(p).__name__


def versao(sistema):
    try:
        if sistema.startswith("autosklearn"):
            import autosklearn
            return autosklearn.__version__
        if sistema == "autogluon":
            import autogluon.tabular
            return autogluon.tabular.__version__
        if sistema == "flaml":
            import flaml
            return flaml.__version__
        import tpot
        return tpot.__version__
    except Exception:  # noqa: BLE001
        return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sistema")
    ap.add_argument("--orcamentos", required=True)
    ap.add_argument("--datasets", default="")
    ap.add_argument("--nucleos", type=int, default=4)
    a = ap.parse_args()
    orcs = [int(x) for x in a.orcamentos.split(",")]
    nomes = a.datasets.split(",") if a.datasets else comum.nomes()
    arq = comum.RES / f"automl_{a.sistema}.jsonl"
    hw = {"cpu": platform.processor() or platform.machine(), "so": platform.platform(), "nucleos": a.nucleos,
          "versao": versao(a.sistema)}
    for orc in orcs:
        for nome in nomes:
            chave = f"{a.sistema}|{nome}|{orc}"
            if comum.ja_feito(arq, chave):
                continue
            Xtr, ytr, Xte, yte = comum.carregar(nome)
            pasta = comum.HPO / "tmp" / a.sistema
            reg = {"chave": chave, "sistema": a.sistema, "dataset": nome, "orcamento": orc, **hw}
            t0 = time.perf_counter()
            try:
                m, desc = ajustar(a.sistema, Xtr, ytr, orc, a.nucleos, pasta)
                reg["seg_ajuste"] = time.perf_counter() - t0
                t1 = time.perf_counter()
                p = np.asarray(m.predict(Xte)).astype(np.int64).ravel()
                reg["seg_predicao"] = time.perf_counter() - t1
                reg["acc_teste"] = float((p == yte).mean())
                reg.update(desc)
            except Exception as e:  # noqa: BLE001
                reg["seg_ajuste"] = time.perf_counter() - t0
                reg["erro"] = repr(e)[:500]
                # como no AMLB: falha vira o preditor da classe majoritária do treino
                reg["acc_teste"] = float((yte == np.bincount(ytr).argmax()).mean())
                reg["falha"] = True
                traceback.print_exc()
            comum.gravar(arq, reg)
            print(json.dumps({k: reg.get(k) for k in ("chave", "acc_teste", "seg_ajuste", "melhor", "erro")},
                             default=str), flush=True)
            shutil.rmtree(comum.HPO / "tmp" / a.sistema, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
