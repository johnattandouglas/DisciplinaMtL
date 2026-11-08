"""Executor comum dos sistemas de AutoML na IR, com um adaptador por sistema.

Para cada dataset: lê a MESMA divisão treino/teste dos 42 pares (automl/divisoes/holdout), entrega só o treino ao AutoML com
orçamento de tempo, semente e número de núcleos fixos, e avalia no teste com o F1 do uba (relevância do alvo inteiro, como na
matriz; ver ir_metrica.py). Quando o sistema aceita métrica própria, ele otimiza o mesmo F1 (limiar de evento 1,0), com a
relevância calculada só no alvo do treino (o AutoML nunca vê o teste).

Registra por dataset: tempo de parede e de CPU do ajuste, razão CPU/parede (> 1,2 = paralelizou), pico de RAM (processo e
filhos), threads e processos filhos máximos, modelo escolhido, se é ensemble e de quê, pré-processamento, se o modelo está
entre os 6 da matriz (BG, DT, MLP, RF, SVR, XGB), F1 de validação interno e F1/SERA no teste.

Uso (Git Bash, PYTHONIOENCODING=utf-8), no ambiente do sistema:
    automl/envs/flaml/Scripts/python.exe automl/rodar_automl.py --sistema flaml --orcamento 10 --nucleos 2 --datasets a1
    automl/envs/flaml/Scripts/python.exe automl/rodar_automl.py --sistema flaml --orcamento 60 --todos --nucleos 2
    (com --todos, vários processos iguais dividem a lista: cada um trava e roda o próximo dataset livre)
Saída: automl/resultados/automl/<sistema>/t<orçamento>/<dataset>.csv (uma linha) e o log do sistema em .../logs/.
"""
from __future__ import annotations

import argparse
import json
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


for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, str(_nucleos_cli()))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comum as C  # noqa: E402
import ir_metrica as M  # noqa: E402

warnings.filterwarnings("ignore")
LIMIAR = 1.0          # limiar de evento do F1 otimizado (o que reproduz a matriz; ver comparar_matriz.py)
SEIS = {"BG", "DT", "MLP", "RF", "SVR", "XGB"}


class MetricaF1:
    """F1 do uba (limiar LIMIAR) com a relevância do treino. Classe de módulo para poder ser serializada (AutoGluon, TPOT
    com dask e joblib copiam a métrica para outros processos)."""

    def __init__(self, rel):
        self.rel = rel
        self.__name__ = "f1_uba"

    def __call__(self, y, p, **kw):
        return M.f1_uba(np.asarray(y, float), np.asarray(p, float).ravel(), self.rel, LIMIAR)


class PerdaF1:
    """1 - F1, serializável (hyperopt-sklearn roda cada avaliação em outro processo)."""

    def __init__(self, metrica):
        self.metrica = metrica

    def __call__(self, yt, yp):
        return 1.0 - self.metrica(yt, yp)


# ---------------------------------------------------------------------------
# Adaptadores
# ---------------------------------------------------------------------------
class Adaptador:
    """Interface: `ajustar` recebe só o treino; `metrica(y, p)` devolve o F1 (maior = melhor) com a relevância do treino."""
    nome = "base"
    aceita_metrica = False

    def ajustar(self, X, y, orcamento: int, nucleos: int, semente: int, metrica, pasta_log: Path) -> None:
        raise NotImplementedError

    def prever(self, X) -> np.ndarray:
        raise NotImplementedError

    def descrever(self) -> dict:
        """modelo, familia_6 (código entre os 6 ou ""), ensemble (bool), composicao, preproc, config, n_configs, f1_val."""
        raise NotImplementedError

    def versao(self) -> str:
        return ""


class FLAML(Adaptador):
    nome = "flaml"
    aceita_metrica = True
    MAPA6 = {"rf": "RF", "xgboost": "XGB", "xgb_limitdepth": "XGB"}   # lgbm, extra_tree, catboost, kneighbor, sgd: fora dos 6

    def ajustar(self, X, y, orcamento, nucleos, semente, metrica, pasta_log):
        from flaml import AutoML

        def f1_flaml(X_val, y_val, estimator, labels, X_train, y_train, *args, **kwargs):
            f1 = metrica(np.asarray(y_val, float), estimator.predict(X_val))
            return 1.0 - f1, {"f1_val": f1}

        self.a = AutoML()
        pasta_log.mkdir(parents=True, exist_ok=True)
        self.a.fit(X_train=X, y_train=y, task="regression", metric=f1_flaml, time_budget=orcamento, n_jobs=nucleos,
                   seed=semente, log_file_name=str(pasta_log / "flaml.log"), verbose=0)

    def prever(self, X):
        return np.asarray(self.a.predict(X), float)

    def descrever(self):
        est = self.a.best_estimator
        modelo = type(self.a.model.estimator).__name__ if self.a.model is not None else ""
        return {"modelo": f"{est} ({modelo})", "familia_6": self.MAPA6.get(est, ""), "ensemble": False,
                "composicao": "", "preproc": "DataTransformer do FLAML (só trata colunas categóricas/datas; aqui tudo é numérico)",
                "config": json.dumps(self.a.best_config, default=str), "n_configs": len(self.a.config_history),
                "f1_val": 1.0 - float(self.a.best_loss)}

    def versao(self):
        import flaml
        return flaml.__version__


class AutoSklearn(Adaptador):
    """A PREENCHER. Só roda em Linux (WSL não está instalado nesta máquina). Pontos já decididos:
      * métrica: autosklearn.metrics.make_scorer("f1_uba", f, greater_is_better=True) com f(y, p) = metrica(y, p);
      * meta-aprendizado ligado: AutoSklearnRegressor(initial_configurations_via_metalearning=25) (padrão); desligado: = 0;
      * ensemble: ensemble_class padrão; composição em `show_models()` / `leaderboard()`; pré-processamento em cada pipeline;
      * sobreposição do meta-conhecimento: os IDs de tarefas OpenML ficam em autosklearn/metalearning/files/
        <métrica>_regression_dense/ (description.txt e algorithm_runs.arff); cruzar com os IDs OpenML dos 215 datasets;
      * n_jobs=nucleos, seed=semente, time_left_for_this_task=orcamento, memory_limit fixo."""
    nome = "autosklearn"
    aceita_metrica = True


class H2O(Adaptador):
    """A PREENCHER. Precisa de Java (não instalado). H2OAutoML(max_runtime_secs=orcamento, seed=semente, nthreads=nucleos);
    não aceita métrica própria para a busca (sort_metric só entre as nativas): registrar isso. Ensembles: StackedEnsemble."""
    nome = "h2o"


def _fam6(nome_modelo: str) -> str:
    """Código entre os 6 da matriz a partir do nome da classe/modelo (vazio se não for um deles)."""
    s = nome_modelo.lower()
    for chave, cod in (("bagging", "BG"), ("decisiontree", "DT"), ("mlp", "MLP"), ("randomforest", "RF"),
                       ("svr", "SVR"), ("xgb", "XGB")):
        if chave in s.replace("_", "").replace(" ", ""):
            return cod
    return ""


def _sk_scorer(metrica):
    from sklearn.metrics import make_scorer
    return make_scorer(metrica, greater_is_better=True)


class AutoGluon(Adaptador):
    """TabularPredictor com o F1 como eval_metric. Variante "portfolio": hyperparameters="zeroshot" (o portfólio aprendido
    de configurações do TabRepo); variante "padrao": hyperparameters="default" (grade fixa, sem portfólio). Ambas com
    presets="medium_quality" (sem bagging), para caber nos orçamentos curtos."""
    nome = "autogluon_portfolio"
    aceita_metrica = True
    hp = "zeroshot"

    def ajustar(self, X, y, orcamento, nucleos, semente, metrica, pasta_log):
        import shutil
        from autogluon.core.metrics import make_scorer
        from autogluon.tabular import TabularPredictor
        df = pd.DataFrame(X, columns=[f"x{j}" for j in range(X.shape[1])])
        df["target"] = y
        sc = make_scorer("f1_uba", metrica, optimum=1, greater_is_better=True)
        caminho = pasta_log / "ag"
        shutil.rmtree(caminho, ignore_errors=True)
        self.cols = list(df.columns[:-1])
        self.p = TabularPredictor(label="target", problem_type="regression", eval_metric=sc, path=str(caminho),
                                  verbosity=0)
        self.p.fit(df, time_limit=orcamento, presets="medium_quality", hyperparameters=self.hp, num_cpus=nucleos,
                   num_gpus=0)

    def prever(self, X):
        return np.asarray(self.p.predict(pd.DataFrame(X, columns=self.cols)), float)

    def descrever(self):
        best = self.p.model_best
        usados = [m for m in self.p._trainer.get_minimum_model_set(best) if m != best]
        lb = self.p.leaderboard(silent=True)
        fam = sorted({_fam6(m) for m in (usados or [best])} - {""})
        return {"modelo": best, "familia_6": ",".join(fam), "ensemble": "Ensemble" in best,
                "composicao": ";".join(usados), "preproc": "AutoMLPipelineFeatureGenerator (tipos, constantes, duplicadas)",
                "config": self.hp, "n_configs": len(lb), "f1_val": float(lb.set_index("model").loc[best, "score_val"])}

    def versao(self):
        import autogluon.tabular
        return autogluon.tabular.__version__


class AutoGluonPadrao(AutoGluon):
    nome = "autogluon_padrao"
    hp = "default"


class TPOT(Adaptador):
    """TPOT 1.x: TPOTRegressor com o F1 como scorer, max_time_mins = orçamento/60."""
    nome = "tpot"
    aceita_metrica = True

    def ajustar(self, X, y, orcamento, nucleos, semente, metrica, pasta_log):
        from tpot import TPOTRegressor
        self.m = TPOTRegressor(scorers=[_sk_scorer(metrica)], scorers_weights=[1.0], max_time_mins=orcamento / 60,
                               n_jobs=nucleos, random_state=semente, verbose=0)
        self.m.fit(X, y)

    def prever(self, X):
        return np.asarray(self.m.predict(X), float)

    def descrever(self):
        pipe = self.m.fitted_pipeline_
        passos = [type(st).__name__ for _, st in pipe.steps] if hasattr(pipe, "steps") else [type(pipe).__name__]
        final = passos[-1]
        return {"modelo": final, "familia_6": _fam6(final), "ensemble": False, "composicao": "",
                "preproc": ";".join(passos[:-1]), "config": str(pipe)[:500], "n_configs": np.nan, "f1_val": np.nan}

    def versao(self):
        import tpot
        return tpot.__version__


class MLJAR(Adaptador):
    """mljar-supervised: AutoML(mode="Compete", total_time_limit=orçamento). A versão 1.3.2 só aceita métrica própria em
    classificação ("user_defined_metric is not allowed in ML task: regression"); aqui otimiza RMSE e o F1 só é medido."""
    nome = "mljar"
    aceita_metrica = False

    def ajustar(self, X, y, orcamento, nucleos, semente, metrica, pasta_log):
        import shutil
        from supervised import AutoML
        caminho = pasta_log / "mljar"
        shutil.rmtree(caminho, ignore_errors=True)
        pasta_log.mkdir(parents=True, exist_ok=True)
        self.a = AutoML(mode="Compete", ml_task="regression", eval_metric="rmse",
                        total_time_limit=max(orcamento, 1), n_jobs=nucleos, random_state=semente,
                        results_path=str(caminho), verbose=0, explain_level=0)
        self.a.fit(pd.DataFrame(X), y)

    def prever(self, X):
        return np.asarray(self.a.predict(pd.DataFrame(X)), float)

    def descrever(self):
        lb = self.a.get_leaderboard()
        best = self.a._best_model.get_name() if self.a._best_model is not None else ""
        comp = ""
        if "Ensemble" in best:
            try:
                comp = ";".join(m["model"].get_name() for m in self.a._best_model.selected_models)
            except Exception:
                comp = ""
        base = comp.split(";") if comp else [best]
        fam = sorted({_fam6(m.replace("Xgboost", "xgb").replace("Random_Forest", "randomforest")
                            .replace("Decision_Tree", "decisiontree").replace("Neural_Network", "mlp")) for m in base} - {""})
        return {"modelo": best, "familia_6": ",".join(fam), "ensemble": "Ensemble" in best, "composicao": comp,
                "preproc": "pré-processamento do mljar (escala, imputação, seleção de atributos opcional)",
                "config": "Compete", "n_configs": len(lb),
                "f1_val": np.nan}

    def versao(self):
        import supervised
        return supervised.__version__


class NaiveAutoML(Adaptador):
    """naiveautoml.NaiveAutoML(task_type="regression", scoring=F1, timeout_overall=orçamento)."""
    nome = "naiveautoml"
    aceita_metrica = True

    def ajustar(self, X, y, orcamento, nucleos, semente, metrica, pasta_log):
        import naiveautoml
        sc = _sk_scorer(metrica)
        self.a = naiveautoml.NaiveAutoML(task_type="regression", scoring=sc, timeout_overall=orcamento,
                                         random_state=semente, show_progress=False)
        self.a.fit(pd.DataFrame(X), pd.Series(y))

    def prever(self, X):
        return np.asarray(self.a.predict(pd.DataFrame(X)), float)

    def descrever(self):
        pipe = self.a.chosen_model
        passos = [type(st).__name__ for _, st in pipe.steps] if hasattr(pipe, "steps") else [type(pipe).__name__]
        hist = getattr(self.a, "history", None)
        return {"modelo": passos[-1], "familia_6": _fam6(passos[-1]), "ensemble": False, "composicao": "",
                "preproc": ";".join(passos[:-1]), "config": str(pipe)[:500],
                "n_configs": len(hist) if hist is not None else np.nan, "f1_val": np.nan}

    def versao(self):
        from importlib.metadata import version
        return version("naiveautoml")


class HyperoptSklearn(Adaptador):
    """hyperopt-sklearn: HyperoptEstimator(any_regressor, any_preprocessing, TPE, loss = 1 - F1). Não tem orçamento total;
    fazemos uma avaliação por vez (fit_iter) até o orçamento acabar."""
    nome = "hpsklearn"
    aceita_metrica = True

    def ajustar(self, X, y, orcamento, nucleos, semente, metrica, pasta_log):
        import time as _t
        from hpsklearn import HyperoptEstimator, any_preprocessing, any_regressor
        from hyperopt import tpe
        self.e = HyperoptEstimator(regressor=any_regressor("r"), preprocessing=any_preprocessing("p"), algo=tpe.suggest,
                                   loss_fn=PerdaF1(metrica),
                                   max_evals=10000, trial_timeout=max(orcamento, 1), seed=semente, n_jobs=nucleos)
        t0 = _t.perf_counter()
        it = self.e.fit_iter(X, y)
        next(it)
        while _t.perf_counter() - t0 < orcamento:
            it.send(1)
        self.e._retrain_best_model_on_full_data(X, y)

    def prever(self, X):
        return np.asarray(self.e.predict(X), float)

    def descrever(self):
        b = self.e.best_model()
        lrn = type(b["learner"]).__name__
        losses = [l for l in self.e.trials.losses() if l is not None]
        return {"modelo": lrn, "familia_6": _fam6(lrn), "ensemble": False, "composicao": "",
                "preproc": ";".join(type(p).__name__ for p in b["preprocs"]), "config": str(b["learner"])[:500],
                "n_configs": len(losses), "f1_val": 1.0 - min(losses) if losses else np.nan}

    def versao(self):
        from importlib.metadata import version
        return version("hyperopt-sklearn")


class LightAutoML(Adaptador):
    """LightAutoML: TabularAutoML(task=Task("reg", metric=F1, greater_is_better=True), timeout, cpu_limit)."""
    nome = "lightautoml"
    aceita_metrica = True

    def ajustar(self, X, y, orcamento, nucleos, semente, metrica, pasta_log):
        from lightautoml.automl.presets.tabular_presets import TabularAutoML
        from lightautoml.tasks import Task
        df = pd.DataFrame(X, columns=[f"x{j}" for j in range(X.shape[1])])
        df["target"] = y
        self.cols = list(df.columns[:-1])
        task = Task("reg", metric=lambda yt, yp, **kw: metrica(np.asarray(yt, float), np.asarray(yp, float)),
                    greater_is_better=True)
        self.a = TabularAutoML(task=task, timeout=orcamento, cpu_limit=nucleos,
                               reader_params={"random_state": semente})
        self.a.fit_predict(df, roles={"target": "target"}, verbose=0)

    def prever(self, X):
        return np.asarray(self.a.predict(pd.DataFrame(X, columns=self.cols)).data[:, 0], float)

    def descrever(self):
        desc = self.a.create_model_str_desc() if hasattr(self.a, "create_model_str_desc") else ""
        return {"modelo": "LightAutoML blend", "familia_6": "", "ensemble": True, "composicao": desc[:500],
                "preproc": "LAMA (codificação, imputação)", "config": "", "n_configs": np.nan, "f1_val": np.nan}

    def versao(self):
        from importlib.metadata import version
        return version("lightautoml")


SISTEMAS = {c.nome: c for c in (FLAML, AutoSklearn, H2O, AutoGluon, AutoGluonPadrao, TPOT, MLJAR, NaiveAutoML,
                                HyperoptSklearn, LightAutoML)}


# ---------------------------------------------------------------------------
# Execução
# ---------------------------------------------------------------------------
def rodar(nome: str, sistema: str, orcamento: int, nucleos: int, pasta: Path) -> dict:
    X, y = C.ler_dataset(nome)
    tr, te = C.divisoes(nome, len(y), "holdout")[0]
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    rel_teste = M.relevancia(y)                 # avaliação final: igual à dos 42 pares (relevância do alvo inteiro)
    rel_treino = M.relevancia(ytr)              # métrica de busca: só o treino

    metrica = MetricaF1(rel_treino)

    ad = SISTEMAS[sistema]()
    linha = {"dataset": nome, "sistema": sistema, "orcamento_s": orcamento, "nucleos": nucleos, "semente": C.SEMENTE,
             "n_treino": len(tr), "n_teste": len(te), "p": X.shape[1], "metrica_propria": ad.aceita_metrica, "erro": ""}
    try:
        with C.Medidor() as med:
            ad.ajustar(Xtr, ytr, orcamento, nucleos, C.SEMENTE, metrica, pasta / "logs" / nome)
        t0 = time.perf_counter()
        p = ad.prever(Xte)
        linha["t_pred_s"] = time.perf_counter() - t0
        linha.update(med.resumo())
        linha["paralelizou"] = linha["cpu_por_parede"] > 1.2
        linha.update(ad.descrever())
        linha["entre_os_6"] = linha["familia_6"] in SEIS
        linha["f1_teste_thr1"] = M.f1_uba(yte, p, rel_teste, 1.0)
        linha["f1_teste_thr08"] = M.f1_uba(yte, p, rel_teste, 0.8)
        linha["sera_teste"] = M.sera(yte, p, rel_teste)
        linha["versao"] = ad.versao()
    except Exception as e:
        linha["erro"] = f"{type(e).__name__}: {str(e)[:300]}"
        traceback.print_exc()
    return linha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sistema", choices=sorted(SISTEMAS), default="flaml")
    ap.add_argument("--orcamento", type=int, required=True, help="segundos")
    ap.add_argument("--nucleos", type=int, default=1)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--todos", action="store_true")
    a = ap.parse_args()
    nomes = C.ordem_fila(C.nomes_ir()) if a.todos or not a.datasets else a.datasets
    pasta = C.RESULTADOS_DIR / "automl" / a.sistema / f"t{a.orcamento}"
    pasta.mkdir(parents=True, exist_ok=True)
    for nome in nomes:                     # vários workers podem rodar este laço: cada um pega o próximo dataset livre
        saida = pasta / f"{nome}.csv"
        if saida.exists() or (a.todos and not C.reivindicar(pasta, nome)):
            continue
        linha = rodar(nome, a.sistema, a.orcamento, a.nucleos, pasta)
        if linha["erro"]:
            print(f"{nome}: ERRO {linha['erro']}", flush=True)
            continue                                         # sem CSV: a fila tenta de novo na próxima vez
        tmp = saida.with_suffix(".tmp")
        pd.DataFrame([linha]).to_csv(tmp, index=False)
        os.replace(tmp, saida)
        print(f"{nome}: {a.sistema} t={a.orcamento}s parede={linha['parede_s']:.1f}s cpu/parede={linha['cpu_por_parede']:.2f} "
              f"RAM={linha['ram_pico_mb']:.0f}MB modelo={linha['modelo']} F1 teste={linha['f1_teste_thr1']:.3f} "
              f"(val {linha['f1_val']:.3f}, {linha['n_configs']} configs)", flush=True)


if __name__ == "__main__":
    main()
