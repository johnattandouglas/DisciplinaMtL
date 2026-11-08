# Sistemas AutoML tabulares para comparar com meta-aprendizado

Levantamento feito em 08/11/2026, sem instalar nem rodar nada. Cada item foi conferido no artigo original, na
documentação oficial, no PyPI ou no Crossref. O que não consegui conferir está marcado como "não verificado". As chaves
BibTeX entre colchetes estão em `relatorio_neurips/automl_refs.bib`.

Escopo: sistemas que escolhem modelo (e em geral hiperparâmetros e pré-processamento) para classificação ou regressão em
dados tabulares. Requisitos do nosso caso: regressão com métrica própria (F1 baseada em relevância para regressão
desbalanceada [ribeiro2020imbalanced]), máquina Windows 11 com 12 núcleos, WSL aceitável.

## O AutoML Benchmark como ponto de partida

O AMLB [gijsbers2024amlb] (JMLR 25(101), 2024) avaliou 9 sistemas em 71 tarefas de classificação e 33 de regressão, todas
do OpenML: AutoGluon, auto-sklearn, auto-sklearn 2, FLAML, GAMA, H2O AutoML, LightAutoML, MLJAR e TPOT. A NaiveAutoML
rodou, mas saiu da análise principal (Apêndice E do artigo). O Auto-WEKA ficou de fora pelo desempenho na avaliação de
2019 e pela falta de atualização. ML-Plan, mlr3automl e MLNET ficaram de fora por problemas técnicos. O AutoXGBoost saiu
a pedido dos autores. AutoKeras e Auto-PyTorch não entraram porque dependem de GPU. O repositório do AMLB
(`frameworks/`) integra ainda FEDOT, hyperopt-sklearn, Oboe e SapientML.

Dois pontos do AMLB importam para nós. Primeiro, a seção 5.3.3 diz que o meta-aprendizado do auto-sklearn usa 208
datasets do OpenML e que o do auto-sklearn 2 não permite excluir datasets; os autores tratam os resultados do
auto-sklearn 2 como provavelmente otimistas por isso. Segundo, o auto-sklearn 2 não faz regressão.

## Sistemas, em ordem cronológica

### Auto-WEKA (2013; 2.0 em 2017)

- **Referência:** Thornton et al., KDD 2013, p. 847-855 [thornton2013autoweka]; Kotthoff et al., JMLR 18(25), 2017
  [kotthoff2017autoweka].
- **Versão atual:** 2.6, julho de 2017 (página do projeto). O GitHub diz que o projeto "não será mais atualizado".
- **Espaço de busca:** algoritmos do WEKA. No artigo de 2013: 27 classificadores base, 10 meta-métodos, 2 métodos de
  ensemble, 3 métodos de busca e 8 avaliadores de seleção de atributos. A Figura 1 da versão 2.0 lista 28 aprendizes,
  2 métodos de ensemble, 7 meta-métodos e 2 buscas de atributos. Só 13 aprendizes fazem regressão (entre eles
  GaussianProcesses, IBk, LinearRegression, MultilayerPerceptron, RandomTree, REPTree, SGD e SMOreg).
- **Estratégia:** otimização bayesiana com SMAC (floresta aleatória como modelo substituto) sobre o espaço CASH.
- **Meta-conhecimento:** nenhum.
- **Modelo final:** uma configuração (algoritmo, hiperparâmetros e seleção de atributos).
- **Orçamento:** `timeLimit` (minutos) e `memLimit` (MB); `parallelRuns` em uma máquina. Cada aprendiz roda no máximo 1/12
  do orçamento.
- **Regressão e métrica própria:** regressão desde a 2.0. Otimiza qualquer métrica do WEKA (`-metric`), mas não aceita
  métrica do usuário sem alterar o código Java.
- **SO e dependências:** Java e WEKA 3.7.13 ou mais novo; roda onde o WEKA roda, inclusive no Windows.
- **Licença:** GPL; o SMAC embutido é AGPLv3.
- **Manutenção:** parada desde 2017.

### Hyperopt-sklearn (2014)

- **Referência:** Komer, Bergstra e Eliasmith, Proc. SciPy 2014, doi:10.25080/Majora-14bd3278-006 [komer2014hyperopt].
- **Versão atual:** 1.1.1, 22/03/2025 (PyPI; exige Python 3.11 ou mais novo).
- **Espaço de busca:** classificadores, regressores e pré-processadores do scikit-learn, mais XGBoost (por exemplo
  `any_regressor`).
- **Estratégia:** Hyperopt. O padrão do código é busca aleatória (`hyperopt.rand.suggest`); o README usa TPE.
- **Meta-conhecimento:** nenhum.
- **Modelo final:** um modelo, retreinado nos dados todos (`refit=True`). Sem ensemble.
- **Orçamento:** `max_evals` e `trial_timeout` (segundos por avaliação). Não há limite global de tempo nem de memória.
- **Regressão e métrica própria:** sim. `loss_fn(y_true, y_pred)` recebe qualquer função a minimizar.
- **SO e dependências:** Python puro; cada avaliação roda num `multiprocessing.Process`. Restrição de SO não declarada.
- **Licença:** BSD.
- **Manutenção:** ativa em 2025 (duas versões em março de 2025).

### Auto-sklearn 1.0 (2015)

- **Referência:** Feurer et al., NeurIPS 28, 2015 [feurer2015efficient].
- **Versão atual:** 0.15.0, 20/09/2022 (PyPI). Último commit no branch `development`: 18/04/2023. Exige
  scikit-learn 0.24.x e Python 3.7 a 3.9 (classificadores do PyPI).
- **Espaço de busca:** no artigo, 15 classificadores, 14 pré-processadores de atributos e 4 de dados, com 110
  hiperparâmetros. A versão atual tem espaço equivalente para regressão (contagem não verificada).
- **Estratégia:** otimização bayesiana com SMAC.
- **Meta-conhecimento:** inicia o SMAC com as melhores configurações dos k datasets mais próximos em meta-atributos
  (`initial_configurations_via_metalearning=25`). No artigo de 2015 a base tinha cerca de 140 datasets (inferido do
  pôster); segundo o AMLB, a versão atual usa 208 datasets do OpenML e 38 meta-atributos. Um dataset alvo do OpenML pode
  estar na base. O AMLB excluiu datasets por nome, mas avisa que cópias com outro nome escapam.
- **Desligar meta-aprendizado:** sim, `initial_configurations_via_metalearning=0`.
- **Modelo final:** ensemble selection de Caruana et al. (2004) sobre as previsões de validação (`ensemble_nbest=50`).
  `ensemble_class=None` ou ensemble de tamanho 1 dá um modelo só.
- **Orçamento:** `time_left_for_this_task` (padrão 3600 s), `per_run_time_limit`, `memory_limit` (padrão 3072 MB por
  processo), `n_jobs` via Dask.
- **Regressão e métrica própria:** sim. `autosklearn.metrics.make_scorer(name, score_func, optimum, greater_is_better)`;
  o exemplo oficial é de classificação, mas o scorer é genérico.
- **SO e dependências:** só Linux. A documentação diz que não é possível rodar no Windows (usa o módulo `resource`) e
  sugere bash do Windows 10, máquina virtual ou Docker. Precisa de compilador C++ e, sem wheel do `pyrfr`, de SWIG.
- **Licença:** BSD-3.
- **Manutenção:** sem versão nova desde 2022; 200 issues abertas.

### TPOT (2016; TPOT 1.x em 2025)

- **Referência:** Olson et al., EvoApplications 2016, LNCS 9597, p. 123-137 [olson2016evoapps]; Olson et al., GECCO 2016,
  p. 485-492 [olson2016gecco]; capítulo no livro de AutoML [olson2019tpot]; TPOT2 em Ribeiro et al., GPTP XX, 2024
  [ribeiro2024tpot2].
- **Versão atual:** 1.1.0, 03/07/2025 (PyPI). A 1.0.0 (26/02/2025) incorporou o TPOT2 e substituiu a linha 0.12
  (última: 0.12.2, 23/02/2024).
- **Espaço de busca:** pipelines do scikit-learn com XGBoost (segundo o AMLB). Na 1.x, `search_space` aceita `linear`
  (padrão), `linear-light`, `graph`, `graph-light`, `mdr` ou um espaço próprio. Faixas de hiperparâmetros adaptadas do
  TPOT antigo e do auto-sklearn.
- **Estratégia:** programação genética multiobjetivo (desempenho e complexidade do pipeline).
- **Meta-conhecimento:** nenhum.
- **Modelo final:** o melhor pipeline (um só; o pipeline pode conter empilhamento interno).
- **Orçamento:** `max_time_mins` (padrão 60), `max_eval_time_mins`, `memory_limit` por worker do Dask, `n_jobs`,
  `early_stop`.
- **Regressão e métrica própria:** sim. `TPOTRegressor(scorers=[...])` aceita scorers do scikit-learn
  (`make_scorer`); a documentação recomenda `functools.partial` para funções com dados extras.
- **SO e dependências:** Python 3.10 a 3.13, Dask. Restrição de SO não declarada; o código precisa de
  `if __name__ == "__main__"` em scripts (relevante no Windows).
- **Licença:** LGPL-3.0.
- **Manutenção:** ativa.

### H2O AutoML (2017; artigo em 2020)

- **Referência:** LeDell e Poirier, 7th ICML Workshop on AutoML, 2020 [ledell2020h2o]. O AutoML apareceu numa versão de
  prévia do H2O em junho de 2017 (blog da H2O.ai).
- **Versão atual:** h2o 3.46.0.12, 12/08/2026 (PyPI).
- **Espaço de busca:** portfólio fixo (3 XGBoost, 5 GBM e um DRF pré-definidos, grade fixa de GLM, XRT, rede profunda
  quase padrão) mais buscas aleatórias em XGBoost, GBM e redes. Pré-processamento opcional: só `target_encoding`
  (experimental).
- **Estratégia:** portfólio escolhido pelos autores mais busca aleatória com parada antecipada, intercalados em grupos de
  prioridade.
- **Meta-conhecimento:** a documentação não diz como o portfólio foi escolhido. Não há meta-aprendizado por dataset.
- **Modelo final:** dois stacked ensembles (todos os modelos; o melhor de cada família) com GLM não negativo como
  meta-aprendiz. O líder do leaderboard pode ser um modelo só ou um ensemble.
- **Orçamento:** `max_runtime_secs` (1 h se nada for dado), `max_models`, `max_runtime_secs_per_model`, `nfolds`;
  memória e núcleos via `h2o.init`.
- **Regressão e métrica própria:** regressão sim. Métrica própria não: `sort_metric` e `stopping_metric` aceitam só
  métricas internas (deviance, MSE, RMSE, MAE, RMSLE etc.). `custom_metric_func` existe para GBM, GLM, DRF, Deep
  Learning, Stacked Ensembles e XGBoost isolados, não para o AutoML. Dá para reordenar o leaderboard depois com a nossa
  métrica, mas a busca não a usa.
- **SO e dependências:** Java 8 a 17 (JRE 64 bits). Windows 7 ou mais novo é suportado, mas o XGBoost do H2O não roda no
  Windows. No Windows nativo o portfólio perde os modelos XGBoost.
- **Licença:** Apache-2.0.
- **Manutenção:** ativa.

### ML-Plan (2018)

- **Referência:** Mohr, Wever e Hüllermeier, Machine Learning 107(8-10):1495-1515, 2018 [mohr2018mlplan].
- **Versão atual:** distribuída em `download.mlplan.org`; data da última versão não verificada. Código no monorepo
  `starlibs/AILibs`.
- **Espaço de busca:** pipelines do WEKA ou do scikit-learn.
- **Estratégia:** planejamento hierárquico (HTN) com busca em grafo e avaliações aleatórias de completamento.
- **Meta-conhecimento:** nenhum (não verificado no artigo).
- **Modelo final:** um pipeline.
- **Orçamento:** tempo total e número de CPUs (`-t`, `-ncpus` no AMLB); memória via JVM.
- **Regressão e métrica própria:** regressão sim (flag `-regression` no AMLB). Métricas fixas (erro, AUC, logloss, F1,
  R2, RMSE, MSE, RMSLE, MAE); métrica própria exigiria código Java.
- **SO e dependências:** Java (o AMLB instala OpenJDK 11).
- **Licença:** AGPL-3.0.
- **Manutenção:** não verificado.

### AutoXGBoost (2018)

- **Referência:** Thomas, Coors e Bischl, ICML 2018 AutoML Workshop, arXiv:1807.03873 [thomas2018autoxgboost].
- **Resumo:** só gradient boosting, com tuning baseado em modelo (mlr/mlrMBO, R), ajuste de limiar e codificação de
  categóricas. Os próprios autores retiraram o sistema do AMLB por depender de software obsoleto. Não recomendado.

### Oboe e TensorOboe (2019; 2020)

- **Referência:** Yang et al., KDD 2019, p. 1173-1183 [yang2019oboe]; Yang et al., KDD 2020, p. 1446-1456
  [yang2020tensoroboe].
- **Versão atual:** 0.2.0, 23/10/2021 (PyPI). Último commit: 23/10/2021.
- **Espaço de busca:** Oboe escolhe estimadores (dados já pré-processados); TensorOboe escolhe pipelines completos
  (imputação, codificação, padronização, redução de dimensão, estimador).
- **Estratégia:** filtragem colaborativa. Uma matriz (ou tensor) de erros de modelos em datasets é fatorada; o sistema
  avalia poucos modelos no alvo e infere o resto, sob orçamento de tempo.
- **Meta-conhecimento:** é o núcleo do método. A matriz de erros vem de meta-treino com datasets do OpenML (quantidade não
  verificada) e vem pronta no pacote.
- **Desligar meta-aprendizado:** não faz sentido; sem a matriz o método não existe.
- **Modelo final:** um modelo ou ensemble dos selecionados.
- **Orçamento:** `runtime_limit`.
- **Regressão e métrica própria:** não. A integração do AMLB levanta erro em regressão ("not yet supported") e ignora a
  métrica.
- **SO e dependências:** Python 3.7 ou mais novo. SO não declarado.
- **Licença:** BSD-3.
- **Manutenção:** parada desde 2021.

### AutoKeras (2019)

- **Referência:** Jin, Song e Hu, KDD 2019, p. 1946-1956 [jin2019autokeras].
- **Resumo:** busca de arquitetura neural (NAS) com Keras. O `StructuredDataRegressor` saiu na versão 2.0 e voltou na 3.0
  (notas de versão do GitHub). Só redes neurais, depende de GPU para ser competitivo (o AMLB o excluiu por isso). Não faz
  escolha entre famílias de modelos tabulares; não recomendado como baseline de seleção de algoritmo.

### GAMA (2019)

- **Referência:** Gijsbers e Vanschoren, JOSS 4(33):1132, 2019 [gijsbers2019gama]; ECML PKDD 2020 Demo Track, LNCS,
  p. 560-564, 2021 [gijsbers2021gama].
- **Versão atual:** 23.0.0.post1, 14/06/2024 (PyPI). Repositório arquivado (somente leitura) em 01/10/2026.
- **Espaço de busca:** pipelines lineares do scikit-learn com número livre de pré-processadores.
- **Estratégia:** programação genética assíncrona com NSGA-II (padrão `AsyncEA`). O artigo de demonstração cita três
  algoritmos de busca; os outros dois não foram verificados.
- **Meta-conhecimento:** nenhum.
- **Modelo final:** `EnsemblePostProcessing` (ensemble selection de Caruana) no preset de desempenho; `BestFit` no preset
  simples.
- **Orçamento:** `max_total_time` (padrão 3600 s), `max_eval_time`, `max_memory_mb`, `n_jobs`.
- **Regressão e métrica própria:** regressão sim. Métrica própria na prática não: `Metric` só aceita scorers registrados
  no `SCORERS` do scikit-learn e com nome conhecido como de classificação ou regressão (conferido em
  `gama/utilities/metrics.py`). Exigiria alterar o código.
- **SO e dependências:** Python 3.9 ou mais novo; SO não declarado.
- **Licença:** Apache-2.0 no repositório, MIT no PyPI (conflito entre as fontes).
- **Manutenção:** encerrada (arquivado).

### AutoGluon-Tabular (2020)

- **Referência:** Erickson et al., arXiv:2003.06505, 2020 [erickson2020autogluon]. Portfólio aprendido: Salinas e
  Erickson, TabRepo, AutoML Conference 2024 [salinas2024tabrepo].
- **Versão atual:** autogluon.tabular 1.6.3, 18/09/2026 (PyPI).
- **Espaço de busca:** famílias fixas de modelos (boosting, florestas, redes neurais e outras; lista exata por versão
  não verificada) e, nas versões novas, modelos de fundação tabulares. O preset `extreme_quality` da 1.6 usa modelos de fundação (TabICL,
  TabDPT e outro) e exige GPU; o preset `noncommercial` acrescenta o TabPFN-3. Pré-processamento automático de tipos.
- **Estratégia:** não faz busca de hiperparâmetros por padrão. Treina um portfólio fixo de configurações com bagging e
  empilhamento em várias camadas.
- **Meta-conhecimento:** nos presets `best_quality` e `high_quality`, o portfólio `zeroshot` (cerca de 100 configurações)
  vem da simulação de ensembles do TabRepo em 200 datasets de classificação e regressão do OpenML. Desde a 1.0 esse
  portfólio aprendido substituiu o portfólio feito à mão. Um alvo do OpenML pode ter ajudado a escolher o portfólio.
- **Desligar meta-aprendizado:** sim, com `hyperparameters='default'` (usado pelo `medium_quality`, o preset padrão) ou
  com um dicionário de modelos próprio. Se o portfólio `default` também foi ajustado em benchmarks: não verificado.
- **Modelo final:** ensemble ponderado (ensemble selection de Caruana) sobre a última camada de empilhamento.
- **Orçamento:** `time_limit` (s), `num_cpus`, `num_gpus`, `memory_limit` (GB, limite brando), `num_bag_folds`,
  `num_stack_levels`.
- **Regressão e métrica própria:** sim. `autogluon.core.metrics.make_scorer(name, score_func, optimum,
  greater_is_better)`; a métrica orienta a validação, a escolha do modelo e os pesos do ensemble. A documentação manda
  definir a métrica num arquivo `.py` separado para o pickle funcionar com Ray.
- **SO e dependências:** Linux, macOS e Windows (Python 3.10 a 3.13). No Windows a documentação recomenda Anaconda.
- **Licença:** Apache-2.0. O TabPFN-3 do preset `noncommercial` tem licença não comercial.
- **Manutenção:** ativa.

### MLJAR-supervised (software; citação de 2021)

- **Referência:** Płońska e Płoński, MLJAR, software citado pelo AMLB como 2021 [plonska2021mljar]. Título exato da
  citação de 2021 não verificado. O README de 2026 cita um artigo "to appear" no PMLR.
- **Versão atual:** 1.3.2, 27/07/2026; 1.4.0rc6 em pré-lançamento (05/11/2026). Ano da primeira versão no PyPI não
  verificado.
- **Espaço de busca:** Baseline, Linear, árvore de decisão, floresta aleatória, Extra Trees, LightGBM, XGBoost, CatBoost,
  rede neural, k-NN; extras opcionais (RealMLP, TabM, TabICLv2). Engenharia de atributos ("golden features", k-means),
  seleção de atributos por atributo aleatório de sonda.
- **Estratégia:** sequência fixa de etapas. Modelos simples, um modelo padrão por algoritmo, busca aleatória em conjuntos
  pré-definidos ("not so random"), engenharia e seleção de atributos, depois hill climbing nos melhores. O modo `Optuna`
  troca isso por TPE do Optuna.
- **Meta-conhecimento:** valores padrão e grades estão no código; a documentação não diz se foram aprendidos de dados.
  Sem meta-aprendizado por dataset.
- **Modelo final:** ensemble guloso ponderado (até 30 rodadas), empilhamento e ensemble dos empilhados no modo `Compete`.
- **Orçamento:** `total_time_limit` (padrão 3600 s), `model_time_limit`, `n_jobs`.
- **Regressão e métrica própria:** sim. `eval_metric` aceita função `f(y_true, y_predicted, sample_weight=None)`; o
  valor é sempre minimizado (passar o negativo da F1).
- **SO e dependências:** Python 3.9 ou mais novo (o README novo pede 3.11). Restrição de SO não declarada; funcionamento
  no Windows não verificado.
- **Licença:** MIT.
- **Manutenção:** ativa.

### Auto-sklearn 2.0 (2020; JMLR 2022)

- **Referência:** Feurer et al., JMLR 23(261):1-61, 2022 [feurer2022autosklearn2].
- **Versão atual:** a mesma do auto-sklearn (0.15.0), módulo `autosklearn.experimental.askl2`.
- **Espaço de busca:** só algoritmos de aprendizado iterativo e quase nenhum pré-processamento (AMLB). Lista exata não
  verificada.
- **Estratégia:** otimização bayesiana com successive halving; um seletor de política escolhe a estratégia de validação e
  de alocação de orçamento.
- **Meta-conhecimento:** portfólio estático de pipelines, construído por seleção gulosa sobre centenas de datasets
  (AMLB); meta-modelo para escolher a política. Não depende de meta-atributos.
- **Desligar meta-aprendizado:** não. O AMLB diz que não dá para excluir datasets do portfólio nem dos meta-modelos.
- **Modelo final:** ensemble selection.
- **Regressão e métrica própria:** só classificação (`AutoSklearn2Classifier`); não há `AutoSklearn2Regressor`.
- **SO, licença e manutenção:** iguais aos do auto-sklearn 1.0.

### FLAML (2021)

- **Referência:** Wang, Wu, Weimer e Zhu, MLSys 2021 [wang2021flaml]; busca CFO em Wu, Wang e Huang, AAAI 2021
  [wu2021cfo]; padrões minerados em Kayali e Wang, arXiv:2202.09927 [kayali2022mining].
- **Versão atual:** 2.8.0, 06/11/2026 (PyPI).
- **Espaço de busca:** lgbm, xgboost, xgb_limitdepth, rf, extra_tree, histgb, catboost, kneighbor (regressão), modelos
  lineares em alguns modos. Pouco pré-processamento.
- **Estratégia:** escolhe o aprendiz pelo custo estimado de melhorar (ECI) e ajusta cada um com CFO (busca local
  aleatorizada que parte de configurações baratas); BlendSearch no modo paralelo.
- **Meta-conhecimento:** opcional. `starting_points="data"` usa padrões dependentes de dados minerados offline
  [kayali2022mining], avaliados em 62 datasets de classificação e regressão; a origem dos datasets de mineração não foi
  verificada. Os exemplos pedem o modo explicitamente.
- **Desligar meta-aprendizado:** sim, `starting_points="static"` ou omitir `"data"`.
- **Modelo final:** o melhor modelo; `ensemble=True` faz empilhamento.
- **Orçamento:** `time_budget` (s), `max_iter`, `n_jobs`, `n_concurrent_trials`.
- **Regressão e métrica própria:** sim. `metric` aceita função com assinatura `(X_val, y_val, estimator, labels,
  X_train, y_train, ...)` que devolve o valor a minimizar; a documentação tem exemplo de regressão.
- **SO e dependências:** "OS Independent" no PyPI; Python 3.10 a 3.13.
- **Licença:** MIT.
- **Manutenção:** ativa.

### LightAutoML (2021)

- **Referência:** Vakhrushev et al., arXiv:2109.01528, 2021 [vakhrushev2021lightautoml].
- **Versão atual:** 0.4.2, 04/12/2025 (PyPI); 0.4.3.dev1 em 15/01/2026.
- **Espaço de busca:** segundo o AMLB, modelos lineares e boosting (LightGBM, CatBoost); redes neurais em presets
  próprios.
- **Estratégia:** regras de especialistas definem o espaço; TPE (Optuna) ajusta o boosting; busca em grade com
  warm-start e parada antecipada ajusta os lineares (AMLB).
- **Meta-conhecimento:** regras fixas de especialistas; sem meta-aprendizado por dataset.
- **Modelo final:** votação ponderada (regressão e binária) ou empilhamento em dois níveis (multiclasse). O preset
  "utilized" repete o AutoML com outras configurações e combina os resultados.
- **Orçamento:** tempo (`timeout`) e CPUs (parâmetros exatos não verificados).
- **Regressão e métrica própria:** sim. `Task('reg', loss=..., metric=callable, greater_is_better=...)`. Assinatura da
  função não documentada na página da API.
- **SO e dependências:** PyPI declara "OS Independent", Python 3.8 ou mais novo. O README traz instruções de instalação
  para Windows (dependências do relatório em PDF). Suporte oficial ao Windows não verificado.
- **Licença:** Apache-2.0.
- **Manutenção:** ativa.

### Auto-PyTorch Tabular (2021)

- **Referência:** Zimmer, Lindauer e Hutter, IEEE TPAMI 43(9):3079-3090, 2021 [zimmer2021autopytorch].
- **Versão atual:** 0.2.1, 23/08/2022 (PyPI). Último commit no `development`: 23/08/2022.
- **Espaço de busca:** redes neurais para dados tabulares e seu pré-processamento. Arquiteturas exatas e presença de
  modelos tradicionais no ensemble não verificadas.
- **Estratégia:** SMAC com Hyperband (multifidelidade).
- **Meta-conhecimento:** portfólio guloso para iniciar o SMAC (`portfolio_selection`). Desligado por padrão.
- **Modelo final:** ensemble selection de Caruana.
- **Orçamento:** `total_walltime_limit`, `func_eval_time_limit_secs`, `memory_limit` (padrão 4096 MB), `budget_type`
  (épocas ou tempo).
- **Regressão e métrica própria:** regressão sim. `optimize_metric` recebe o nome de uma métrica interna; métrica própria
  não documentada.
- **SO e dependências:** só Linux, compilador C++ e SWIG 3.0.x.
- **Licença:** Apache-2.0 no README, BSD no PyPI.
- **Manutenção:** parada desde 2022.

### mlr3automl (2021)

- **Referência:** repositório `a-hanf/mlr3automl` e palestra no useR! 2021 [hanf2021mlr3automl].
- **Resumo:** AutoML em R sobre o mlr3; classificação e regressão; portfólio inicial e Hyperband (commits e
  documentação do repositório). Último commit em 20/02/2022. LGPL-3.0. O AMLB não conseguiu avaliá-lo. Métrica
  própria via medidas do mlr3 (não verificado).

### FEDOT (2021; FGCS 2022)

- **Referência:** Nikitin et al., Future Generation Computer Systems 127:109-125, 2022 [nikitin2022fedot].
- **Versão atual:** 0.7.5, 10/03/2025 (PyPI).
- **Espaço de busca:** pipelines em forma de grafo com blocos de pré-processamento e modelos.
- **Estratégia:** algoritmo evolutivo que compõe o grafo, mais tuning de hiperparâmetros.
- **Meta-conhecimento:** não declarado.
- **Modelo final:** um pipeline composto (que pode conter vários modelos).
- **Orçamento:** `timeout` (unidade não verificada), presets (`best_quality` etc.).
- **Regressão e métrica própria:** regressão sim. O README diz que o tuning aceita métricas próprias; uso de métrica
  própria na composição não verificado.
- **SO e dependências:** Python 3.8 a 3.10 (classificadores do PyPI); SO não declarado.
- **Licença:** BSD-3.
- **Manutenção:** última versão em março de 2025.

### NaiveAutoML (2022; Machine Learning 2023)

- **Referência:** Mohr e Wever, Machine Learning 112(4):1131-1170, publicado online em 29/09/2022, edição de abril de
  2023 [mohr2023naive].
- **Versão atual:** 0.1.6, 25/01/2025 (PyPI).
- **Espaço de busca:** pipelines do scikit-learn (escalonamento, seleção de atributos, estimador).
- **Estratégia:** sequencial e gulosa, como faria um cientista de dados. Escolhe o aprendiz com hiperparâmetros padrão,
  depois otimiza escalonamento, seleção de atributos e, por fim, hiperparâmetros. Para sozinho quando não espera
  melhora.
- **Meta-conhecimento:** nenhum.
- **Modelo final:** um pipeline.
- **Orçamento:** `timeout_overall`, `timeout_candidate` (nem sempre respeitado), `max_hpo_iterations`.
- **Regressão e métrica própria:** sim. `scoring` aceita `make_scorer` ou a tupla `(nome, scorer)`.
- **SO e dependências:** Python; SO não declarado.
- **Licença:** MIT.
- **Manutenção:** última versão em janeiro de 2025.

### SapientML (2022)

- **Referência:** Saha et al., ICSE 2022, p. 1932-1944 [saha2022sapientml].
- **Resumo:** sintetiza pipelines aprendendo de soluções humanas minadas do Kaggle: meta-aprendizado prevê componentes,
  depois poucos pipelines concretos são avaliados. Classificação e regressão. Python 3.10 a 3.13, Apache-2.0. A
  integração do AMLB não passa métrica nem limite de tempo. Métrica própria não verificada. O meta-conhecimento vem de
  competições do Kaggle, então há risco de sobreposição com datasets públicos.

### ML.NET AutoML (C#)

- **Referência:** documentação da Microsoft [mlnet2024automl].
- **Resumo:** AutoML do ML.NET (pacote `Microsoft.ML.AutoML`, API nova a partir da 0.20). Classificação binária,
  multiclasse e regressão. Sintonizadores: Eci Cost Frugal (padrão), Cost Frugal, SMAC, grade e aleatório. Métrica
  própria só escrevendo um `ITrialRunner` em C#. Exige .NET. O AMLB não conseguiu avaliá-lo. Fora do nosso ecossistema
  Python.

### Fora do escopo, mas citáveis

- **TabPFN v2** (Hollmann et al., Nature 637(8045):319-326, 2025 [hollmann2025tabpfn]): modelo de fundação; prevê em
  contexto, sem busca. Não é AutoML. Já é baseline no nosso paper e aparece como modelo dentro do AutoGluon.
- **CAAFE** (Hollmann, Müller e Hutter, NeurIPS 2023 [hollmann2023caafe]): LLM gera atributos; não escolhe modelo.
- **AutoML-Agent** (Trirat, Jeong e Hwang, ICML 2025 [trirat2025automlagent]): agentes LLM para o pipeline inteiro,
  incluindo busca de modelo. Depende de API de LLM e não é reprodutível no mesmo sentido; o uso em tabular não foi
  verificado no texto completo.
- Levantamentos gerais: He, Zhao e Chu, Knowledge-Based Systems 212:106622, 2021 [he2021automl]; Baratchi et al.,
  Artificial Intelligence Review 57(5), 2024 [baratchi2024automl].

## Tabela resumo

"Meta" indica uso de conhecimento vindo de outros datasets. "Métrica própria" refere-se a regressão com função do
usuário.

| Sistema | 1º artigo | Versão atual (data) | Estratégia | Meta | Desliga meta | Modelo final | Métrica própria em regressão | Windows nativo | Licença | Manutenção |
|---|---|---|---|---|---|---|---|---|---|---|
| Auto-WEKA | 2013 | 2.6 (07/2017) | BO (SMAC) | não | n/a | um modelo | não (só métricas do WEKA) | sim (Java) | GPL | parada |
| Hyperopt-sklearn | 2014 | 1.1.1 (03/2025) | aleatória ou TPE | não | n/a | um modelo | sim (`loss_fn`) | sem restrição declarada | BSD | ativa |
| Auto-sklearn 1.0 | 2015 | 0.15.0 (09/2022) | BO (SMAC) | kNN warm-start, OpenML | sim | ensemble selection | sim (`make_scorer`) | não (WSL/Docker) | BSD-3 | parada |
| TPOT | 2016 | 1.1.0 (07/2025) | programação genética | não | n/a | um pipeline | sim (scorer) | sem restrição declarada | LGPL-3 | ativa |
| H2O AutoML | 2017/2020 | 3.46.0.12 (08/2026) | portfólio + aleatória | portfólio fixo | n/a | stacked ensemble | não | sim, sem XGBoost | Apache-2 | ativa |
| ML-Plan | 2018 | não verificado | planejamento HTN | não | n/a | um pipeline | não (lista fixa) | sim (Java) | AGPL-3 | não verificado |
| Oboe/TensorOboe | 2019 | 0.2.0 (10/2021) | filtragem colaborativa | matriz de erros, OpenML | não | modelo ou ensemble | não (só classificação) | não declarado | BSD-3 | parada |
| GAMA | 2019 | 23.0.0.post1 (06/2024) | GP assíncrona (NSGA-II) | não | n/a | ensemble selection | não (só scorers registrados) | não declarado | Apache-2/MIT | arquivado |
| AutoGluon | 2020 | 1.6.3 (09/2026) | portfólio + bagging + stacking | portfólio do TabRepo (200 datasets OpenML) | sim (`default`) | ensemble ponderado | sim (`make_scorer`) | sim | Apache-2 | ativa |
| Auto-sklearn 2.0 | 2020/2022 | 0.15.0 (09/2022) | BO + successive halving | portfólio + seletor de política | não | ensemble selection | não (sem regressão) | não | BSD-3 | parada |
| FLAML | 2021 | 2.8.0 (11/2026) | ECI + CFO | padrões minerados (opcional) | sim | um modelo ou stacking | sim (função) | sim ("OS Independent") | MIT | ativa |
| LightAutoML | 2021 | 0.4.2 (12/2025) | regras + TPE + grade | regras de especialistas | n/a | votação ponderada | sim (callable) | não verificado | Apache-2 | ativa |
| MLJAR-supervised | 2021 (software) | 1.3.2 (07/2026) | aleatória + hill climbing | não | n/a | ensemble guloso + stacking | sim (função, minimiza) | não verificado | MIT | ativa |
| Auto-PyTorch | 2021 | 0.2.1 (08/2022) | SMAC + Hyperband | portfólio (opcional) | sim | ensemble selection | não documentada | não (só Linux) | Apache-2/BSD | parada |
| mlr3automl | 2021 | sem versão (02/2022) | portfólio + Hyperband | portfólio | não verificado | não verificado | não verificado | R | LGPL-3 | parada |
| FEDOT | 2021/2022 | 0.7.5 (03/2025) | evolutivo em grafo | não declarado | n/a | pipeline composto | parcial (no tuning) | não declarado | BSD-3 | lenta |
| NaiveAutoML | 2022/2023 | 0.1.6 (01/2025) | gulosa sequencial | não | n/a | um pipeline | sim (`make_scorer`) | não declarado | MIT | lenta |
| SapientML | 2022 | não verificado | síntese de programas | corpus do Kaggle | não | um pipeline | não verificado | não declarado | Apache-2 | não verificado |
| ML.NET AutoML | n/a | não verificado | Eci-CFO, SMAC, grade | não | n/a | um modelo | só via C# | sim (.NET) | MIT (não verificado) | não verificado |

## Conjunto recomendado para Windows 11 com 12 núcleos

Critérios: fazer regressão, aceitar a F1 por relevância como métrica de busca, cobrir anos e estratégias diferentes, e
incluir sistemas com e sem meta-conhecimento. O comparador central é o auto-sklearn 1.0, porque é o caso clássico de
meta-aprendizado por similaridade de datasets e permite desligar esse meta-aprendizado. Rodar com e sem
`initial_configurations_via_metalearning` isola o efeito do meta-conhecimento externo.

Núcleo (seis sistemas):

1. **Auto-sklearn 1.0 (2015), no WSL.** BO com SMAC e warm-start por k vizinhos; ensemble selection. Duas versões:
   meta ligado e desligado. Ambiente isolado com Python 3.9 e scikit-learn 0.24 (exigências do `requirements.txt`).
   `n_jobs=12` e `memory_limit` por processo; com 12 processos de 3 GB são 36 GB, então ajustar à RAM da máquina.
2. **TPOT 1.x (2016), Windows nativo.** Programação genética, sem meta-conhecimento. `n_jobs=12`, scorer via
   `make_scorer`.
3. **AutoGluon (2020), Windows nativo.** Portfólio fixo com bagging e stacking. Duas versões: `best_quality`
   (portfólio aprendido no TabRepo) e `hyperparameters='default'` (sem portfólio aprendido). `num_cpus=12`, métrica em
   módulo separado. Evitar presets que exigem GPU.
4. **FLAML (2021), Windows nativo.** Busca de baixo custo (ECI + CFO). Duas versões: `starting_points="static"` e
   `"data"` (padrões minerados).
5. **MLJAR-supervised (2021), Windows nativo, se instalar sem erro.** Busca aleatória mais hill climbing com engenharia
   de atributos; modo `Compete`. A métrica é minimizada, então passar `-F1`. Se falhar no Windows, rodar no WSL.
6. **NaiveAutoML (2022), Windows nativo.** Linha de base gulosa e sem meta-conhecimento, proposta pelos autores como
   referência mínima para AutoML.

Complementos opcionais:

- **Hyperopt-sklearn (2014):** o sistema mais antigo que aceita métrica própria em Python; TPE puro sobre o
  scikit-learn. Útil para cobrir 2014.
- **LightAutoML (2021), no WSL:** regras de especialistas mais TPE; aceita métrica própria. Prefiro o WSL até
  confirmar o Windows.
- **H2O AutoML (2017/2020):** portfólio mais busca aleatória e stacking. Busca com RMSE ou MAE e reordenação do
  leaderboard pela F1 depois. Deve aparecer no texto como "busca com métrica interna". No Windows perde o XGBoost; no WSL
  não.

Cuidados comuns. A F1 por relevância depende da função de relevância φ, que é ajustada sobre a distribuição do alvo. A φ
deve ser calculada uma vez no treino de cada fold externo e passada à métrica por `functools.partial` ou fechamento,
para que as validações internas de cada sistema usem a mesma φ. Todos devem receber o mesmo orçamento de tempo por fold
e os mesmos 12 núcleos. Para auto-sklearn e AutoGluon com meta ligado, registrar quais dos nossos datasets estão no
OpenML; ambos podem ter visto esses datasets no meta-treino.

## Sistemas que não servem e por quê

Sem métrica própria em regressão:

- **Auto-sklearn 2.0:** não faz regressão.
- **Oboe/TensorOboe:** a integração do AMLB recusa regressão e ignora a métrica.
- **H2O AutoML:** só métricas internas na busca e no leaderboard; o `custom_metric_func` vale para algoritmos isolados.
- **Auto-WEKA e ML-Plan:** só métricas pré-definidas; métrica nova exige código Java.
- **GAMA:** só aceita scorers registrados no scikit-learn com nome conhecido; além disso, foi arquivado em 10/2026.
- **Auto-PyTorch:** métrica por nome interno; métrica própria não documentada.

Sem Windows nativo:

- **Auto-sklearn 1.0 e 2.0:** usam o módulo `resource`; a documentação diz que não rodam no Windows. Solução: WSL ou
  Docker.
- **Auto-PyTorch:** só Linux, com SWIG 3.0.x. Além disso, está parado desde 2022.
- **H2O AutoML:** roda, mas o XGBoost do H2O não existe no Windows.
- **LightAutoML e MLJAR-supervised:** sem restrição declarada, mas o funcionamento no Windows não foi verificado.

Fora por outros motivos: AutoXGBoost (software obsoleto, segundo os autores), AutoKeras (só NAS, depende de GPU),
mlr3automl (R, parado desde 2022), SapientML (meta-conhecimento do Kaggle, sem controle de tempo na integração do AMLB),
ML.NET (C#), CAAFE (não escolhe modelo), TabPFN (não faz busca) e agentes LLM (dependem de API externa).
