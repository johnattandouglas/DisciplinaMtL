# Escolha dos sistemas de AutoML (Atividade 4, HPO)

Comparamos três sistemas: Auto-sklearn (principal), AutoGluon e FLAML (suplementares, nos apêndices). Usamos dois
critérios: popularidade, medida por citações do artigo de cada sistema, e presença no AutoML Benchmark (AMLB; Gijsbers et al.,
2024, `gijsbers2024amlb`).

| Sistema | Artigo (chave em `automl_refs.bib`) | Citações (Semantic Scholar, 09/11/2026) | No AMLB |
|---|---|---|---|
| Auto-sklearn | Feurer et al., NeurIPS 2015 (`feurer2015efficient`) | 2.048 | sim |
| AutoGluon | Erickson et al., 2020 (`erickson2020autogluon`) | 1.142 | sim |
| FLAML | Wang et al., MLSys 2021 (`wang2021flaml`) | 364 | sim |
| TPOT (reserva) | Olson et al., GECCO 2016 (`olson2016gecco`) | 657 | sim |
| AMLB | Gijsbers et al., JMLR 2024 (`gijsbers2024amlb`) | 137 | |

As contagens vêm da API do Semantic Scholar (campo `citationCount`), consultada em 09/11/2026. O Google Scholar costuma dar
números maiores; não o consultamos porque ele bloqueia acesso automatizado. O TPOT tem mais citações que o FLAML, mas ficou
fora: ele não respeita orçamentos curtos de tempo (termina a geração em curso), e o orçamento curto é o que se compara ao custo
online da meta-aprendizagem. Ele seria o substituto do Auto-sklearn se o WSL não estivesse disponível. H2O e Auto-WEKA ficaram
fora por decisão do autor (exigem Java).

Os três cobrem estratégias diferentes. O Auto-sklearn faz otimização bayesiana (SMAC) sobre pipelines do scikit-learn, com
*warm-start* por meta-características: começa pelas 25 configurações que foram boas nos datasets do OpenML mais próximos do
alvo, e termina com um *ensemble* guloso (Feurer et al., 2015). É o mesmo princípio do SMBO-Similarity da Atividade 3, o que
permite a comparação direta; por isso rodamos o Auto-sklearn com a meta-aprendizagem ligada e desligada
(`initial_configurations_via_metalearning=0`). O AutoGluon não busca hiperparâmetros: treina um portfólio fixo de modelos e
os combina por empilhamento e *ensemble* ponderado (Erickson et al., 2020). O FLAML faz busca de baixo custo, que começa pelas
configurações baratas e só cresce o custo quando o ganho compensa (Wang et al., 2021).

Atenção: o meta-dataset do Auto-sklearn vem do OpenML, que contém vários datasets do UCI. Parte dos 120 datasets alvo pode
estar no conhecimento prévio do Auto-sklearn; o cruzamento de nomes com o OpenML está em `resultados/sobreposicao_openml.csv`.
