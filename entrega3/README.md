# Atividade Prática 3 — Warm-starting da otimização de hiperparâmetros

Código, dados e resultados da Atividade 3 (IN1097). O relatório está em [`relatorio/Relatorio3.pdf`](relatorio/Relatorio3.pdf).
O trabalho está dividido em **quatro notebooks, um por etapa, com as saídas executadas**; cada um grava o que o próximo lê, então
dá para abrir e avaliar qualquer etapa sem rodar os experimentos de novo:

| Notebook | Etapa | Grava |
|---|---|---|
| [`01_Etapa1-MetaDados.ipynb`](01_Etapa1-MetaDados.ipynb) | Etapa 1: Θ, configuração padrão, matriz de perdas K × n, custo | `results/matriz_de_perdas.csv`, `results/configuracoes.csv` |
| [`02_MetaCaracteristicas-OurTrained.ipynb`](02_MetaCaracteristicas-OurTrained.ipynb) | meta-características; treino da Our-Trained (1024 dims) | `representations/Our-Trained_split{k}.csv`, `results/models/` |
| [`03_Etapa2-Buscas.ipynb`](03_Etapa2-Buscas.ipynb) | Etapa 2: **roda e salva** as buscas (métodos 1–5) | `results/curvas/*.npz` |
| [`04_Analises.ipynb`](04_Analises.ipynb) | Análises 1, 2 e 3 — **só lê** os resultados | `results/tables/`, `results/figures/` |

**Matriz de perdas (K × n = 3456 × 120):** [`results/matriz_de_perdas.csv`](results/matriz_de_perdas.csv) — uma linha por
configuração (`config_id`), uma coluna por dataset; perda = 1 − acurácia de validação na época 50. As configurações
(`config_id` → hiperparâmetros, com a marca da configuração padrão) estão em [`results/configuracoes.csv`](results/configuracoes.csv).

## Problema

Trocamos o problema das Atividades 1 e 2 pela meta-task de *warm-start* de HPO do paper do Dataset2Vec (Jomaa et al., 2021,
seção 6.3): 120 datasets de classificação do UCI (a coleção pré-processada de Fernández-Delgado et al., a mesma da Atividade 1)
× 3456 configurações de uma rede feed-forward. A pergunta passa a ser *com qual configuração começar a busca* da rede.

| Item | Valor |
|---|---|
| Algoritmo base | rede feed-forward (MLP), 50 épocas |
| Θ (7 hiperparâmetros) | ativação {relu, leaky_relu, selu} · forma {ASC, DES, ENC, SQU, SYM} · dropout {0, 0.2, 0.5} · camadas {1, 3, 5, 7} · neurônios {4, 8, 16, 32} · batch-norm {0, 1} · otimizador {adam, rmsprop, sgd} |
| K | 3456 (grid completo; com 1 camada só existe a forma SQU) |
| Configuração padrão | relu, SQU, dropout 0, 1 camada de 32 neurônios, sem batch-norm, adam (ponto do grid mais próximo do default do `MLPClassifier`) |
| Protocolo | hold-out fixo do meta-dataset (treino → validação), igual para as K × n células |
| T / n_init / sementes | 100 / 20 / 5 |
| SMBO (métodos 3–5) | GP Matérn 3/2 com ARD + Expected Improvement |
| Meta-características | clássicas: PyMFE, MF1 (Wistuba et al.), MF2 (Feurer et al.); aprendidas: D2V (Jomaa et al.), **Our-Trained** (1024 dims, treinada para esta meta-task) |
| Validação | leave-one-dataset-out (D2V e Our-Trained: extrator do split em que o alvo é de teste, que nunca viu o alvo) |

## Resultados principais

Perda normalizada média da melhor configuração até a avaliação i (120 alvos × 5 sementes; menor = melhor), warm-start com k = 1:

| Método | t = 20 | t = 50 | t = 100 | AUC |
|---|--:|--:|--:|--:|
| WS-PyMFE | **0,073** | **0,045** | **0,034** | **0,055** |
| WS-D2V | 0,076 | 0,048 | 0,035 | 0,058 |
| WS-MF2 | 0,077 | 0,047 | 0,037 | 0,059 |
| WS-MF1 | 0,077 | 0,049 | 0,036 | 0,059 |
| WS-Our-Trained (1024) | 0,085 | 0,048 | 0,037 | 0,061 |
| WS independente (rank médio) | 0,076 | 0,053 | 0,038 | 0,060 |
| SMBO frio | 0,084 | 0,052 | 0,038 | 0,065 |
| Random Search | 0,083 | 0,061 | 0,047 | 0,072 |
| Padrão | 0,170 | 0,170 | 0,170 | 0,170 |

- **Fase da busca:** o warm-start ajuda logo após os pontos iniciais (t = 20: WS-PyMFE e WS-D2V vencem o SMBO frio, Wilcoxon-Holm
  p ≈ 0,03–0,04); a vantagem some até T = 100, quando todos os SMBO empatam.
- **Meta-características × portfólio:** com k = 1 nenhuma representação supera o WS independente nem vizinhos arbitrários
  (controle WS-Random). Com k = 5, as clássicas (PyMFE, MF1, MF2) vencem o WS-Random na AUC (p<sub>Holm</sub> ≤ 0,015); as
  aprendidas (D2V, Our-Trained) não. O ganho vem sobretudo de agregar boas configurações; a similaridade acrescenta pouco.
- **Our-Trained (1024):** o pior warm-start — a CE de treino é atingida por um preditor constante, então z não separa datasets.
- **Falhas:** o ganho não se relaciona com a distância ao vizinho (|ρ| ≤ 0,11), mas sim com a similaridade real entre as paisagens
  de perda do alvo e do vizinho (ρ ≈ 0,28–0,38); as falhas se concentram em datasets pequenos e ruidosos.

## Como reproduzir

Ambiente: Python 3.12 com `numpy`, `pandas`, `scipy`, `scikit-learn`, `statsmodels`, `matplotlib`, `joblib`, `tqdm`, `torch`
(o treino da Our-Trained usa GPU se houver) e `jupyter`.

Rode os notebooks em ordem (01 → 04). Tudo usa cache: o que já está em `representations/` (Our-Trained) e
`results/curvas/` (buscas) é lido em vez de recalculado — por isso o `04_Analises.ipynb` roda em segundos. Para refazer do
zero, use `FORCE_TRAIN = True` no 02 (~3 min em GPU) e `FORCE_SEARCH = True` no 03 (~2 h em 11 núcleos: 17 métodos SMBO ×
120 alvos × 5 sementes). O `03_rodar_buscas.py` faz o mesmo que o notebook 03 pela linha de comando (útil para rodar em
segundo plano).

## Estrutura

```
01_Etapa1-MetaDados.ipynb                 Etapa 1 (executado)
02_MetaCaracteristicas-OurTrained.ipynb   meta-características e treino da Our-Trained (executado)
03_Etapa2-Buscas.ipynb                    Etapa 2: roda e salva as buscas (executado)
04_Analises.ipynb                         Análises 1, 2 e 3 (executado; só lê resultados)
03_rodar_buscas.py                        o mesmo que o notebook 03, pela linha de comando
src/config.py                 constantes (T, n_init, sementes, k, caminhos)
src/metatask_data.py          leitura/download da meta-task (grid canonizado: o índice da config é o mesmo em todos os datasets)
src/our_trained.py            Our-Trained (Attention set-invariant -> MLP -> z de 1024 dims), CE guiada pela meta-task
src/search.py                 matriz de perdas, desenhos de warm-start (LODO), SMBO GP+EI, execução com cache
src/analysis.py               AUC, rank médio, Friedman + Nemenyi (diagrama CD), Wilcoxon-Holm, análises da Seção 2.1
data/                         meta-task (respostas, grid, splits, datasets brutos usados pela Our-Trained)
representations/             meta-características (PyMFE, MF1, MF2, D2V por split, Our-Trained por split)
results/                      matriz de perdas, curvas (results/curvas/*.npz), modelos, tabelas e figuras
relatorio/                    relatório: Relatorio3.pdf, fonte HTML e versão LaTeX (Relatorio3.tex + figuras/, para o Overleaf)
```

## Origem

O carregamento da meta-task, as meta-características de referência (MF1, MF2, D2V) e o modelo da Our-Trained vêm do projeto
de pesquisa (`DoutoradoIR/NEW-meta-task-HPO`); a única mudança no treino da Our-Trained é a dimensão do embedding (32 → 1024).
