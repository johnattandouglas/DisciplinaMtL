"""Caminhos e constantes da Atividade 3 (warm-starting de HPO) sobre a meta-task do Dataset2Vec.

Ordem de execução (um notebook por etapa; cada um grava o que o próximo lê):

    01_Etapa1-MetaDados.ipynb             meta-task (baixa se faltar) -> matriz de perdas K x n em results/
    02_MetaCaracteristicas-PyMFE.ipynb    meta-características clássicas (pymfe) dos 120 datasets -> representations/
    03_Etapa2-Buscas.ipynb                roda e salva as buscas simuladas (métodos 1-5) -> results/curvas/
    04_Analises.ipynb                     análises 1-3 (só lê results/curvas/) -> results/tables/, results/figures/

Pastas:

    data/                 meta-task (respostas das 3456 configs x 120 datasets, grid e datasets brutos)
    representations/      meta-características PyMFE (uma linha por dataset)
    results/              matriz de perdas, curvas das buscas, tabelas e figuras
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]          # .../entrega3
DATA_DIR = ROOT / "data"
DATASETS_DIR = DATA_DIR / "datasets"                # <nome>.npz (X, y) por dataset
REPR_DIR = ROOT / "representations"
PYMFE_CSV = REPR_DIR / "PyMFE.csv"
RESULTS_DIR = ROOT / "results"
CURVES_DIR = RESULTS_DIR / "curvas"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
LOSS_MATRIX_CSV = RESULTS_DIR / "matriz_de_perdas.csv"   # K x n (linhas = configurações, colunas = datasets)
CONFIGS_CSV = RESULTS_DIR / "configuracoes.csv"          # as K configurações de Θ (id = linha da matriz)

# ---------------------------------------------------------------------------
# Fonte dos dados (pacote `ismlldataset` / repositório hadijomaa/dataset2vec)
# ---------------------------------------------------------------------------
ISMLL_SERVER = "https://www.ismll.uni-hildesheim.de/personen/hsjomaa/"

# ---------------------------------------------------------------------------
# Espaço Θ: grid de redes feed-forward (7 hiperparâmetros) e resposta
# ---------------------------------------------------------------------------
CONFIG_COLS = ["ACTIVATION", "ARCHITECTURE", "DROPOUT", "LAYERS", "NEURONS", "NORMALIZATION", "OPTIMZERS"]
RESPONSE_IDX = 9            # acurácia na época 50 (colunas `*_acc_0..9` = épocas 5,10,...,50)
RESPONSE_DECIMALS = 4

# Configuração "padrão": o ponto do grid mais próximo do default do MLPClassifier do scikit-learn
# (1 camada oculta de 100 neurônios, ReLU, Adam, sem dropout e sem batch-norm). 100 neurônios não existe no
# grid: usa-se a maior largura disponível (32). Com 1 camada a única arquitetura é SQU.
DEFAULT_CONFIG = {"ACTIVATION": "relu", "ARCHITECTURE": "SQU", "DROPOUT": 0.0, "LAYERS": 1, "NEURONS": 32,
                  "NORMALIZATION": 0, "OPTIMZERS": "adam"}

# ---------------------------------------------------------------------------
# Protocolo de busca (Etapa 2)
# ---------------------------------------------------------------------------
SEED = 42
T_BUDGET = 100              # T: avaliações por busca (os pontos iniciais contam)
N_INIT = 20                 # n_init: pontos iniciais
N_SEEDS = 5                 # sementes por (método, dataset alvo)
K_NEIGHBORS = (1, 3, 5)     # nº de vizinhos k do warm-start por similaridade (Análise 2)
K_MAIN = 1                  # k da comparação principal (Análise 1), fixado a priori
CHECKPOINTS = (N_INIT, 50, T_BUDGET)   # logo após os pontos iniciais, intermediário e T

# ---------------------------------------------------------------------------
# Meta-características: só as clássicas do pymfe
# ---------------------------------------------------------------------------
REP = "PyMFE"
PYMFE_MAX_ROWS = 5000       # subamostra estratificada para datasets maiores (custo do pymfe)


def make_dirs() -> None:
    for d in (DATA_DIR, DATASETS_DIR, REPR_DIR, CURVES_DIR, TABLES_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
