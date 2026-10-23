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

