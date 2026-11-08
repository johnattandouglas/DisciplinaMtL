# Entrega 4: AutoML versus meta-aprendizagem

Material da Atividade 4 de IN1097 e do artigo "Popularity, Not Similarity: Why Dataset Representations Rarely Help
Meta-Learning" (formato IJCNN). O artigo compara sete representações de datasets (TabPFN, Dataset2Vec, PyMFE, ECoL,
MetaBu, Dida, TRIO) com recomendações que não usam representação, em três meta-tarefas (HPO, PIPES, IR), e compara a
meta-aprendizagem com sistemas de AutoML tabular no IR.

## Pastas

- `figuras/`: figuras do artigo. `fig_k.pdf` mostra o valor top-1 do k-NN contra k.
- `resultados/`: números do artigo.
  - `METRIC.txt`: análise completa (popularidade, folga, kDN, oráculo q).
  - `ijcnn_numeros.txt`: valores com k = 1, 11 e 30 e os testes contra vizinhos ao acaso.
- `levantamento/`: levantamento dos sistemas de AutoML tabular (ano, espaço de busca, estratégia, meta-conhecimento,
  modelo final, orçamento, sistema operacional) e as referências em BibTeX. É a base da Etapa 1 da atividade.
- `codigo/`: scripts que geram os números e as figuras. Eles importam módulos do repositório de pesquisa do autor
  (`avaliar`, `knn_fixo`, `topk`, `kdn`, `destilacao`), que ainda não estão aqui.

## Em andamento

- Experimentos de AutoML no IR, com a constante, a recomendação por meta-aprendizagem e o oráculo reexecutados na mesma
  partição de cada dataset. Resultados brutos por dataset vão para `resultados/automl/`.
- Análises extras (k-NN ponderado, floresta, simulação sintética da folga, k por validação aninhada, oráculo do resíduo).
- Cinco sementes e curvas de aprendizado de MetaBu, Dida e TRIO.
