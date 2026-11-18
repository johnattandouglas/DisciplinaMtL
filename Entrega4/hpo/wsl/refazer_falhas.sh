#!/usr/bin/env bash
# Refaz, com o script corrigido (_ask_desc aceita ensemble só com o dummy), as 21 tentativas das campanhas de 30 e 120 s
# que ficaram sem resultado pelo AttributeError('int' object has no attribute 'get'). n_jobs=1, 4096 MB, como as campanhas.
cd /mnt/d/Projetos/DisciplinaMtL/Entrega4/hpo
PY=~/autosklearn-env/bin/python
S7=planning,miniboone,adult,post-operative,magic,congressional-voting,connect-4
$PY rodar_automl_hpo.py autosklearn_ml   --orcamentos 30  --nucleos 1 --datasets $S7 >> logs/refazer.log 2>&1 &
$PY rodar_automl_hpo.py autosklearn_noml --orcamentos 30  --nucleos 1 --datasets $S7,breast-cancer-wisc-prog,fertility >> logs/refazer.log 2>&1 &
$PY rodar_automl_hpo.py autosklearn_ml   --orcamentos 120 --nucleos 1 --datasets planning,post-operative >> logs/refazer.log 2>&1 &
$PY rodar_automl_hpo.py autosklearn_noml --orcamentos 120 --nucleos 1 --datasets planning,post-operative,fertility >> logs/refazer.log 2>&1 &
wait
