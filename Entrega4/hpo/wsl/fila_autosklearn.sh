#!/usr/bin/env bash
# Fila do Auto-sklearn (dentro do WSL): meta-aprendizagem ligada e desligada em paralelo, 4 núcleos cada. Retomável.
# Uso (Windows): wsl.exe -e bash -lc "nohup bash /mnt/d/Projetos/DisciplinaMtL/Entrega4/hpo/wsl/fila_autosklearn.sh > /dev/null 2>&1 &"
cd /mnt/d/Projetos/DisciplinaMtL/Entrega4/hpo
ORC=30,120,300
PY=~/autosklearn-env/bin/python
echo "INICIO $(date)" >> logs/fila_autosklearn.log
for s in autosklearn_ml autosklearn_noml; do
  $PY rodar_automl_hpo.py $s --orcamentos $ORC --nucleos 4 >> logs/$s.log 2>&1 &
done
wait
echo "FIM $(date)" >> logs/fila_autosklearn.log
