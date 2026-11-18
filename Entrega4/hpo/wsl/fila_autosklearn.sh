#!/usr/bin/env bash
# Fila do Auto-sklearn (WSL): orçamentos de 30, 60 e 120 s (ORC), n_jobs=1, com e sem meta-aprendizagem; cada condição em 2 processos
# (metade dos datasets cada). Retomável. Uso (Windows): wsl.exe -e bash -lc "bash .../wsl/fila_autosklearn.sh"
cd /mnt/d/Projetos/DisciplinaMtL/Entrega4/hpo
PY=~/autosklearn-env/bin/python
ORC=${ORC:-30,60,120}
mapfile -t D < <(ls /mnt/d/Projetos/DisciplinaMtL/entrega3/data/datasets/*.npz | xargs -n1 basename | sed 's/\.npz$//')
A=$(IFS=,; echo "${D[*]:0:60}"); B=$(IFS=,; echo "${D[*]:60}")
echo "INICIO $(date)" >> logs/fila_autosklearn.log
for s in autosklearn_ml autosklearn_noml; do
  for L in "$A" "$B"; do
    $PY rodar_automl_hpo.py $s --orcamentos $ORC --nucleos 1 --datasets "$L" >> logs/$s.log 2>&1 &
  done
done
wait
echo "FIM $(date)" >> logs/fila_autosklearn.log
