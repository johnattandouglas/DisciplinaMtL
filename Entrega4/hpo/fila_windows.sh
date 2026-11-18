#!/usr/bin/env bash
# Fila do Windows (Git Bash): AutoGluon e FLAML em paralelo, 3 núcleos cada, na GPU. Retomável: basta rodar de novo.
# (As MLPs já foram reexecutadas por mlp_gpu.py.)  Uso: nohup bash fila_windows.sh > logs/fila_windows.log 2>&1 &
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
ORC=30,120,300
E=/d/Projetos/DoutoradoIR/automl/envs
echo "INICIO $(date)"
$E/autogluon/Scripts/python.exe rodar_automl_hpo.py autogluon --orcamentos $ORC --nucleos 3 >> logs/autogluon.log 2>&1 &
$E/flaml/Scripts/python.exe rodar_automl_hpo.py flaml --orcamentos $ORC --nucleos 3 >> logs/flaml.log 2>&1 &
wait
echo "FIM $(date)"
