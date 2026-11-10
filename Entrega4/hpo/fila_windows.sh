#!/usr/bin/env bash
# Fila do Windows (Git Bash): MLPs na GPU, depois AutoGluon (GPU) e FLAML (XGBoost na GPU). Retomável: basta rodar de novo.
# Uso: nohup bash fila_windows.sh > logs/fila_windows.log 2>&1 &
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
ORC=30,120,300
E=/d/Projetos/DoutoradoIR/automl/envs
echo "INICIO $(date)"
/d/Projetos/DoutoradoIR/.venv/Scripts/python.exe mlp_gpu.py >> logs/mlp.log 2>&1; echo "mlp $? $(date)"
$E/autogluon/Scripts/python.exe rodar_automl_hpo.py autogluon --orcamentos $ORC --nucleos 4 >> logs/autogluon.log 2>&1; echo "autogluon $? $(date)"
$E/flaml/Scripts/python.exe rodar_automl_hpo.py flaml --orcamentos $ORC --nucleos 4 >> logs/flaml.log 2>&1; echo "flaml $? $(date)"
echo "FIM $(date)"
