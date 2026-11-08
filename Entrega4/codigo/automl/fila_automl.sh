#!/usr/bin/env bash
# Fila de CPU do experimento "o AutoML escapa da popularidade?" na IR (215 datasets). NÃO usa GPU.
#
#   1. os 42 pares na divisão holdout 70/30 (semente 42) de todos os datasets (rodar_pares.py);
#   2. a comparação com a matriz original (comparar_matriz.py);
#   3. o FLAML nos três orçamentos, do menor para o maior (rodar_automl.py).
#
# Cada etapa abre WORKERS processos iguais; cada processo pega o próximo dataset livre (maior primeiro, trava atômica em
# .travas/) e roda um dataset por vez com NUCLEOS threads. Total de núcleos = WORKERS x NUCLEOS. Retomável: datasets com CSV
# pronto são pulados; as travas de uma execução interrompida são apagadas no começo de cada etapa.
#
# Uso (Git Bash, a partir de qualquer pasta):
#   nohup bash automl/fila_automl.sh 5 2 > /dev/null 2>&1 &              # 5 workers x 2 núcleos = 10 núcleos
#   ORCAMENTOS="10 60 300" bash automl/fila_automl.sh 5 2                # orçamentos do FLAML em segundos
#   ETAPAS="flaml" bash automl/fila_automl.sh 5 2                        # só o FLAML (pares | comparar | flaml)
# Progresso: automl/resultados/fila/fila_automl.log (uma linha por etapa; FIM no final) e um log por worker.
set -u
WORKERS=${1:-5}
NUCLEOS=${2:-2}
ORCAMENTOS=${ORCAMENTOS:-"10 60 300"}
ETAPAS=${ETAPAS:-"pares comparar flaml"}

AQUI=$(cd "$(dirname "$0")" && pwd)
cd "$AQUI/.."                                   # raiz do repositório
export PYTHONIOENCODING=utf-8 TQDM_DISABLE=1
export OMP_NUM_THREADS=$NUCLEOS MKL_NUM_THREADS=$NUCLEOS OPENBLAS_NUM_THREADS=$NUCLEOS NUMEXPR_NUM_THREADS=$NUCLEOS
PY_PARES=automl/envs/pares/Scripts/python.exe
PY_FLAML=automl/envs/flaml/Scripts/python.exe
LOGS=automl/resultados/fila
mkdir -p "$LOGS"
LOG=$LOGS/fila_automl.log
msg() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> "$LOG"; }

# etapa <nome> <pasta de resultados> <comando...>: WORKERS cópias do comando em paralelo; espera todas
etapa() {
  local nome=$1 pasta=$2; shift 2
  rm -rf "$pasta/.travas"
  msg "INÍCIO $nome ($WORKERS workers x $NUCLEOS núcleos)"
  local t0=$SECONDS
  for ((w = 0; w < WORKERS; w++)); do
    "$@" >> "$LOGS/${nome}_w$w.log" 2>&1 &
  done
  wait
  local n
  n=$(ls "$pasta"/*.csv 2>/dev/null | wc -l)
  msg "FIM $nome: $n datasets com resultado, $((SECONDS - t0)) s"
}

msg "fila iniciada: WORKERS=$WORKERS NUCLEOS=$NUCLEOS ORCAMENTOS='$ORCAMENTOS' ETAPAS='$ETAPAS'"
for e in $ETAPAS; do
  case $e in
    pares)
      etapa pares automl/resultados/pares/holdout \
        "$PY_PARES" automl/rodar_pares.py --todos --protocolo holdout --nucleos "$NUCLEOS" ;;
    comparar)
      "$PY_PARES" automl/comparar_matriz.py --protocolo holdout >> "$LOG" 2>&1 ;;
    flaml)
      for t in $ORCAMENTOS; do
        etapa "flaml_t$t" "automl/resultados/automl/flaml/t$t" \
          "$PY_FLAML" automl/rodar_automl.py --sistema flaml --orcamento "$t" --todos --nucleos "$NUCLEOS"
      done ;;
  esac
done
msg "FIM"
