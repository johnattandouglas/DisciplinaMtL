#!/usr/bin/env bash
# Segunda fila de CPU: os outros sistemas de AutoML que funcionaram no piloto, nos mesmos orçamentos (10/60/300 s).
# Espera a primeira fila terminar (linha "FIM" sozinha em automl/resultados/fila/fila_automl.log) antes de começar.
# Cada sistema roda no seu ambiente automl/envs/<amb>/. Mesmo esquema da primeira fila: WORKERS processos, cada um pega o
# próximo dataset livre (trava atômica), NUCLEOS threads por processo; retomável.
#
# Uso (Git Bash): nohup bash automl/fila_automl2.sh 4 2 > /dev/null 2>&1 &
#   SISTEMAS="autogluon_portfolio:autogluon autogluon_padrao:autogluon" ORCAMENTOS="10 60" bash automl/fila_automl2.sh 4 2
# Progresso: automl/resultados/fila/fila_automl2.log
set -u
WORKERS=${1:-4}
NUCLEOS=${2:-2}
ORCAMENTOS=${ORCAMENTOS:-"10 60 300"}
# sistema:ambiente (só os que passaram no piloto; ver automl/versoes.csv)
SISTEMAS=${SISTEMAS:-"autogluon_portfolio:autogluon autogluon_padrao:autogluon lightautoml:lightautoml mljar:mljar hpsklearn:hpsklearn tpot:tpot naiveautoml:naiveautoml"}
# NaiveAutoML não completa nenhuma avaliação em 10 s (piloto): pulado nesse orçamento
PULAR=${PULAR:-"naiveautoml:10"}

AQUI=$(cd "$(dirname "$0")" && pwd)
cd "$AQUI/.."
export PYTHONIOENCODING=utf-8 TQDM_DISABLE=1
export OMP_NUM_THREADS=$NUCLEOS MKL_NUM_THREADS=$NUCLEOS OPENBLAS_NUM_THREADS=$NUCLEOS NUMEXPR_NUM_THREADS=$NUCLEOS
LOGS=automl/resultados/fila
mkdir -p "$LOGS"
LOG=$LOGS/fila_automl2.log
msg() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> "$LOG"; }

msg "aguardando FIM da primeira fila"
until grep -qx '.*[0-9] FIM' "$LOGS/fila_automl.log" 2>/dev/null; do sleep 60; done
msg "fila iniciada: WORKERS=$WORKERS NUCLEOS=$NUCLEOS ORCAMENTOS='$ORCAMENTOS' SISTEMAS='$SISTEMAS'"

for t in $ORCAMENTOS; do
  for s in $SISTEMAS; do
    sis=${s%%:*}; amb=${s#*:}
    case " $PULAR " in *" $sis:$t "*) msg "pulado ${sis}_t$t"; continue ;; esac
    pasta=automl/resultados/automl/$sis/t$t
    mkdir -p "$pasta"; rm -rf "$pasta/.travas"
    msg "INÍCIO ${sis}_t$t"
    t0=$SECONDS
    for ((w = 0; w < WORKERS; w++)); do
      "automl/envs/$amb/Scripts/python.exe" automl/rodar_automl.py --sistema "$sis" --orcamento "$t" --todos \
        --nucleos "$NUCLEOS" >> "$LOGS/${sis}_t${t}_w$w.log" 2>&1 &
    done
    wait
    msg "FIM ${sis}_t$t: $(ls "$pasta"/*.csv 2>/dev/null | wc -l) datasets, $((SECONDS - t0)) s"
  done
done
msg "FIM"
