#!/usr/bin/env bash
# Instala o Auto-sklearn 0.15.0 no Ubuntu do WSL sem sudo (Miniforge, tudo por HTTPS; a rede bloqueia HTTP na porta 80).
# Uso (dentro do WSL): bash setup_autosklearn.sh  -> ~/miniforge e o ambiente ~/autosklearn-env
set -e
cd ~
if [ ! -x ~/miniforge/bin/conda ]; then
  [ -f mf.sh ] || cp /mnt/d/Projetos/DisciplinaMtL/Entrega4/hpo/wsl/mf.sh mf.sh   # baixado pelo Windows (o WSL não alcança o github.com)
  bash mf.sh -b -p ~/miniforge
fi
if [ ! -x ~/autosklearn-env/bin/python ]; then
  ~/miniforge/bin/conda create -y -p ~/autosklearn-env -c conda-forge python=3.9 swig gxx_linux-64 make \
    "numpy<1.24" "scipy<1.10" "pandas<2" "scikit-learn=0.24.2" psutil
fi
~/autosklearn-env/bin/pip install "auto-sklearn==0.15.0"
~/autosklearn-env/bin/python -c "import autosklearn, sklearn; print('autosklearn', autosklearn.__version__, 'sklearn', sklearn.__version__)"
