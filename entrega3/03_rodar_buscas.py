""" Faz o mesmo que o `03_Etapa2-Buscas.ipynb`, roda as buscas simuladas da Etapa 2 (métodos 1 a 5) e guarda as curvas em results/curvas/<método>.npz,
fiz aqui pra deixar rodando na linha de comando. Pra rodar:
    python 03_rodar_buscas.py              # os 5 métodos (warm-start por similaridade com k = 1, 3 e 5)
    python 03_rodar_buscas.py --force      # refaz tudo
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from src import config as cfg
from src import search 
from src.metatask_data import load_meta_task


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--jobs", type=int, default=-2, help="processos em paralelo (joblib; -2 = todos menos 1)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    meta = load_meta_task()
    t0 = time.perf_counter()
    ctx = search.build_context(meta)
    print(f"{meta.n_ds} alvos | K={meta.n_cfg} | T={ctx['T']} | n_init={ctx['n_init']} | sementes={ctx['n_seeds']} | "
          f"k={list(cfg.K_NEIGHBORS)}")
    for key in search.all_method_keys():
        search.run_method(key, ctx, n_jobs=args.jobs, force=args.force)
    print(f"Pronto em {(time.perf_counter() - t0) / 60:.1f} min. Curvas em {cfg.CURVES_DIR}")


if __name__ == "__main__":
    main()
