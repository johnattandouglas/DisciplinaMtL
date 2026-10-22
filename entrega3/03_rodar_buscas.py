"""03 - Roda as buscas simuladas da Etapa 2 (métodos 1 a 5) e guarda as curvas em results/curvas/<método>.npz.

Faz o mesmo que o `03_Etapa2-Buscas.ipynb`, pela linha de comando (útil para deixar as ~2 h de buscas em segundo plano).
Cada método é rodado nos 120 alvos (leave-one-dataset-out) x N_SEEDS sementes, com T avaliações e n_init pontos
iniciais (ver src/config.py). O que já está em cache é carregado; use --force para refazer.

    python 03_rodar_buscas.py                      # todos os métodos
    python 03_rodar_buscas.py --reps PyMFE D2V     # só o warm-start por similaridade dessas representações (+ baselines)
    python 03_rodar_buscas.py --ks 1               # só k=1 no warm-start por similaridade
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

from src import config as cfg  # noqa: E402
from src import search  # noqa: E402
from src.metatask_data import load_meta_task  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reps", nargs="*", default=cfg.REP_ALL, help="representações do warm-start por similaridade")
    ap.add_argument("--ks", nargs="*", type=int, default=list(cfg.K_NEIGHBORS), help="nº de vizinhos k")
    ap.add_argument("--jobs", type=int, default=-2, help="processos em paralelo (joblib; -2 = todos menos 1)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    meta = load_meta_task()
    t0 = time.perf_counter()
    ctx = search.build_context(meta, reps=args.reps)
    print(f"{meta.n_ds} alvos | K={meta.n_cfg} | T={ctx['T']} | n_init={ctx['n_init']} | sementes={ctx['n_seeds']} | "
          f"representações={args.reps} | k={args.ks} | contexto em {time.perf_counter() - t0:.1f}s")

    keys = ["default", "random_search", "smbo_cold", "smbo_ws_avgrank"]
    # k da comparação principal primeiro, depois os demais (a Análise 2 usa todos)
    for k in sorted(args.ks, key=lambda v: (v != cfg.K_MAIN, v)):
        keys += [search.method_key("ws_knn", r, k) for r in args.reps]
    for key in keys:
        search.run_method(key, ctx, n_jobs=args.jobs, force=args.force)
    print(f"Pronto em {(time.perf_counter() - t0) / 60:.1f} min. Curvas em {cfg.CURVES_DIR}")


if __name__ == "__main__":
    main()
