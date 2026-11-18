"""Cruza os 120 datasets da meta-task com os 208 datasets do OpenML do meta-dataset do Auto-sklearn 0.15.0
(accuracy_multiclass.classification_dense/feature_values.arff; a 1ª coluna é o nome do dataset no OpenML).
A lista foi extraída no WSL para resultados/autosklearn_tarefas.txt. Grava resultados/sobreposicao_openml.csv.
Uso: python wsl/sobreposicao_openml.py
"""
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import comum  # noqa: E402

base = (comum.RES / "autosklearn_tarefas.txt").read_text().split()


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


nb = {b: norm(b) for b in base}
res = pd.DataFrame({"dataset": comum.nomes()})
res["igual"] = res.dataset.map(lambda n: ";".join(b for b, x in nb.items() if x == norm(n)))
res["parecido"] = res.dataset.map(
    lambda n: ";".join(b for b, x in nb.items() if len(x) > 3 and x != norm(n) and (x in norm(n) or norm(n) in x)))
res.to_csv(comum.RES / "sobreposicao_openml.csv", index=False)
print(len(base), "datasets na base;", (res.igual != "").sum(), "nomes iguais;", (res.parecido != "").sum(), "parecidos")
print(res[(res.igual != "") | (res.parecido != "")].to_string(index=False))

# Revisão manual dos "parecidos" (autor/Claude, 18/11/2026): mesmos dados no OpenML. waveform-5000 tem 40 atributos,
# ou seja, é o waveform-noise. Falsos positivos: bank, car, iris (irish), plant-texture, wine, thyroid (hypothyroid).
MESMOS = {"annealing", "cardiotocography-10clases", "cardiotocography-3clases", "conn-bench-vowel-deterding",
          "ilpd-indian-liver", "led-display", "magic", "molec-biol-splice", "musk-1", "musk-2", "ozone", "steel-plates"}
res["na_base"] = (res.igual != "") | res.dataset.isin(MESMOS) | (res.dataset == "waveform-noise")
res.to_csv(comum.RES / "sobreposicao_openml.csv", index=False)
print("na base do Auto-sklearn (após revisão):", int(res.na_base.sum()), "de", len(res))
