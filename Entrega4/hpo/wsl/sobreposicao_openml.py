"""Cruza os 120 datasets da meta-task com as tarefas do OpenML do meta-dataset do Auto-sklearn 0.15.0
(accuracy_multiclass.classification_dense/feature_values.arff). Roda no WSL (precisa do pacote autosklearn) e usa a API
REST do OpenML por HTTPS. Grava resultados/sobreposicao_openml.csv.
Uso: ~/autosklearn-env/bin/python wsl/sobreposicao_openml.py
"""
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

import autosklearn
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import comum  # noqa: E402

arq = Path(os.path.dirname(autosklearn.__file__)) / "metalearning/files/accuracy_multiclass.classification_dense/feature_values.arff"
tarefas = [l.split(",")[0] for l in arq.read_text().splitlines() if l and not l.startswith(("@", "%"))]


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


linhas = []
for t in tarefas:
    try:
        tk = json.load(urllib.request.urlopen(f"https://www.openml.org/api/v1/json/task/{t}", timeout=30))["task"]
        did = [i for i in tk["input"] if i["name"] == "source_data"][0]["data_set"]["data_set_id"]
        ds = json.load(urllib.request.urlopen(f"https://www.openml.org/api/v1/json/data/{did}", timeout=30))
        linhas.append({"task_id": t, "data_id": did, "nome_openml": ds["data_set_description"]["name"]})
    except Exception as e:  # noqa: BLE001
        linhas.append({"task_id": t, "erro": repr(e)[:100]})
om = pd.DataFrame(linhas)
nos = {norm(n): n for n in comum.nomes()}
om["norm"] = om.nome_openml.fillna("").map(norm)
res = pd.DataFrame({"dataset": comum.nomes()})
res["nome_openml_igual"] = res.dataset.map(lambda n: ";".join(om.nome_openml[om.norm == norm(n)].astype(str)))
res["nome_openml_contem"] = res.dataset.map(
    lambda n: ";".join(om.nome_openml[[bool(x) and (x in norm(n) or norm(n) in x) and len(x) > 3 for x in om.norm]]
                       .astype(str)))
res.to_csv(comum.RES / "sobreposicao_openml.csv", index=False)
om.to_csv(comum.RES / "autosklearn_tarefas_openml.csv", index=False)
print(len(tarefas), "tarefas;", (res.nome_openml_igual != "").sum(), "nomes iguais;",
      (res.nome_openml_contem != "").sum(), "nomes parecidos (conferir à mão)")
