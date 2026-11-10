"""Treina na GPU (PyTorch) as configs de MLP escolhidas em resultados/configs_por_abordagem.csv, no treino (75%) da nossa
divisão, e mede a acurácia no teste (25%). Retomável: cada (dataset, config) vira uma linha em resultados/mlp.jsonl.

Implementação nossa do espaço do Dataset2Vec (a dos autores não é pública): 50 épocas, lote 64, taxas padrão do Keras
(adam e rmsprop 1e-3, sgd 1e-2), entropia cruzada. Larguras por camada a partir de w = NEURONS e L = LAYERS:
  SQU  w em todas;  ASC  w, 2w, 4w, ...;  DES  o inverso de ASC;  ENC  larga-estreita-larga (funil: w·2^|i-c|);
  SYM  estreita-larga-estreita (w·2^(c-|i-c|)),  com c = (L-1)/2 arredondado para baixo.
Batch-norm (NORMALIZATION=1) depois de cada linear; dropout depois da ativação.

Uso (.venv do DoutoradoIR, que tem torch com CUDA):  python mlp_gpu.py [--datasets a,b] [--limite N]
"""
import argparse
import json
import platform
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

import comum

ATIV = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU, "selu": nn.SELU}


def larguras(forma, L, w):
    c = (L - 1) // 2
    if forma == "SQU" or L == 1:
        return [w] * L
    if forma == "ASC":
        return [w * 2 ** i for i in range(L)]
    if forma == "DES":
        return [w * 2 ** (L - 1 - i) for i in range(L)]
    if forma == "ENC":
        return [w * 2 ** abs(i - c) for i in range(L)]
    if forma == "SYM":
        return [w * 2 ** (c - abs(i - c)) for i in range(L)]
    raise ValueError(forma)


def montar(cfg, d, k):
    camadas, entra = [], d
    for h in larguras(cfg["ARCHITECTURE"], int(cfg["LAYERS"]), int(cfg["NEURONS"])):
        camadas.append(nn.Linear(entra, h))
        if int(cfg["NORMALIZATION"]):
            camadas.append(nn.BatchNorm1d(h))
        camadas.append(ATIV[cfg["ACTIVATION"]]())
        if float(cfg["DROPOUT"]) > 0:
            camadas.append(nn.Dropout(float(cfg["DROPOUT"])))
        entra = h
    camadas.append(nn.Linear(entra, k))
    return nn.Sequential(*camadas)


def treinar(cfg, Xtr, ytr, Xte, k, semente=0, epocas=50, lote=64, dev="cuda"):
    torch.manual_seed(semente)
    g = torch.Generator(device="cpu").manual_seed(semente)
    m = montar(cfg, Xtr.shape[1], k).to(dev)
    opt = {"adam": lambda p: torch.optim.Adam(p, 1e-3), "rmsprop": lambda p: torch.optim.RMSprop(p, 1e-3),
           "sgd": lambda p: torch.optim.SGD(p, 1e-2)}[cfg["OPTIMZERS"]](m.parameters())
    X = torch.tensor(Xtr, device=dev)
    y = torch.tensor(ytr, device=dev)
    n = len(X)
    perda = nn.CrossEntropyLoss()
    m.train()
    for _ in range(epocas):
        perm = torch.randperm(n, generator=g).to(dev)
        for s in range(0, n, lote):
            b = perm[s:s + lote]
            if len(b) < 2 and int(cfg["NORMALIZATION"]):   # batch-norm não aceita lote de 1
                continue
            opt.zero_grad(set_to_none=True)
            perda(m(X[b]), y[b]).backward()
            opt.step()
    m.eval()
    with torch.no_grad():
        return m(torch.tensor(Xte, device=dev)).argmax(1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="")
    ap.add_argument("--limite", type=int, default=0)
    a = ap.parse_args()
    sel = pd.read_csv(comum.RES / "configs_por_abordagem.csv")
    cols = ["ACTIVATION", "ARCHITECTURE", "DROPOUT", "LAYERS", "NEURONS", "NORMALIZATION", "OPTIMZERS"]
    pares = sel.drop_duplicates(["dataset", "config"])
    if a.datasets:
        pares = pares[pares.dataset.isin(a.datasets.split(","))]
    arq = comum.RES / "mlp.jsonl"
    hw = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cpu": platform.processor()}
    feitos = 0
    for nome, grupo in pares.groupby("dataset", sort=True):
        Xtr, ytr, Xte, yte = comum.carregar(nome)
        k = int(max(ytr.max(), yte.max()) + 1)
        for _, r in grupo.iterrows():
            chave = f"{nome}|{int(r.config)}"
            if comum.ja_feito(arq, chave):
                continue
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            p = treinar(r[cols].to_dict(), Xtr, ytr, Xte, k)
            torch.cuda.synchronize()
            seg = time.perf_counter() - t0
            reg = {"chave": chave, "dataset": nome, "config": int(r.config), "acc_teste": float((p == yte).mean()),
                   "seg_treino": seg, "n_treino": len(ytr), "n_teste": len(yte), **hw}
            comum.gravar(arq, reg)
            print(json.dumps({k_: reg[k_] for k_ in ("chave", "acc_teste", "seg_treino")}), flush=True)
            feitos += 1
            if a.limite and feitos >= a.limite:
                return


if __name__ == "__main__":
    main()
