"""Our-Trained (embedding de 1024 dimensões) treinada para a meta-task de warm-start de HPO.

Adaptação de `NEW-meta-task-HPO/train_our_trained2.py` (projeto DoutoradoIR). Mudanças em relação a ele:
  * dimensão do embedding z: 32 -> 1024 (`cfg.EMB_DIM_OUR`);
  * caminhos/nomes desta entrega (Our-Trained_split{k}.csv, our_trained_split{k}.pt) e registro de tempo simplificado.
Arquitetura, alvo, perda e protocolo são os mesmos:

  * arquitetura: Attention set-invariant (`TransformerSetInvariant_WITH_MAPS`, emb 8) -> MLP 256-256 -> z (1024) -> cabeça
    linear de CE (descartada depois do treino; a representação é z). Entrada: o dataset bruto como tensor
    (1, n_inst, n_feat, 1), rótulo na última coluna, até 256 instâncias e 256 features; Adam, lr=1e-3; seed 42.
  * alvo da CE (o que torna a representação específica da meta-task): para cada um dos 7 hiperparâmetros do grid, a
    distribuição desse hiperparâmetro entre as TOP-20 configurações do dataset (empates no limiar incluídos) — CE com alvo
    suave, média das 7 cabeças. Datasets cujas melhores configurações se parecem ficam próximos em z, que é o que o
    warm-start por vizinhos usa.
  * protocolo: os 5 splits oficiais do D2V (train/valid/test = 80/16/24 datasets). No split k treina nos 80 de treino,
    escolhe a época pela CE nos 16 de validação e extrai o embedding dos 120 datasets. Os 24 de teste-k são os alvos da
    busca e nunca participam do treino nem da seleção de época (logo, o alvo d_t nunca é visto pelo extrator).

Uso (notebook `02_MetaCaracteristicas-OurTrained.ipynb`):
    from src import our_trained
    embs = our_trained.run_all_splits(meta)          # treina os 5 splits (pula os que já estão em cache)
"""
import copy
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm.auto import tqdm

from . import config as cfg
from .metatask_data import MetaTask

SEED = cfg.SEED

OUTPUT_EMB_SIZE = cfg.EMB_DIM_OUR  # 1024
MAX_INSTANCES_PER_DATASET = 256   # atenção é O(n_inst x n_feat^2)
MAX_FEATURES_PER_DATASET = 256

FT_LR = 1e-3
FT_MAX_EPOCHS = 80
FT_BATCH_DATASETS = 8             # datasets por passo do otimizador (mini-batch de datasets)
FT_VAL_EVERY = 2                  # valida a cada N épocas
FT_PATIENCE = 8                   # nº de validações seguidas sem melhora da CE de validação
FT_GRAD_CLIP = 5.0
TARGET_TOP_K = cfg.N_INIT         # top-k configs que definem o alvo suave (= n_init do warm-start)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TIMES_CSV = cfg.TABLES_DIR / "our_trained_treino.csv"


# ============================================================================
# Modelo (idêntico ao de train_our_trained2.py; só OUTPUT_EMB_SIZE muda)
# ============================================================================
class MLPRefine(nn.Module):
    def __init__(self, input_dim: int, out_dim: int = OUTPUT_EMB_SIZE):
        super().__init__()
        self.expand1 = nn.Linear(input_dim, 256)
        self.expand2 = nn.Linear(256, 256)
        self.to_out = nn.Linear(256, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.expand1(x))
        h = F.gelu(self.expand2(h))
        return self.to_out(h)


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.norm_q = nn.LayerNorm(dim_Q)
            self.norm_0 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)
        self.ff = nn.Sequential(
            nn.Linear(dim_Q, dim_Q * 4),
            nn.GELU(),
            nn.Linear(dim_Q * 4, dim_Q),
        )

    def forward(self, Q, K):
        orig_shape = Q.shape
        orig_dims = Q.dim()
        if orig_dims == 2:
            Q = Q.unsqueeze(0)
            K = K.unsqueeze(0)
        elif orig_dims == 4:
            ds, inst, seq, emb = Q.shape
            Q = Q.view(ds * inst, seq, emb)
            K = K.view(ds * inst, seq, emb)
        Q = self.norm_q(Q)
        K = self.norm_q(K)
        Q = self.fc_q(Q)
        K = self.fc_k(K)
        V = self.fc_v(K)
        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, dim=2), dim=0)
        K_ = torch.cat(K.split(dim_split, dim=2), dim=0)
        V_ = torch.cat(V.split(dim_split, dim=2), dim=0)
        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), dim=2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), dim=0), dim=2)
        O = O + F.relu(self.fc_o(O))
        if orig_dims == 2:
            O = O.squeeze(0)
        elif orig_dims == 4:
            O = O.view(orig_shape[0], orig_shape[1], orig_shape[2], self.dim_V)
        h = self.norm_0(Q + O)
        return h + self.ff(h)


class MyModule(nn.Module):
    def __init__(self, n_mhas, emb_size, num_heads, ln):
        super().__init__()
        self.layes = nn.ModuleList([
            MAB(emb_size, emb_size, emb_size, num_heads, ln) for _ in range(n_mhas)
        ])

    def forward(self, x, y):
        for layer in self.layes:
            x = layer(x, x)
        return x


class TransformerSetInvariant_WITH_MAPS(nn.Module):
    def __init__(self, emb_size=8, output_size=8, num_heads=2, n_mha=1, instance_agg_learned_vectors=4):
        super().__init__()
        self.emb_size = emb_size
        self.instance_agg_learned_vectors = instance_agg_learned_vectors
        self.emb_proj = nn.Linear(1, emb_size)
        self.emb_type = nn.Parameter(torch.rand(2, emb_size))
        self.instance_agg_vectors = nn.Parameter(torch.rand(emb_size, instance_agg_learned_vectors))
        self.mha_matrix_fet = MyModule(n_mha, emb_size, num_heads, True)
        self.instance_mlp = nn.Sequential(
            nn.Linear(emb_size * instance_agg_learned_vectors, emb_size * instance_agg_learned_vectors),
            nn.GELU(),
            nn.Linear(emb_size * instance_agg_learned_vectors, output_size),
            nn.GELU(),
            nn.Linear(output_size, output_size),
        )
        self.output_norm = nn.LayerNorm(emb_size)

    def forward(self, x, idx_feature=None):
        feature_type = torch.zeros(x.size(0), x.size(1), x.size(2), dtype=torch.long, device=x.device)
        if idx_feature is not None:
            feature_type[:, :, idx_feature.argsort()[-1]] = 1
        else:
            feature_type[:, :, -1] = 1
        x_sum = self.emb_proj(x) + self.emb_type[feature_type]
        x_sum = self.mha_matrix_fet(x_sum, x_sum)
        lines_aggregated = x_sum.mean(dim=-2)
        importance = torch.sigmoid(lines_aggregated @ self.instance_agg_vectors)
        product = lines_aggregated.unsqueeze(3) * importance.unsqueeze(2)
        aggregated = product.mean(dim=1).permute(0, 2, 1)
        aggregated = self.output_norm(aggregated)
        dataset_representation = aggregated.reshape(1, -1)
        return self.instance_mlp(dataset_representation)


class AttReprModel(nn.Module):
    """Attention + MLP + cabeça de CE. A CE retropropaga no transformer e na MLP.

    `group_sizes` = nº de valores de cada hiperparâmetro; a cabeça produz um logit por valor
    (softmax separado por hiperparâmetro)."""

    def __init__(self, group_sizes, att_output_size: int = 8, emb_size: int = 8):
        super().__init__()
        self.att = TransformerSetInvariant_WITH_MAPS(
            emb_size=emb_size,
            output_size=att_output_size,
            num_heads=2,
            n_mha=1,
            instance_agg_learned_vectors=4,
        )
        self.mlp = MLPRefine(att_output_size, OUTPUT_EMB_SIZE)
        self.head = nn.Linear(OUTPUT_EMB_SIZE, int(sum(group_sizes)))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.att(x)
        return self.mlp(emb).squeeze(0)

    def forward(self, x: torch.Tensor):
        z = self.encode(x).unsqueeze(0)
        return z, self.head(z)


def build_init_state(group_sizes, seed: int = SEED) -> dict:
    """Estado inicial (sem treino) compartilhado por todos os splits."""
    torch.manual_seed(seed)
    net = AttReprModel(group_sizes)
    return copy.deepcopy(net.state_dict())


# ============================================================================
# Dados: datasets brutos -> tensores (1, n_inst, n_feat, 1) com o rótulo na última coluna
# ============================================================================
def load_arrays(meta: MetaTask) -> dict:
    """Features padronizadas por dataset (z-score) e rótulo normalizado em [0,1] (y / (C-1))."""
    arrays = {}
    for name in meta.names:
        X, y = meta.load_xy(name)
        X = X.astype(np.float64)
        sd = X.std(axis=0)
        sd[sd == 0] = 1.0
        X = np.nan_to_num((X - X.mean(axis=0)) / sd).astype(np.float32)
        yn = (y / max(int(y.max()), 1)).astype(np.float32)
        arrays[name] = (X, yn)
    return arrays


def make_tensor(X: np.ndarray, y: np.ndarray, rng: np.random.Generator,
                max_instances: int = MAX_INSTANCES_PER_DATASET,
                max_features: int = MAX_FEATURES_PER_DATASET) -> torch.Tensor:
    n, d = X.shape
    rows = rng.choice(n, size=max_instances, replace=False) if n > max_instances else np.arange(n)
    cols = np.sort(rng.choice(d, size=max_features - 1, replace=False)) if d > max_features - 1 else np.arange(d)
    arr = np.concatenate([X[np.ix_(rows, cols)], y[rows][:, None]], axis=1)
    return torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).unsqueeze(-1)


def make_eval_tensors(meta: MetaTask, arrays: dict) -> list:
    """Uma subamostra fixa (seed) por dataset, usada em toda extração de embedding."""
    return [make_tensor(*arrays[n], np.random.default_rng(SEED)) for n in meta.names]


# ============================================================================
# Alvo da CE: distribuição de cada hiperparâmetro entre as top-k configs do dataset
# ============================================================================
def build_group_info(configs: pd.DataFrame):
    cats = [sorted(configs[c].unique().tolist()) for c in cfg.CONFIG_COLS]
    sizes = [len(c) for c in cats]
    codes = np.stack(
        [pd.Categorical(configs[c], categories=cats[j]).codes for j, c in enumerate(cfg.CONFIG_COLS)], axis=1
    )
    return sizes, codes, cats


def soft_targets(y_row: np.ndarray, codes: np.ndarray, sizes, top_k: int = TARGET_TOP_K) -> np.ndarray:
    thr = np.sort(y_row)[::-1][top_k - 1]
    top = y_row >= thr                       # empates no limiar entram (muitos datasets têm acc=1.0 em várias configs)
    parts = []
    for j, sz in enumerate(sizes):
        counts = np.bincount(codes[top, j], minlength=sz).astype(np.float32)
        parts.append(counts / counts.sum())
    return np.concatenate(parts)


def soft_multihead_ce(logits: torch.Tensor, target: torch.Tensor, sizes) -> torch.Tensor:
    """CE com alvo suave, média entre as cabeças (um softmax por hiperparâmetro)."""
    total, start = 0.0, 0
    for sz in sizes:
        logp = F.log_softmax(logits[..., start:start + sz], dim=-1)
        total = total - (target[..., start:start + sz] * logp).sum(dim=-1)
        start += sz
    return (total / len(sizes)).mean()


# ============================================================================
# Extração de embeddings e diagnóstico de warm-start
# ============================================================================
@torch.no_grad()
def encode_all(net: nn.Module, eval_tensors: list, indices=None) -> np.ndarray:
    net.eval()
    indices = range(len(eval_tensors)) if indices is None else indices
    rows = [net.encode(eval_tensors[i].to(DEVICE)).cpu().numpy() for i in indices]
    return np.asarray(rows, dtype=np.float32)


@torch.no_grad()
def eval_ce(net: nn.Module, eval_tensors: list, targets: dict, indices, sizes) -> float:
    net.eval()
    vals = []
    for i in indices:
        _, logits = net(eval_tensors[i].to(DEVICE))
        tgt = torch.from_numpy(targets[i]).to(DEVICE).unsqueeze(0)
        vals.append(float(soft_multihead_ce(logits, tgt, sizes)))
    return float(np.mean(vals))


def _normalize_responses(Y: np.ndarray) -> np.ndarray:
    lo, hi = Y.min(axis=1, keepdims=True), Y.max(axis=1, keepdims=True)
    return (Y - lo) / np.where(hi - lo > 0, hi - lo, 1.0)


def warmstart_regret(Z: np.ndarray, Yn: np.ndarray, train_idx, eval_idx, k: int = cfg.N_INIT) -> float:
    """Diagnóstico: regret médio do warm-start (1 vizinho, top-k configs) dos `eval_idx` usando só os `train_idx`."""
    rows = np.r_[train_idx, eval_idx]
    lo, span = Z[rows].min(axis=0), Z[rows].max(axis=0) - Z[rows].min(axis=0)
    span[span == 0] = 1.0
    Zn = (Z - lo) / span
    cand = np.asarray(train_idx)
    regs = []
    for i in eval_idx:
        j = cand[int(np.argmin(np.linalg.norm(Zn[cand] - Zn[i], axis=1)))]
        top = np.argsort(Yn[j], kind="stable")[::-1][:k]
        regs.append(1.0 - Yn[i, top].max())
    return float(np.mean(regs))


# ============================================================================
# Treino de um split
# ============================================================================
def train_split(split: int, meta: MetaTask, arrays: dict, eval_tensors: list, init_state: dict, sizes, codes,
                max_epochs: int = FT_MAX_EPOCHS, show_bar: bool = True) -> dict:
    """Treina a Attention+MLP no split `split` (train), seleciona a época pela CE de validação (valid)
    e devolve os embeddings de todos os datasets. Os datasets de teste do split não são usados."""
    idx = meta.split_indices(split)
    train_idx, valid_idx = idx["train"], idx["valid"]
    Yn = _normalize_responses(meta.Y_valid)
    targets = {i: soft_targets(meta.Y_valid[i], codes, sizes) for i in train_idx + valid_idx}

    net = AttReprModel(sizes).to(DEVICE)
    net.load_state_dict(init_state)
    opt = Adam(net.parameters(), lr=FT_LR)
    rng = np.random.default_rng(SEED + split)

    history, best = [], {"val_ce": float("inf"), "epoch": 0, "state": copy.deepcopy(net.state_dict())}
    t_val = [0.0]

    def validate(epoch, train_ce):
        t0 = time.perf_counter()
        Z = encode_all(net, eval_tensors)
        row = {
            "epoch": epoch, "train_ce": train_ce,
            "val_ce": eval_ce(net, eval_tensors, targets, valid_idx, sizes),
            "val_regret_warmstart": warmstart_regret(Z, Yn, train_idx, valid_idx),
        }
        history.append(row)
        t_val[0] += time.perf_counter() - t0
        return row

    t_start = time.perf_counter()
    row0 = validate(0, float("nan"))                 # estado inicial (sem treino)
    best["val_ce"] = row0["val_ce"]
    wait = 0

    bar = tqdm(range(1, max_epochs + 1), desc=f"Our-Trained split {split}", disable=not show_bar, leave=False)
    for epoch in bar:
        net.train()
        order = rng.permutation(train_idx)
        loss_sum = 0.0
        for b in range(0, len(order), FT_BATCH_DATASETS):
            batch = order[b:b + FT_BATCH_DATASETS]
            opt.zero_grad(set_to_none=True)
            for j in batch:
                x = make_tensor(*arrays[meta.names[j]], rng).to(DEVICE)
                _, logits = net(x)
                tgt = torch.from_numpy(targets[j]).to(DEVICE).unsqueeze(0)
                loss_j = soft_multihead_ce(logits, tgt, sizes) / len(batch)
                loss_j.backward()                    # acumula gradientes; libera o grafo a cada dataset
                loss_sum += float(loss_j.detach()) * len(batch)
            torch.nn.utils.clip_grad_norm_(net.parameters(), FT_GRAD_CLIP)
            opt.step()
        train_ce = loss_sum / len(order)

        if epoch % FT_VAL_EVERY == 0 or epoch == max_epochs:
            row = validate(epoch, train_ce)
            bar.set_postfix(train_ce=f"{train_ce:.3f}", val_ce=f"{row['val_ce']:.3f}",
                            val_regret=f"{row['val_regret_warmstart']:.3f}", best_ep=best["epoch"])
            if row["val_ce"] < best["val_ce"] - 1e-6:
                best.update(val_ce=row["val_ce"], epoch=epoch, state=copy.deepcopy(net.state_dict()))
                wait = 0
            else:
                wait += 1
                if wait >= FT_PATIENCE:
                    break
    bar.close()
    t_loop = time.perf_counter() - t_start

    net.load_state_dict(best["state"])
    t0 = time.perf_counter()
    Z = encode_all(net, eval_tensors)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t_extract = time.perf_counter() - t0
    role = {i: "train" for i in train_idx}
    role.update({i: "valid" for i in valid_idx})
    role.update({i: "test" for i in idx["test"]})
    emb = pd.DataFrame(Z, index=meta.names, columns=[str(j) for j in range(OUTPUT_EMB_SIZE)])
    emb.insert(0, "role", [role[i] for i in range(meta.n_ds)])
    emb.index.name = "ds_n"
    return {"embeddings": emb, "state": best["state"], "best_epoch": best["epoch"],
            "history": pd.DataFrame(history), "sizes": list(sizes),
            "time_train_s": t_loop - t_val[0], "time_val_s": t_val[0], "time_extract_120_s": t_extract,
            "epochs_run": epoch, "n_train": len(train_idx)}


# ============================================================================
# Orquestração + cache
# ============================================================================
def _paths(split: int):
    return (cfg.REPR_DIR / f"Our-Trained_split{split}.csv",
            cfg.MODELS_DIR / f"our_trained_split{split}.pt",
            cfg.MODELS_DIR / f"our_trained_split{split}_history.csv")


def _log_times(row: dict) -> None:
    df = pd.read_csv(TIMES_CSV) if TIMES_CSV.exists() else pd.DataFrame()
    if len(df):
        df = df[df["split"] != row["split"]]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True).sort_values("split")
    df.to_csv(TIMES_CSV, index=False)


def run_all_splits(meta: MetaTask, splits=range(cfg.N_SPLITS), force: bool = False,
                   max_epochs: int = FT_MAX_EPOCHS) -> dict:
    """Treina (ou carrega do cache) a Our-Trained de cada split. Retorna {split: DataFrame de embeddings}."""
    cfg.make_dirs()
    sizes, codes, _ = build_group_info(meta.configs)
    init_state = build_init_state(sizes)
    arrays = eval_tensors = None
    out = {}
    for k in splits:
        emb_path, model_path, hist_path = _paths(k)
        if emb_path.exists() and model_path.exists() and not force:
            out[k] = pd.read_csv(emb_path, index_col=0)
            continue
        if arrays is None:
            print(f"Device: {DEVICE}")
            arrays = load_arrays(meta)
            eval_tensors = make_eval_tensors(meta, arrays)
        t0 = time.time()
        res = train_split(k, meta, arrays, eval_tensors, init_state, sizes, codes, max_epochs=max_epochs)
        res["embeddings"].to_csv(emb_path)
        res["history"].to_csv(hist_path, index=False)
        torch.save({"state_dict": {n: v.cpu() for n, v in res["state"].items()}, "best_epoch": res["best_epoch"],
                    "group_sizes": res["sizes"], "split": k, "emb_dim": OUTPUT_EMB_SIZE}, model_path)
        h = res["history"]
        best_row = h.loc[h["epoch"] == res["best_epoch"]].iloc[0]
        _log_times({"split": k, "dispositivo": str(DEVICE), "epocas_rodadas": res["epochs_run"],
                    "melhor_epoca": res["best_epoch"], "n_treino": res["n_train"],
                    "treino_s": round(res["time_train_s"], 2), "validacoes_s": round(res["time_val_s"], 2),
                    "extracao_120_datasets_s": round(res["time_extract_120_s"], 3),
                    "val_ce_inicial": h["val_ce"].iloc[0], "val_ce_melhor": best_row["val_ce"],
                    "val_regret_ws_inicial": h["val_regret_warmstart"].iloc[0],
                    "val_regret_ws_melhor": best_row["val_regret_warmstart"]})
        print(f"[split {k}] melhor época={res['best_epoch']} | val_ce {h['val_ce'].iloc[0]:.3f} -> "
              f"{best_row['val_ce']:.3f} | regret warm-start (valid) {h['val_regret_warmstart'].iloc[0]:.3f} -> "
              f"{best_row['val_regret_warmstart']:.3f} | {time.time() - t0:.0f}s")
        out[k] = res["embeddings"]
    return out
