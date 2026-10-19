import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pathlib
import warnings
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import HypergraphConv
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score, average_precision_score)
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data")
IMBALANCED_DIR = os.path.join(BASE_DIR, "imbalanced_test_Data")

class Config:
    def __init__(self):
        self.train_data_path = os.path.join(DATA_DIR, "train_14400.csv")
        self.val_data_path = os.path.join(DATA_DIR, "val_14400.csv")
        self.embedding_dim = 128
        self.hidden_dim = 64
        self.num_classes = 10
        self.num_epochs = 100
        self.batch_size = 256
        self.learning_rate = 1e-3
        self.weight_decay = 1e-5
        self.dropout = 0.2
        self.patience = 3
        self.min_improvement = 0.001
        self.alpha_hgcn = 0.7
        self.num_heads = 4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = Config()

def set_seed(seed=42):
    np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

class HyperedgeDataset(Dataset):
    def __init__(self, triples, labels, node_to_idx):
        self.triples = triples; self.labels = labels; self.node_to_idx = node_to_idx
    def __len__(self): return len(self.triples)
    def __getitem__(self, idx):
        d, g, s = self.triples[idx]; y = self.labels[idx]
        return torch.tensor([self.node_to_idx[d], self.node_to_idx[g], self.node_to_idx[s]], dtype=torch.long), torch.tensor(y, dtype=torch.long)

def build_mode_hyperedge_indices(all_triples, all_rel, node_to_idx, num_classes, device):
    mode_to_triples = {c: [] for c in range(num_classes)}
    for trip, y in zip(all_triples, all_rel):
        mode_to_triples[int(y)].append(trip)
    hyperedge_indices = []
    for c in range(num_classes):
        triples_c = mode_to_triples.get(c, [])
        if len(triples_c) == 0:
            hyperedge_indices.append(None); continue
        node_ids, hedge_ids = [], []
        for e, (d, g, s) in enumerate(triples_c):
            node_ids += [node_to_idx[d], node_to_idx[g], node_to_idx[s]]
            hedge_ids += [e, e, e]
        hyperedge_indices.append(torch.tensor([node_ids, hedge_ids], dtype=torch.long, device=device))
    return hyperedge_indices

class IMF_Layer(nn.Module):
    def __init__(self, in_dim, dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.W_q = nn.Linear(in_dim, in_dim); self.W_k = nn.Linear(in_dim, in_dim); self.W_v = nn.Linear(in_dim, in_dim)
        self.dropout = nn.Dropout(dropout); self.layer_norm = nn.LayerNorm(in_dim)
    def forward(self, x):
        x_trans = x.permute(1, 0, 2); residual = x_trans
        Q = self.W_q(x_trans); K = self.W_k(x_trans); V = self.W_v(x_trans)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.in_dim ** 0.5)
        attn = self.dropout(F.softmax(scores, dim=-1))
        x_new = torch.matmul(attn, V)
        return self.layer_norm(x_new + residual).permute(1, 0, 2)

class ChannelAttention(nn.Module):
    def __init__(self, in_dim, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(in_dim, in_dim // reduction, bias=False), nn.ReLU(),
                                nn.Linear(in_dim // reduction, in_dim, bias=False), nn.Sigmoid())
    def forward(self, x):
        y = x.mean(dim=1, keepdim=True); w = self.fc(y); return x * w

class HighwayGatingLayer(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.H_layer = nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU())
        self.T_gate = nn.Sequential(nn.Linear(in_dim, in_dim), nn.Sigmoid())
        nn.init.constant_(self.T_gate[0].bias, -1.0)
    def forward(self, x):
        return self.H_layer(x) * self.T_gate(x) + x * (1.0 - self.T_gate(x))

class DHGCN(nn.Module):
    def __init__(self, num_nodes, embedding_dim, hidden_dim, num_classes, hyperedge_index_list, num_heads=4, alpha=0.7, dropout=0.2):
        super().__init__()
        self.num_modes = num_classes; self.alpha = alpha; self.num_heads = num_heads
        self.node_embedding = nn.Embedding(num_nodes, embedding_dim)
        self.hconvs1 = nn.ModuleList([HypergraphConv(embedding_dim, embedding_dim) for _ in range(self.num_modes)])
        self.hconvs2 = nn.ModuleList([HypergraphConv(embedding_dim, embedding_dim) for _ in range(self.num_modes)])
        self._register_edges(hyperedge_index_list)
        self.imf_layer = IMF_Layer(embedding_dim, dropout)
        self.channel_attn = ChannelAttention(embedding_dim)
        self.gating_layer = HighwayGatingLayer(embedding_dim)
        self.mode_embeddings = nn.Parameter(torch.randn(self.num_heads, self.num_modes, embedding_dim))
        self.act = nn.ReLU(); self.dropout = nn.Dropout(dropout)
        self.head_hgcn = nn.Sequential(nn.Linear(embedding_dim * 3, hidden_dim), nn.ReLU(),
                                        nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
                                        nn.Linear(hidden_dim // 2, num_classes))
        self.head_mlp = nn.Sequential(nn.Linear(embedding_dim * 3, hidden_dim), nn.ReLU(),
                                      nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
                                      nn.Linear(hidden_dim // 2, num_classes))
        self._init_weights()
    def _register_edges(self, hyperedge_index_list):
        dev = torch.device("cpu")
        for hed in hyperedge_index_list:
            if hed is not None: dev = hed.device; break
        for i, hedge_idx in enumerate(hyperedge_index_list):
            name = f"hyperedge_index_mode_{i}"
            if hedge_idx is None: self.register_buffer(name, torch.empty(2, 0, dtype=torch.long, device=dev))
            else: self.register_buffer(name, hedge_idx)
    def _init_weights(self):
        nn.init.xavier_uniform_(self.node_embedding.weight); nn.init.xavier_uniform_(self.mode_embeddings)
        for m in list(self.head_hgcn) + list(self.head_mlp):
            if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight)
    def forward(self, triple_idx):
        x0 = self.node_embedding.weight; mode_feats = []
        for i in range(self.num_modes):
            hedge = getattr(self, f"hyperedge_index_mode_{i}")
            if hedge.numel() == 0: mode_feats.append(x0); continue
            h1 = self.dropout(self.act(self.hconvs1[i](x0, hedge)))
            h2 = self.hconvs2[i](h1, hedge); mode_feats.append(h2)
        X_modes = torch.stack(mode_feats, dim=0)
        X_processed = self.imf_layer(X_modes)
        X_processed = self.channel_attn(X_processed)
        X_processed = self.gating_layer(X_processed)
        Xm = X_processed.unsqueeze(0); Sm = self.mode_embeddings.unsqueeze(2)
        attn_logits = (Xm * Sm).sum(dim=-1); attn = torch.softmax(attn_logits, dim=1)
        Z_per_head = (attn.unsqueeze(-1) * Xm).sum(dim=1); z = Z_per_head.mean(dim=0)
        d, g, s = triple_idx[:, 0], triple_idx[:, 1], triple_idx[:, 2]
        logits_hgcn = self.head_hgcn(torch.cat([z[d], z[g], z[s]], dim=1))
        logits_mlp = self.head_mlp(torch.cat([x0[d], x0[g], x0[s]], dim=1))
        return logits_hgcn, logits_mlp
    def inference(self, triple_idx):
        l_h, l_a = self.forward(triple_idx)
        return self.alpha * l_h + (1 - self.alpha) * l_a

def load_df(path):
    df = pd.read_csv(path)
    cols = df.columns
    df = df[[cols[0], cols[1], cols[2], cols[-1]]].copy()
    df.columns = ["drug", "gene", "disease", "label"]
    df["label"] = df["label"].astype(int)
    return df

def compute_metrics(labels, preds, probs, num_classes):
    metrics = {
        "ACC": accuracy_score(labels, preds),
        "PRE": precision_score(labels, preds, average='macro', zero_division=0),
        "REC": recall_score(labels, preds, average='macro', zero_division=0),
        "F1": f1_score(labels, preds, average='macro', zero_division=0)
    }
    try:
        y_oh = OneHotEncoder(sparse_output=False, categories=[range(num_classes)]).fit_transform(
            np.array(labels).reshape(-1, 1))
        metrics["AUC"] = roc_auc_score(y_oh, np.array(probs), multi_class="ovr", average="macro")
        metrics["AUPR"] = average_precision_score(y_oh, np.array(probs), average="macro")
    except:
        metrics["AUC"], metrics["AUPR"] = 0.0, 0.0
    return metrics

def evaluate(model, loader, num_classes, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for idx, labels in loader:
            idx, labels = idx.to(device), labels.to(device)
            logits = model.inference(idx)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(F.softmax(logits, 1).cpu().numpy())
    return compute_metrics(all_labels, all_preds, all_probs, num_classes)

def main():
    df_train = load_df(args.train_data_path)
    df_val = load_df(args.val_data_path)

    node_to_idx, cnt = {}, 0
    for df in [df_train, df_val]:
        for _, r in df.iterrows():
            for c in ['drug', 'gene', 'disease']:
                if r[c] not in node_to_idx: node_to_idx[r[c]] = cnt; cnt += 1

    def get_loader(df, shuffle):
        t = list(zip(df['drug'], df['gene'], df['disease']))
        y = df['label'].values
        return DataLoader(HyperedgeDataset(t, y, node_to_idx), batch_size=args.batch_size, shuffle=shuffle)

    train_loader = get_loader(df_train, True)
    val_loader = get_loader(df_val, False)

    train_triples = list(zip(df_train['drug'], df_train['gene'], df_train['disease']))
    he_list = build_mode_hyperedge_indices(train_triples, df_train['label'].values, node_to_idx, args.num_classes, args.device)

    cw = torch.tensor(compute_class_weight('balanced', classes=np.arange(args.num_classes), y=df_train['label']),
                      dtype=torch.float32).to(args.device)

    ratios = ["1to5", "1to10", "1to20", "1to50", "1to100"]
    all_results = {}

    for ratio in ratios:
        print(f"\n{'='*25} 负采样比例 1:{ratio[3:]} {'='*25}")
        df_test = pd.read_csv(os.path.join(IMBALANCED_DIR, f"test_{ratio}.csv"))
        cols = df_test.columns
        df_test = df_test[[cols[0], cols[1], cols[2], cols[-1]]].copy()
        df_test.columns = ["drug", "gene", "disease", "label"]
        df_test["label"] = df_test["label"].astype(int)

        n_pos = (df_test["label"] != 9).sum()
        n_neg = (df_test["label"] == 9).sum()
        print(f"  测试集: 总={len(df_test)}, 正={n_pos}, 负={n_neg}")

        for _, r in df_test.iterrows():
            for c in ['drug', 'gene', 'disease']:
                if r[c] not in node_to_idx:
                    node_to_idx[r[c]] = cnt; cnt += 1

        test_loader = get_loader(df_test, False)

        run_results = []
        for run in range(2):
            print(f"\n  --- Run {run+1}/2 ---")
            set_seed(42 + run)
            model = DHGCN(len(node_to_idx), args.embedding_dim, args.hidden_dim, args.num_classes,
                          he_list, args.num_heads, args.alpha_hgcn, args.dropout).to(args.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
            criterion = nn.CrossEntropyLoss(weight=cw)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

            best_val_acc = 0.0; patience_cnt = 0; best_weights = None

            for epoch in range(args.num_epochs):
                model.train()
                train_loss = 0
                for idx, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}", leave=False):
                    idx, labels = idx.to(args.device), labels.to(args.device)
                    optimizer.zero_grad()
                    out_m, out_a = model(idx)
                    loss = args.alpha_hgcn * criterion(out_m, labels) + (1 - args.alpha_hgcn) * criterion(out_a, labels)
                    loss.backward(); optimizer.step()
                    train_loss += loss.item()

                model.eval()
                val_preds, val_trues, val_probs = [], [], []
                with torch.no_grad():
                    for idx, labels in val_loader:
                        idx, labels = idx.to(args.device), labels.to(args.device)
                        logits = model.inference(idx)
                        val_preds.extend(logits.argmax(1).cpu().numpy())
                        val_trues.extend(labels.cpu().numpy())
                        val_probs.extend(F.softmax(logits, 1).cpu().numpy())
                val_metrics = compute_metrics(val_trues, val_preds, val_probs, args.num_classes)
                val_acc = val_metrics["ACC"]
                scheduler.step(train_loss / len(train_loader))

                print(f"Epoch {epoch+1}: Val ACC={val_acc:.4f} | AUPR={val_metrics['AUPR']:.4f} | F1={val_metrics['F1']:.4f}")

                if val_acc - best_val_acc > args.min_improvement:
                    best_val_acc = val_acc; best_weights = model.state_dict(); patience_cnt = 0
                else:
                    patience_cnt += 1
                    if patience_cnt >= args.patience:
                        print(f"  Early stop at epoch {epoch+1}")
                        break

            if best_weights: model.load_state_dict(best_weights)
            metrics = evaluate(model, test_loader, args.num_classes, args.device)
            run_results.append(metrics)
            print(f"  Run {run+1} Test: ACC={metrics['ACC']:.4f} AUPR={metrics['AUPR']:.4f} F1={metrics['F1']:.4f} AUC={metrics['AUC']:.4f}")

        mean_metrics = {k: np.mean([r[k] for r in run_results]) for k in run_results[0]}
        std_metrics = {k: np.std([r[k] for r in run_results]) for k in run_results[0]}
        all_results[f"1:{ratio[3:]}"] = {k: f"{mean_metrics[k]:.4f} ± {std_metrics[k]:.4f}" for k in mean_metrics}
        print(f"  Mean±Std: " + " | ".join(f"{k}={v}" for k, v in all_results[f"1:{ratio[3:]}"].items()))

    df_res = pd.DataFrame(all_results).T[["ACC", "PRE", "AUPR", "REC", "F1", "AUC"]]
    out_path = os.path.join(BASE_DIR, "class_imbalance_sensitivity_results.csv")
    df_res.to_csv(out_path, encoding="utf-8-sig")
    print(f"\n结果保存到: {out_path}")
    print(df_res)

if __name__ == "__main__":
    main()
