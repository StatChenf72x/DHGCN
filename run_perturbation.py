import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import warnings
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import HypergraphConv
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data")


class Config:
    embedding_dim = 128
    hidden_dim = 64
    num_classes = 10
    num_epochs = 50
    batch_size = 256
    learning_rate = 1e-3
    weight_decay = 1e-5
    dropout = 0.2
    patience = 3
    min_improvement = 0.001
    alpha_hgcn = 0.7
    num_heads = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    def inference(self, triple_idx):
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
        return self.alpha * logits_hgcn + (1 - self.alpha) * logits_mlp


def load_df(path):
    df = pd.read_csv(path)
    cols = df.columns
    df = df[[cols[0], cols[1], cols[2], cols[-1]]].copy()
    df.columns = ["drug", "gene", "disease", "label"]
    df["label"] = df["label"].astype(int)
    return df


def main():
    set_seed(42)
    print("加载数据...")
    df_train = load_df(os.path.join(DATA_DIR, "train_14400.csv"))
    df_val = load_df(os.path.join(DATA_DIR, "val_14400.csv"))
    df_test = load_df(os.path.join(DATA_DIR, "test_14400.csv"))

    node_to_idx, cnt = {}, 0
    for df in [df_train, df_val, df_test]:
        for _, r in df.iterrows():
            for c in ['drug', 'gene', 'disease']:
                if r[c] not in node_to_idx: node_to_idx[r[c]] = cnt; cnt += 1
    num_nodes = cnt
    print(f"总节点数: {num_nodes}")

    def get_loader(df, shuffle):
        t = list(zip(df['drug'], df['gene'], df['disease']))
        y = df['label'].values
        return DataLoader(HyperedgeDataset(t, y, node_to_idx), batch_size=args.batch_size, shuffle=shuffle)

    train_loader = get_loader(df_train, True)
    val_loader = get_loader(df_val, False)
    test_loader = get_loader(df_test, False)

    train_triples = list(zip(df_train['drug'], df_train['gene'], df_train['disease']))
    he_list = build_mode_hyperedge_indices(train_triples, df_train['label'].values, node_to_idx, args.num_classes, args.device)

    cw = torch.tensor(compute_class_weight('balanced', classes=np.arange(args.num_classes), y=df_train['label']),
                      dtype=torch.float32).to(args.device)

    print("训练模型...")
    model = DHGCN(num_nodes, args.embedding_dim, args.hidden_dim, args.num_classes,
                  he_list, args.num_heads, args.alpha_hgcn, args.dropout).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=cw)

    best_val_acc = 0.0; patience_cnt = 0; best_weights = None
    for epoch in range(args.num_epochs):
        model.train()
        for idx, labels in train_loader:
            idx, labels = idx.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            logits = model.inference(idx)
            loss = criterion(logits, labels)
            loss.backward(); optimizer.step()

        model.eval()
        val_correct = 0; val_total = 0
        with torch.no_grad():
            for idx, labels in val_loader:
                idx, labels = idx.to(args.device), labels.to(args.device)
                pred = model.inference(idx).argmax(1)
                val_correct += (pred == labels).sum().item()
                val_total += len(labels)
        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1}: Val ACC={val_acc:.4f}")

        if val_acc - best_val_acc > args.min_improvement:
            best_val_acc = val_acc; best_weights = model.state_dict(); patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"Early stop at epoch {epoch+1}")
                break

    if best_weights: model.load_state_dict(best_weights)
    print(f"最佳验证集ACC: {best_val_acc:.4f}")

    print("\n挑样本做扰动分析...")
    model.eval()
    correct_samples = []
    with torch.no_grad():
        for idx, labels in test_loader:
            idx, labels = idx.to(args.device), labels.to(args.device)
            logits = model.inference(idx)
            prob = F.softmax(logits, dim=1)
            pred = prob.argmax(1)
            for i in range(len(idx)):
                if pred[i].item() == labels[i].item():
                    correct_samples.append({
                        'triple': idx[i].cpu(),
                        'true': labels[i].item(),
                        'orig_prob': prob[i, pred[i]].item()
                    })
    print(f"正确预测: {len(correct_samples)}")

    random.seed(42)
    chosen = random.sample(correct_samples, min(200, len(correct_samples)))

    all_node_ids = list(range(num_nodes))

    drop_drug = []
    drop_gene = []
    drop_disease = []

    print("做扰动分析...")
    for s in chosen:
        orig_d, orig_g, orig_s = s['triple'][0].item(), s['triple'][1].item(), s['triple'][2].item()
        orig_p = s['orig_prob']
        true_y = s['true']

        for _ in range(5):
            new_d = random.choice(all_node_ids)
            with torch.no_grad():
                t = torch.tensor([[new_d, orig_g, orig_s]], dtype=torch.long).to(args.device)
                p = F.softmax(model.inference(t), dim=1)[0, true_y].item()
            drop_drug.append(orig_p - p)

        for _ in range(5):
            new_g = random.choice(all_node_ids)
            with torch.no_grad():
                t = torch.tensor([[orig_d, new_g, orig_s]], dtype=torch.long).to(args.device)
                p = F.softmax(model.inference(t), dim=1)[0, true_y].item()
            drop_gene.append(orig_p - p)

        for _ in range(5):
            new_s = random.choice(all_node_ids)
            with torch.no_grad():
                t = torch.tensor([[orig_d, orig_g, new_s]], dtype=torch.long).to(args.device)
                p = F.softmax(model.inference(t), dim=1)[0, true_y].item()
            drop_disease.append(orig_p - p)

    print(f"Drug perturbed: mean drop = {np.mean(drop_drug):.4f}")
    print(f"Gene perturbed: mean drop = {np.mean(drop_gene):.4f}")
    print(f"Disease perturbed: mean drop = {np.mean(drop_disease):.4f}")

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12
    fig, ax = plt.subplots(figsize=(7, 5))

    data = [drop_drug, drop_gene, drop_disease]
    labels = ['Drug', 'Gene', 'Disease']
    colors = ['#4a7ba6', '#c0504d', '#5b9b5b']

    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5,
                    medianprops=dict(color='black', linewidth=1.5),
                    flierprops=dict(marker='o', markerfacecolor='gray', markersize=3, alpha=0.5))

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('Drop in Predicted Probability', fontsize=13, fontweight='bold')
    ax.set_xlabel('Perturbed Component', fontsize=13, fontweight='bold')
    ax.set_title('Component Perturbation Analysis', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    plt.tight_layout()
    out_path = os.path.join(BASE_DIR, "perturbation_boxplot.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\n图已保存: {out_path}")

    with open(os.path.join(BASE_DIR, "perturbation_stats.txt"), "w") as f:
        f.write(f"n_samples={len(chosen)}\n")
        f.write(f"drug_mean_drop={np.mean(drop_drug):.4f}\n")
        f.write(f"gene_mean_drop={np.mean(drop_gene):.4f}\n")
        f.write(f"disease_mean_drop={np.mean(drop_disease):.4f}\n")


if __name__ == "__main__":
    main()
