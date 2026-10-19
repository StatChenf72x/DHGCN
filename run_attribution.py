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
from sklearn.metrics import accuracy_score, f1_score
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

    def forward_with_attn(self, triple_idx):
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
        attn_logits = (Xm * Sm).sum(dim=-1)
        attn = torch.softmax(attn_logits, dim=1)
        Z_per_head = (attn.unsqueeze(-1) * Xm).sum(dim=1)
        z = Z_per_head.mean(dim=0)
        d, g, s = triple_idx[:, 0], triple_idx[:, 1], triple_idx[:, 2]
        logits_hgcn = self.head_hgcn(torch.cat([z[d], z[g], z[s]], dim=1))
        logits_mlp = self.head_mlp(torch.cat([x0[d], x0[g], x0[s]], dim=1))
        return logits_hgcn, logits_mlp, attn

    def inference(self, triple_idx):
        l_h, l_a, _ = self.forward_with_attn(triple_idx)
        return self.alpha * l_h + (1 - self.alpha) * l_a


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

    print(f"  train triples={len(train_triples)}, device={args.device}")
    print("训练模型...")
    model = DHGCN(len(node_to_idx), args.embedding_dim, args.hidden_dim, args.num_classes,
                  he_list, args.num_heads, args.alpha_hgcn, args.dropout).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=cw)

    best_val_acc = 0.0; patience_cnt = 0; best_weights = None
    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0
        for idx, labels in train_loader:
            idx, labels = idx.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            logits = model.inference(idx)
            loss = criterion(logits, labels)
            loss.backward(); optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_correct = 0; val_total = 0
        with torch.no_grad():
            for idx, labels in val_loader:
                idx, labels = idx.to(args.device), labels.to(args.device)
                logits = model.inference(idx)
                pred = logits.argmax(1)
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

    print("\n在测试集上挑样本...")
    model.eval()
    all_info = []
    with torch.no_grad():
        for idx, labels in test_loader:
            idx, labels = idx.to(args.device), labels.to(args.device)
            l_h, l_a, attn = model.forward_with_attn(idx)
            logits = model.alpha * l_h + (1 - model.alpha) * l_a
            prob = F.softmax(logits, dim=1)
            pred = prob.argmax(1)
            attn_avg = attn.mean(dim=0).cpu().numpy()
            for i in range(len(idx)):
                d, g, s = idx[i, 0].item(), idx[i, 1].item(), idx[i, 2].item()
                true_y = labels[i].item()
                pred_y = pred[i].item()
                confidence = prob[i, pred_y].item()
                node_attn = (attn_avg[:, d] + attn_avg[:, g] + attn_avg[:, s]) / 3.0
                all_info.append({
                    'true': true_y, 'pred': pred_y, 'confidence': confidence,
                    'attn': node_attn, 'triple': (d, g, s)
                })

    correct = [x for x in all_info if x['true'] == x['pred']]
    print(f"测试集预测正确: {len(correct)}/{len(all_info)}")

    valid = [x for x in correct if int(np.argmax(x['attn'])) == x['true']]
    print(f"  其中最高注意力在对应mode: {len(valid)}")

    by_class = {}
    for x in valid:
        by_class.setdefault(x['true'], []).append(x)
    for c in by_class:
        by_class[c].sort(key=lambda x: -x['confidence'])

    chosen = []
    classes_sorted = sorted(by_class.keys(), key=lambda c: -by_class[c][0]['confidence'])
    for c in classes_sorted[:3]:
        chosen.append(by_class[c][0])

    medium = [x for x in valid if 0.6 < x['confidence'] < 0.8]
    if medium:
        medium.sort(key=lambda x: x['confidence'])
        chosen.append(medium[len(medium)//2])
    elif len(by_class) > 3:
        chosen.append(by_class[classes_sorted[3]][0])

    print(f"\n挑了{len(chosen)}个样本:")
    for i, x in enumerate(chosen):
        print(f"  样本{i+1}: true=class{x['true']}, pred=class{x['pred']}, conf={x['confidence']:.4f}")

    attn_matrix = np.array([x['attn'] for x in chosen])
    true_labels = [x['true'] for x in chosen]
    pred_labels = [x['pred'] for x in chosen]

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 11
    fig, ax = plt.subplots(figsize=(10, 3.5))
    sns.heatmap(attn_matrix, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=[f'Mode {i}' for i in range(10)],
                yticklabels=[f'Sample {i+1}\n(true=Class {true_labels[i]})' for i in range(len(chosen))],
                cbar_kws={'label': 'Attention weight'},
                ax=ax, linewidths=0.5)
    for i in range(len(chosen)):
        j = true_labels[i]
        ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='blue', lw=2.5))

    ax.set_xlabel('Semantic Mode (Class)')
    ax.set_ylabel('Test Sample')
    ax.set_title('Mode-level Attention Attribution')
    plt.tight_layout()
    out_path = os.path.join(BASE_DIR, "attribution_heatmap.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\n图已保存: {out_path}")

    df_attn = pd.DataFrame(attn_matrix, columns=[f'Mode_{i}' for i in range(10)])
    df_attn.insert(0, 'TrueClass', true_labels)
    df_attn.insert(1, 'PredClass', pred_labels)
    df_attn.insert(2, 'Confidence', [x['confidence'] for x in chosen])
    csv_path = os.path.join(BASE_DIR, "attribution_data.csv")
    df_attn.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"数据已保存: {csv_path}")


if __name__ == "__main__":
    main()
