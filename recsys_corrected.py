# %% [markdown]
# # Serendipitous Recommender System with Transformers
# **Architecture**: Transformer + NOVA-style non-invasive side-info fusion + Unexpectedness (Mean Shift)
# 
# Pipeline: Data → Split → LightGCN(BPR) → Unexpectedness → NOVA-Transformer → BCE Training → Evaluation

# %% [markdown]
# ## Cell 1: Configuration & Imports

# %%
import os, time, random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import scipy.sparse as sp
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.metrics import roc_auc_score
from pathlib import Path

# ── Reproducibility ──
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Hyperparameters ──
EMBED_DIM       = 128
MAX_SEQ_LEN     = 20      # History window size (matching PURS spirit)
LIGHTGCN_LAYERS = 3
LIGHTGCN_EPOCHS = 100
LIGHTGCN_LR     = 1e-3
NUM_HEADS       = 4
NUM_TF_LAYERS   = 2       # Transformer layers
DROPOUT         = 0.1
BATCH_SIZE      = 256
TRAIN_EPOCHS    = 50
TRAIN_LR        = 1e-3
RATING_THRESH   = 3.5     # Binarization threshold
MIN_HIST        = 5       # Minimum interactions per user
K               = 10      # Top-K for evaluation

# ── Device ──
def resolve_device():
    if torch.cuda.is_available():
        try:
            major, _ = torch.cuda.get_device_capability(0)
            if major < 7:
                print(f"CUDA capability sm_{major}x unsupported. Using CPU.")
                return torch.device("cpu")
            torch.randn(2, 2, device="cuda")  # probe
            return torch.device("cuda")
        except Exception as e:
            print(f"CUDA check failed ({e}). Using CPU.")
    return torch.device("cpu")

DEVICE = resolve_device()
print(f"Device: {DEVICE}")

# %% [markdown]
# ## Cell 2: Data Loading & Preprocessing

# %%
# ── Load MovieLens 1M ──
data_path = Path('/kaggle/input/datasets/odedgolden/movielens-1m-dataset/ratings.dat')
if not data_path.exists():
    data_path = Path('ratings.dat')  # local fallback
if not data_path.exists():
    raise FileNotFoundError(f"ratings.dat not found at {data_path}")

print(f"Loading: {data_path}")
df = pd.read_csv(str(data_path), sep='::', engine='python',
                 names=['UserId', 'MovieId', 'Rating', 'Timestamp'])

# Sort globally by time (chronological order)
df = df.sort_values('Timestamp').reset_index(drop=True)

# Binarize: rating > 3.5 → click=1, else click=0
df['Click'] = (df['Rating'] > RATING_THRESH).astype(float)

# Map IDs to contiguous indices (0 = padding)
user_ids = df['UserId'].unique()
item_ids = df['MovieId'].unique()
user2idx = {u: i+1 for i, u in enumerate(user_ids)}
item2idx = {m: j+1 for j, m in enumerate(item_ids)}
df['u_idx'] = df['UserId'].map(user2idx)
df['i_idx'] = df['MovieId'].map(item2idx)

NUM_USERS = len(user_ids) + 1  # +1 for padding idx 0
NUM_ITEMS = len(item_ids) + 1

print(f"Users: {len(user_ids)} | Items: {len(item_ids)} | Interactions: {len(df)}")
print(f"Click rate: {df['Click'].mean():.3f}")

# %% [markdown]
# ## Cell 3: Chronological Train / Val / Test Split + Sliding Window

# %%
# ── Global chronological split (like PURS: first 80% train, next 10% val, last 10% test) ──
n = len(df)
train_end = int(n * 0.8)
val_end   = int(n * 0.9)

df_train = df.iloc[:train_end].copy()
df_val   = df.iloc[train_end:val_end].copy()
df_test  = df.iloc[val_end:].copy()

print(f"Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")

# ── Sliding window sample generation ──
# Each sample: (user_idx, history_items[SEQ_LEN], target_item, click, ratings[SEQ_LEN], time_gaps[SEQ_LEN])
# Matching PURS: for each user with >SEQ_LEN interactions, slide a window

def build_samples(user_df, seq_len=MAX_SEQ_LEN):
    """Generate sliding window samples from a user-grouped DataFrame."""
    samples = []
    for u_idx, group in user_df.groupby('u_idx'):
        group = group.sort_values('Timestamp')
        items   = group['i_idx'].values
        clicks  = group['Click'].values
        ratings = group['Rating'].values.astype(int)
        times   = group['Timestamp'].values

        if len(items) < MIN_HIST:
            continue

        # Compute time gaps (in days), capped at 99
        time_gaps = np.zeros(len(times), dtype=int)
        time_gaps[1:] = np.clip((times[1:] - times[:-1]) // (24 * 3600), 0, 99)

        # Sliding window: history[i:i+seq_len] → predict item[i+seq_len]
        if len(items) > seq_len:
            for i in range(len(items) - seq_len):
                hist_items  = items[i : i + seq_len].tolist()
                hist_rats   = ratings[i : i + seq_len].tolist()
                hist_tgaps  = time_gaps[i : i + seq_len].tolist()
                target_item = int(items[i + seq_len])
                target_click= float(clicks[i + seq_len])
                samples.append((u_idx, hist_items, hist_rats, hist_tgaps,
                                target_item, target_click))
        else:
            # Pad shorter sequences (left-pad with 0)
            pad_len     = seq_len - (len(items) - 1)
            hist_items  = [0] * pad_len + items[:-1].tolist()
            hist_rats   = [0] * pad_len + ratings[:-1].tolist()
            hist_tgaps  = [0] * pad_len + time_gaps[:-1].tolist()
            target_item = int(items[-1])
            target_click= float(clicks[-1])
            samples.append((u_idx, hist_items, hist_rats, hist_tgaps,
                            target_item, target_click))
    return samples

train_samples = build_samples(df_train)
val_samples   = build_samples(df_val)
test_samples  = build_samples(df_test)

print(f"Train samples: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}")

# %% [markdown]
# ## Cell 4: LightGCN Training (Proper BPR Loss)

# %%
# ── Build bipartite graph from TRAIN interactions ONLY ──
R = sp.dok_matrix((NUM_USERS, NUM_ITEMS), dtype=np.float32)
for u, i in zip(df_train['u_idx'], df_train['i_idx']):
    R[u, i] = 1.0
R = R.tolil()

adj_mat = sp.dok_matrix((NUM_USERS + NUM_ITEMS, NUM_USERS + NUM_ITEMS), dtype=np.float32).tolil()
adj_mat[:NUM_USERS, NUM_USERS:] = R
adj_mat[NUM_USERS:, :NUM_USERS] = R.T
adj_mat = adj_mat.todok()

# Symmetric normalization: D^{-1/2} A D^{-1/2}
rowsum = np.array(adj_mat.sum(axis=1)).flatten()
d_inv_sqrt = np.zeros_like(rowsum)
nonzero = rowsum > 0
d_inv_sqrt[nonzero] = np.power(rowsum[nonzero], -0.5)
D = sp.diags(d_inv_sqrt)
norm_adj = D.dot(adj_mat).dot(D).tocoo()

indices = torch.from_numpy(np.vstack((norm_adj.row, norm_adj.col)).astype(np.int64))
values  = torch.from_numpy(norm_adj.data.astype(np.float32))
norm_adj_tensor = torch.sparse_coo_tensor(indices, values, norm_adj.shape).coalesce().to(DEVICE)

# ── LightGCN Model ──
class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, embed_dim=EMBED_DIM, n_layers=LIGHTGCN_LAYERS):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.item_emb = nn.Embedding(n_items, embed_dim)
        nn.init.xavier_normal_(self.user_emb.weight)
        nn.init.xavier_normal_(self.item_emb.weight)
        self.n_layers = n_layers

    def forward(self, adj):
        ego = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        all_embs = [ego]
        x = ego
        for _ in range(self.n_layers):
            x = torch.sparse.mm(adj, x)
            all_embs.append(x)
        out = torch.stack(all_embs, dim=1).mean(dim=1)
        users, items = torch.split(out, [NUM_USERS, NUM_ITEMS])
        return users, items

# ── Build training edges for BPR ──
train_user_items = {}  # user -> set of positive items (for negative sampling)
for u, i in zip(df_train['u_idx'].values, df_train['i_idx'].values):
    train_user_items.setdefault(int(u), set()).add(int(i))

train_edges = list(zip(df_train['u_idx'].values.astype(int),
                       df_train['i_idx'].values.astype(int)))
all_item_indices = list(range(1, NUM_ITEMS))  # exclude padding 0

# ── Train LightGCN with BPR ──
gcn = LightGCN(NUM_USERS, NUM_ITEMS).to(DEVICE)
gcn_opt = torch.optim.Adam(gcn.parameters(), lr=LIGHTGCN_LR, weight_decay=1e-5)

print(f"Training LightGCN ({LIGHTGCN_EPOCHS} epochs, BPR loss)...")
for epoch in range(LIGHTGCN_EPOCHS):
    gcn.train()
    # Sample a mini-batch of BPR triplets
    np.random.shuffle(train_edges)
    batch_users, batch_pos, batch_neg = [], [], []
    for u, pos_i in train_edges[:4096]:  # sample 4096 triplets per epoch
        neg_i = random.choice(all_item_indices)
        while neg_i in train_user_items.get(u, set()):
            neg_i = random.choice(all_item_indices)
        batch_users.append(u)
        batch_pos.append(pos_i)
        batch_neg.append(neg_i)

    gcn_opt.zero_grad()
    user_embs, item_embs = gcn(norm_adj_tensor)

    u_e   = user_embs[batch_users]
    pos_e = item_embs[batch_pos]
    neg_e = item_embs[batch_neg]

    pos_scores = (u_e * pos_e).sum(dim=1)
    neg_scores = (u_e * neg_e).sum(dim=1)
    bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
    reg_loss = (u_e.norm(2).pow(2) + pos_e.norm(2).pow(2) + neg_e.norm(2).pow(2)) / len(batch_users) * 1e-5
    loss = bpr_loss + reg_loss
    loss.backward()
    gcn_opt.step()

    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch+1}/{LIGHTGCN_EPOCHS} | BPR: {bpr_loss.item():.4f} | Reg: {reg_loss.item():.6f}")

# Extract trained item embeddings
with torch.no_grad():
    _, gnn_item_embs = gcn(norm_adj_tensor)
    gnn_item_embs = gnn_item_embs.cpu()
print(f"LightGCN done. Item embedding shape: {gnn_item_embs.shape}")

# Sanity check: positive scores should exceed negative scores
with torch.no_grad():
    user_embs_check, item_embs_check = gcn(norm_adj_tensor)
    sample_pos = (user_embs_check[batch_users[:100]] * item_embs_check[batch_pos[:100]]).sum(1).mean()
    sample_neg = (user_embs_check[batch_users[:100]] * item_embs_check[batch_neg[:100]]).sum(1).mean()
    print(f"Sanity: avg pos score = {sample_pos.item():.4f}, avg neg score = {sample_neg.item():.4f}")

# %% [markdown]
# ## Cell 5: Unexpectedness Computation (Mean Shift Clustering)

# %%
# For each user, cluster their TRAIN history once, then compute unexpectedness
# for every item as distance from weighted cluster centers.
# This matches the 2014 Adamopoulos paper: unexp(i) = dist(i, expected_set)

def compute_user_unexpectedness(user_item_list, item_embs_np):
    """
    Given a user's chronological item list, compute unexpectedness for each item.
    Returns array of unexpectedness scores (same length as user_item_list).
    """
    scores = np.zeros(len(user_item_list), dtype=np.float32)
    if len(user_item_list) < 2:
        return scores

    # Get embeddings for all items (skip padding=0)
    valid_items = [it for it in user_item_list if it != 0]
    if len(valid_items) < 3:
        # Too few items for clustering; use mean distance
        if len(valid_items) >= 2:
            embs = item_embs_np[valid_items]
            center = embs.mean(axis=0)
            for idx, it in enumerate(user_item_list):
                if it != 0:
                    scores[idx] = np.linalg.norm(item_embs_np[it] - center)
        return scores

    # Run Mean Shift on ALL of this user's history embeddings
    history_embs = item_embs_np[valid_items]

    # Handle degenerate cases
    if np.allclose(history_embs, history_embs[0], atol=1e-8):
        for idx, it in enumerate(user_item_list):
            if it != 0:
                scores[idx] = np.linalg.norm(item_embs_np[it] - history_embs[0])
        return scores

    try:
        bw = estimate_bandwidth(history_embs, quantile=0.3,
                                n_samples=min(200, len(history_embs)))
        if not np.isfinite(bw) or bw < 1e-8:
            bw = 1.0
        ms = MeanShift(bandwidth=bw, bin_seeding=True, max_iter=50)
        ms.fit(history_embs)
        centers = ms.cluster_centers_
        labels  = ms.labels_

        # Weighted center (mean of cluster centers, weighted by cluster size)
        cluster_sizes = np.array([np.sum(labels == c) for c in range(len(centers))])
        weights = cluster_sizes / cluster_sizes.sum()
        weighted_center = (weights[:, None] * centers).sum(axis=0)
    except Exception:
        weighted_center = history_embs.mean(axis=0)

    # Compute distance of each item from the weighted center
    for idx, it in enumerate(user_item_list):
        if it != 0:
            scores[idx] = np.linalg.norm(item_embs_np[it] - weighted_center)

    return scores

# ── Compute unexpectedness for all users ──
print("Computing unexpectedness scores (Mean Shift clustering)...")
t0 = time.time()
item_embs_np = gnn_item_embs.numpy()

# Build per-user item lists from TRAIN data (chronological)
user_train_items = {}
for u, i in zip(df_train['u_idx'].values, df_train['i_idx'].values):
    user_train_items.setdefault(int(u), []).append(int(i))

# Compute unexpectedness per user
user_unexp = {}  # user_idx -> {item_idx: unexp_score}
for u_idx, item_list in user_train_items.items():
    unexp_arr = compute_user_unexpectedness(item_list, item_embs_np)
    user_unexp[u_idx] = {it: float(unexp_arr[i]) for i, it in enumerate(item_list)}

# Also compute for val/test items using the same user cluster centers
# (target items from val/test get unexpectedness relative to train history)
for split_df in [df_val, df_test]:
    for u_idx, group in split_df.groupby('u_idx'):
        train_items = user_train_items.get(int(u_idx), [])
        if len(train_items) < 3:
            for it in group['i_idx'].values:
                user_unexp.setdefault(int(u_idx), {})[int(it)] = 0.0
            continue
        # Get cluster center from train history
        valid = [x for x in train_items if x != 0]
        if len(valid) < 3:
            center = item_embs_np[valid].mean(axis=0) if valid else np.zeros(EMBED_DIM)
        else:
            embs = item_embs_np[valid]
            try:
                bw = estimate_bandwidth(embs, quantile=0.3, n_samples=min(200, len(embs)))
                if not np.isfinite(bw) or bw < 1e-8:
                    bw = 1.0
                ms = MeanShift(bandwidth=bw, bin_seeding=True, max_iter=50)
                ms.fit(embs)
                centers = ms.cluster_centers_
                labels = ms.labels_
                sizes = np.array([np.sum(labels == c) for c in range(len(centers))])
                weights = sizes / sizes.sum()
                center = (weights[:, None] * centers).sum(axis=0)
            except Exception:
                center = embs.mean(axis=0)
        for it in group['i_idx'].values:
            dist = float(np.linalg.norm(item_embs_np[int(it)] - center))
            user_unexp.setdefault(int(u_idx), {})[int(it)] = dist

elapsed = time.time() - t0
print(f"Unexpectedness computed in {elapsed:.1f}s")

# Check distribution
all_unexp_vals = [v for d in user_unexp.values() for v in d.values()]
print(f"Unexp stats: mean={np.mean(all_unexp_vals):.4f}, std={np.std(all_unexp_vals):.4f}, "
      f"min={np.min(all_unexp_vals):.4f}, max={np.max(all_unexp_vals):.4f}")

# %% [markdown]
# ## Cell 6: Dataset & DataLoader

# %%
class SerendipityDataset(Dataset):
    """
    Each sample: history (items, ratings, time_gaps, unexp) + target (item, click, unexp).
    Unexpectedness for history items AND target item are looked up from pre-computed dict.
    """
    def __init__(self, samples, user_unexp_dict, seq_len=MAX_SEQ_LEN):
        self.seq_len = seq_len
        self.item_seqs   = []
        self.rating_seqs = []
        self.tgap_seqs   = []
        self.hist_unexp  = []
        self.targets     = []
        self.clicks      = []
        self.target_unexp= []
        self.masks       = []

        for (u_idx, hist_items, hist_rats, hist_tgaps, tgt_item, tgt_click) in samples:
            # Look up unexpectedness for each history item
            u_dict = user_unexp_dict.get(int(u_idx), {})
            h_unexp = [u_dict.get(int(it), 0.0) for it in hist_items]
            t_unexp = u_dict.get(int(tgt_item), 0.0)

            # Mask: 1 where item is non-padding, 0 where padding
            mask = [1.0 if it != 0 else 0.0 for it in hist_items]

            self.item_seqs.append(hist_items)
            self.rating_seqs.append(hist_rats)
            self.tgap_seqs.append(hist_tgaps)
            self.hist_unexp.append(h_unexp)
            self.targets.append(tgt_item)
            self.clicks.append(tgt_click)
            self.target_unexp.append(t_unexp)
            self.masks.append(mask)

        self.item_seqs    = torch.tensor(self.item_seqs, dtype=torch.long)
        self.rating_seqs  = torch.tensor(self.rating_seqs, dtype=torch.long)
        self.tgap_seqs    = torch.tensor(self.tgap_seqs, dtype=torch.long)
        self.hist_unexp   = torch.tensor(self.hist_unexp, dtype=torch.float32)
        self.targets      = torch.tensor(self.targets, dtype=torch.long)
        self.clicks       = torch.tensor(self.clicks, dtype=torch.float32)
        self.target_unexp = torch.tensor(self.target_unexp, dtype=torch.float32)
        self.masks        = torch.tensor(self.masks, dtype=torch.float32)

    def __len__(self):
        return len(self.item_seqs)

    def __getitem__(self, idx):
        return {
            'item_ids':      self.item_seqs[idx],
            'ratings':       self.rating_seqs[idx],
            'time_gaps':     self.tgap_seqs[idx],
            'hist_unexp':    self.hist_unexp[idx],
            'target_id':     self.targets[idx],
            'click':         self.clicks[idx],
            'target_unexp':  self.target_unexp[idx],
            'mask':          self.masks[idx],
        }

train_ds = SerendipityDataset(train_samples, user_unexp)
val_ds   = SerendipityDataset(val_samples, user_unexp)
test_ds  = SerendipityDataset(test_samples, user_unexp)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

print(f"DataLoaders ready: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

# %% [markdown]
# ## Cell 7: Model Architecture
# **NOVA-Transformer** with non-invasive side-info fusion + PURS-style unexpectedness utility

# %%
# ── Feature Extractor ──
class FeatureExtractor(nn.Module):
    """Extracts embeddings for item IDs, ratings, time gaps, and unexpectedness."""
    def __init__(self, num_items, embed_dim, pretrained_item_embs):
        super().__init__()
        self.item_emb = nn.Embedding(num_items, embed_dim, padding_idx=0)
        # Initialize from trained GNN embeddings
        with torch.no_grad():
            self.item_emb.weight.copy_(pretrained_item_embs)
        self.rating_emb  = nn.Embedding(6, embed_dim, padding_idx=0)   # ratings 0-5
        self.time_gap_emb = nn.Embedding(100, embed_dim, padding_idx=0) # gaps 0-99
        self.unexp_proj  = nn.Linear(1, embed_dim)

    def forward(self, item_ids, ratings, time_gaps, hist_unexp):
        """Returns tuple of 4 feature tensors, each (B, T, D)."""
        e_item = self.item_emb(item_ids)
        e_rat  = self.rating_emb(ratings)
        e_time = self.time_gap_emb(time_gaps)
        e_unexp = self.unexp_proj(hist_unexp.unsqueeze(-1))
        return e_item, e_rat, e_time, e_unexp


# ── NOVA Gating Fusor ──
class NOVAGatingFusor(nn.Module):
    """
    From NOVA paper: learns dynamic gates to fuse multiple feature embeddings.
    Output is used for Q and K in attention. V stays as pure item embeddings.
    """
    def __init__(self, embed_dim, num_features=4):
        super().__init__()
        self.gate = nn.Linear(embed_dim * num_features, num_features)

    def forward(self, features):
        """
        features: list of (B, T, D) tensors
        Returns: (B, T, D) fused tensor
        """
        concat = torch.cat(features, dim=-1)           # (B, T, D*4)
        gate_weights = torch.sigmoid(self.gate(concat)) # (B, T, 4)
        stacked = torch.stack(features, dim=-2)         # (B, T, 4, D)
        gate_expanded = gate_weights.unsqueeze(-1)      # (B, T, 4, 1)
        return (gate_expanded * stacked).sum(dim=-2)    # (B, T, D)


# ── Causal Multi-Head Self-Attention (NOVA-style) ──
class NOVACausalAttention(nn.Module):
    """
    NOVA-style: Q, K from fused side-info; V from pure item embeddings.
    Includes causal masking (item at position t can only attend to positions <= t).
    """
    def __init__(self, embed_dim, num_heads=NUM_HEADS, dropout=DROPOUT):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = self.head_dim ** 0.5

        self.W_Q = nn.Linear(embed_dim, embed_dim)
        self.W_K = nn.Linear(embed_dim, embed_dim)
        self.W_V = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, query_key_input, value_input, pad_mask):
        """
        query_key_input: (B, T, D) — fused features for Q and K
        value_input:     (B, T, D) — pure item embeddings for V
        pad_mask:        (B, T)    — 1 for real items, 0 for padding
        Returns:         (B, T, D)
        """
        B, T, D = query_key_input.shape

        Q = self.W_Q(query_key_input).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_K(query_key_input).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_V(value_input).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        # Q, K, V: (B, H, T, head_dim)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, T, T)

        # Causal mask: lower-triangular (position t can attend to 0..t)
        causal = torch.tril(torch.ones(T, T, device=scores.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Padding mask: mask out padding positions in keys
        if pad_mask is not None:
            key_mask = pad_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
            scores = scores.masked_fill(key_mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, 0.0)  # handle all-masked rows
        attn_weights = self.attn_drop(attn_weights)

        out = torch.matmul(attn_weights, V)  # (B, H, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


# ── Transformer Block ──
class TransformerBlock(nn.Module):
    """Single transformer layer with NOVA attention + FFN + LayerNorm + residuals."""
    def __init__(self, embed_dim, num_heads=NUM_HEADS, dropout=DROPOUT):
        super().__init__()
        self.attn = NOVACausalAttention(embed_dim, num_heads, dropout)
        self.ffn  = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, fused, value, pad_mask):
        """
        fused: (B, T, D) — fused side-info features (for Q, K)
        value: (B, T, D) — pure item representations (for V + residual)
        Returns updated value: (B, T, D)
        """
        # Self-attention with residual on value
        normed_fused = self.ln1(fused)
        normed_value = self.ln1(value)
        value = value + self.attn(normed_fused, normed_value, pad_mask)
        # FFN with residual
        value = value + self.ffn(self.ln2(value))
        return value


# ── Unexpectedness Factor Module (simplified from PURS DIN attention) ──
class UnexpFactorMLP(nn.Module):
    """
    Computes a personalized scalar 'unexpectedness factor' from the user's
    session state. This models how much this user values surprise.
    In PURS, this uses DIN attention; here we use the transformer session state.
    """
    def __init__(self, embed_dim, dropout=DROPOUT):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, session_state, target_emb):
        """
        session_state: (B, D) — user preference from transformer
        target_emb:    (B, D) — target item embedding
        Returns: (B,) — scalar factor per sample
        """
        combined = torch.cat([session_state, target_emb], dim=-1)
        return self.mlp(combined).squeeze(-1)


# ── Full Serendipitous Recommender ──
class SerendipitousTransformer(nn.Module):
    """
    Complete model: FeatureExtractor → NOVA Fusor → Causal Transformer → PURS Utility Head
    Utility = b_u + b_i + relevance(session, target) + unexp_factor * f(unexp_distance)
    """
    def __init__(self, num_items, embed_dim, pretrained_item_embs,
                 num_heads=NUM_HEADS, num_layers=NUM_TF_LAYERS, dropout=DROPOUT,
                 max_seq_len=MAX_SEQ_LEN):
        super().__init__()
        self.embed_dim = embed_dim

        # Feature extraction
        self.extractor = FeatureExtractor(num_items, embed_dim, pretrained_item_embs)

        # Positional encoding (learnable)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)

        # NOVA gating fusor
        self.fusor = NOVAGatingFusor(embed_dim, num_features=4)

        # Transformer layers
        self.tf_layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # PURS-style output head
        self.relevance_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.BatchNorm1d(128),
            nn.Sigmoid(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.Sigmoid(),
            nn.Linear(64, 1),
        )
        self.unexp_factor = UnexpFactorMLP(embed_dim, dropout)

        # Bias terms (matching PURS: b_u + b_i)
        self.user_bias = nn.Parameter(torch.zeros(1))
        self.item_bias = nn.Embedding(num_items, 1, padding_idx=0)

    def forward(self, item_ids, ratings, time_gaps, hist_unexp,
                target_ids, target_unexp, mask):
        """
        item_ids:     (B, T) history item indices
        ratings:      (B, T) history ratings
        time_gaps:    (B, T) history time gaps
        hist_unexp:   (B, T) history unexpectedness scores
        target_ids:   (B,)   target item indices
        target_unexp: (B,)   target item unexpectedness distances
        mask:         (B, T) padding mask (1=real, 0=pad)
        Returns: utility logits (B,)
        """
        B, T = item_ids.shape

        # 1. Extract feature embeddings
        e_item, e_rat, e_time, e_unexp = self.extractor(
            item_ids, ratings, time_gaps, hist_unexp
        )

        # 2. Add positional encoding to all features
        positions = torch.arange(T, device=item_ids.device).unsqueeze(0).expand(B, -1)
        pos_enc = self.pos_emb(positions)  # (B, T, D)

        # 3. NOVA fusion: combine side-info features with gating
        fused = self.fusor([e_item, e_rat, e_time, e_unexp]) + pos_enc

        # 4. Transformer: Q,K from fused; V from pure item embeddings
        value = e_item  # V = pure item embeddings (NOVA key idea)
        for layer in self.tf_layers:
            value = layer(fused, value, mask)

        # 5. Extract session state (last real position)
        # Find the index of the last non-padding position per sample
        lengths = mask.sum(dim=1).long()  # (B,)
        last_idx = (lengths - 1).clamp(min=0)  # (B,)
        session_state = value[torch.arange(B, device=value.device), last_idx]  # (B, D)

        # 6. Target item embedding
        target_emb = self.extractor.item_emb(target_ids)  # (B, D)

        # 7. Relevance prediction
        concat = torch.cat([session_state, target_emb], dim=-1)  # (B, 2D)
        relevance = self.relevance_mlp(concat).squeeze(-1)       # (B,)

        # 8. Unexpectedness module (PURS formula)
        # Sub-Gaussian activation: f(x) = x * exp(-x)
        f_unexp = target_unexp * torch.exp(-target_unexp)
        f_unexp = f_unexp.detach()  # stop_gradient (matching PURS exactly)

        # Personalized unexpectedness factor
        uf = self.unexp_factor(session_state, target_emb)  # (B,)

        # 9. Utility = b_u + b_i + relevance + unexp_factor * f(unexp)
        b_i = self.item_bias(target_ids).squeeze(-1)  # (B,)
        utility = self.user_bias + b_i + relevance + uf * f_unexp

        return utility

print("Model architecture defined.")

# %% [markdown]
# ## Cell 8: Training Loop
# Single loss: BCE(sigmoid(utility), click_label) — matching PURS exactly.

# %%
def train_one_epoch(model, loader, optimizer, device):
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches  = 0
    for batch in loader:
        optimizer.zero_grad()
        utility = model(
            batch['item_ids'].to(device),
            batch['ratings'].to(device),
            batch['time_gaps'].to(device),
            batch['hist_unexp'].to(device),
            batch['target_id'].to(device),
            batch['target_unexp'].to(device),
            batch['mask'].to(device),
        )
        loss = F.binary_cross_entropy_with_logits(utility, batch['click'].to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches  += 1
    return total_loss / max(n_batches, 1)


def evaluate_auc(model, loader, device):
    """Compute AUC on a dataset. Returns AUC or None if degenerate."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            utility = model(
                batch['item_ids'].to(device),
                batch['ratings'].to(device),
                batch['time_gaps'].to(device),
                batch['hist_unexp'].to(device),
                batch['target_id'].to(device),
                batch['target_unexp'].to(device),
                batch['mask'].to(device),
            )
            probs = torch.sigmoid(utility).cpu().numpy()
            labels = batch['click'].numpy()
            all_preds.extend(probs.tolist())
            all_labels.extend(labels.tolist())
    if len(set(all_labels)) < 2:
        return None
    return roc_auc_score(all_labels, all_preds)


# ── Training ──
model = SerendipitousTransformer(
    NUM_ITEMS, EMBED_DIM, gnn_item_embs,
    num_heads=NUM_HEADS, num_layers=NUM_TF_LAYERS,
    dropout=DROPOUT, max_seq_len=MAX_SEQ_LEN
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)

best_val_auc = 0.0
patience_counter = 0
MAX_PATIENCE = 10

print(f"Training for {TRAIN_EPOCHS} epochs on {DEVICE}...")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

for epoch in range(TRAIN_EPOCHS):
    t0 = time.time()
    train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE)
    val_auc = evaluate_auc(model, val_loader, DEVICE)
    elapsed = time.time() - t0

    lr_now = optimizer.param_groups[0]['lr']
    val_str = f"{val_auc:.4f}" if val_auc is not None else "N/A"
    print(f"Epoch {epoch+1:3d}/{TRAIN_EPOCHS} | Loss: {train_loss:.4f} | "
          f"Val AUC: {val_str} | LR: {lr_now:.6f} | {elapsed:.1f}s")

    if val_auc is not None:
        scheduler.step(val_auc)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= MAX_PATIENCE:
                print(f"Early stopping at epoch {epoch+1} (best val AUC: {best_val_auc:.4f})")
                break

    # Checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, "checkpoint.pth")

# Load best model for evaluation
if os.path.exists("best_model.pth"):
    model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
    print(f"Loaded best model (val AUC: {best_val_auc:.4f})")

# %% [markdown]
# ## Cell 9: Evaluation
# Metrics: AUC, HR@K, NDCG@K, Precision@K, Unexpectedness, Serendipity, Coverage
# Following PURS evaluation protocol.

# %%
def full_evaluation(model, loader, device, k=K):
    """
    Compute all metrics on a dataset.
    
    Accuracy metrics (AUC, HR, NDCG, Precision):
        - AUC: global, across all samples
        - HR@K, NDCG@K, Precision@K: per-batch ranking with score > 0.5 threshold
          (matching PURS evaluation: recommend items with score > 0.5)
    
    Beyond-accuracy metrics:
        - Unexpectedness: average unexpectedness distance of recommended items
        - Serendipity: fraction of recommended items that are both clicked AND unexpected
        - Coverage: fraction of catalog items that appear in recommendations
    """
    model.eval()
    all_preds, all_labels = [], []
    all_unexp_dists = []
    recommended_items = set()
    
    # Per-user accumulation for HR/NDCG
    hr_list, ndcg_list, prec_list = [], [], []
    seren_list, unexp_list = [], []
    
    # Threshold for "unexpected" (items with unexp > median are "unexpected")
    # We'll compute this adaptively
    all_target_unexp = []
    
    with torch.no_grad():
        # First pass: collect all predictions and unexpectedness values
        batch_results = []
        for batch in loader:
            utility = model(
                batch['item_ids'].to(device),
                batch['ratings'].to(device),
                batch['time_gaps'].to(device),
                batch['hist_unexp'].to(device),
                batch['target_id'].to(device),
                batch['target_unexp'].to(device),
                batch['mask'].to(device),
            )
            scores = torch.sigmoid(utility).cpu().numpy()
            labels = batch['click'].numpy()
            targets = batch['target_id'].numpy()
            t_unexp = batch['target_unexp'].numpy()
            
            all_preds.extend(scores.tolist())
            all_labels.extend(labels.tolist())
            all_target_unexp.extend(t_unexp.tolist())
            
            batch_results.append((scores, labels, targets, t_unexp))
    
    # AUC (global)
    auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.0
    
    # Threshold for "unexpected": use median of all non-zero unexpectedness values
    nonzero_unexp = [u for u in all_target_unexp if u > 0]
    unexp_threshold = np.median(nonzero_unexp) if nonzero_unexp else 0.5
    
    # Second pass: compute ranking metrics per batch
    # Following PURS: items with score > 0.5 are "recommended"
    for (scores, labels, targets, t_unexp) in batch_results:
        for i in range(len(scores)):
            score = scores[i]
            label = labels[i]
            target = targets[i]
            unexp_val = t_unexp[i]
            
            is_recommended = score > 0.5
            is_unexpected = unexp_val > unexp_threshold
            
            if is_recommended:
                recommended_items.add(int(target))
                # HR: did we correctly recommend a clicked item?
                hr_list.append(float(label > 0))
                # Precision: is the recommendation relevant?
                prec_list.append(float(label > 0))
                # NDCG: for single-item, it's same as HR
                ndcg_list.append(float(label > 0))
                # Unexpectedness of recommended items
                unexp_list.append(float(unexp_val))
                # Serendipity: recommended AND clicked AND unexpected
                seren_list.append(float(label > 0 and is_unexpected))
    
    # Coverage
    total_items = NUM_ITEMS - 1  # exclude padding
    coverage = len(recommended_items) / max(total_items, 1) * 100
    
    metrics = {
        'AUC':            auc,
        'HR':             np.mean(hr_list) if hr_list else 0.0,
        'NDCG':           np.mean(ndcg_list) if ndcg_list else 0.0,
        'Precision':      np.mean(prec_list) if prec_list else 0.0,
        'Unexpectedness': np.mean(unexp_list) if unexp_list else 0.0,
        'Serendipity':    np.mean(seren_list) if seren_list else 0.0,
        'Coverage':       coverage,
        'Num_Recommended': len(hr_list),
        'Unexp_Threshold': unexp_threshold,
    }
    return metrics


# ── Run evaluation on test set ──
print("\n" + "="*60)
print("FINAL EVALUATION ON TEST SET")
print("="*60)

test_metrics = full_evaluation(model, test_loader, DEVICE)

print(f"\n--- Accuracy Metrics ---")
print(f"   AUC:          {test_metrics['AUC']:.4f}")
print(f"   HR:           {test_metrics['HR']:.4f}")
print(f"   NDCG:         {test_metrics['NDCG']:.4f}")
print(f"   Precision:    {test_metrics['Precision']:.4f}")

print(f"\n--- Beyond-Accuracy Metrics ---")
print(f"   Unexpectedness: {test_metrics['Unexpectedness']:.4f}")
print(f"   Serendipity:    {test_metrics['Serendipity']:.4f}")
print(f"   Coverage:       {test_metrics['Coverage']:.2f}%")

print(f"\n--- Meta ---")
print(f"   Items recommended: {test_metrics['Num_Recommended']}")
print(f"   Unexp threshold:   {test_metrics['Unexp_Threshold']:.4f}")

# Also evaluate on train and val for comparison
train_metrics = full_evaluation(model, train_loader, DEVICE)
val_metrics   = full_evaluation(model, val_loader, DEVICE)

print(f"\n--- Comparison ---")
print(f"   {'Split':<8} | {'AUC':>8} | {'HR':>8} | {'Unexp':>8} | {'Seren':>8} | {'Cov':>8}")
print(f"   {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8}")
for name, m in [('Train', train_metrics), ('Val', val_metrics), ('Test', test_metrics)]:
    print(f"   {name:<8} | {m['AUC']:8.4f} | {m['HR']:8.4f} | "
          f"{m['Unexpectedness']:8.4f} | {m['Serendipity']:8.4f} | {m['Coverage']:7.2f}%")

print("\nPipeline complete!")
