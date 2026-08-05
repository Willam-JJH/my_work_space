"""
Grand Experiment: Path Signatures + Multimodal Pretraining + Upper Transformer
==============================================================================
Phase 1: Contrastive pretraining (price ↔ sig ↔ tech)
Phase 2: Base Transformer → return predictions + attention patterns
Phase 3: Upper Transformer fuses pretrained embeddings + patterns → error

Runs on max available data.
"""
import numpy as np; import pandas as pd; import torch; import torch.nn as nn
import torch.nn.functional as F; from torch.utils.data import DataLoader
import signatory; from scipy.stats import spearmanr; import math, pickle, os, time, warnings
warnings.filterwarnings('ignore')

GPU = torch.device("xpu" if torch.xpu.device_count() > 0 else "cpu"); CPU = torch.device("cpu")
print(f"GPU: {torch.xpu.get_device_name(0) if GPU.type=='xpu' else 'CPU'} | VRAM: {torch.xpu.get_device_properties(0).total_memory/1e9:.1f}GB" if GPU.type=='xpu' else '')
torch.manual_seed(42); np.random.seed(42)

# ============================================================
# CONFIG
# ============================================================
N_STOCKS = 500  # top 250 US + 250 CN by completeness
LOOKBACK = 30; SIG_DEPTH = 1
D_EMBED = 128; D_MODEL = 64; N_LAYERS = 4; N_HEADS = 4; D_FF = 128; DROPOUT = 0.1
BATCH = 24; PRETRAIN_EP = 60; BASE_EP = 60; UPPER_EP = 100
TEMP = 0.07

# ============================================================
# DATA: top 100 US + 100 CN, aligned dates
# ============================================================
print("[1/6] Loading data...")
us = pd.read_parquet("D:/code/data/us_all_returns.parquet")
cn = pd.read_parquet("D:/code/data/cn_all_returns.parquet")

us_comp = us.notna().sum()/len(us); us_top = us_comp.nlargest(min(250,len(us_comp))).index.tolist()
cn_comp = cn.notna().sum()/len(cn); cn_top = cn_comp.nlargest(min(250,len(cn_comp))).index.tolist()

ret = pd.concat([us[us_top], cn[cn_top]], axis=1).dropna().ffill().fillna(0)
ret_vals = ret.values.astype(np.float32); n_assets = len(ret.columns)
n_samp = ret_vals.shape[0] - LOOKBACK
print(f"  {n_assets} stocks (US + CN) x {ret_vals.shape[0]} days")

X = np.zeros((n_samp, n_assets, LOOKBACK), dtype=np.float32)
y = np.zeros((n_samp, n_assets), dtype=np.float32)
for i in range(n_samp): X[i] = ret_vals[i:i+LOOKBACK].T; y[i] = ret_vals[i+LOOKBACK]
mu = X.mean(axis=-1, keepdims=True); st = X.std(axis=-1, keepdims=True) + 1e-8
X = (X - mu) / st
y = np.clip(y, -np.percentile(np.abs(y), 99), np.percentile(np.abs(y), 99))
split = int(n_samp * 0.7)

# Signatures
print("  Computing path signatures (depth=1)...")
sig_dim_full = signatory.signature_channels(n_assets, SIG_DEPTH)
X_sig_t = torch.FloatTensor(X).transpose(1,2).to(CPU)
sig_full = np.zeros((n_samp, sig_dim_full), dtype=np.float32)
for i in range(0, n_samp, 64):
    sig_full[i:i+64] = signatory.signature(X_sig_t[i:i+64], SIG_DEPTH, basepoint=True).cpu().numpy()
from sklearn.preprocessing import StandardScaler; from sklearn.decomposition import PCA
sig_full = StandardScaler().fit_transform(sig_full)
sig_dim = 128
sig = PCA(sig_dim).fit_transform(sig_full)
print(f"  Sig: {sig_dim_full:,} → PCA → {sig_dim}")

# Technical features
v5 = X[:,:,-5:].std(axis=-1, keepdims=True); v10 = X[:,:,-10:].std(axis=-1, keepdims=True)
m5 = X[:,:,-5:].mean(axis=-1, keepdims=True); m10 = X[:,:,-10:].mean(axis=-1, keepdims=True)
rsi = (X[:,:,-5:] > 0).mean(axis=-1, keepdims=True)
tech = np.concatenate([v5, v10, m5, m10, rsi], axis=-1)

X_tr, X_te = X[:split], X[split:]; y_tr, y_te = y[:split], y[split:]
sig_tr, sig_te = sig[:split], sig[split:]; tech_tr, tech_te = tech[:split], tech[split:]
print(f"  Train: {split} | Test: {n_samp - split}")

# ============================================================
# BUILDING BLOCKS
# ============================================================
class MHA(nn.Module):
    def __init__(self, dm, nh, dp=0.1):
        super().__init__(); self.nh = nh; self.dk = dm // nh
        self.q = nn.Linear(dm, dm); self.k = nn.Linear(dm, dm)
        self.v = nn.Linear(dm, dm); self.out = nn.Linear(dm, dm)
        self.drop = nn.Dropout(dp); self._w = None
    def forward(self, x):
        B, N, E = x.shape
        q = self.q(x).view(B, N, self.nh, self.dk).transpose(1, 2)
        k = self.k(x).view(B, N, self.nh, self.dk).transpose(1, 2)
        v = self.v(x).view(B, N, self.nh, self.dk).transpose(1, 2)
        w = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk); w = F.softmax(w, -1)
        self._w = w.detach(); w = self.drop(w)
        return self.out((w @ v).transpose(1, 2).contiguous().view(B, N, E))

class EncLayer(nn.Module):
    def __init__(self, d, nh, df, dp):
        super().__init__()
        self.n1 = nn.LayerNorm(d); self.attn = MHA(d, nh, dp)
        self.n2 = nn.LayerNorm(d); self.ff = nn.Sequential(nn.Linear(d, df), nn.GELU(), nn.Dropout(dp), nn.Linear(df, d), nn.Dropout(dp))
    def forward(self, x): return x + self.ff(self.n2(x + self.attn(self.n1(x))))

# ============================================================
# PHASE 1: MULTIMODAL PRETRAINING
# ============================================================
print("[2/6] Multimodal contrastive pretraining...")

class PriceEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(LOOKBACK, D_EMBED * 2), nn.GELU(), nn.Linear(D_EMBED * 2, D_EMBED))
        self.pos = nn.Parameter(torch.randn(1, n_assets, D_EMBED) * 0.02)
        self.attn = nn.MultiheadAttention(D_EMBED, 4, batch_first=True)
        self.out = nn.Sequential(nn.Linear(D_EMBED, D_EMBED), nn.LayerNorm(D_EMBED))
    def forward(self, x):
        B, N, L = x.shape; h = self.proj(x) + self.pos[:, :N, :]
        h, _ = self.attn(h, h, h); return self.out(h.mean(1))

class SigEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(sig_dim, D_EMBED * 4), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(D_EMBED * 4, D_EMBED * 2), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(D_EMBED * 2, D_EMBED), nn.LayerNorm(D_EMBED))
    def forward(self, x): return self.net(x)

class TechEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(5, D_EMBED * 2), nn.GELU(), nn.Linear(D_EMBED * 2, D_EMBED))
        self.pos = nn.Parameter(torch.randn(1, n_assets, D_EMBED) * 0.02)
        self.attn = nn.MultiheadAttention(D_EMBED, 4, batch_first=True)
        self.out = nn.Sequential(nn.Linear(D_EMBED, D_EMBED), nn.LayerNorm(D_EMBED))
    def forward(self, x):
        B, N, _ = x.shape; h = self.proj(x) + self.pos[:, :N, :]
        h, _ = self.attn(h, h, h); return self.out(h.mean(1))

p_enc = PriceEnc().to(GPU); s_enc = SigEnc().to(GPU); t_enc = TechEnc().to(GPU)
all_pt_params = list(p_enc.parameters()) + list(s_enc.parameters()) + list(t_enc.parameters())
opt_pt = torch.optim.AdamW(all_pt_params, lr=5e-4, weight_decay=1e-4)
pt_ds = torch.utils.data.TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(sig_tr), torch.FloatTensor(tech_tr))
pt_ld = DataLoader(pt_ds, BATCH, shuffle=True, drop_last=True)

for ep in range(PRETRAIN_EP):
    tl = 0; nb = 0
    for xb, sb, tb in pt_ld:
        xb, sb, tb = xb.to(GPU), sb.to(GPU), tb.to(GPU)
        zp = F.normalize(p_enc(xb), -1); zs = F.normalize(s_enc(sb), -1); zt = F.normalize(t_enc(tb), -1)
        B = xb.shape[0]; loss = 0
        for za, zb in [(zp, zs), (zp, zt), (zs, zt)]:
            sim = (za @ zb.T) / TEMP; labels = torch.arange(B, device=GPU)
            loss += (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
        loss /= 3
        opt_pt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(all_pt_params, 1.0); opt_pt.step()
        tl += loss.item(); nb += 1
    if (ep + 1) % 15 == 0:
        with torch.no_grad():
            zpa = F.normalize(p_enc(torch.FloatTensor(X_tr[:128]).to(GPU)), -1)
            zsa = F.normalize(s_enc(torch.FloatTensor(sig_tr[:128]).to(GPU)), -1)
            align = (zpa * zsa).sum(-1).mean().item()
        print(f"  Pretrain {ep+1:3d} | Loss: {tl/nb:.4f} | Align: {align:.4f}")

# Extract pretrained embeddings
p_enc.eval(); s_enc.eval(); t_enc.eval()
with torch.no_grad():
    zp_tr = p_enc(torch.FloatTensor(X_tr).to(GPU)).cpu().numpy()
    zs_tr = s_enc(torch.FloatTensor(sig_tr).to(GPU)).cpu().numpy()
    zt_tr = t_enc(torch.FloatTensor(tech_tr).to(GPU)).cpu().numpy()
    zp_te = p_enc(torch.FloatTensor(X_te).to(GPU)).cpu().numpy()
    zs_te = s_enc(torch.FloatTensor(sig_te).to(GPU)).cpu().numpy()
    zt_te = t_enc(torch.FloatTensor(tech_te).to(GPU)).cpu().numpy()
mm_tr = np.concatenate([zp_tr, zs_tr, zt_tr], axis=-1)
mm_te = np.concatenate([zp_te, zs_te, zt_te], axis=-1)
mm_dim = mm_tr.shape[1]
print(f"  Multimodal embeddings: {mm_dim}-dim saved")

# ============================================================
# PHASE 2: BASE TRANSFORMER
# ============================================================
print("[3/6] Base Transformer...")

class BaseTrans(nn.Module):
    def __init__(self):
        super().__init__(); self.nl = N_LAYERS
        self.proj = nn.Linear(LOOKBACK, D_MODEL)
        self.pos = nn.Parameter(torch.randn(1, 500, D_MODEL) * 0.02)
        self.drop = nn.Dropout(DROPOUT)
        self.layers = nn.ModuleList([EncLayer(D_MODEL, N_HEADS, D_FF, DROPOUT) for _ in range(N_LAYERS)])
        self.head = nn.Sequential(nn.Linear(D_MODEL, D_MODEL // 2), nn.GELU(), nn.Linear(D_MODEL // 2, 1))
    def forward(self, x, get_pat=False):
        B, N, L = x.shape; h = self.drop(self.proj(x) + self.pos[:, :N, :]); pats = {}
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if get_pat: pats[i] = layer.attn._w
        pred = self.head(h).squeeze(-1)
        return (pred, pats) if get_pat else pred

class FinDS(torch.utils.data.Dataset):
    def __init__(self, X, y, mm):
        self.X = torch.FloatTensor(X); self.y = torch.FloatTensor(y); self.mm = torch.FloatTensor(mm)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i], self.mm[i]

tr_ds = FinDS(X_tr, y_tr, mm_tr); te_ds = FinDS(X_te, y_te, mm_te)
tr_ld = DataLoader(tr_ds, BATCH, shuffle=True, drop_last=True)
te_ld = DataLoader(te_ds, BATCH, shuffle=False)

base = BaseTrans().to(GPU)
opt_b = torch.optim.AdamW(base.parameters(), lr=1e-3, weight_decay=1e-4)
sch_b = torch.optim.lr_scheduler.CosineAnnealingLR(opt_b, BASE_EP)
for ep in range(BASE_EP):
    base.train(); tl = 0
    for x, y, _ in tr_ld:
        pred = base(x.to(GPU)); loss = F.huber_loss(pred, y.to(GPU), delta=1.0)
        opt_b.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(base.parameters(), 2.0); opt_b.step(); tl += loss.item()
    sch_b.step()
    if (ep + 1) % 15 == 0: print(f"  Base {ep+1:3d} | Loss: {tl/len(tr_ld):.6f}")

base.eval()
with torch.no_grad():
    preds, ys = [], []
    for x, y, _ in te_ld: preds.append(base(x.to(GPU)).cpu().numpy()); ys.append(y.numpy())
    preds = np.concatenate(preds); ys = np.concatenate(ys)
base_mse = float(np.mean((preds - ys)**2)); base_err = np.abs(preds - ys).mean(1); base_abs = np.abs(preds).mean(1)
print(f"  Base MSE: {base_mse:.6f} | |Pred| r: {spearmanr(base_abs, base_err)[0]:.4f}")

# ============================================================
# PHASE 3: UPPER TRANSFORMER
# ============================================================
print("[4/6] Upper Transformer...")
base.eval()
all_pats, all_errs, all_mm = [], [], []
with torch.no_grad():
    for x, y, mm in tr_ld:
        pred, pats = base(x.to(GPU), get_pat=True)
        stacked = torch.cat([pats[i].cpu() for i in range(N_LAYERS)], dim=1)
        all_pats.append(stacked.numpy()); all_mm.append(mm.numpy())
        all_errs.append((pred - y.to(GPU)).abs().mean(1).cpu().numpy())
p_train = np.concatenate(all_pats); e_train = np.concatenate(all_errs); m_train = np.concatenate(all_mm)

total_heads = N_LAYERS * N_HEADS; attn_dim = total_heads * n_assets

class UpperTrans(nn.Module):
    def __init__(self, n_ht, attn_d, mm_d, d=128, nl=4, nh=4, dff=256, do=0.15):
        super().__init__()
        self.attn_proj = nn.Sequential(nn.Linear(attn_d, d), nn.LayerNorm(d))
        self.pos = nn.Parameter(torch.randn(1, n_ht, d) * 0.02)
        self.mm_proj = nn.Sequential(nn.Linear(mm_d, d*2), nn.GELU(), nn.Dropout(0.1), nn.Linear(d*2, d), nn.LayerNorm(d))
        self.mm_tok = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.layers = nn.ModuleList([EncLayer(d, nh, dff, do) for _ in range(nl)])
        self.err_head = nn.Sequential(nn.Linear(d, d*2), nn.GELU(), nn.Dropout(do), nn.Linear(d*2, d), nn.GELU(), nn.Dropout(do), nn.Linear(d, 1))
    def forward(self, patterns, mm_emb):
        B, H, N, _ = patterns.shape
        pooled = patterns.mean(dim=-1).reshape(B, H, N); flat = pooled.reshape(B, H * N)
        a_tok = self.attn_proj(flat).unsqueeze(1).expand(B, H, -1) + self.pos[:, :H, :]
        m_tok = self.mm_proj(mm_emb).unsqueeze(1) + self.mm_tok
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, m_tok, a_tok], dim=1)
        for layer in self.layers: x = layer(x)
        return self.err_head(x[:, 0]).squeeze(-1)

upper = UpperTrans(total_heads, attn_dim, mm_dim).to(GPU)
opt_u = torch.optim.AdamW(upper.parameters(), lr=3e-3, weight_decay=1e-5)
sch_u = torch.optim.lr_scheduler.CosineAnnealingLR(opt_u, UPPER_EP)

for ep in range(UPPER_EP):
    upper.train(); tl = 0; nb = 0
    for i in range(0, len(p_train), BATCH):
        pb = torch.FloatTensor(p_train[i:i+BATCH]).to(GPU)
        mb = torch.FloatTensor(m_train[i:i+BATCH]).to(GPU)
        eb = torch.FloatTensor(e_train[i:i+BATCH]).to(GPU)
        pe = upper(pb, mb); loss = F.mse_loss(pe, eb)
        opt_u.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(upper.parameters(), 2.0)
        opt_u.step(); tl += loss.item(); nb += 1
    sch_u.step()
    if (ep + 1) % 20 == 0:
        with torch.no_grad():
            s = upper(torch.FloatTensor(p_train[:256]).to(GPU), torch.FloatTensor(m_train[:256]).to(GPU)).std().item()
        print(f"  Upper {ep+1:3d} | Loss: {tl/nb:.6f} | std: {s:.6f}")

# ============================================================
# EVALUATION
# ============================================================
print("[5/6] Evaluation...")
base.eval(); upper.eval()
all_up, all_true, all_abs, all_mm_te = [], [], [], []
with torch.no_grad():
    for x, y, mm in te_ld:
        pred, pats = base(x.to(GPU), get_pat=True)
        stacked = torch.cat([pats[i].cpu() for i in range(N_LAYERS)], dim=1)
        up_err = upper(stacked.to(GPU), mm.to(GPU)).cpu().numpy()
        true_err = (pred - y.to(GPU)).abs().mean(1).cpu().numpy()
        all_up.append(up_err); all_true.append(true_err)
        all_abs.append(np.abs(pred.cpu().numpy()).mean(1))
up_err = np.concatenate(all_up); true_err = np.concatenate(all_true); abs_bl = np.concatenate(all_abs)

# Baselines
class MMOnly(nn.Module):
    def __init__(self, in_d, d=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_d, d*4), nn.GELU(), nn.Dropout(0.2), nn.Linear(d*4, d*2), nn.GELU(), nn.Dropout(0.2), nn.Linear(d*2, d), nn.GELU(), nn.Linear(d, 1))
    def forward(self, x): return self.net(x).squeeze(-1)

# Multimodal-only
mm_model = MMOnly(mm_dim).to(GPU); opt_mm = torch.optim.AdamW(mm_model.parameters(), lr=1e-3)
for ep in range(50):
    mm_model.train()
    for i in range(0, len(m_train), BATCH):
        mb = torch.FloatTensor(m_train[i:i+BATCH]).to(GPU); eb = torch.FloatTensor(e_train[i:i+BATCH]).to(GPU)
        loss = F.mse_loss(mm_model(mb), eb); opt_mm.zero_grad(); loss.backward(); opt_mm.step()
mm_model.eval()
with torch.no_grad():
    mm_only = np.concatenate([mm_model(torch.FloatTensor(mm_te[i:i+BATCH]).to(GPU)).cpu().numpy() for i in range(0, len(mm_te), BATCH)])

# Attention-only
attn_train = p_train.mean(axis=-1).reshape(len(p_train), -1)
attn_test_list = []
with torch.no_grad():
    for x, y, _ in te_ld:
        _, pats = base(x.to(GPU), get_pat=True)
        stacked = torch.cat([pats[i].cpu() for i in range(N_LAYERS)], dim=1)
        attn_test_list.append(stacked.mean(-1).reshape(x.shape[0], -1).numpy())
attn_test = np.concatenate(attn_test_list)

class AttnOnly(nn.Module):
    def __init__(self, in_d, d=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_d, d*2), nn.GELU(), nn.Dropout(0.15), nn.Linear(d*2, d), nn.GELU(), nn.Linear(d, 1))
    def forward(self, x): return self.net(x).squeeze(-1)

attn_model = AttnOnly(attn_train.shape[1]).to(GPU); opt_a = torch.optim.AdamW(attn_model.parameters(), lr=1e-3)
for ep in range(50):
    attn_model.train()
    for i in range(0, len(attn_train), BATCH):
        ab = torch.FloatTensor(attn_train[i:i+BATCH]).to(GPU); eb = torch.FloatTensor(e_train[i:i+BATCH]).to(GPU)
        loss = F.mse_loss(attn_model(ab), eb); opt_a.zero_grad(); loss.backward(); opt_a.step()
attn_model.eval()
with torch.no_grad(): attn_only = attn_model(torch.FloatTensor(attn_test).to(GPU)).cpu().numpy()

# Signature-only baseline
class SigOnlyMLP(nn.Module):
    def __init__(self, in_d, d=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_d, d*4), nn.GELU(), nn.Dropout(0.2), nn.Linear(d*4, d*2), nn.GELU(), nn.Dropout(0.2), nn.Linear(d*2, d), nn.GELU(), nn.Linear(d, 1))
    def forward(self, x): return self.net(x).squeeze(-1)

sig_model = SigOnlyMLP(sig_dim).to(GPU); opt_s = torch.optim.AdamW(sig_model.parameters(), lr=1e-3)
for ep in range(50):
    sig_model.train()
    for i in range(0, len(sig_tr), BATCH):
        sb = torch.FloatTensor(sig_tr[i:i+BATCH]).to(GPU); eb = torch.FloatTensor(e_train[i:i+BATCH]).to(GPU)
        loss = F.mse_loss(sig_model(sb), eb); opt_s.zero_grad(); loss.backward(); opt_s.step()
sig_model.eval()
with torch.no_grad(): sig_only = sig_model(torch.FloatTensor(sig_te).to(GPU)).cpu().numpy()

def sr(a, b): return float(spearmanr(a, b)[0])

print("[6/6] Final Results")
print("=" * 65)
print(f"  GRAND EXPERIMENT — {n_assets} Stocks (US+CN)")
print(f"  Path Signatures + Upper Transformer + Pretraining")
print("=" * 65)
print(f"  {'Method':<35} {'Spearman r':>10} {'vs |Pred|':>10}")
print(f"  {'-'*55}")
methods = [
    ("|Prediction| Baseline (heuristic)", sr(abs_bl, true_err)),
    ("Signature-only MLP", sr(sig_only, true_err)),
    ("Attention-only MLP", sr(attn_only, true_err)),
    ("Multimodal-only MLP (pretrained)", sr(mm_only, true_err)),
    ("Upper Trans + MM + Attn (FULL)", sr(up_err, true_err)),
]
best_baseline = sr(abs_bl, true_err)
for name, val in methods:
    delta = val - best_baseline
    mark = " ← BEST" if val == max(v for _, v in methods) else ""
    print(f"  {name:<35} {val:>10.4f} {delta:>+10.4f}{mark}")
print(f"  {'-'*55}")
print(f"  Base Transformer MSE: {base_mse:.6f}")
print("=" * 65)
