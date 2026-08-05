"""
Direct Feature Fusion on 99 US Stocks (No Pretraining)
=======================================================
Skip contrastive pretraining — directly fuse raw features through Upper Transformer.
Features: Returns (99×30 flatten), Signature (128 PCA), Technical (99×6 flatten)
"""
import numpy as np; import pandas as pd; import torch; import torch.nn as nn; import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader; import signatory; from scipy.stats import spearmanr
import math, warnings; warnings.filterwarnings('ignore')

GPU = torch.device("xpu" if torch.xpu.device_count() > 0 else "cpu"); CPU = torch.device("cpu")
print(f"GPU: {torch.xpu.get_device_name(0) if GPU.type=='xpu' else 'CPU'}")

LOOKBACK, SIG_DEPTH = 30, 2
D_MODEL, N_LAYERS, N_HEADS, D_FF, DROPOUT = 64, 4, 4, 128, 0.1
BATCH, BASE_EP, UPPER_EP = 64, 50, 80
torch.manual_seed(42); np.random.seed(42)

# ============================================================
# DATA
# ============================================================
print("[1/4] Loading data...")
returns = pd.read_parquet("D:/code/data/us_returns.parquet")
returns = returns.dropna(axis=1, thresh=int(len(returns)*0.4)).ffill().fillna(0)
ret_vals = returns.values.astype(np.float32)
n_assets = len(returns.columns); n_samp = ret_vals.shape[0] - LOOKBACK
print(f"  {n_assets} stocks x {ret_vals.shape[0]} days")

X_raw = np.zeros((n_samp, n_assets, LOOKBACK), dtype=np.float32)
y_raw = np.zeros((n_samp, n_assets), dtype=np.float32)
for i in range(n_samp):
    X_raw[i] = ret_vals[i:i+LOOKBACK].T; y_raw[i] = ret_vals[i+LOOKBACK]
mu = X_raw.mean(axis=-1, keepdims=True); st = X_raw.std(axis=-1, keepdims=True) + 1e-8
X_raw = (X_raw - mu) / st
y_raw = np.clip(y_raw, -np.percentile(np.abs(y_raw), 99), np.percentile(np.abs(y_raw), 99))
split = int(n_samp * 0.7)

# Signatures + PCA
print("  Computing signatures...")
sig_dim_full = signatory.signature_channels(n_assets, SIG_DEPTH)
X_sig = torch.FloatTensor(X_raw).transpose(1,2).to(CPU)
sig = np.zeros((n_samp, sig_dim_full), dtype=np.float32)
for i in range(0, n_samp, 64):
    sig[i:i+64] = signatory.signature(X_sig[i:i+64], SIG_DEPTH, basepoint=True).cpu().numpy()
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
sig = StandardScaler().fit_transform(sig); sig = PCA(n_components=128).fit_transform(sig)

# Technical
def tech_feat(r):
    v5=r[:,:,-5:].std(-1,keepdims=1); v10=r[:,:,-10:].std(-1,keepdims=1)
    v20=r[:,:,-20:].std(-1,keepdims=1); m5=r[:,:,-5:].mean(-1,keepdims=1)
    m10=r[:,:,-10:].mean(-1,keepdims=1); rsi=(r[:,:,-5:]>0).mean(-1,keepdims=1)
    return np.concatenate([v5,v10,v20,m5,m10,rsi],-1)
tech = tech_feat(X_raw)

X_tr, X_te = X_raw[:split], X_raw[split:]
y_tr, y_te = y_raw[:split], y_raw[split:]
sig_tr, sig_te = sig[:split], sig[split:]
tech_tr, tech_te = tech[:split], tech[split:]
print(f"  Train: {split} | Test: {n_samp-split}")

# Feature dimensions
feat_price_dim = n_assets * LOOKBACK   # 2970
feat_sig_dim = 128
feat_tech_dim = n_assets * 6           # 594
feat_total = feat_price_dim + feat_sig_dim + feat_tech_dim  # 3692
print(f"  Feature dim: {feat_price_dim} + {feat_sig_dim} + {feat_tech_dim} = {feat_total}")

# ============================================================
# BASE TRANSFORMER (cross-asset attention)
# ============================================================
class MHA(nn.Module):
    def __init__(self, dm, nh, dp=0.1):
        super().__init__(); self.nh=nh; self.dk=dm//nh
        self.q=nn.Linear(dm,dm); self.k=nn.Linear(dm,dm)
        self.v=nn.Linear(dm,dm); self.out=nn.Linear(dm,dm)
        self.drop=nn.Dropout(dp); self._w=None
    def forward(self, x):
        B,N,E=x.shape
        q=self.q(x).view(B,N,self.nh,self.dk).transpose(1,2)
        k=self.k(x).view(B,N,self.nh,self.dk).transpose(1,2)
        v=self.v(x).view(B,N,self.nh,self.dk).transpose(1,2)
        w=(q@k.transpose(-2,-1))/math.sqrt(self.dk); w=F.softmax(w,-1)
        self._w=w.detach(); w=self.drop(w)
        return self.out((w@v).transpose(1,2).contiguous().view(B,N,E))

class EncLayer(nn.Module):
    def __init__(self,d,nh,df,dp):
        super().__init__(); self.n1=nn.LayerNorm(d); self.attn=MHA(d,nh,dp)
        self.n2=nn.LayerNorm(d); self.ff=nn.Sequential(nn.Linear(d,df),nn.GELU(),nn.Dropout(dp),nn.Linear(df,d),nn.Dropout(dp))
    def forward(self,x): return x+self.ff(self.n2(x+self.attn(self.n1(x))))

class BaseTrans(nn.Module):
    def __init__(self):
        super().__init__(); self.nl=N_LAYERS
        self.proj=nn.Linear(LOOKBACK,D_MODEL); self.pos=nn.Parameter(torch.randn(1,500,D_MODEL)*0.02)
        self.drop=nn.Dropout(DROPOUT)
        self.layers=nn.ModuleList([EncLayer(D_MODEL,N_HEADS,D_FF,DROPOUT) for _ in range(N_LAYERS)])
        self.head=nn.Sequential(nn.Linear(D_MODEL,D_MODEL//2),nn.GELU(),nn.Linear(D_MODEL//2,1))
    def forward(self,x,get_pat=False):
        B,N,L=x.shape; h=self.drop(self.proj(x)+self.pos[:,:N,:]); pats={}
        for i,layer in enumerate(self.layers):
            h=layer(h)
            if get_pat: pats[i]=layer.attn._w
        pred=self.head(h).squeeze(-1)
        return (pred,pats) if get_pat else pred

# ============================================================
# DIRECT FUSION UPPER TRANSFORMER
# ============================================================
class FusionUpper(nn.Module):
    def __init__(self, n_ht, attn_dim, feat_dim, d=128, nl=4, nh=4, dff=256, do=0.15):
        super().__init__()
        self.attn_proj = nn.Sequential(nn.Linear(attn_dim, d), nn.LayerNorm(d))
        self.pos = nn.Parameter(torch.randn(1, n_ht, d) * 0.02)
        self.feat_proj = nn.Sequential(nn.Linear(feat_dim, d*4), nn.GELU(), nn.Dropout(0.2),
                                       nn.Linear(d*4, d*2), nn.GELU(), nn.Dropout(0.2),
                                       nn.Linear(d*2, d), nn.LayerNorm(d))
        self.feat_tok = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.layers = nn.ModuleList([EncLayer(d, nh, dff, do) for _ in range(nl)])
        self.err_head = nn.Sequential(nn.Linear(d, d*2), nn.GELU(), nn.Dropout(do),
                                      nn.Linear(d*2, d), nn.GELU(), nn.Dropout(do),
                                      nn.Linear(d, 1))
    def forward(self, patterns, features):
        B, H, N, _ = patterns.shape
        pooled = patterns.mean(dim=-1).reshape(B, H, N); flat = pooled.reshape(B, H*N)
        a_tok = self.attn_proj(flat).unsqueeze(1).expand(B, H, -1) + self.pos[:, :H, :]
        f_tok = self.feat_proj(features).unsqueeze(1) + self.feat_tok
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, f_tok, a_tok], dim=1)
        for layer in self.layers: x = layer(x)
        return self.err_head(x[:, 0]).squeeze(-1)

# Dataset
class FinDS(Dataset):
    def __init__(self,X,y,feat): self.X,self.y,self.feat=torch.FloatTensor(X),torch.FloatTensor(y),torch.FloatTensor(feat)
    def __len__(self): return len(self.X)
    def __getitem__(self,i): return self.X[i],self.y[i],self.feat[i]

# Flatten features
price_flat_tr = X_tr.reshape(len(X_tr), -1); price_flat_te = X_te.reshape(len(X_te), -1)
feat_tr = np.concatenate([price_flat_tr, sig_tr, tech_tr.reshape(len(tech_tr), -1)], axis=-1)
feat_te = np.concatenate([price_flat_te, sig_te, tech_te.reshape(len(tech_te), -1)], axis=-1)
from sklearn.preprocessing import StandardScaler
feat_scaler = StandardScaler(); feat_tr = feat_scaler.fit_transform(feat_tr); feat_te = feat_scaler.transform(feat_te)

tr_ds = FinDS(X_tr, y_tr, feat_tr); te_ds = FinDS(X_te, y_te, feat_te)
tr_ld = DataLoader(tr_ds, BATCH, shuffle=True, drop_last=True)
te_ld = DataLoader(te_ds, BATCH, shuffle=False)

# ============================================================
# TRAIN
# ============================================================
print("[2/4] Training Base Transformer...")
base = BaseTrans().to(GPU); opt_b = torch.optim.AdamW(base.parameters(), lr=1e-3, weight_decay=1e-4)
sch_b = torch.optim.lr_scheduler.CosineAnnealingLR(opt_b, BASE_EP)
for ep in range(BASE_EP):
    base.train(); tl = 0
    for x, y, _ in tr_ld:
        pred = base(x.to(GPU)); loss = F.huber_loss(pred, y.to(GPU), delta=1.0)
        opt_b.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(base.parameters(), 2.0)
        opt_b.step(); tl += loss.item()
    sch_b.step()
    if (ep+1) % 15 == 0: print(f"  Base {ep+1:3d} | Loss: {tl/len(tr_ld):.6f}")

base.eval()
with torch.no_grad():
    preds, ys = [], []
    for x, y, _ in te_ld: preds.append(base(x.to(GPU)).cpu().numpy()); ys.append(y.numpy())
    preds = np.concatenate(preds); ys = np.concatenate(ys)
base_mse = float(np.mean((preds-ys)**2)); base_err = np.abs(preds-ys).mean(1); base_abs = np.abs(preds).mean(1)
print(f"  Base MSE: {base_mse:.6f} | |Pred| r: {spearmanr(base_abs, base_err)[0]:.4f}")

# Extract patterns
print("[3/4] Extracting patterns + training Fusion Upper...")
base.eval()
all_pats, all_errs, all_feat = [], [], []
with torch.no_grad():
    for x, y, feat in tr_ld:
        pred, pats = base(x.to(GPU), get_pat=True)
        stacked = torch.cat([pats[i].cpu() for i in range(N_LAYERS)], dim=1)
        all_pats.append(stacked.numpy()); all_feat.append(feat.numpy())
        all_errs.append((pred - y.to(GPU)).abs().mean(1).cpu().numpy())
p_train = np.concatenate(all_pats); e_train = np.concatenate(all_errs); f_train = np.concatenate(all_feat)

total_heads = N_LAYERS * N_HEADS; attn_dim = total_heads * n_assets
upper = FusionUpper(total_heads, attn_dim, feat_total).to(GPU)
opt_u = torch.optim.AdamW(upper.parameters(), lr=3e-3, weight_decay=1e-5)
sch_u = torch.optim.lr_scheduler.CosineAnnealingLR(opt_u, UPPER_EP)

for ep in range(UPPER_EP):
    upper.train(); tl = 0; nb = 0
    for i in range(0, len(p_train), BATCH):
        pb = torch.FloatTensor(p_train[i:i+BATCH]).to(GPU)
        fb = torch.FloatTensor(f_train[i:i+BATCH]).to(GPU)
        eb = torch.FloatTensor(e_train[i:i+BATCH]).to(GPU)
        pe = upper(pb, fb); loss = F.mse_loss(pe, eb)
        opt_u.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(upper.parameters(), 2.0)
        opt_u.step(); tl += loss.item(); nb += 1
    sch_u.step()
    if (ep+1) % 10 == 0:
        with torch.no_grad():
            s = upper(torch.FloatTensor(p_train[:256]).to(GPU), torch.FloatTensor(f_train[:256]).to(GPU)).std().item()
        print(f"  Upper {ep+1:3d} | Loss: {tl/nb:.6f} | std: {s:.6f}")

# ============================================================
# EVALUATION
# ============================================================
print("[4/4] Evaluation...")
base.eval(); upper.eval()
all_up, all_true, all_abs = [], [], []
with torch.no_grad():
    for x, y, feat in te_ld:
        pred, pats = base(x.to(GPU), get_pat=True)
        stacked = torch.cat([pats[i].cpu() for i in range(N_LAYERS)], dim=1)
        up_err = upper(stacked.to(GPU), feat.to(GPU)).cpu().numpy()
        true_err = (pred - y.to(GPU)).abs().mean(1).cpu().numpy()
        all_up.append(up_err); all_true.append(true_err)
        all_abs.append(np.abs(pred.cpu().numpy()).mean(1))
up_err = np.concatenate(all_up); true_err = np.concatenate(all_true); abs_bl = np.concatenate(all_abs)

# Feature-only baseline
class FeatMLP(nn.Module):
    def __init__(self, in_d, d=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_d, d*4), nn.GELU(), nn.Dropout(0.2),
                                 nn.Linear(d*4, d*2), nn.GELU(), nn.Dropout(0.2),
                                 nn.Linear(d*2, d), nn.GELU(), nn.Linear(d, 1))
    def forward(self, x): return self.net(x).squeeze(-1)

feat_model = FeatMLP(feat_total).to(GPU); opt_f = torch.optim.AdamW(feat_model.parameters(), lr=1e-3)
for ep in range(50):
    feat_model.train()
    for i in range(0, len(f_train), BATCH):
        fb = torch.FloatTensor(f_train[i:i+BATCH]).to(GPU)
        eb = torch.FloatTensor(e_train[i:i+BATCH]).to(GPU)
        loss = F.mse_loss(feat_model(fb), eb); opt_f.zero_grad(); loss.backward(); opt_f.step()
feat_model.eval()
with torch.no_grad():
    feat_only = np.concatenate([feat_model(torch.FloatTensor(feat_te[i:i+BATCH]).to(GPU)).cpu().numpy() for i in range(0, len(feat_te), BATCH)])

def sr(a, b): return float(spearmanr(a, b)[0])

print("=" * 65)
print(f"  DIRECT FUSION on {n_assets} US Stocks (No Pretraining)")
print("=" * 65)
print(f"  |Pred| Baseline:         {sr(abs_bl, true_err):.4f}")
print(f"  Feature-only MLP:       {sr(feat_only, true_err):.4f}")
print(f"  Fusion Upper + Feat:    {sr(up_err, true_err):.4f}")
delta = sr(up_err, true_err) - max(sr(abs_bl, true_err), sr(feat_only, true_err))
print(f"  ---")
print(f"  Δ vs best baseline:     {delta:+.4f}")
print(f"  Base MSE:               {base_mse:.6f}")
print("=" * 65)
