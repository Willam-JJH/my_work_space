"""
Meta-Transformer + Proposal Signature Pipeline
===============================================
Proposal d=3 embedding (logP, logV, vol_innov) → Signatory depth=3 (39-dim)
→ Base Transformer cross-asset attention → Upper Meta-Transformer fusion
→ Return prediction + uncertainty

Designed for RTX 4090. Data: 2000-2024, 5000+ stocks.
"""
import numpy as np; import pandas as pd; import torch; import torch.nn as nn; import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader; import signatory; from scipy.stats import spearmanr
import math, time, os, warnings; warnings.filterwarnings('ignore')

GPU = torch.device("cuda" if torch.cuda.is_available() else "cpu"); CPU = torch.device("cpu")
print(f"GPU: {torch.cuda.get_device_name(0) if GPU.type=='cuda' else 'CPU'} | "
      f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB" if GPU.type=='cuda' else '')

# ============================================================
# CONFIG — Tuned for 4090 48GB
# ============================================================
N_STOCKS_US = 3000        # ALL US stocks
N_STOCKS_CN = 3000  # max CN fitting 48GB      # top CN stocks (faster test)
WINDOW = 250            # trading days per month (proposal standard)
PRED_HORIZON = 21       # predict next month (~21 trading days)
SIG_DEPTH = 3           # proposal d=3 → 39 components
BATCH = 1               # minimal batch for 2000+ tokens
D_MODEL = 128           # transformer dim
N_LAYERS = 4; N_HEADS = 4; D_FF = 256; DROPOUT = 0.1
BASE_EP = 40; UPPER_EP = 60
torch.manual_seed(42); np.random.seed(42)

# ============================================================
# DATA LOADING
# ============================================================
def load_market(name, n_stocks, price_path, vol_path):
    """Load price+volume, compute proposal d=3 path, return dataset."""
    print(f"\n{'='*50}\n  {name}: loading {n_stocks} stocks\n{'='*50}")

    # Load
    price = pd.read_parquet(price_path)
    vol = pd.read_parquet(vol_path)
    # Align dates and columns
    common_cols = sorted(set(price.columns) & set(vol.columns))
    common_idx = price.index.intersection(vol.index)
    price = price.loc[common_idx, common_cols].ffill().fillna(0)
    vol = vol.loc[common_idx, common_cols].ffill().fillna(1)  # fill missing vol with 1

    # Select top by data completeness
    comp = price.notna().sum() / len(price)
    top_cols = comp.nlargest(min(n_stocks, len(comp))).index.tolist()
    price = price[top_cols]; vol = vol[top_cols]
    n_a = len(top_cols); n_days = len(price)
    print(f"  {n_a} stocks x {n_days} days | {price.index[0].date()} → {price.index[-1].date()}")

    # Compute log prices and log volumes
    logP = np.log(np.maximum(price.values.astype(np.float64),1e-8))
    logV = np.log(np.maximum(vol.values.astype(np.float64),1e-8))

    # Volume innovation: logV - MA_20(logV)
    ma20 = pd.DataFrame(logV).rolling(20, min_periods=1).mean().values
    vol_innov = logV - ma20

    # Proposal d=3 path: (logP, logV, vol_innov)
    paths = np.stack([logP, logV, vol_innov], axis=-1)  # (n_days, n_assets, 3)
    paths = paths.astype(np.float32)

    # Sliding windows: 250-day windows → predict next-month return
    n_samp = n_days - WINDOW - PRED_HORIZON + 1
    X_paths = np.zeros((n_samp, n_a, WINDOW, 3), dtype=np.float32)  # (samples, assets, days, channels)
    y_fwd = np.zeros((n_samp, n_a), dtype=np.float32)
    for i in range(n_samp):
        X_paths[i] = paths[i:i+WINDOW].transpose(1,0,2)  # (WINDOW,n_a,3)→(n_a,WINDOW,3)
        # Forward return: sum of daily returns over next PRED_HORIZON days
        y_fwd[i] = logP[i+WINDOW+PRED_HORIZON-1] - logP[i+WINDOW-1]

    # Also compute 30-day returns for Base Transformer
    X_ret = np.zeros((n_samp, n_a, 30), dtype=np.float32)
    logP_returns = np.diff(logP, axis=0)  # daily log returns
    logP_returns = np.vstack([np.zeros((1,n_a)), logP_returns])  # pad first day
    for i in range(n_samp):
        end_idx = i + WINDOW
        start_idx = end_idx - 30
        X_ret[i] = logP_returns[start_idx:end_idx].T  # (n_a, 30)
    # Standardize returns per window
    mu = X_ret.mean(axis=-1, keepdims=True); st = X_ret.std(axis=-1, keepdims=True) + 1e-8
    X_ret = (X_ret - mu) / st

    # Handle NaN + clip targets
    y_fwd = np.nan_to_num(y_fwd, nan=0.0, posinf=0.0, neginf=0.0)
    y_fwd = np.clip(y_fwd, -np.percentile(np.abs(y_fwd), 99), np.percentile(np.abs(y_fwd), 99))

    # Train/test split
    # 3-way split: train 2000-2015, val 2015-2020, test 2020-2024
    # Find split indices by date
    train_end = np.where(price.index >= '2015-01-01')[0][0] if '2015-01-01' in str(price.index) else int(n_samp * 0.5)
    val_end = np.where(price.index >= '2020-01-01')[0][0] if '2020-01-01' in str(price.index) else int(n_samp * 0.75)
    # Adjust to window offset
    train_end = max(WINDOW + PRED_HORIZON, train_end)
    val_end = max(train_end + WINDOW, val_end)
    split = train_end - WINDOW - PRED_HORIZON + 1  # for compatibility: split = train samples
    val_split = val_end - WINDOW - PRED_HORIZON + 1  # val split point
    return (X_ret[:split], X_paths[:split], y_fwd[:split]), \
           (X_ret[split:], X_paths[split:], y_fwd[split:]), n_a

# ============================================================
# PER-STOCK SIGNATURE COMPUTATION
# ============================================================
def compute_signatures(paths, depth=SIG_DEPTH, device=CPU):
    """
    paths: (n_samples, n_assets, WINDOW, 3) — d=3 channels
    Returns: (n_samples, n_assets, sig_dim) signatures per stock
    """
    n_samp, n_a, L, C = paths.shape
    sig_dim = signatory.signature_channels(C, depth)  # 39 for C=3, depth=3
    print(f"  Signatures: {C} channels, depth={depth} → {sig_dim} dim")

    sigs = np.zeros((n_samp, n_a, sig_dim), dtype=np.float32)
    paths_t = torch.FloatTensor(paths).to(device)  # (n_samp, n_a, L, C)

    for a in range(n_a):
        # For each asset: (n_samp, L, C) → signature
        asset_path = paths_t[:, a]  # (n_samp, L, C)
        for i in range(0, n_samp, 128):
            batch = asset_path[i:i+128]
            sigs[i:i+128, a] = signatory.signature(batch, depth, basepoint=True).cpu().numpy()
        if (a+1) % 100 == 0:
            print(f"    asset {a+1}/{n_a}")

    # Handle inf/nan then standardize
    from sklearn.preprocessing import RobustScaler
    sigs = np.nan_to_num(sigs, nan=0.0, posinf=0.0, neginf=0.0)
    sigs_flat = sigs.reshape(-1, sig_dim)
    sigs_flat = np.nan_to_num(sigs_flat, nan=0.0, posinf=0.0, neginf=0.0)
    p1 = np.percentile(sigs_flat, 1, axis=0); p99 = np.percentile(sigs_flat, 99, axis=0)
    sigs_flat = np.clip(sigs_flat, p1, p99)
    sigs_flat = RobustScaler().fit_transform(sigs_flat)
    sigs = sigs_flat.reshape(n_samp, n_a, sig_dim)

    return sigs, sig_dim

# ============================================================
# CLASSICAL FACTORS
# ============================================================
def compute_classical(X_ret):
    """X_ret: (n_samp, n_a, 30) standardized returns"""
    n_samp, n_a, L = X_ret.shape
    factors = []
    # Momentum: last 5d, 10d, 20d mean
    for h in [5, 10, 20]:
        factors.append(X_ret[:,:, -h:].mean(axis=-1))
    # Volatility: last 10d, 20d std
    for h in [10, 20]:
        factors.append(X_ret[:,:, -h:].std(axis=-1))
    # RSI proxy
    factors.append((X_ret[:,:, -5:] > 0).mean(axis=-1))
    # Stack: (n_samp, n_a, n_factors)
    stacked = np.stack(factors, axis=-1)
    return stacked  # (n_samp, n_a, 6)

# ============================================================
# MODEL COMPONENTS
# ============================================================
class MHA(nn.Module):
    def __init__(self, dm, nh, dp=0.1):
        super().__init__(); self.nh=nh; self.dk=dm//nh
        self.q=nn.Linear(dm,dm); self.k=nn.Linear(dm,dm); self.v=nn.Linear(dm,dm)
        self.out=nn.Linear(dm,dm); self.drop=nn.Dropout(dp); self._w=None
    def forward(self,x):
        B,N,E=x.shape; q=self.q(x).view(B,N,self.nh,self.dk).transpose(1,2)
        k=self.k(x).view(B,N,self.nh,self.dk).transpose(1,2); v=self.v(x).view(B,N,self.nh,self.dk).transpose(1,2)
        w=(q@k.transpose(-2,-1))/math.sqrt(self.dk); w=F.softmax(w,-1); self._w=w.detach()
        return self.out((self.drop(w)@v).transpose(1,2).contiguous().view(B,N,E))

class EncLayer(nn.Module):
    def __init__(self,d,nh,df,dp):
        super().__init__(); self.n1=nn.LayerNorm(d); self.attn=MHA(d,nh,dp)
        self.n2=nn.LayerNorm(d); self.ff=nn.Sequential(nn.Linear(d,df),nn.GELU(),nn.Dropout(dp),nn.Linear(df,d),nn.Dropout(dp))
    def forward(self,x): return x+self.ff(self.n2(x+self.attn(self.n1(x))))

class BaseTrans(nn.Module):
    """Cross-asset attention on 30-day returns."""
    def __init__(self, n_a):
        super().__init__(); self.nl=N_LAYERS
        self.proj=nn.Linear(30,D_MODEL); self.pos=nn.Parameter(torch.randn(1,2000,D_MODEL)*0.02)
        self.drop=nn.Dropout(DROPOUT)
        self.layers=nn.ModuleList([EncLayer(D_MODEL,N_HEADS,D_FF,DROPOUT) for _ in range(N_LAYERS)])
        self.head=nn.Sequential(nn.Linear(D_MODEL,D_MODEL//2),nn.GELU(),nn.Linear(D_MODEL//2,1))
    def forward(self,x,get_pat=False):
        B,N,L=x.shape; h=self.drop(self.proj(x)+self.pos[:,:N,:]); pats={}
        for i,layer in enumerate(self.layers): h=layer(h)
        pred=self.head(h).squeeze(-1)
        return pred  # simplified — no pattern capture to save memory

class MetaTransformer(nn.Module):
    """
    Upper Meta-Transformer: fuses signature tokens + classical factors.
    Token sequence: [CLS] [SIG_1...SIG_N] [FACTOR_pooled]
    """
    def __init__(self, n_a, sig_dim, cl_dim, d=D_MODEL, nl=4, nh=4, dff=256, do=0.1):
        super().__init__()
        self.n_a = n_a
        # Signature token per asset
        self.sig_proj = nn.Sequential(nn.Linear(sig_dim, d), nn.LayerNorm(d))
        self.sig_pos = nn.Parameter(torch.randn(1, 2000, d) * 0.02)
        # Classical factor pooled token
        self.cl_proj = nn.Sequential(nn.Linear(cl_dim, d), nn.LayerNorm(d))
        self.cl_tok = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        # CLS
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        # Transformer
        self.layers = nn.ModuleList([EncLayer(d, nh, dff, do) for _ in range(nl)])
        # Heads
        self.ret_head = nn.Linear(d, n_a)          # per-asset return prediction
        self.unc_head = nn.Sequential(nn.Linear(d, d*2), nn.GELU(), nn.Dropout(do),
                                      nn.Linear(d*2, d), nn.GELU(), nn.Linear(d, 1))
    def forward(self, sigs, classical):
        """
        sigs: (B, n_a, sig_dim) — per-asset signatures
        classical: (B, n_a, cl_dim) — per-asset classical factors
        """
        B = sigs.shape[0]
        # Project signatures to tokens
        sig_tok = self.sig_proj(sigs) + self.sig_pos[:, :self.n_a, :]  # (B, n_a, d)
        # Classical: mean pool across assets → 1 token
        cl_pooled = classical.mean(dim=1)  # (B, cl_dim)
        cl_tok = self.cl_proj(cl_pooled).unsqueeze(1) + self.cl_tok  # (B, 1, d)
        # CLS
        cls_tok = self.cls.expand(B, -1, -1)  # (B, 1, d)
        # Full sequence: [CLS] + [SIG_1...SIG_N] + [FACTOR]
        x = torch.cat([cls_tok, sig_tok, cl_tok], dim=1)  # (B, 1+N+1, d)
        for layer in self.layers: x = layer(x)
        # Predictions
        ret_pred = self.ret_head(x[:, 0])          # from CLS
        unc_pred = self.unc_head(x[:, 0]).squeeze(-1)  # from CLS
        return ret_pred, unc_pred

# ============================================================
# TRAINING
# ============================================================
def run_experiment(name, train_data, val_data, test_data, n_a):
    (Xr_tr, Xp_tr, y_tr), (Xr_val, Xp_val, y_val), (Xr_te, Xp_te, y_te) = train_data, val_data, test_data
    print(f"  Train: {Xr_tr.shape[0]} | Val: {Xr_val.shape[0]} | Test: {Xr_te.shape[0]}")

    # Compute signatures
    t0 = time.time()
    sig_tr, sig_dim = compute_signatures(Xp_tr)
    sig_val, _ = compute_signatures(Xp_val)
    sig_te, _ = compute_signatures(Xp_te)
    print(f"  Sigs computed in {time.time()-t0:.0f}s")

    # Classical factors
    cl_tr = compute_classical(Xr_tr); cl_te = compute_classical(Xr_te)
    cl_dim = cl_tr.shape[-1]

    # Normalize features
    from sklearn.preprocessing import RobustScaler
    sig_flat = sig_tr.reshape(-1, sig_dim); sig_tr_n = RobustScaler().fit_transform(sig_flat).reshape(sig_tr.shape)
    sig_flat = sig_te.reshape(-1, sig_dim); sig_te_n = RobustScaler().fit_transform(sig_flat).reshape(sig_te.shape)
    cl_flat = cl_tr.reshape(-1, cl_dim); cl_tr_n = RobustScaler().fit_transform(cl_flat).reshape(cl_tr.shape)
    cl_flat = cl_te.reshape(-1, cl_dim); cl_te_n = RobustScaler().fit_transform(cl_flat).reshape(cl_te.shape)

    # Dataset
    class FinDS(Dataset):
        def __init__(self,s,c,y): self.s,self.c,self.y=torch.FloatTensor(s),torch.FloatTensor(c),torch.FloatTensor(y)
        def __len__(self): return len(self.s)
        def __getitem__(self,i): return self.s[i],self.c[i],self.y[i]
    tr_ds=FinDS(sig_tr_n,cl_tr_n,y_tr); te_ds=FinDS(sig_te_n,cl_te_n,y_te)
    tr_ld=DataLoader(tr_ds,BATCH,shuffle=True,drop_last=True); te_ld=DataLoader(te_ds,BATCH,shuffle=False)

    # Model
    model = MetaTransformer(n_a, sig_dim, cl_dim).to(GPU)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Meta-Transformer: {n_params:,} params")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, BASE_EP)

    print(f"  Training ({BASE_EP} epochs)...")
    for ep in range(BASE_EP):
        model.train(); tl = 0
        for s, c, y in tr_ld:
            ret_pred, unc_pred = model(s.to(GPU), c.to(GPU))
            loss_ret = F.huber_loss(ret_pred, y.to(GPU), delta=1.0)
            # Uncertainty: predict |error|
            with torch.no_grad():
                true_err = (ret_pred - y.to(GPU)).abs().mean(1)
            loss_unc = F.mse_loss(unc_pred, true_err)
            loss = loss_ret + 0.1 * loss_unc
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step()
            tl += loss.item()
        sch.step()
        if (ep+1) % 10 == 0: print(f"    Epoch {ep+1:3d} | Loss: {tl/len(tr_ld):.6f}")

    # Evaluate
    model.eval()
    all_ret, all_unc, all_y = [], [], []
    with torch.no_grad():
        for s, c, y in te_ld:
            rp, up = model(s.to(GPU), c.to(GPU))
            all_ret.append(rp.cpu().numpy()); all_unc.append(up.cpu().numpy()); all_y.append(y.numpy())
    pred_r = np.concatenate(all_ret); pred_u = np.concatenate(all_unc); true_y = np.concatenate(all_y)

    mse = float(np.mean((pred_r - true_y)**2))
    err_true = np.abs(pred_r - true_y).mean(axis=1)
    abs_bl = np.abs(pred_r).mean(axis=1)

    def sr(a,b): return float(spearmanr(a,b)[0])

    # Baselines
    from sklearn.linear_model import Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(sig_tr_n.reshape(len(sig_tr_n), -1), y_tr)
    pred_ridge = ridge.predict(sig_te_n.reshape(len(sig_te_n), -1))
    ridge_err = np.abs(pred_ridge - true_y).mean(1); ridge_abs = np.abs(pred_ridge).mean(1)

    return {
        'name': name, 'n_stocks': n_a, 'sig_dim': sig_dim,
        'mse': mse,
        'pred_baseline': sr(abs_bl, err_true),
        'model_unc': sr(pred_u, err_true),
        'ridge_mse': float(np.mean((pred_ridge-true_y)**2)),
        'ridge_pred_r': sr(ridge_abs, ridge_err),
    }

# ============================================================
# MAIN
# ============================================================
print("="*60)
print("  META-TRANSFORMER + PROPOSAL SIGNATURES")
print("  d=3 path (logP, logV, vol_innov), depth=3 → 39-dim")
print("="*60)

results = []

# US market
us_train, us_val, us_test, us_n = load_market("US", N_STOCKS_US,
    "D:/code/data/us_price_expanded.parquet", "D:/code/data/us_volume_expanded.parquet")
results.append(run_experiment("US", us_train, us_val, us_test, us_n))

# CN market
cn_train, cn_val, cn_test, cn_n = load_market("CN", N_STOCKS_CN,
    "D:/code/data/cn_price.parquet", "D:/code/data/cn_volume.parquet")
results.append(run_experiment("CN", cn_train, cn_val, cn_test, cn_n))

# Summary
print("\n" + "="*65)
print("  FINAL RESULTS: Meta-Transformer + Proposal Signatures")
print("="*65)
for r in results:
    print(f"\n  {r['name']} ({r['n_stocks']} stocks, sig={r['sig_dim']}d)")
    print(f"  {'Method':<25} {'MSE':>10} {'Spearman r':>10}")
    print(f"  {'-'*45}")
    print(f"  {'Meta-Transformer':<25} {r['mse']:>10.6f} {r['model_unc']:>10.4f}")
    print(f"  {'|Pred| Baseline':<25} {'':>10} {r['pred_baseline']:>10.4f}")
    print(f"  {'Ridge (sig only)':<25} {r['ridge_mse']:>10.6f} {r['ridge_pred_r']:>10.4f}")
print("="*65)
