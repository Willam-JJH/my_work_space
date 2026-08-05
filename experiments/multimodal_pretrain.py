"""
Multimodal Representation Pretraining for Return Prediction
============================================================
Phase 1: Contrastive pretraining — align price, signature, technical embeddings
Phase 2: Fine-tune on return prediction
Compare: pretrained vs from-scratch vs baseline
"""
import numpy as np; import pandas as pd; import torch; import torch.nn as nn
import torch.nn.functional as F; from torch.utils.data import Dataset, DataLoader
import signatory; import yfinance as yf; from scipy.stats import spearmanr
import math, warnings; warnings.filterwarnings('ignore')

GPU = torch.device("xpu" if torch.xpu.device_count() > 0 else "cpu"); CPU = torch.device("cpu")
print(f"GPU: {torch.xpu.get_device_name(0) if GPU.type=='xpu' else 'CPU'}")

# Config
TICKERS = ["AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","JPM","V","JNJ",
           "WMT","PG","MA","UNH","HD","BAC","NFLX","ADBE","CRM","XOM"]
LOOKBACK = 30; SIG_DEPTH = 3
BATCH = 64; D_EMBED = 128; TEMP = 0.07
PRETRAIN_EP = 60; FINETUNE_EP = 60
torch.manual_seed(42); np.random.seed(42)

# ============================================================
# DATA
# ============================================================
print("[1/6] Loading data...")
data = yf.download(TICKERS, start="2015-01-01", end="2024-12-31", auto_adjust=True, progress=False)
close = data["Close"].dropna(axis=1, thresh=int(len(data)*0.8)).dropna(axis=0)
returns = np.log(close / close.shift(1)).dropna().values.astype(np.float32)
tickers = list(close.columns); n_assets = len(tickers)
print(f"  {n_assets} stocks x {returns.shape[0]} days")

n_samp = returns.shape[0] - LOOKBACK
X_raw = np.zeros((n_samp, n_assets, LOOKBACK), dtype=np.float32)
y_raw = np.zeros((n_samp, n_assets), dtype=np.float32)
for i in range(n_samp):
    X_raw[i] = returns[i:i+LOOKBACK].T; y_raw[i] = returns[i+LOOKBACK]
mu = X_raw.mean(axis=-1, keepdims=True); st = X_raw.std(axis=-1, keepdims=True) + 1e-8
X_raw = (X_raw - mu) / st
clip = np.percentile(np.abs(y_raw), 99); y_raw = np.clip(y_raw, -clip, clip)

# Signatures
print("[2/6] Computing path signatures...")
sig_dim = signatory.signature_channels(n_assets, SIG_DEPTH)
X_sig_t = torch.FloatTensor(X_raw).transpose(1,2).to(CPU)
signatures = np.zeros((n_samp, sig_dim), dtype=np.float32)
for i in range(0, n_samp, 128):
    sig = signatory.signature(X_sig_t[i:i+128], SIG_DEPTH, basepoint=True)
    signatures[i:i+128] = sig.cpu().numpy()
print(f"  Sig dim: {sig_dim:,}")

# Technical features
def compute_tech(returns_2d):
    """returns_2d: (n_samp, n_assets, lookback) → tech: (n_samp, n_assets, 6)"""
    r = returns_2d
    vol_5 = r[:,:,-5:].std(axis=-1, keepdims=True)
    vol_10 = r[:,:,-10:].std(axis=-1, keepdims=True)
    vol_20 = r[:,:,-20:].std(axis=-1, keepdims=True)
    mom_5 = r[:,:,-5:].mean(axis=-1, keepdims=True)
    mom_10 = r[:,:,-10:].mean(axis=-1, keepdims=True)
    rsi = (r[:,:,-5:] > 0).mean(axis=-1, keepdims=True)
    return np.concatenate([vol_5,vol_10,vol_20,mom_5,mom_10,rsi], axis=-1)

X_tech = compute_tech(X_raw)  # (n_samp, n_assets, 6)

# Scale features
from sklearn.preprocessing import StandardScaler
sig_scaler = StandardScaler(); signatures = sig_scaler.fit_transform(signatures)

# Split
split = int(n_samp * 0.7)
X_tr, X_te = X_raw[:split], X_raw[split:]
y_tr, y_te = y_raw[:split], y_raw[split:]
sig_tr, sig_te = signatures[:split], signatures[split:]
tech_tr, tech_te = X_tech[:split], X_tech[split:]
print(f"  Train: {split} | Test: {n_samp - split}")

# ============================================================
# MODALITY ENCODERS
# ============================================================
class PriceEncoder(nn.Module):
    """Encode 30-day returns into embedding."""
    def __init__(self, n_assets, d=D_EMBED):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(LOOKBACK, d*2), nn.GELU(), nn.Linear(d*2, d))
        self.asset_pos = nn.Parameter(torch.randn(1, n_assets, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, 4, batch_first=True)
        self.out = nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d))
    def forward(self, x):
        B, N, L = x.shape
        h = self.proj(x) + self.asset_pos[:, :N, :]
        h, _ = self.attn(h, h, h)
        return self.out(h.mean(dim=1))  # (B, d) pooled

class SigEncoder(nn.Module):
    """Encode signature features into embedding."""
    def __init__(self, sig_d, d=D_EMBED):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(sig_d, d*4), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(d*4, d*2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(d*2, d), nn.LayerNorm(d))
    def forward(self, x): return self.net(x)  # (B, d)

class TechEncoder(nn.Module):
    """Encode technical features into embedding."""
    def __init__(self, n_assets, d=D_EMBED):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(6, d*2), nn.GELU(), nn.Linear(d*2, d))
        self.asset_pos = nn.Parameter(torch.randn(1, n_assets, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, 4, batch_first=True)
        self.out = nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d))
    def forward(self, x):
        B, N, _ = x.shape
        h = self.proj(x) + self.asset_pos[:, :N, :]
        h, _ = self.attn(h, h, h)
        return self.out(h.mean(dim=1))

# ============================================================
# PHASE 1: CONTRASTIVE PRETRAINING
# ============================================================
print("[3/6] Contrastive pretraining...")

price_enc = PriceEncoder(n_assets).to(GPU)
sig_enc = SigEncoder(sig_dim).to(GPU)
tech_enc = TechEncoder(n_assets).to(GPU)
all_params = list(price_enc.parameters()) + list(sig_enc.parameters()) + list(tech_enc.parameters())
opt_pretrain = torch.optim.AdamW(all_params, lr=5e-4, weight_decay=1e-4)

# Pretrain dataset: all samples (no target needed)
pretrain_ds = torch.utils.data.TensorDataset(
    torch.FloatTensor(X_tr), torch.FloatTensor(sig_tr), torch.FloatTensor(tech_tr))
pretrain_ld = DataLoader(pretrain_ds, BATCH, shuffle=True, drop_last=True)

for ep in range(PRETRAIN_EP):
    tl = 0; nb = 0
    for x_b, sig_b, tech_b in pretrain_ld:
        x_b = x_b.to(GPU); sig_b = sig_b.to(GPU); tech_b = tech_b.to(GPU)

        # Compute embeddings
        z_price = F.normalize(price_enc(x_b), dim=-1)
        z_sig = F.normalize(sig_enc(sig_b), dim=-1)
        z_tech = F.normalize(tech_enc(tech_b), dim=-1)

        # InfoNCE: price<->sig, price<->tech, sig<->tech
        B = x_b.shape[0]
        loss = 0
        for za, zb in [(z_price, z_sig), (z_price, z_tech), (z_sig, z_tech)]:
            sim = (za @ zb.T) / TEMP  # (B, B)
            labels = torch.arange(B, device=GPU)
            loss += (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
        loss /= 3

        opt_pretrain.zero_grad(); loss.backward(); opt_pretrain.step()
        tl += loss.item(); nb += 1
    if (ep+1) % 15 == 0:
        # Compute alignment metric
        with torch.no_grad():
            z_p = F.normalize(price_enc(torch.FloatTensor(X_tr[:128]).to(GPU)), dim=-1)
            z_s = F.normalize(sig_enc(torch.FloatTensor(sig_tr[:128]).to(GPU)), dim=-1)
            align = (z_p * z_s).sum(-1).mean().item()
        print(f"  Epoch {ep+1:3d} | Loss: {tl/nb:.4f} | Align: {align:.4f}")

# Save embeddings + models
import pickle, os
os.makedirs("D:/code/experiments", exist_ok=True)
with torch.no_grad():
    zp_tr_save = price_enc(torch.FloatTensor(X_tr).to(GPU)).cpu().numpy()
    zs_tr_save = sig_enc(torch.FloatTensor(sig_tr).to(GPU)).cpu().numpy()
    zt_tr_save = tech_enc(torch.FloatTensor(tech_tr).to(GPU)).cpu().numpy()
    zp_te_save = price_enc(torch.FloatTensor(X_te).to(GPU)).cpu().numpy()
    zs_te_save = sig_enc(torch.FloatTensor(sig_te).to(GPU)).cpu().numpy()
    zt_te_save = tech_enc(torch.FloatTensor(tech_te).to(GPU)).cpu().numpy()
with open("D:/code/experiments/pretrained_embeddings.pkl", "wb") as f:
    pickle.dump({"zp_tr":zp_tr_save,"zs_tr":zs_tr_save,"zt_tr":zt_tr_save,
                 "zp_te":zp_te_save,"zs_te":zs_te_save,"zt_te":zt_te_save}, f)
torch.save({"price":price_enc.state_dict(),"sig":sig_enc.state_dict(),"tech":tech_enc.state_dict()},
           "D:/code/experiments/pretrained_models.pt")
print("  Pretrained embeddings saved!")

# Freeze encoders for fine-tuning evaluation
price_enc.eval(); sig_enc.eval(); tech_enc.eval()
for p in price_enc.parameters(): p.requires_grad_(False)
for p in sig_enc.parameters(): p.requires_grad_(False)
for p in tech_enc.parameters(): p.requires_grad_(False)

# ============================================================
# PHASE 2: PREDICTION HEAD
# ============================================================
print("[4/6] Training prediction heads...")

class FusionPredictor(nn.Module):
    """Fuse pretrained embeddings → predict returns."""
    def __init__(self, d=D_EMBED, n_out=20):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Linear(d * 3, d * 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(d * 2, d), nn.GELU(),
            nn.Linear(d, n_out))
    def forward(self, z_price, z_sig, z_tech):
        z = torch.cat([z_price, z_sig, z_tech], dim=-1)
        return self.fuse(z)

# Compute all embeddings (frozen encoders)
print("  Computing embeddings...")
with torch.no_grad():
    Zp_tr = price_enc(torch.FloatTensor(X_tr).to(GPU)).cpu()
    Zs_tr = sig_enc(torch.FloatTensor(sig_tr).to(GPU)).cpu()
    Zt_tr = tech_enc(torch.FloatTensor(tech_tr).to(GPU)).cpu()
    Zp_te = price_enc(torch.FloatTensor(X_te).to(GPU)).cpu()
    Zs_te = sig_enc(torch.FloatTensor(sig_te).to(GPU)).cpu()
    Zt_te = tech_enc(torch.FloatTensor(tech_te).to(GPU)).cpu()

class EmbDS(Dataset):
    def __init__(self, zp, zs, zt, y):
        self.zp = zp; self.zs = zs; self.zt = zt; self.y = y
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.zp[i], self.zs[i], self.zt[i], self.y[i]

ft_tr_ds = EmbDS(Zp_tr, Zs_tr, Zt_tr, torch.FloatTensor(y_tr))
ft_te_ds = EmbDS(Zp_te, Zs_te, Zt_te, torch.FloatTensor(y_te))
ft_tr_ld = DataLoader(ft_tr_ds, BATCH, shuffle=True, drop_last=True)

# ----- Pretrained fusion -----
fusion_pt = FusionPredictor().to(GPU)
opt_pt = torch.optim.AdamW(fusion_pt.parameters(), lr=1e-3, weight_decay=1e-4)
sch_pt = torch.optim.lr_scheduler.CosineAnnealingLR(opt_pt, FINETUNE_EP)
for ep in range(FINETUNE_EP):
    fusion_pt.train(); tl = 0
    for zp, zs, zt, y in ft_tr_ld:
        pred = fusion_pt(zp.to(GPU), zs.to(GPU), zt.to(GPU))
        loss = F.huber_loss(pred, y.to(GPU), delta=1.0)
        opt_pt.zero_grad(); loss.backward(); opt_pt.step(); tl += loss.item()
    sch_pt.step()
    if (ep+1) % 20 == 0: print(f"  Pretrained {ep+1:3d} | Loss: {tl/len(ft_tr_ld):.6f}")

# ----- From-scratch fusion (no pretraining) -----
fusion_fs = FusionPredictor().to(GPU)
opt_fs = torch.optim.AdamW(fusion_fs.parameters(), lr=1e-3, weight_decay=1e-4)
sch_fs = torch.optim.lr_scheduler.CosineAnnealingLR(opt_fs, FINETUNE_EP)
for ep in range(FINETUNE_EP):
    fusion_fs.train(); tl = 0
    for zp, zs, zt, y in ft_tr_ld:
        pred = fusion_fs(zp.to(GPU), zs.to(GPU), zt.to(GPU))
        loss = F.huber_loss(pred, y.to(GPU), delta=1.0)
        opt_fs.zero_grad(); loss.backward(); opt_fs.step(); tl += loss.item()
    sch_fs.step()
    if (ep+1) % 20 == 0: print(f"  FromScratch {ep+1:3d} | Loss: {tl/len(ft_tr_ld):.6f}")

# ----- Ridge baseline -----
print("[5/6] Ridge baselines...")
from sklearn.linear_model import Ridge
ridge_sig = Ridge(alpha=1.0).fit(sig_tr, y_tr)
ridge_raw = Ridge(alpha=1.0).fit(X_tr.reshape(len(X_tr), -1), y_tr)

# ============================================================
# EVALUATION
# ============================================================
print("[6/6] Evaluation...")
with torch.no_grad():
    pred_pt = fusion_pt(Zp_te.to(GPU), Zs_te.to(GPU), Zt_te.to(GPU)).cpu().numpy()
    pred_fs = fusion_fs(Zp_te.to(GPU), Zs_te.to(GPU), Zt_te.to(GPU)).cpu().numpy()
pred_ridge_sig = ridge_sig.predict(sig_te)
pred_ridge_raw = ridge_raw.predict(X_te.reshape(len(X_te), -1))

def mse(p, t): return float(np.mean((p - t)**2))
def sr(pred_err, true_err): return float(spearmanr(pred_err, true_err)[0])

print("=" * 65)
print("  MULTIMODAL PRETRAINING RESULTS")
print("=" * 65)
print(f"  {'Method':<30} {'MSE':>10} {'|Pred| r':>10}")
print(f"  {'-'*50}")
print(f"  {'Ridge (raw returns)':<30} {mse(pred_ridge_raw, y_te):>10.6f} {sr(np.abs(pred_ridge_raw).mean(1), np.abs(pred_ridge_raw-y_te).mean(1)):>10.4f}")
print(f"  {'Ridge (signature)':<30} {mse(pred_ridge_sig, y_te):>10.6f} {sr(np.abs(pred_ridge_sig).mean(1), np.abs(pred_ridge_sig-y_te).mean(1)):>10.4f}")
print(f"  {'Fusion From-Scratch':<30} {mse(pred_fs, y_te):>10.6f} {sr(np.abs(pred_fs).mean(1), np.abs(pred_fs-y_te).mean(1)):>10.4f}")
print(f"  {'Fusion PRETRAINED':<30} {mse(pred_pt, y_te):>10.6f} {sr(np.abs(pred_pt).mean(1), np.abs(pred_pt-y_te).mean(1)):>10.4f}")
print(f"  {'-'*50}")
delta_mse = (mse(pred_fs, y_te) - mse(pred_pt, y_te)) / mse(pred_fs, y_te) * 100
print(f"  Pretrain MSE improvement: {delta_mse:+.1f}%")
delta_r = sr(np.abs(pred_pt).mean(1), np.abs(pred_pt-y_te).mean(1)) - sr(np.abs(pred_fs).mean(1), np.abs(pred_fs-y_te).mean(1))
print(f"  Pretrain |Pred| r gain:   {delta_r:+.4f}")
print("=" * 65)
