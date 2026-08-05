"""
Path Signature Factors for Financial Prediction
================================================
signatory on CPU (signature computation) + Intel GPU training
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import signatory
import yfinance as yf
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
import warnings; warnings.filterwarnings('ignore')

GPU = torch.device("xpu" if torch.xpu.device_count() > 0 else "cpu")
CPU = torch.device("cpu")
print(f"GPU: {torch.xpu.get_device_name(0) if GPU.type=='xpu' else 'CPU'}")

TICKERS = ["AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","JPM","V","JNJ",
           "WMT","PG","MA","UNH","HD","BAC","NFLX","ADBE","CRM","XOM"]

# ============================================================
# DATA
# ============================================================
print("[1/4] Downloading S&P 500 data...")
data = yf.download(TICKERS, start="2015-01-01", end="2024-12-31", auto_adjust=True, progress=False)
close = data["Close"].dropna(axis=1, thresh=int(len(data)*0.8)).dropna(axis=0)
returns = np.log(close / close.shift(1)).dropna().values.astype(np.float32)
tickers_kept = list(close.columns)
n_assets = len(tickers_kept)
print(f"  {n_assets} stocks x {returns.shape[0]} days")

# Sliding windows: (n_samples, n_assets, window)
window = 30
n_samples = returns.shape[0] - window
X_raw = np.zeros((n_samples, n_assets, window), dtype=np.float32)
y_raw = np.zeros((n_samples, n_assets), dtype=np.float32)
for i in range(n_samples):
    X_raw[i] = returns[i:i+window].T
    y_raw[i] = returns[i+window]
# Standardize
mean = X_raw.mean(axis=-1, keepdims=True)
std = X_raw.std(axis=-1, keepdims=True) + 1e-8
X_raw = (X_raw - mean) / std
clip = np.percentile(np.abs(y_raw), 99)
y_raw = np.clip(y_raw, -clip, clip)

# ============================================================
# PATH SIGNATURES
# ============================================================
print(f"[2/4] Computing path signatures (depth=3, {n_assets} channels)...")
sig_dim = signatory.signature_channels(n_assets, 3)
print(f"  Signature dimension: {sig_dim:,}")

X_t = torch.FloatTensor(X_raw).transpose(1, 2).to(CPU)  # (N, window, assets)
signatures = np.zeros((n_samples, sig_dim), dtype=np.float32)
bs = 128
for i in range(0, n_samples, bs):
    batch = X_t[i:i+bs]
    sig = signatory.signature(batch, 3, basepoint=True)
    signatures[i:i+bs] = sig.cpu().numpy()
    if (i // bs + 1) % 20 == 0:
        print(f"  {i+bs}/{n_samples}")

print(f"  Signatures: {signatures.shape}")

# Train/test split
split = int(n_samples * 0.7)
sig_tr, sig_te = signatures[:split], signatures[split:]
y_tr, y_te = y_raw[:split], y_raw[split:]

scaler = StandardScaler()
sig_tr = scaler.fit_transform(sig_tr)
sig_te = scaler.transform(sig_te)

print(f"  Train: {sig_tr.shape[0]} | Test: {sig_te.shape[0]}")

# ============================================================
# MODEL
# ============================================================
class SigMLP(nn.Module):
    def __init__(self, in_dim, out_dim, h=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h*4), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(h*4, h*2), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(h*2, h), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(h, out_dim))
    def forward(self, x): return self.net(x)

class SigDS(Dataset):
    def __init__(self, sig, y): self.sig, self.y = torch.FloatTensor(sig), torch.FloatTensor(y)
    def __len__(self): return len(self.sig)
    def __getitem__(self, i): return self.sig[i], self.y[i]

print("[3/4] Training...")
model = SigMLP(sig_dim, n_assets).to(GPU)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 80)
tr_ld = DataLoader(SigDS(sig_tr, y_tr), batch_size=64, shuffle=True)

for ep in range(80):
    model.train(); tl = 0
    for s, y in tr_ld:
        pred = model(s.to(GPU))
        loss = F.huber_loss(pred, y.to(GPU), delta=1.0)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step(); tl += loss.item()
    sch.step()
    if (ep+1) % 20 == 0:
        print(f"  Epoch {ep+1:3d} | Loss: {tl/len(tr_ld):.6f}")

# ============================================================
# EVALUATION
# ============================================================
print("[4/4] Results")
model.eval()
with torch.no_grad():
    pred_sig = model(torch.FloatTensor(sig_te).to(GPU)).cpu().numpy()
    pred_tr = model(torch.FloatTensor(sig_tr).to(GPU)).cpu().numpy()

mse = float(np.mean((pred_sig - y_te)**2))
mae = float(np.mean(np.abs(pred_sig - y_te)))
err_te = np.abs(pred_sig - y_te).mean(axis=1)
err_tr = np.abs(pred_tr - y_tr).mean(axis=1)
abs_pred = np.abs(pred_sig).mean(axis=1)

# Linear baseline
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1.0)
ridge.fit(sig_tr, y_tr)
pred_r = ridge.predict(sig_te)
err_r = np.abs(pred_r - y_te).mean(axis=1)
abs_pred_r = np.abs(pred_r).mean(axis=1)

# Train uncertainty model on errors
unc = SigMLP(sig_dim, 1, h=128).to(GPU)
unc_opt = torch.optim.AdamW(unc.parameters(), lr=1e-3)
unc_ds = SigDS(sig_tr, err_tr)
unc_ld = DataLoader(unc_ds, batch_size=64, shuffle=True)
for ep in range(30):
    unc.train()
    for s, e in unc_ld:
        p = unc(s.to(GPU)); loss = F.mse_loss(p.squeeze(), e.to(GPU))
        unc_opt.zero_grad(); loss.backward(); unc_opt.step()
unc.eval()
with torch.no_grad():
    unc_pred = unc(torch.FloatTensor(sig_te).to(GPU)).cpu().numpy().squeeze()

print("=" * 50)
print(f"  Signature MLP    | MSE={mse:.6f}  MAE={mae:.6f}")
print(f"  Ridge Baseline   | MSE={float(np.mean((pred_r - y_te)**2)):.6f}")
print(f"  Sig Uncertainty  | Spearman r={spearmanr(unc_pred, err_te)[0]:.4f}")
print(f"  |Pred| Baseline  | Spearman r={spearmanr(abs_pred, err_te)[0]:.4f}")
print(f"  Ridge |Pred|     | Spearman r={spearmanr(abs_pred_r, err_r)[0]:.4f}")
print("=" * 50)
