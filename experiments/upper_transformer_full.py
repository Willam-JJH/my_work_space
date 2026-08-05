"""
Upper Transformer + Signature V2: Fixed model, smaller & better tuned.
"""
import numpy as np; import pandas as pd; import torch; import torch.nn as nn
import torch.nn.functional as F; from torch.utils.data import Dataset, DataLoader
import signatory; import yfinance as yf; from scipy.stats import spearmanr
import math, warnings; warnings.filterwarnings('ignore')

GPU = torch.device("xpu" if torch.xpu.device_count() > 0 else "cpu"); CPU = torch.device("cpu")
print(f"GPU: {torch.xpu.get_device_name(0) if GPU.type=='xpu' else 'CPU'}")

TICKERS = ["AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","JPM","V","JNJ",
           "WMT","PG","MA","UNH","HD","BAC","NFLX","ADBE","CRM","XOM"]
LOOKBACK, SIG_DEPTH = 30, 3
D_MODEL, N_LAYERS, N_HEADS, D_FF, DROPOUT = 64, 4, 4, 128, 0.1
BATCH, BASE_EP, UPPER_EP = 64, 50, 80
torch.manual_seed(42); np.random.seed(42)

# ----- DATA -----
print("[1/5] Loading data...")
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
split = int(n_samp * 0.7)
print(f"  Train: {split} | Test: {n_samp - split}")

# ----- SIGNATURES -----
print("[2/5] Computing path signatures (depth=3)...")
sig_dim = signatory.signature_channels(n_assets, SIG_DEPTH)
print(f"  Channels={n_assets} | Sig dim={sig_dim:,}")

X_sig = torch.FloatTensor(X_raw).transpose(1,2).to(CPU)
signatures = np.zeros((n_samp, sig_dim), dtype=np.float32)
for i in range(0, n_samp, 128):
    sig = signatory.signature(X_sig[i:i+128], SIG_DEPTH, basepoint=True)
    signatures[i:i+128] = sig.cpu().numpy()
    if (i//128 + 1) % 20 == 0: print(f"  batch {i+128}/{n_samp}")
print(f"  Signatures: {signatures.shape}")

from sklearn.preprocessing import StandardScaler
signatures = StandardScaler().fit_transform(signatures)

class FinDS(Dataset):
    def __init__(self, X, y, sig):
        self.X = torch.FloatTensor(X); self.y = torch.FloatTensor(y); self.sig = torch.FloatTensor(sig)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i], self.sig[i]

tr_ds = FinDS(X_raw[:split], y_raw[:split], signatures[:split])
te_ds = FinDS(X_raw[split:], y_raw[split:], signatures[split:])
tr_ld = DataLoader(tr_ds, BATCH, shuffle=True, drop_last=True)
te_ld = DataLoader(te_ds, BATCH, shuffle=False)

# ----- BASE TRANSFORMER (smaller) -----
class MHA(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__(); self.n_heads=n_heads; self.d_k=d_model//n_heads
        self.q=nn.Linear(d_model,d_model); self.k=nn.Linear(d_model,d_model)
        self.v=nn.Linear(d_model,d_model); self.out=nn.Linear(d_model,d_model)
        self.drop=nn.Dropout(dropout); self._w=None
    def forward(self, x):
        B,N,E=x.shape
        q=self.q(x).view(B,N,self.n_heads,self.d_k).transpose(1,2)
        k=self.k(x).view(B,N,self.n_heads,self.d_k).transpose(1,2)
        v=self.v(x).view(B,N,self.n_heads,self.d_k).transpose(1,2)
        w=(q@k.transpose(-2,-1))/math.sqrt(self.d_k); w=F.softmax(w,dim=-1)
        self._w=w.detach(); w=self.drop(w)
        return self.out((w@v).transpose(1,2).contiguous().view(B,N,E))

class EncLayer(nn.Module):
    def __init__(self,d,n_h,d_ff,dp):
        super().__init__()
        self.n1=nn.LayerNorm(d); self.attn=MHA(d,n_h,dp)
        self.n2=nn.LayerNorm(d); self.ff=nn.Sequential(nn.Linear(d,d_ff),nn.GELU(),nn.Dropout(dp),nn.Linear(d_ff,d),nn.Dropout(dp))
    def forward(self,x): return x+self.ff(self.n2(x+self.attn(self.n1(x))))

class BaseTrans(nn.Module):
    def __init__(self, d=D_MODEL, nl=N_LAYERS, nh=N_HEADS, dff=D_FF, do=DROPOUT):
        super().__init__(); self.nl=nl
        self.proj=nn.Linear(LOOKBACK,d)
        self.pos=nn.Parameter(torch.randn(1,500,d)*0.02)
        self.drop=nn.Dropout(do)
        self.layers=nn.ModuleList([EncLayer(d,nh,dff,do) for _ in range(nl)])
        self.head=nn.Sequential(nn.Linear(d,d//2),nn.GELU(),nn.Linear(d//2,1))
    def forward(self, x, get_pat=False):
        B,N,L=x.shape; h=self.drop(self.proj(x)+self.pos[:,:N,:]); pats={}
        for i,layer in enumerate(self.layers):
            h=layer(h)
            if get_pat: pats[i]=layer.attn._w
        pred=self.head(h).squeeze(-1)
        return (pred,pats) if get_pat else pred

# ----- UPPER TRANSFORMER (smaller, tuned) -----
class UpperTrans(nn.Module):
    def __init__(self, n_heads_total, attn_dim, sig_d, d=128, nl=3, nh=4, dff=256, do=0.15):
        super().__init__()
        # Pool attention: (B,H,N,N) → (B,H,N) via row-mean → concat → (B, H*N)
        self.pattern_dim_reduced = n_heads_total * 20  # H * N (20 stocks)
        self.attn_pool = nn.Sequential(nn.Linear(attn_dim, d), nn.LayerNorm(d))
        self.sig_proj = nn.Sequential(nn.Linear(sig_d, d), nn.LayerNorm(d))
        self.pos = nn.Parameter(torch.randn(1, n_heads_total, d) * 0.02)
        self.sig_tok = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.layers = nn.ModuleList([EncLayer(d, nh, dff, do) for _ in range(nl)])
        self.err_head = nn.Sequential(
            nn.Linear(d, d*2), nn.GELU(), nn.Dropout(do),
            nn.Linear(d*2, d), nn.GELU(), nn.Dropout(do),
            nn.Linear(d, 1))
    def forward(self, patterns, sig_feat):
        B, H, N, _ = patterns.shape
        # Pool each NxN matrix to N-dim vector, then project
        pooled = patterns.mean(dim=-1).reshape(B, H, N)  # (B, H, N)
        flat = pooled.reshape(B, H * N)  # (B, H*N)
        a_tok = self.attn_pool(flat).unsqueeze(1).expand(B, H, -1) + self.pos[:, :H, :]  # (B, H, d)
        s_tok = self.sig_proj(sig_feat).unsqueeze(1) + self.sig_tok  # (B, 1, d)
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, s_tok, a_tok], dim=1)  # (B, 1+1+H, d)
        for layer in self.layers: x = layer(x)
        return self.err_head(x[:, 0]).squeeze(-1)

# ----- TRAIN BASE -----
print("[3/5] Training Base Transformer...")
base = BaseTrans().to(GPU)
opt_b = torch.optim.AdamW(base.parameters(), lr=1e-3, weight_decay=1e-4)
sch_b = torch.optim.lr_scheduler.CosineAnnealingLR(opt_b, BASE_EP)
for ep in range(BASE_EP):
    base.train(); tl = 0
    for x, y, _ in tr_ld:
        pred = base(x.to(GPU)); loss = F.huber_loss(pred, y.to(GPU), delta=1.0)
        opt_b.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(base.parameters(), 2.0); opt_b.step(); tl += loss.item()
    sch_b.step()
    if (ep+1) % 15 == 0: print(f"  Epoch {ep+1:3d} | Loss: {tl/len(tr_ld):.6f}")

base.eval()
with torch.no_grad():
    preds, ys = [], []
    for x, y, _ in te_ld: preds.append(base(x.to(GPU)).cpu().numpy()); ys.append(y.numpy())
    preds = np.concatenate(preds); ys = np.concatenate(ys)
base_mse = float(np.mean((preds-ys)**2)); base_err = np.abs(preds-ys).mean(axis=1)
base_abs = np.abs(preds).mean(axis=1)
print(f"  Base MSE: {base_mse:.6f} | |Pred| r: {spearmanr(base_abs, base_err)[0]:.4f}")

# ----- EXTRACT PATTERNS + TRAIN UPPER -----
print("[4/5] Extracting patterns + training Upper Transformer...")
base.eval()
all_pats, all_errs, all_sig = [], [], []
with torch.no_grad():
    for x, y, s in tr_ld:
        pred, pats = base(x.to(GPU), get_pat=True)
        stacked = torch.cat([pats[i].cpu() for i in range(N_LAYERS)], dim=1)
        all_pats.append(stacked.numpy()); all_sig.append(s.numpy())
        all_errs.append((pred - y.to(GPU)).abs().mean(1).cpu().numpy())
p_train = np.concatenate(all_pats); e_train = np.concatenate(all_errs); s_train = np.concatenate(all_sig)

total_heads = N_LAYERS * N_HEADS
attn_dim = total_heads * n_assets  # pooled: H*N instead of H*N*N

upper = UpperTrans(total_heads, attn_dim, sig_dim).to(GPU)
opt_u = torch.optim.AdamW(upper.parameters(), lr=3e-3, weight_decay=1e-5)
sch_u = torch.optim.lr_scheduler.CosineAnnealingLR(opt_u, UPPER_EP)

for ep in range(UPPER_EP):
    upper.train(); tl = 0; nb = 0
    for i in range(0, len(p_train), BATCH):
        pb = torch.FloatTensor(p_train[i:i+BATCH]).to(GPU)
        sb = torch.FloatTensor(s_train[i:i+BATCH]).to(GPU)
        eb = torch.FloatTensor(e_train[i:i+BATCH]).to(GPU)
        pred_e = upper(pb, sb)
        loss = F.mse_loss(pred_e, eb)
        opt_u.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(upper.parameters(), 2.0); opt_u.step(); tl += loss.item(); nb += 1
    sch_u.step()
    if (ep+1) % 10 == 0:
        # Check prediction variance
        with torch.no_grad():
            pb_test = torch.FloatTensor(p_train[:256]).to(GPU)
            sb_test = torch.FloatTensor(s_train[:256]).to(GPU)
            pe_test = upper(pb_test, sb_test)
            std_out = pe_test.std().item()
        print(f"  Epoch {ep+1:3d} | Loss: {tl/nb:.6f} | std: {std_out:.6f}")

# ----- EVALUATE -----
print("[5/5] Evaluation...")
base.eval(); upper.eval()
all_up, all_true, all_abs = [], [], []
with torch.no_grad():
    for x, y, s in te_ld:
        pred, pats = base(x.to(GPU), get_pat=True)
        stacked = torch.cat([pats[i].cpu() for i in range(N_LAYERS)], dim=1)
        up_err = upper(stacked.to(GPU), s.to(GPU)).cpu().numpy()
        true_err = (pred - y.to(GPU)).abs().mean(1).cpu().numpy()
        all_up.append(up_err); all_true.append(true_err)
        all_abs.append(np.abs(pred.cpu().numpy()).mean(axis=1))
up_err = np.concatenate(all_up); true_err = np.concatenate(all_true); abs_bl = np.concatenate(all_abs)

# Sig-only baseline
class SigMLP(nn.Module):
    def __init__(self, in_d, d=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_d, d*4), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(d*4, d*2), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(d*2, d), nn.GELU(),
            nn.Linear(d, 1))
    def forward(self, x): return self.net(x).squeeze(-1)

sig_model = SigMLP(sig_dim).to(GPU)
opt_s = torch.optim.AdamW(sig_model.parameters(), lr=1e-3, weight_decay=1e-4)
for ep in range(50):
    sig_model.train()
    for i in range(0, len(s_train), BATCH):
        sb = torch.FloatTensor(s_train[i:i+BATCH]).to(GPU)
        eb = torch.FloatTensor(e_train[i:i+BATCH]).to(GPU)
        loss = F.mse_loss(sig_model(sb), eb)
        opt_s.zero_grad(); loss.backward(); opt_s.step()
sig_model.eval()
all_sig_pred = []
with torch.no_grad():
    for _, _, s in te_ld: all_sig_pred.append(sig_model(s.to(GPU)).cpu().numpy())
sig_only = np.concatenate(all_sig_pred)

# Attention-only baseline (no sig)
class AttnOnlyMLP(nn.Module):
    def __init__(self, in_d, d=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_d, d*2), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(d*2, d), nn.GELU(),
            nn.Linear(d, 1))
    def forward(self, x): return self.net(x).squeeze(-1)

# Pool patterns: mean over last dimension → (N, H*N)
attn_train = p_train.mean(axis=-1).reshape(p_train.shape[0], -1)

attn_te = []
with torch.no_grad():
    for x, y, s in te_ld:
        _, pats = base(x.to(GPU), get_pat=True)
        stacked = torch.cat([pats[i].cpu() for i in range(N_LAYERS)], dim=1)
        pooled = stacked.mean(dim=-1).reshape(stacked.shape[0], -1)
        attn_te.append(pooled.numpy())
attn_te = np.concatenate(attn_te)

attn_model = AttnOnlyMLP(attn_train.shape[1]).to(GPU)
opt_a = torch.optim.AdamW(attn_model.parameters(), lr=1e-3)
for ep in range(50):
    attn_model.train()
    for i in range(0, len(attn_train), BATCH):
        ab = torch.FloatTensor(attn_train[i:i+BATCH]).to(GPU)
        eb = torch.FloatTensor(e_train[i:i+BATCH]).to(GPU)
        loss = F.mse_loss(attn_model(ab), eb)
        opt_a.zero_grad(); loss.backward(); opt_a.step()
attn_model.eval()
with torch.no_grad():
    attn_only = attn_model(torch.FloatTensor(attn_te).to(GPU)).cpu().numpy()

def sr(a, b): return float(spearmanr(a, b)[0])

print("=" * 60)
print("  FINAL RESULTS")
print("=" * 60)
print(f"  Stocks: {n_assets} | Sig dim: {sig_dim:,} | GPU: {torch.xpu.get_device_name(0)}")
print(f"  Base Transformer MSE:        {base_mse:.6f}")
print(f"  |Prediction| Baseline r:     {sr(abs_bl, true_err):.4f}")
print(f"  Signature-only MLP r:        {sr(sig_only, true_err):.4f}")
print(f"  Attention-only MLP r:        {sr(attn_only, true_err):.4f}")
print(f"  Upper Trans + Sig + Attn r:  {sr(up_err, true_err):.4f}")
print(f"  Upper vs |Pred|:             {sr(up_err, true_err) - sr(abs_bl, true_err):+.4f}")
print(f"  Upper vs Sig-only:           {sr(up_err, true_err) - sr(sig_only, true_err):+.4f}")
print(f"  Upper vs Attn-only:          {sr(up_err, true_err) - sr(attn_only, true_err):+.4f}")
print("=" * 60)
