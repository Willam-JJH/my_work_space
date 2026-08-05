"""
MLP reconstruction test: Can a nonlinear MLP predict attention patterns from input data?
If yes (high R²) → Pattern is largely derivable, weakening Meta-Attention premise.
If no  (low R²)  → Pattern has genuine independent structure.
"""
import numpy as np, pandas as pd, time, warnings
warnings.filterwarnings('ignore')
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr

SEED = 42; DEV = 'xpu'; B = 256; L = 30

# Load CN A-share
df = pd.read_parquet('D:/code/data/cn_market/daily_returns.parquet')
rm = df.pivot(index='trddt', columns='stkcd', values='dretwd')
rm = rm.dropna(axis=1, thresh=int(len(rm)*0.8)).ffill().dropna(axis=0)
R = rm.values.astype(np.float32); N = R.shape[1]
R = np.nan_to_num((R - np.nanmean(R,0,keepdims=True)) / (np.nanstd(R,0,keepdims=True)+1e-8), 0)
n = len(R) - L - 1
X = np.lib.stride_tricks.sliding_window_view(R, L, axis=0)[:n].transpose(0,2,1).astype(np.float32)
y = R[L+1:][:n].astype(np.float32)
tr = int(n*0.7)
Xt = torch.FloatTensor(X[:tr]).to(DEV); yt = torch.FloatTensor(y[:tr]).to(DEV)
Xe = torch.FloatTensor(X[tr:]).to(DEV); ye = torch.FloatTensor(y[tr:]).to(DEV)
ld = DataLoader(TensorDataset(Xt, yt), batch_size=B, shuffle=True)
ld_test = DataLoader(TensorDataset(Xe, ye), batch_size=B, shuffle=False)
print(f"CN A-Share: {N} stocks | Train:{len(Xt)} Test:{len(Xe)}")

# Train base transformer
class MHA(nn.Module):
    def __init__(self,d=128,h=8,dp=.1):
        super().__init__(); self.h=h; self.dk=d//h
        self.Wqkv=nn.Linear(d,3*d,0); self.Wo=nn.Linear(d,d,0); self.drop=nn.Dropout(dp); self.pats={}
    def forward(self,x,store=True,nm='L0'):
        B,S,D=x.shape; H,K=self.h,self.dk; qkv=self.Wqkv(x).view(B,S,3,H,K).permute(2,0,3,1,4)
        w=self.drop(F.softmax((qkv[0]@qkv[1].transpose(-2,-1))/K**0.5,dim=-1))
        if store: self.pats[nm]=w.detach()
        return self.Wo((w@qkv[2]).transpose(1,2).contiguous().view(B,S,D))
class Block(nn.Module):
    def __init__(self,d,h,ff,dp):
        super().__init__(); self.attn=MHA(d,h,dp); self.n1=nn.LayerNorm(d); self.n2=nn.LayerNorm(d)
        self.ffn=nn.Sequential(nn.Linear(d,ff),nn.GELU(),nn.Dropout(dp),nn.Linear(ff,d),nn.Dropout(dp))
    def forward(self,x,store=True,nm='L0'): x=self.n1(x+self.attn(x,store,nm)); return self.n2(x+self.ffn(x))
class Base(nn.Module):
    def __init__(self,ns,d=128,h=8,nl=4,ff=256,dp=.1):
        super().__init__(); self.h=h; self.nl=nl
        self.proj=nn.Linear(ns,d); self.pe=nn.Parameter(torch.randn(1,L,d)*0.02)
        self.blocks=nn.ModuleList([Block(d,h,ff,dp) for _ in range(nl)])
        self.head=nn.Sequential(nn.Linear(d,d//2),nn.GELU(),nn.Linear(d//2,ns))
    def forward(self,x,store=False):
        x=self.proj(x)+self.pe[:,:x.shape[1],:]; [b(x,store=store,nm=str(i)) for i,b in enumerate(self.blocks)]
        return self.head(x[:,-1,:]),x
    def get_pats(self): return {k:v for blk in self.blocks for k,v in blk.attn.pats.items()}

print("Training base transformer...")
np.random.seed(SEED); torch.manual_seed(SEED)
base = Base(N).to(DEV)
opt = torch.optim.AdamW(base.parameters(), lr=3e-4, weight_decay=1e-5)
crit = nn.HuberLoss(delta=1.0); t0 = time.time()
for ep in range(120):
    base.train()
    for bx, by in ld: opt.zero_grad(); l=crit(base(bx,False)[0],by); l.backward(); torch.nn.utils.clip_grad_norm_(base.parameters(),2.0); opt.step()
print(f"  Done: {time.time()-t0:.0f}s")

# Extract patterns
print("Extracting patterns...")
base.eval()
all_patterns = []
all_inputs = []
with torch.no_grad():
    for bx, _ in ld_test:
        _ = base(bx, True)
        pat = base.get_pats()['3'].cpu().numpy()  # (B, 8, 30, 30)
        inp = bx.cpu().numpy()  # (B, 30, 2308)
        all_patterns.append(pat)
        all_inputs.append(inp)

patterns = np.concatenate(all_patterns, axis=0)  # (750, 8, 30, 30)
inputs = np.concatenate(all_inputs, axis=0)       # (750, 30, 2308)

P = patterns.shape[0]; H = patterns.shape[1]; S = patterns.shape[2]
print(f"  Patterns: {patterns.shape}")
print(f"  Inputs: {inputs.shape}")

# ============================================================
# MLP Reconstruction: X_t → A_t  (per head)
# ============================================================
# For each head, train an MLP: (B, 30*2308) → (B, 30*30)
# vs the original Ridge+PCA baseline: (B, 16) → (B, 16) in PCA space

print(f"\n{'='*60}")
print("MLP RECONSTRUCTION TEST")
print(f"{'='*60}")

# Prepare data
X_flat = inputs.reshape(P, -1)  # (750, 30*2308) = (750, 69240)
# Normalize input
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X_flat)

results = []

for h in range(H):
    A_h = patterns[:, h, :, :].reshape(P, -1)  # (750, 900)
    scaler_A = StandardScaler()
    A_scaled = scaler_A.fit_transform(A_h)

    # Split
    X_tr, X_te, A_tr, A_te = train_test_split(X_scaled, A_scaled, test_size=0.2, random_state=SEED)
    X_tr_t = torch.FloatTensor(X_tr).to(DEV); A_tr_t = torch.FloatTensor(A_tr).to(DEV)
    X_te_t = torch.FloatTensor(X_te).to(DEV); A_te_t = torch.FloatTensor(A_te).to(DEV)

    # MLP: deep enough to capture nonlinearity
    mlp = nn.Sequential(
        nn.Linear(X_flat.shape[1], 2048), nn.GELU(), nn.Dropout(0.3),
        nn.Linear(2048, 1024), nn.GELU(), nn.Dropout(0.3),
        nn.Linear(1024, 512), nn.GELU(), nn.Dropout(0.3),
        nn.Linear(512, 256), nn.GELU(),
        nn.Linear(256, A_h.shape[1]),
    ).to(DEV)

    opt = torch.optim.AdamW(mlp.parameters(), lr=1e-4, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    train_loader_mlp = DataLoader(TensorDataset(X_tr_t, A_tr_t), batch_size=128, shuffle=True)

    # Train
    best_loss = float('inf')
    for ep in range(200):
        mlp.train()
        for bx, ba in train_loader_mlp:
            opt.zero_grad(); l=loss_fn(mlp(bx), ba); l.backward(); opt.step()

    # Evaluate
    mlp.eval()
    with torch.no_grad():
        A_pred_scaled = mlp(X_te_t).cpu().numpy()
    A_pred = scaler_A.inverse_transform(A_pred_scaled)
    A_true = scaler_A.inverse_transform(A_te)

    # R² on original scale
    ss_res = np.sum((A_true - A_pred)**2)
    ss_tot = np.sum((A_true - np.mean(A_true))**2)
    r2 = 1 - ss_res / (ss_tot + 1e-10)

    results.append(r2)
    print(f"  Head {h}: MLP R² = {r2:+.4f}")

avg_mlp = np.mean(results); std_mlp = np.std(results)
print(f"\n  MLP Average R² = {avg_mlp:.4f} ± {std_mlp:.4f}")

# ============================================================
# COMPARISON: Ridge+PCA vs MLP
# ============================================================
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

pca_results = []
for h in range(H):
    A_h = patterns[:, h, :, :].reshape(P, -1)
    pA = PCA(16).fit_transform(A_h); pX = PCA(16).fit_transform(X_scaled)
    Xtr,Xte,Atr,Ate = train_test_split(pX, pA, test_size=0.2, random_state=SEED)
    pred = Ridge(1.0).fit(Xtr, Atr).predict(Xte)
    pca_results.append(r2_score(Ate, pred))

avg_pca = np.mean(pca_results); std_pca = np.std(pca_results)

print(f"\n{'='*60}")
print("Ridge+PCA vs MLP Comparison")
print(f"{'='*60}")
print(f"  {'Method':<20} {'Avg R²':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
print(f"  {'-'*60}")
print(f"  {'Ridge+PCA (16 dims)':<20} {avg_pca:>10.4f} {std_pca:>10.4f} {min(pca_results):>10.4f} {max(pca_results):>10.4f}")
print(f"  {'MLP (4 layers)':<20} {avg_mlp:>10.4f} {std_mlp:>10.4f} {min(results):>10.4f} {max(results):>10.4f}")

if avg_mlp < 0.1:
    verdict = "MLP also CANNOT reconstruct patterns → signal is genuinely independent"
elif avg_mlp > 0.5:
    verdict = "MLP CAN reconstruct patterns → Meta-Attention premise is weakened"
else:
    verdict = "MLP partially reconstructs → moderate nonlinear dependence, but substantial residual"

print(f"\n  >>> {verdict}")
print(f"  >>> Implication for Meta-Attention: {'WEAKENED' if avg_mlp > 0.3 else 'SUPPORTED' if avg_mlp < 0.1 else 'MIXED'}")
