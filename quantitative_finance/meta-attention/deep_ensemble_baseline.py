"""
Deep Ensemble baseline vs Meta-Attention.
Trains N=5 independent base models, uses ensemble variance as uncertainty,
compares directly to Meta-Attention on the same test set.
"""
import numpy as np, pandas as pd, time, warnings
warnings.filterwarnings('ignore')
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEEDS = [42, 123, 456, 789, 1024]  # 5 ensemble members
DEV = 'xpu'; B = 64; L = 30
print(f"Device: {DEV} | Ensemble size: {len(SEEDS)}")
print("="*60)

# Load CN A-share data
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
print(f"CN A-Share: {N} stocks | Train:{len(Xt)} Test:{len(Xe)}")

# Models
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

class MetaPred(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc=nn.Sequential(nn.Linear(900,256),nn.LayerNorm(256),nn.GELU(),nn.Dropout(.1),nn.Linear(256,32))
        self.ha=nn.MultiheadAttention(32,1,batch_first=True,dropout=.1)
        self.pred=nn.Sequential(nn.Linear(32,64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,16),nn.GELU(),nn.Dropout(.1),nn.Linear(16,1),nn.Softplus())
    def forward(self,p):
        B,H,S,_=p.shape; e=self.enc(p.reshape(B,H,S*S).view(B*H,S*S)).view(B,H,-1)
        ao,aw=self.ha(e,e,e); return self.pred((ao*aw.mean(1).unsqueeze(-1)).sum(1)).squeeze(-1)

# ============================================================
# APPROACH 1: Deep Ensemble (N=5 models)
# ============================================================
print("\n=== APPROACH 1: Deep Ensemble (N=5) ===")
ensemble_preds = []
t0 = time.time()

for i, seed in enumerate(SEEDS):
    print(f"  Training member {i+1}/{len(SEEDS)} (seed={seed})...", end=' ', flush=True)
    np.random.seed(seed); torch.manual_seed(seed)
    base = Base(N).to(DEV)
    opt = torch.optim.AdamW(base.parameters(), lr=3e-4, weight_decay=1e-5)
    crit = nn.HuberLoss(delta=1.0)
    for ep in range(120):
        base.train()
        for bx, by in ld: opt.zero_grad(); l=crit(base(bx,False)[0],by); l.backward(); torch.nn.utils.clip_grad_norm_(base.parameters(),2.0); opt.step()
    base.eval()
    with torch.no_grad():
        preds, _ = base(Xe, False)
        ensemble_preds.append(preds.cpu().numpy())
    print(f"done ({time.time()-t0:.0f}s)")

# Stack: (N_members, N_samples, N_stocks)
ensemble_stack = np.stack(ensemble_preds, axis=0)  # (5, 750, 2308)

# Ensemble uncertainty: per-sample std across members, averaged over stocks
ensemble_mean = ensemble_stack.mean(axis=0)  # (750, 2308)
ensemble_std = ensemble_stack.std(axis=0)     # (750, 2308)
# Aggregate: mean std across stocks
ensemble_uncertainty = ensemble_std.mean(axis=1)  # (750,)

# Actual errors of ensemble mean
actual_err_de = np.abs(ensemble_mean - ye.cpu().numpy()).mean(axis=1)  # (750,)

r_de, p_de = spearmanr(ensemble_uncertainty, actual_err_de)
print(f"  Ensemble uncertainty vs actual error: r={r_de:+.4f} p={p_de:.4f}")

dt_de = time.time() - t0
print(f"  Total Deep Ensemble time: {dt_de:.0f}s")

# ============================================================
# APPROACH 2: Meta-Attention (single base + pattern predictor)
# ============================================================
print(f"\n=== APPROACH 2: Meta-Attention ===")
t1 = time.time()
np.random.seed(42); torch.manual_seed(42)

base_m = Base(N).to(DEV)
opt_m = torch.optim.AdamW(base_m.parameters(), lr=3e-4, weight_decay=1e-5)
crit = nn.HuberLoss(delta=1.0)
print("  Training base...")
for ep in range(120):
    base_m.train()
    for bx, by in ld: opt_m.zero_grad(); l=crit(base_m(bx,False)[0],by); l.backward(); torch.nn.utils.clip_grad_norm_(base_m.parameters(),2.0); opt_m.step()

base_m.eval()
with torch.no_grad():
    _, ht = base_m(Xt, True); pt = base_m.get_pats()['3'].clone()
    _, he = base_m(Xe, True); pp = base_m.get_pats()['3'].clone()
    rt, _ = base_m(Xt, False); re_meta, _ = base_m(Xe, False)
at = torch.abs(rt-yt).mean(-1); ae_meta = torch.abs(re_meta-ye).mean(-1)

vn = len(pt)//5; f_t,f_v = pt[:-vn],pt[-vn:]; e_t,e_v = at[:-vn],at[-vn:]
meta = MetaPred().to(DEV); opt = torch.optim.AdamW(meta.parameters(),lr=1e-4,weight_decay=1e-3)
ec = nn.HuberLoss(delta=0.5); best_r, best_state = -1, None
print("  Training Meta-Attention predictor...")
for ep in range(150):
    meta.train(); opt.zero_grad(); l=ec(meta(f_t),e_t); l.backward(); torch.nn.utils.clip_grad_norm_(meta.parameters(),2.0); opt.step()
    if (ep+1)%10==0:
        meta.eval()
        with torch.no_grad(): r_v,_=spearmanr(meta(f_v).cpu().numpy(),e_v.cpu().numpy())
        if r_v>best_r: best_r=r_v; best_state = {k:v.clone() for k,v in meta.state_dict().items()}
        if r_v<best_r-0.15: break
if best_state: meta.load_state_dict(best_state)
meta.eval()
with torch.no_grad(): me = meta(pp).cpu().numpy()
r_meta, p_meta = spearmanr(me, ae_meta.cpu().numpy())
dt_meta = time.time() - t1
print(f"  Meta-Attention: r={r_meta:+.4f} p={p_meta:.4f} | Time: {dt_meta:.0f}s")

# ============================================================
# APPROACH 3: MC Dropout (single model, 10 forward passes)
# ============================================================
print(f"\n=== APPROACH 3: MC Dropout (10 passes) ===")
# Use the Meta-Attention base model, enable dropout at inference
base_m.train()  # Keep dropout active
mc_preds = []
with torch.no_grad():  # No grad needed for inference
    for _ in range(10):
        preds, _ = base_m(Xe, False)
        mc_preds.append(preds.cpu().numpy())
mc_stack = np.stack(mc_preds, axis=0)  # (10, 750, 2308)
mc_std = mc_stack.std(axis=0).mean(axis=1)  # uncertainty per sample
r_mc, p_mc = spearmanr(mc_std, actual_err_de)  # same actual error as ensemble
print(f"  MC Dropout uncertainty vs error: r={r_mc:+.4f} p={p_mc:.4f}")

# ============================================================
# APPROACH 4: Temperature Scaling (simple baseline)
# ============================================================
print(f"\n=== APPROACH 4: Prediction Magnitude as Uncertainty ===")
# Simple heuristic: |prediction| as proxy for confidence
pred_magnitude = np.abs(re_meta.cpu().numpy()).mean(axis=1)  # larger pred = more confident
r_mag, p_mag = spearmanr(pred_magnitude, ae_meta.cpu().numpy())
print(f"  |Prediction| vs error: r={r_mag:+.4f} p={p_mag:.4f}")

# ============================================================
# COMPARISON TABLE
# ============================================================
print(f"\n{'='*65}")
print("FINAL COMPARISON: All Uncertainty Estimation Methods")
print(f"{'='*65}")
print(f"  {'Method':<35} {'Spearman r':>10} {'p-value':>10} {'Time':>10}")
print(f"  {'-'*65}")
results = [
    ("Meta-Attention (pattern->error)", r_meta, p_meta, dt_meta),
    ("Deep Ensemble (N=5, variance)", r_de, p_de, dt_de),
    ("MC Dropout (10 passes)", r_mc, p_mc, 0),
    ("|Prediction| heuristic", r_mag, p_mag, 0),
]
for name, r, p, t in sorted(results, key=lambda x: x[1], reverse=True):
    print(f"  {name:<35} {r:>10.4f} {p:>10.4f} {t:>8.0f}s")

best = max(results, key=lambda x: x[1])
second = sorted(results, key=lambda x: x[1], reverse=True)[1]
print(f"\n  Best: {best[0]} (r={best[1]:.4f})")
print(f"  Meta vs 2nd best delta: {best[1]-second[1]:+.4f}")
print(f"  Meta-Attention {'BEATS' if best[0].startswith('Meta') else 'LOSES TO'} the best baseline")

# Scatter plot
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
fig.suptitle('Uncertainty vs Actual Error: Method Comparison (CN A-Share)', fontsize=13, fontweight='bold')

data = [
    (me, ae_meta.cpu().numpy(), f'Meta-Attention\nr={r_meta:.3f}'),
    (ensemble_uncertainty, actual_err_de, f'Deep Ensemble\nr={r_de:.3f}'),
    (mc_std, actual_err_de, f'MC Dropout\nr={r_mc:.3f}'),
    (pred_magnitude, ae_meta.cpu().numpy(), f'|Prediction|\nr={r_mag:.3f}'),
]
for ax, (x, y, title) in zip(axes, data):
    ax.scatter(x[:500], y[:500], alpha=0.3, s=5)
    z = np.polyfit(x, y, 1)
    xl = np.linspace(x.min(), x.max(), 100)
    ax.plot(xl, np.polyval(z, xl), 'r-', alpha=0.5)
    ax.set_xlabel('Uncertainty'); ax.set_ylabel('Actual |Error|')
    ax.set_title(title, fontsize=10)

plt.tight_layout()
plt.savefig('uncertainty_comparison.png', dpi=150, bbox_inches='tight')
print("  Saved: uncertainty_comparison.png")
