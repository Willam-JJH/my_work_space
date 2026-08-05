"""
Gradient Boosting Pattern Reconstruction on all 6 groups (FIXED).
Uses PCA→HistGradientBoosting→inverse PCA approach.
"""
import numpy as np, pandas as pd, time, warnings
warnings.filterwarnings('ignore')
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor

DEV='xpu'; B=128; L=30; SEED=42
PCA_DIM = 16  # reduce 900-dim pattern to 16 PC components per head (speed-optimized)
def log(*a): print(' '.join(str(x) for x in a), flush=True)
log("Gradient Boosting Pattern Reconstruction (fixed multi-output)")

class MHA(nn.Module):
    def __init__(self,d=128,h=8,dp=.1):
        super().__init__(); self.h,self.dk=h,d//h
        self.Wqkv=nn.Linear(d,3*d,0); self.Wo=nn.Linear(d,d,0); self.drop=nn.Dropout(dp); self.pats={}
    def forward(self,x,store=True,nm='0'):
        B,S,D=x.shape; H,K=self.h,self.dk; qkv=self.Wqkv(x).view(B,S,3,H,K).permute(2,0,3,1,4)
        w=self.drop(F.softmax((qkv[0]@qkv[1].transpose(-2,-1))/K**0.5,dim=-1))
        if store: self.pats[nm]=w.detach()
        return self.Wo((w@qkv[2]).transpose(1,2).contiguous().view(B,S,D))
class Block(nn.Module):
    def __init__(self,d,h,ff,dp):
        super().__init__(); self.attn=MHA(d,h,dp); self.n1=nn.LayerNorm(d); self.n2=nn.LayerNorm(d)
        self.ffn=nn.Sequential(nn.Linear(d,ff),nn.GELU(),nn.Dropout(dp),nn.Linear(ff,d),nn.Dropout(dp))
    def forward(self,x,store=True,nm='0'): x=self.n1(x+self.attn(x,store,nm)); return self.n2(x+self.ffn(x))
class Base(nn.Module):
    def __init__(self,ns,d=128,h=8,nl=4,ff=256,dp=.1):
        super().__init__(); self.h,self.nl=h,nl
        self.proj=nn.Linear(ns,d); self.pe=nn.Parameter(torch.randn(1,L,d)*0.02)
        self.blocks=nn.ModuleList([Block(d,h,ff,dp) for _ in range(nl)])
        self.head=nn.Sequential(nn.Linear(d,d//2),nn.GELU(),nn.Linear(d//2,ns))
    def forward(self,x,store=False):
        x=self.proj(x)+self.pe[:,:x.shape[1],:]
        for i,b in enumerate(self.blocks): x=b(x,store=store,nm=str(i))
        return self.head(x[:,-1,:]),x
    def get_pats(self): return {k:v for blk in self.blocks for k,v in blk.attn.pats.items()}

# Load data
us=pd.read_parquet('D:/code/data/us_market/log_returns.parquet')
ns=pd.read_parquet('D:/code/data/log_returns_nonstock.parquet')
cn_raw=pd.read_parquet('D:/code/data/cn_market/daily_returns.parquet')
cn=cn_raw.pivot(index='trddt',columns='stkcd',values='dretwd')
cn=cn.dropna(axis=1,thresh=int(len(cn)*0.6)).ffill().dropna(axis=0)

groups={
    'US Stocks':(us,list(us.columns[:200])),
    'CN A-Share':(cn,list(cn.columns[:200])),
    'Forex':(ns,[c for c in ns.columns if '=X' in c][:50]),
    'Crypto':(ns,[c for c in ns.columns if '-USD' in c][:30]),
    'Commodities':(ns,[c for c in ns.columns if '=F' in c][:30]),
    'Indices':(ns,[c for c in ns.columns if c.startswith('^')][:30]),
}

all_r2s={}; t_total=time.time()
for gi,(gname,(src,tickers)) in enumerate(groups.items()):
    if len(tickers)<5: continue
    t0=time.time()
    log(f"\n[{'='*50}]")
    log(f"[{gi+1}/{len(groups)}] {gname} ({len(tickers)} assets)")
    log(f"[{'='*50}]")
    sub=src[tickers].ffill().dropna(axis=0)
    R=sub.values.astype(np.float32); N_stocks=R.shape[1]
    R=np.nan_to_num((R-np.nanmean(R,0,keepdims=True))/(np.nanstd(R,0,keepdims=True)+1e-8),0)
    n=len(R)-L-1
    X=np.lib.stride_tricks.sliding_window_view(R,L,axis=0)[:n].transpose(0,2,1).astype(np.float32)
    y=R[L+1:][:n].astype(np.float32); tr=int(n*0.7)
    log(f"  Train:{tr} Test:{n-tr} | {N_stocks} stocks")

    np.random.seed(SEED); torch.manual_seed(SEED)
    Xt=torch.FloatTensor(X[:tr]).to(DEV); yt=torch.FloatTensor(y[:tr]).to(DEV)
    ld=DataLoader(TensorDataset(Xt,yt),batch_size=B,shuffle=True)

    # Train base transformer
    base=Base(N_stocks).to(DEV); opt=torch.optim.AdamW(base.parameters(),lr=3e-4,weight_decay=1e-5)
    for ep in range(60):
        base.train()
        for bx,by in ld: opt.zero_grad(); l=nn.HuberLoss(delta=1.0)(base(bx,False)[0],by); l.backward(); torch.nn.utils.clip_grad_norm_(base.parameters(),2.0); opt.step()

    # Extract patterns
    base.eval()
    with torch.no_grad():
        _,ht=base(Xt,True); p_tr=base.get_pats()['3'].cpu().numpy()       # (tr, 8, 30, 30)
        _,he=base(torch.FloatTensor(X[tr:]).to(DEV),True); p_te=base.get_pats()['3'].cpu().numpy()
    del base; torch.xpu.empty_cache()

    X_tr_flat=X[:tr].reshape(tr,-1); X_te_flat=X[tr:].reshape(n-tr,-1)
    log(f"  Input dim: {X_tr_flat.shape[1]} | Pattern: {p_tr.shape}")

    # Per-head: PCA(900→{PCA_DIM}) → sequential HGBR per component → invPCA → R²
    head_r2s=[]
    for h in range(8):
        y_tr_h=p_tr[:,h].reshape(tr,-1); y_te_h=p_te[:,h].reshape(n-tr,-1)  # (n, 900)

        # PCA reduce target dimensionality
        pca=PCA(n_components=PCA_DIM, random_state=SEED)
        y_tr_pca=pca.fit_transform(y_tr_h)
        y_te_pca=pca.transform(y_te_h)

        # Sequential: one HGBR per PCA component (avoids nested parallelism deadlock)
        # HGBR uses internal OpenMP threading efficiently
        pred_pca=np.zeros_like(y_te_pca)
        for k in range(PCA_DIM):
            hgbr=HistGradientBoostingRegressor(
                max_iter=60, max_depth=4, learning_rate=0.1,
                min_samples_leaf=40, random_state=SEED
            )
            hgbr.fit(X_tr_flat, y_tr_pca[:, k])
            pred_pca[:, k]=hgbr.predict(X_te_flat)
            if k%4==0:
                log(f"    H{h} PC{k}/{PCA_DIM} done ({time.time()-t0:.0f}s)")

        # Inverse PCA & compute R²
        pred_full=pca.inverse_transform(pred_pca)
        r2=r2_score(y_te_h, pred_full)
        head_r2s.append(r2)
        log(f"  Head {h}: R²={r2:+.4f}  ({time.time()-t0:.0f}s)")

    avg=np.mean(head_r2s); all_r2s[gname]=avg
    heads_str=' | '.join([f'H{h}: {r:+.4f}' for h,r in enumerate(head_r2s)])
    log(f"  {heads_str}")
    tag='⚠ RECONSTRUCTIBLE' if avg>0.3 else ('✓ NOT reconstructible' if avg<0.1 else '◈ Marginal')
    log(f"  Avg R² = {avg:.4f} → {tag}  ({time.time()-t0:.0f}s)")

log(f"\n{'='*60}")
log("GRADIENT BOOSTING RECONSTRUCTION — ALL GROUPS")
log(f"{'='*60}")
for g,r in all_r2s.items():
    tag='⚠ WARNING' if r>0.3 else '✓ OK'
    log(f"  {g:<20} R² = {r:+.4f}  [{tag}]")
avg_all=np.mean(list(all_r2s.values())) if all_r2s else 0
log(f"\n  Overall Avg R² = {avg_all:.4f}")
if avg_all>0.3:
    log("  >>> WARNING: GB can reconstruct patterns → independence claim weakened")
elif avg_all>0.1:
    log("  >>> Marginal: GB partially reconstructs → some dependence exists")
else:
    log("  >>> Patterns NOT reconstructible via GB → independence strongly supported")
log(f"  Total time: {time.time()-t_total:.0f}s")
