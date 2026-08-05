"""
Random init vs Trained: Pattern reconstruction gap.
50 random seeds vs 1 trained seed, ALL 6 groups.
Hypothesis: Trained patterns are LESS reconstructible (Pattern info is learned).
"""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

DEV='xpu'; B=128; L=30; SEED=42; N_SEEDS=50

def log(*a): print(' '.join(str(x) for x in a), flush=True)

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

# Data
us=pd.read_parquet('D:/code/data/us_market/log_returns.parquet')
ns=pd.read_parquet('D:/code/data/log_returns_nonstock.parquet')
cn_df=pd.read_parquet('D:/code/data/cn_market/daily_returns.parquet')
cn=cn_df.pivot(index='trddt',columns='stkcd',values='dretwd').dropna(axis=1,thresh=int(len(cn_df['trddt'].unique())*0.6)).ffill()
fx=[c for c in ns.columns if '=X' in c]; crypto=[c for c in ns.columns if '-USD' in c]
fut=[c for c in ns.columns if '=F' in c]; idx=[c for c in ns.columns if c.startswith('^')]
groups={
    'US Stocks':(us,list(us.columns[:200])),
    'CN A-Share':(cn,list(cn.columns[:200])),
    'Forex':(ns,fx[:50]),'Crypto':(ns,crypto[:30]),
    'Commodities':(ns,fut[:30]),'Indices':(ns,idx[:30]),
}
groups={k:v for k,v in groups.items() if len(v[1])>=10}

all_results={}
for gi,(gname,(src,tickers)) in enumerate(groups.items()):
    log(f"\n[{'='*50}]")
    log(f"[{gi+1}/{len(groups)}] {gname} ({len(tickers)} assets)")
    log(f"[{'='*50}]")
    sub=src[tickers].ffill().dropna(axis=0)
    R=sub.values.astype(np.float32); N=R.shape[1]
    R=np.nan_to_num((R-np.nanmean(R,0,keepdims=True))/(np.nanstd(R,0,keepdims=True)+1e-8),0)
    n=len(R)-L-1
    X=np.lib.stride_tricks.sliding_window_view(R,L,axis=0)[:n].transpose(0,2,1).astype(np.float32)
    y=R[L+1:][:n].astype(np.float32); tr=int(n*0.7)
    Xt_t=torch.FloatTensor(X[:tr]).to(DEV); Xe_t=torch.FloatTensor(X[tr:]).to(DEV)
    yt_t=torch.FloatTensor(y[:tr]).to(DEV)
    X_tr_flat=X[:tr].reshape(tr,-1); X_te_flat=X[tr:].reshape(n-tr,-1)

    # 50 random seeds
    recon_rand=[]
    for si in range(N_SEEDS):
        seed=100+si; np.random.seed(seed); torch.manual_seed(seed)
        base_r=Base(N).to(DEV); base_r.eval()
        with torch.no_grad():
            _,ht_r=base_r(Xt_t,True); pt_r=base_r.get_pats()['3'].cpu().numpy()
            _,he_r=base_r(Xe_t,True); p_te_r=base_r.get_pats()['3'].cpu().numpy()
        p_tr_r=pt_r.reshape(len(pt_r),-1); p_te_rf=p_te_r.reshape(len(p_te_r),-1)
        pca_r=PCA(16, random_state=SEED)
        p_tr_pca=pca_r.fit_transform(p_tr_r); p_te_pca=pca_r.transform(p_te_rf)
        # Train Ridge on train, evaluate on test
        ridge_r=Ridge(1.0).fit(X_tr_flat, p_tr_pca)
        pred_te=pca_r.inverse_transform(ridge_r.predict(X_te_flat))
        recon_rand.append(r2_score(p_te_rf, pred_te))
        del base_r
        if (si+1)%10==0: log(f"  Random {si+1}/{N_SEEDS} done")

    # Trained base (60 epochs)
    np.random.seed(SEED); torch.manual_seed(SEED)
    ld=DataLoader(TensorDataset(Xt_t,yt_t),batch_size=B,shuffle=True)
    base_t=Base(N).to(DEV); opt=torch.optim.AdamW(base_t.parameters(),lr=3e-4,weight_decay=1e-5)
    for ep in range(60):
        base_t.train()
        for bx,by in ld: opt.zero_grad(); l=nn.HuberLoss(delta=1.0)(base_t(bx,False)[0],by); l.backward(); torch.nn.utils.clip_grad_norm_(base_t.parameters(),2.0); opt.step()
    base_t.eval()
    with torch.no_grad():
        _,ht_t=base_t(Xt_t,True); pt_t=base_t.get_pats()['3'].cpu().numpy()
        _,he_t=base_t(Xe_t,True); p_te_t=base_t.get_pats()['3'].cpu().numpy()
    p_tr_tf=pt_t.reshape(len(pt_t),-1); p_te_tf=p_te_t.reshape(len(p_te_t),-1)
    pca_t=PCA(16, random_state=SEED)
    p_tr_t_pca=pca_t.fit_transform(p_tr_tf); p_te_t_pca=pca_t.transform(p_te_tf)
    ridge_t=Ridge(1.0).fit(X_tr_flat, p_tr_t_pca)
    r2_trained=r2_score(p_te_tf, pca_t.inverse_transform(ridge_t.predict(X_te_flat)))
    del base_t; torch.xpu.empty_cache()

    r_mean=np.mean(recon_rand); r_std=np.std(recon_rand)
    log(f"  Random (50s): Recon R²={r_mean:+.4f}±{r_std:.4f}  Trained: {r2_trained:+.4f}  Δ={r_mean-r2_trained:+.4f}")
    all_results[gname]={'random_mean':r_mean,'random_std':r_std,'trained':r2_trained}

log(f"\n{'='*70}")
log("RANDOM vs TRAINED — ALL 6 GROUPS (50 seeds)")
log(f"{'='*70}")
log(f"  {'Group':<16} {'Random':>18} {'Trained':>10} {'Δ':>10} {'Verdict'}")
log("-"*70)
for g,r in all_results.items():
    d=r['random_mean']-r['trained']
    v='LEARNED ✓' if d>0.01 else ('NO DIFF' if abs(d)<0.01 else 'REVERSE ✗')
    log(f"  {g:<16} {r['random_mean']:>+8.4f}±{r['random_std']:.3f} {r['trained']:>+10.4f} {d:>+10.4f} {v}")
