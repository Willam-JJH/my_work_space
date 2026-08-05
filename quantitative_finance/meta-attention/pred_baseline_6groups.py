"""
|Prediction| baseline: ALL 6 groups, fresh base training.
|Prediction| = mean(|output|) across all assets.
"""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import spearmanr

DEV='xpu'; B=128; L=30; SEED=42

def log(*a): print(' '.join(str(x) for x in a), flush=True)

class MHA(nn.Module):
    def __init__(self,d=128,h=8,dp=.1):
        super().__init__(); self.h,self.dk=h,d//h
        self.Wqkv=nn.Linear(d,3*d,0); self.Wo=nn.Linear(d,d,0); self.drop=nn.Dropout(dp)
    def forward(self,x):
        B,S,D=x.shape; H,K=self.h,self.dk; qkv=self.Wqkv(x).view(B,S,3,H,K).permute(2,0,3,1,4)
        w=self.drop(F.softmax((qkv[0]@qkv[1].transpose(-2,-1))/K**0.5,dim=-1))
        return self.Wo((w@qkv[2]).transpose(1,2).contiguous().view(B,S,D))
class Block(nn.Module):
    def __init__(self,d,h,ff,dp):
        super().__init__(); self.attn=MHA(d,h,dp); self.n1=nn.LayerNorm(d); self.n2=nn.LayerNorm(d)
        self.ffn=nn.Sequential(nn.Linear(d,ff),nn.GELU(),nn.Dropout(dp),nn.Linear(ff,d),nn.Dropout(dp))
    def forward(self,x): return self.n2(x+self.ffn(self.n1(x+self.attn(x))))
class Base(nn.Module):
    def __init__(self,ns,d=128,h=8,nl=4,ff=256,dp=.1):
        super().__init__()
        self.proj=nn.Linear(ns,d); self.pe=nn.Parameter(torch.randn(1,L,d)*0.02)
        self.blocks=nn.ModuleList([Block(d,h,ff,dp) for _ in range(nl)])
        self.head=nn.Linear(d,ns)
    def forward(self,x):
        x=self.proj(x)+self.pe[:,:x.shape[1],:]
        for b in self.blocks: x=b(x)
        return self.head(x[:,-1,:])

# Data
us=pd.read_parquet('D:/code/data/us_market/log_returns.parquet')
ns=pd.read_parquet('D:/code/data/log_returns_nonstock.parquet')
cn_df=pd.read_parquet('D:/code/data/cn_market/daily_returns.parquet')
cn=cn_df.pivot(index='trddt',columns='stkcd',values='dretwd').dropna(axis=1,thresh=int(len(cn_df['trddt'].unique())*0.6)).ffill()
fx=[c for c in ns.columns if '=X' in c]; crypto=[c for c in ns.columns if '-USD' in c]
fut=[c for c in ns.columns if '=F' in c]; idx=[c for c in ns.columns if c.startswith('^')]
groups={
    'US Stocks':(us,list(us.columns[:300])),
    'CN A-Share':(cn,list(cn.columns[:300])),
    'Forex':(ns,fx),'Crypto':(ns,crypto),
    'Commodities':(ns,fut),'Indices':(ns,idx),
}
groups={k:v for k,v in groups.items() if len(v[1])>=10}

log("|Prediction| BASELINE — ALL 6 GROUPS")
log("="*60)

for gi,(gname,(src,tickers)) in enumerate(groups.items()):
    sub=src[tickers].ffill().dropna(axis=0); R=sub.values.astype(np.float32); N=R.shape[1]
    R=np.nan_to_num((R-np.nanmean(R,0,keepdims=True))/(np.nanstd(R,0,keepdims=True)+1e-8),0)
    n=len(R)-L-1
    X=np.lib.stride_tricks.sliding_window_view(R,L,axis=0)[:n].transpose(0,2,1).astype(np.float32)
    y=R[L+1:][:n].astype(np.float32); tr=int(n*0.7)

    np.random.seed(SEED); torch.manual_seed(SEED)
    Xt=torch.FloatTensor(X[:tr]).to(DEV); yt=torch.FloatTensor(y[:tr]).to(DEV)
    Xe=torch.FloatTensor(X[tr:]).to(DEV); ye=torch.FloatTensor(y[tr:]).to(DEV)
    ld=DataLoader(TensorDataset(Xt,yt),batch_size=B,shuffle=True)

    base=Base(N).to(DEV); opt=torch.optim.AdamW(base.parameters(),lr=3e-4,weight_decay=1e-5)
    for ep in range(60):
        base.train()
        for bx,by in ld: opt.zero_grad(); l=nn.HuberLoss(delta=1.0)(base(bx),by); l.backward(); torch.nn.utils.clip_grad_norm_(base.parameters(),2.0); opt.step()

    base.eval()
    with torch.no_grad():
        re=base(Xe); ae=torch.abs(re-ye).mean(-1).cpu().numpy()
        abs_re=torch.abs(re).mean(-1).cpu().numpy()

    r_pred,p_pred=spearmanr(abs_re, ae)
    del base; torch.xpu.empty_cache()
    log(f"  [{gi+1}/6] {gname:<16} N={N:>4}  Train={tr} Test={n-tr}  |Pred| r={r_pred:+.4f}  p={p_pred:.4f}")

log(f"\n{'='*60}")
log("|Prediction| is the simplest error predictor — mean absolute output.")
log("It requires ZERO additional model, ZERO extra training.")
