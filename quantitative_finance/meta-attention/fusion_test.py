"""
Nonlinear fusion: Can an MLP combining |Prediction| + Pattern beat |Prediction| alone?
Tests the "fusion is bottleneck" hypothesis. ALL 6 GROUPS.
"""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DEV='xpu'; B=128; L=30; SEED=42

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

class FusionMLP(nn.Module):
    """Nonlinear fusion: |Prediction| + Pattern PCA → error"""
    def __init__(self, in_dim):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(in_dim,128), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(128,64), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(64,32), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(32,1), nn.Softplus()
        )
    def forward(self,x): return self.net(x).squeeze(-1)

us=pd.read_parquet('D:/code/data/us_market/log_returns.parquet')
ns=pd.read_parquet('D:/code/data/log_returns_nonstock.parquet')
cn_df=pd.read_parquet('D:/code/data/cn_market/daily_returns.parquet')
cn=cn_df.pivot(index='trddt',columns='stkcd',values='dretwd').dropna(axis=1,thresh=int(len(cn_df['trddt'].unique())*0.6)).ffill()
fx=[c for c in ns.columns if '=X' in c]; crypto=[c for c in ns.columns if '-USD' in c]
fut=[c for c in ns.columns if '=F' in c]; idx=[c for c in ns.columns if c.startswith('^')]
groups={'US Stocks':(us,list(us.columns[:200])),'CN A-Share':(cn,list(cn.columns[:200])),
    'Forex':(ns,fx),'Crypto':(ns,crypto),'Commodities':(ns,fut),'Indices':(ns,idx)}
groups={k:v for k,v in groups.items() if len(v[1])>=10}

log("NONLINEAR FUSION: |Prediction| + Pattern → MLP → Error")
log("="*60)
all_results={}

for gi,(gname,(src,tickers)) in enumerate(groups.items()):
    log(f"\n[{gi+1}/{len(groups)}] {gname} ({len(tickers)} assets)")
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
        for bx,by in ld: opt.zero_grad(); l=nn.HuberLoss(delta=1.0)(base(bx,False)[0],by); l.backward(); torch.nn.utils.clip_grad_norm_(base.parameters(),2.0); opt.step()
    base.eval()
    with torch.no_grad():
        rt,_=base(Xt,False); re,_=base(Xe,False)
        _,ht=base(Xt,True); pt=base.get_pats()['3'].cpu().numpy()
        _,he=base(Xe,True); p_te=base.get_pats()['3'].cpu().numpy()
    at=torch.abs(rt-yt).mean(-1).cpu().numpy(); ae=torch.abs(re-ye).mean(-1).cpu().numpy()
    abs_re_tr=torch.abs(rt).mean(-1).cpu().numpy(); abs_re_te=torch.abs(re).mean(-1).cpu().numpy()
    r_pred,_=spearmanr(abs_re_te, ae)
    # Pattern PCA
    p_tr_flat=pt.reshape(len(pt),-1); p_te_flat=p_te.reshape(len(p_te),-1)
    pca=PCA(32, random_state=SEED)
    p_tr_pca=pca.fit_transform(p_tr_flat); p_te_pca=pca.transform(p_te_flat)
    # Baseline: Ridge fusion (linear)
    joint_tr_lin=np.concatenate([abs_re_tr.reshape(-1,1), p_tr_pca],1)
    joint_te_lin=np.concatenate([abs_re_te.reshape(-1,1), p_te_pca],1)
    j1,j2,jr1,jr2=train_test_split(StandardScaler().fit_transform(joint_tr_lin), at, test_size=0.2, random_state=SEED)
    r_lin,_=spearmanr(Ridge(1.0).fit(j1,jr1).predict(StandardScaler().fit_transform(joint_te_lin)), ae)
    # MLP fusion
    joint_tr_s=StandardScaler().fit_transform(joint_tr_lin); joint_te_s=StandardScaler().fit_transform(joint_te_lin)
    jt_tr=torch.FloatTensor(joint_tr_s); jt_te=torch.FloatTensor(joint_te_s); a_t=torch.FloatTensor(at)
    split=int(len(jt_tr)*0.8)
    mlp=FusionMLP(jt_tr.shape[1]).to('cpu')
    om=torch.optim.AdamW(mlp.parameters(),lr=3e-4,weight_decay=1e-3)
    best=-1; bs=None
    for ep_ in range(200):
        mlp.train(); om.zero_grad()
        l=nn.HuberLoss(delta=0.5)(mlp(jt_tr[:split]), a_t[:split]); l.backward(); om.step()
        if (ep_+1)%20==0:
            mlp.eval()
            with torch.no_grad(): rv,_=spearmanr(mlp(jt_tr[split:]).numpy(), a_t[split:].numpy())
            if rv>best: best=rv; bs={k:v.clone() for k,v in mlp.state_dict().items()}
    if bs: mlp.load_state_dict(bs)
    mlp.eval()
    with torch.no_grad(): r_mlp,_=spearmanr(mlp(jt_te).numpy(), ae)
    del base; torch.xpu.empty_cache()
    all_results[gname]={'|Pred|':r_pred,'Linear':r_lin,'MLP Fusion':r_mlp}
    log(f"  |Pred|={r_pred:+.4f}  Linear={r_lin:+.4f}  MLP Fusion={r_mlp:+.4f}  {'★ BEATS |Pred|!' if r_mlp>r_pred+0.01 else ''}")

log(f"\n{'='*60}")
log("FUSION TEST — ALL 6 GROUPS")
log(f"{'='*60}")
for g,r in all_results.items():
    best=max(r.values()); beats='★ FUSION WINS' if r['MLP Fusion']>r['|Pred|']+0.01 else ('LINEAR WINS' if r['Linear']>r['|Pred|']+0.01 else '|Pred| WINS')
    log(f"  {g:<16} |Pred|={r['|Pred|']:+.4f} Linear={r['Linear']:+.4f} MLP={r['MLP Fusion']:+.4f}  {beats}")
