"""
Residualization test: Does Pattern have incremental info BEYOND |Prediction|?
Regress error on |Pred|, take residuals, test if Pattern can predict residuals.
If yes → Pattern adds value. If no → Pattern is redundant with |Pred|.
ALL 6 GROUPS.
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

us=pd.read_parquet('D:/code/data/us_market/log_returns.parquet')
ns=pd.read_parquet('D:/code/data/log_returns_nonstock.parquet')
cn_df=pd.read_parquet('D:/code/data/cn_market/daily_returns.parquet')
cn=cn_df.pivot(index='trddt',columns='stkcd',values='dretwd').dropna(axis=1,thresh=int(len(cn_df['trddt'].unique())*0.6)).ffill()
fx=[c for c in ns.columns if '=X' in c]; crypto=[c for c in ns.columns if '-USD' in c]
fut=[c for c in ns.columns if '=F' in c]; idx=[c for c in ns.columns if c.startswith('^')]
groups={'US Stocks':(us,list(us.columns[:200])),'CN A-Share':(cn,list(cn.columns[:200])),
    'Forex':(ns,fx[:50]),'Crypto':(ns,crypto[:30]),'Commodities':(ns,fut[:30]),'Indices':(ns,idx[:30])}
groups={k:v for k,v in groups.items() if len(v[1])>=10}

log("Residualization Test: Pattern info beyond |Prediction|")
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
    del base; torch.xpu.empty_cache()

    # Step 1: |Prediction| → error (baseline)
    r_pred,_=spearmanr(abs_re_te, ae)

    # Step 2: Regress error on |Prediction|, take residuals
    pred_model=Ridge(1.0).fit(abs_re_tr.reshape(-1,1), at)
    resid_tr=at-pred_model.predict(abs_re_tr.reshape(-1,1))
    resid_te=ae-pred_model.predict(abs_re_te.reshape(-1,1))

    # Step 3: Pattern PCA → Ridge → predict residuals
    p_tr_flat=pt.reshape(len(pt),-1); p_te_flat=p_te.reshape(len(p_te),-1)
    pca=PCA(64, random_state=SEED)
    p_tr_pca=pca.fit_transform(p_tr_flat); p_te_pca=pca.transform(p_te_flat)
    s1,s2,r1,r2=train_test_split(p_tr_pca, resid_tr, test_size=0.2, random_state=SEED)
    r_pattern_resid,_=spearmanr(Ridge(1.0).fit(s1,r1).predict(p_te_pca), resid_te)

    # Step 4: Pattern + |Pred| jointly
    joint_tr=np.concatenate([abs_re_tr.reshape(-1,1), p_tr_pca],1)
    joint_te=np.concatenate([abs_re_te.reshape(-1,1), p_te_pca],1)
    j1,j2,jr1,jr2=train_test_split(joint_tr, at, test_size=0.2, random_state=SEED)
    r_joint,_=spearmanr(Ridge(1.0).fit(j1,jr1).predict(joint_te), ae)

    inc=r_joint-r_pred
    res_info=r_pattern_resid
    log(f"  |Pred| r={r_pred:+.4f} | Pattern→resid r={res_info:+.4f} | Joint r={r_joint:+.4f} | Δ={inc:+.4f}")
    all_results[gname]={'|Pred|':r_pred,'Pattern→resid':res_info,'Joint':r_joint,'Δ':inc}

log(f"\n{'='*60}")
log("RESIDUALIZATION — ALL 6 GROUPS")
log(f"{'='*60}")
log(f"  {'Group':<16} {'|Pred|':>8} {'Pat→resid':>10} {'Joint':>8} {'Δ':>8} {'Verdict'}")
log("-"*60)
for g,r in all_results.items():
    v='INCREMENTAL ✓' if r['Δ']>0.02 else ('MARGINAL' if r['Δ']>0.005 else 'REDUNDANT')
    log(f"  {g:<16} {r['|Pred|']:>+8.4f} {r['Pattern→resid']:>+10.4f} {r['Joint']:>+8.4f} {r['Δ']:>+8.4f} {v}")
