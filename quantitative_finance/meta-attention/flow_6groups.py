"""
Attention Flow on ALL 6 groups (not just CN 200).
Tests whether Flow's "breakthrough" generalizes.
"""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import spearmanr
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

log("Attention Flow — ALL 6 GROUPS")
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
        _,ht=base(Xt,True); pt_all=base.get_pats()
        _,he=base(Xe,True); pte_all=base.get_pats()
    at=torch.abs(rt-yt).mean(-1).cpu().numpy(); ae=torch.abs(re-ye).mean(-1).cpu().numpy()
    abs_re_te=torch.abs(re).mean(-1).cpu().numpy()
    r_pred,_=spearmanr(abs_re_te, ae)
    del base; torch.xpu.empty_cache()

    # Compute Flow: |Δ| per layer transition per head
    flow_tr=[]; flow_te=[]
    for i in range(3):
        d_tr=pt_all[str(i+1)].cpu().numpy()-pt_all[str(i)].cpu().numpy()
        d_te=pte_all[str(i+1)].cpu().numpy()-pte_all[str(i)].cpu().numpy()
        flow_tr.append(np.linalg.norm(d_tr.reshape(len(d_tr),8,-1),axis=2))
        flow_te.append(np.linalg.norm(d_te.reshape(len(d_te),8,-1),axis=2))
    flow_tr=np.concatenate(flow_tr,1); flow_te=np.concatenate(flow_te,1)

    # Base Flow + head interactions
    flow_3d_tr=flow_tr.reshape(len(flow_tr),3,8); flow_3d_te=flow_te.reshape(len(flow_te),3,8)
    pair_tr=[]; pair_te=[]
    for d in range(3):
        for h1 in range(8):
            for h2 in range(h1+1,8):
                pair_tr.append(flow_3d_tr[:,d,h1]*flow_3d_tr[:,d,h2])
                pair_te.append(flow_3d_te[:,d,h1]*flow_3d_te[:,d,h2])
    interact_tr=np.concatenate([flow_tr, np.stack(pair_tr,1)],1)
    interact_te=np.concatenate([flow_te, np.stack(pair_te,1)],1)

    # Train & evaluate
    is_tr=StandardScaler().fit_transform(interact_tr); is_te=StandardScaler().fit_transform(interact_te)
    s1,s2,a1,a2=train_test_split(is_tr, at, test_size=0.2, random_state=SEED)
    r_flow,_=spearmanr(Ridge(1.0).fit(s1,a1).predict(is_te), ae)
    gap=r_flow-r_pred

    log(f"  |Pred|={r_pred:+.4f} | Flow+Interact={r_flow:+.4f} | Δ={gap:+.4f}")
    all_results[gname]={'|Pred|':r_pred,'Flow':r_flow,'Δ':gap}

log(f"\n{'='*60}")
log("ATTENTION FLOW — ALL 6 GROUPS")
log(f"{'='*60}")
log(f"  {'Group':<16} {'|Pred|':>8} {'Flow+Int':>10} {'Δ':>8} {'Verdict'}")
log("-"*60)
for g,r in all_results.items():
    v='BEATS |Pred| ✓' if r['Δ']>0.02 else ('CLOSE' if r['Δ']>-0.03 else 'BEHIND')
    log(f"  {g:<16} {r['|Pred|']:>+8.4f} {r['Flow']:>+10.4f} {r['Δ']:>+8.4f} {v}")
