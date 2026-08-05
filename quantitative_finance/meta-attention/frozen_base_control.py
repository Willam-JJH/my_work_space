"""
Frozen base control: Does Pattern info come from static forward-pass structure
or from training dynamics? Compare: frozen (pretrained) vs trainable base.
ALL 6 GROUPS.
"""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

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

class MetaPred(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc=nn.Sequential(nn.Linear(900,256),nn.LayerNorm(256),nn.GELU(),nn.Dropout(.1),nn.Linear(256,32))
        self.ha=nn.MultiheadAttention(32,1,batch_first=True,dropout=.1)
        self.pred=nn.Sequential(nn.Linear(32,64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,16),nn.GELU(),nn.Dropout(.1),nn.Linear(16,1),nn.Softplus())
    def forward(self,p):
        B,H,S,_=p.shape; e=self.enc(p.reshape(B,H,S*S).view(B*H,S*S)).view(B,H,-1)
        ao,aw=self.ha(e,e,e); return self.pred((ao*aw.mean(1).unsqueeze(-1)).sum(1)).squeeze(-1)

us=pd.read_parquet('D:/code/data/us_market/log_returns.parquet')
ns=pd.read_parquet('D:/code/data/log_returns_nonstock.parquet')
cn_df=pd.read_parquet('D:/code/data/cn_market/daily_returns.parquet')
cn=cn_df.pivot(index='trddt',columns='stkcd',values='dretwd').dropna(axis=1,thresh=int(len(cn_df['trddt'].unique())*0.6)).ffill()
fx=[c for c in ns.columns if '=X' in c]; crypto=[c for c in ns.columns if '-USD' in c]
fut=[c for c in ns.columns if '=F' in c]; idx=[c for c in ns.columns if c.startswith('^')]
groups={'US Stocks':(us,list(us.columns[:200])),'CN A-Share':(cn,list(cn.columns[:200])),
    'Forex':(ns,fx[:50]),'Crypto':(ns,crypto[:30]),'Commodities':(ns,fut[:30]),'Indices':(ns,idx[:30])}
groups={k:v for k,v in groups.items() if len(v[1])>=10}

log("Frozen Base Control: Static structure vs Training dynamics")
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

    # Train base
    base=Base(N).to(DEV); opt=torch.optim.AdamW(base.parameters(),lr=3e-4,weight_decay=1e-5)
    for ep in range(60):
        base.train()
        for bx,by in ld: opt.zero_grad(); l=nn.HuberLoss(delta=1.0)(base(bx,False)[0],by); l.backward(); torch.nn.utils.clip_grad_norm_(base.parameters(),2.0); opt.step()
    base.eval()
    with torch.no_grad():
        rt,_=base(Xt,False); re,_=base(Xe,False)
        _,ht=base(Xt,True); pt=base.get_pats()['3'].clone()
        _,he=base(Xe,True); p_te=base.get_pats()['3'].clone()
    at=torch.abs(rt-yt).mean(-1); ae=torch.abs(re-ye).mean(-1)
    abs_re_te=torch.abs(re).mean(-1).cpu().numpy()
    r_pred,_=spearmanr(abs_re_te, ae.cpu().numpy())

    # Condition A: Trainable base (baseline) - Meta trained on patterns from trained base
    meta_trainable=MetaPred().to(DEV)
    oa=torch.optim.AdamW(meta_trainable.parameters(),lr=1e-4,weight_decay=1e-3)
    vn=len(pt)//5; p_tr,p_v=pt[:-vn],pt[-vn:]; e_tr,e_v=at[:-vn],at[-vn:]
    best=-1; bs=None
    for ep in range(50):
        meta_trainable.train(); oa.zero_grad()
        lo=nn.HuberLoss(delta=0.5)(meta_trainable(p_tr),e_tr); lo.backward(); torch.nn.utils.clip_grad_norm_(meta_trainable.parameters(),2.0); oa.step()
        if (ep+1)%10==0:
            meta_trainable.eval()
            with torch.no_grad(): rv,_=spearmanr(meta_trainable(p_v).cpu().numpy(),e_v.cpu().numpy())
            if rv>best: best=rv; bs={k:v.clone() for k,v in meta_trainable.state_dict().items()}
    if bs: meta_trainable.load_state_dict(bs)
    with torch.no_grad(): r_trainable,_=spearmanr(meta_trainable(p_te).cpu().numpy(), ae.cpu().numpy())

    # Condition B: Frozen base → patterns from fixed forward pass
    base.eval()
    for param in base.parameters(): param.requires_grad=False
    meta_frozen=MetaPred().to(DEV)
    of_=torch.optim.AdamW(meta_frozen.parameters(),lr=1e-4,weight_decay=1e-3)
    best2=-1; bs2=None
    for ep in range(50):
        meta_frozen.train(); of_.zero_grad()
        lo=nn.HuberLoss(delta=0.5)(meta_frozen(p_tr),e_tr); lo.backward(); torch.nn.utils.clip_grad_norm_(meta_frozen.parameters(),2.0); of_.step()
        if (ep+1)%10==0:
            meta_frozen.eval()
            with torch.no_grad(): rv,_=spearmanr(meta_frozen(p_v).cpu().numpy(),e_v.cpu().numpy())
            if rv>best2: best2=rv; bs2={k:v.clone() for k,v in meta_frozen.state_dict().items()}
    if bs2: meta_frozen.load_state_dict(bs2)
    with torch.no_grad(): r_frozen,_=spearmanr(meta_frozen(p_te).cpu().numpy(), ae.cpu().numpy())

    del base; torch.xpu.empty_cache()
    delta=r_trainable-r_frozen
    log(f"  |Pred|={r_pred:+.4f} | Trainable Meta={r_trainable:+.4f} | Frozen Meta={r_frozen:+.4f} | Δ={delta:+.4f}")
    all_results[gname]={'|Pred|':r_pred,'trainable':r_trainable,'frozen':r_frozen,'Δ':delta}

log(f"\n{'='*60}")
log("FROZEN vs TRAINABLE BASE — ALL 6 GROUPS")
log(f"{'='*60}")
log(f"  {'Group':<16} {'|Pred|':>8} {'Trainable':>10} {'Frozen':>10} {'Δ':>8} {'Verdict'}")
log("-"*60)
for g,r in all_results.items():
    v='STATIC OK' if abs(r['Δ'])<0.02 else ('TRAINING MATTERS' if r['Δ']>0.02 else 'FROZEN BETTER')
    log(f"  {g:<16} {r['|Pred|']:>+8.4f} {r['trainable']:>+10.4f} {r['frozen']:>+10.4f} {r['Δ']:>+8.4f} {v}")
