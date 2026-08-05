"""
Layer-2 Attention: |Prediction|-gated Pattern Deviation.
Compute Pattern deviation from rolling baseline, gate by |Prediction|.
Tests whether "how much the model's thinking changed" predicts error.
ALL 6 GROUPS, full data.
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

DEV='xpu' if hasattr(torch,'xpu') and torch.xpu.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
B=128; L=30; SEED=42; BASELINE_WINDOW=20
print(f"Device: {DEV}", flush=True)

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
if 'trddt' in cn_df.columns:
    cn=cn_df.pivot(index='trddt',columns='stkcd',values='dretwd').dropna(axis=1,thresh=int(len(cn_df['trddt'].unique())*0.6)).ffill()
else:
    cn=cn_df.dropna(axis=1,thresh=int(len(cn_df)*0.6)).ffill()  # already pivoted
fx=[c for c in ns.columns if '=X' in c]; crypto=[c for c in ns.columns if '-USD' in c]
fut=[c for c in ns.columns if '=F' in c]; idx=[c for c in ns.columns if c.startswith('^')]
groups={'US Stocks':(us,list(us.columns[:300])),'CN A-Share':(cn,list(cn.columns[:300])),
    'Forex':(ns,fx),'Crypto':(ns,crypto),'Commodities':(ns,fut),'Indices':(ns,idx)}
groups={k:v for k,v in groups.items() if len(v[1])>=10}

log("LAYER-2 ATTENTION: |Prediction|-gated Pattern Deviation")
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

    # Extract patterns and predictions for ALL data (train+test)
    base.eval()
    all_pats=[]; all_preds=[]
    with torch.no_grad():
        for i in range(0,len(X),B):
            batch=torch.FloatTensor(X[i:i+B]).to(DEV)
            _,h=base(batch,True)
            all_pats.append(base.get_pats()['3'].cpu().numpy())
            all_preds.append(base(batch,False)[0].cpu().numpy())
    pats=np.concatenate(all_pats,0)  # (n, 8, 30, 30)
    preds=np.concatenate(all_preds,0)  # (n, N)
    del base; torch.xpu.empty_cache()

    # True errors
    errors=np.abs(preds-y).mean(1)  # (n,)

    # |Prediction| baseline
    abs_pred=np.abs(preds).mean(1)
    r_pred,_=spearmanr(abs_pred[tr:], errors[tr:])

    # Pattern deviation from rolling baseline
    pats_flat=pats.reshape(n,-1)  # (n, 7200)
    # Exponential moving average baseline
    baseline=np.zeros_like(pats_flat[0])
    alpha=2.0/(BASELINE_WINDOW+1)
    deviations=np.zeros(n)
    for t in range(n):
        if t>0:
            baseline=alpha*pats_flat[t-1]+(1-alpha)*baseline
        else:
            baseline=pats_flat[0]
        deviations[t]=np.linalg.norm(pats_flat[t]-baseline)  # ||P_t - P_baseline||

    # Layer-2 Attention: |Prediction| * Pattern Deviation
    # Gate: high |Pred| + high deviation → model is "confident but confused"
    layer2_signal=abs_pred*deviations
    r_layer2,_=spearmanr(layer2_signal[tr:], errors[tr:])

    # PCA on deviation patterns for richer signal
    pca=PCA(32, random_state=SEED)
    dev_tr=pca.fit_transform(pats_flat[:tr]-baseline.reshape(1,-1).repeat(tr,0))
    dev_te=pca.transform(pats_flat[tr:])
    # Ridge on deviation PCA
    s1,s2,e1,e2=train_test_split(dev_tr, errors[:tr], test_size=0.2, random_state=SEED)
    r_dev,_=spearmanr(Ridge(1.0).fit(s1,e1).predict(dev_te), errors[tr:])

    # Joint: |Pred| + deviation PCA
    joint_tr=np.concatenate([abs_pred[:tr].reshape(-1,1), dev_tr],1)
    joint_te=np.concatenate([abs_pred[tr:].reshape(-1,1), dev_te],1)
    s1,s2,e1,e2=train_test_split(StandardScaler().fit_transform(joint_tr), errors[:tr], test_size=0.2, random_state=SEED)
    r_joint,_=spearmanr(Ridge(1.0).fit(s1,e1).predict(StandardScaler().fit_transform(joint_te)), errors[tr:])

    all_results[gname]={'|Pred|':r_pred,'Layer2':r_layer2,'Dev PCA':r_dev,'Joint':r_joint}
    log(f"  |Pred|={r_pred:+.3f}  Layer2(|Pred|×Dev)={r_layer2:+.3f}  Dev PCA={r_dev:+.3f}  Joint={r_joint:+.3f}")

log(f"\n{'='*70}")
log("LAYER-2 ATTENTION — ALL 6 GROUPS")
log(f"{'='*70}")
log(f"  {'Group':<16} {'|Pred|':>8} {'Layer2':>8} {'Dev PCA':>8} {'Joint':>8} {'Winner'}")
log("-"*60)
for g,r in all_results.items():
    vals={'|Pred|':r['|Pred|'],'Layer2':r['Layer2'],'Dev PCA':r['Dev PCA'],'Joint':r['Joint']}
    w=max(vals,key=vals.get)
    log(f"  {g:<16} {r['|Pred|']:>+8.4f} {r['Layer2']:>+8.4f} {r['Dev PCA']:>+8.4f} {r['Joint']:>+8.4f}  {w}")
