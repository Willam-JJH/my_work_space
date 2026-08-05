"""
P0: Enhanced base features — volatility, momentum beyond just returns.
Tests whether richer input improves base prediction AND Pattern incremental value.
ALL 6 GROUPS. Compares: basic (1D returns) vs enhanced (5D features).
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

def build_features(returns, windows=[5,10,20]):
    """Expand returns (N_stocks, T) → feature matrix (T, N_stocks*F)."""
    N,T=returns.shape
    feats=[returns]  # raw returns (1 feature)
    for w in windows:
        # Rolling volatility
        vol=pd.DataFrame(returns.T).rolling(w,min_periods=w//2).std().fillna(0).values.T
        feats.append(vol)
        # Momentum (cumulative return)
        mom=pd.DataFrame(returns.T).rolling(w,min_periods=w//2).apply(lambda x: np.prod(1+x)-1).fillna(0).values.T
        feats.append(mom)
    return np.concatenate(feats, axis=0)  # (F*N, T)

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
    'Forex':(ns,fx),'Crypto':(ns,crypto),'Commodities':(ns,fut),'Indices':(ns,idx)}
groups={k:v for k,v in groups.items() if len(v[1])>=10}

log("P0: ENHANCED FEATURES — Basic(1D) vs Enhanced(5D)")
log("="*60)
all_results={}

for gi,(gname,(src,tickers)) in enumerate(groups.items()):
    log(f"\n[{gi+1}/{len(groups)}] {gname} ({len(tickers)} assets)")
    sub=src[tickers].ffill().dropna(axis=0)
    R_basic=sub.values.astype(np.float32).T  # (N_stocks, T)
    R_enh=build_features(R_basic)  # (N_stocks*5, T)
    N_basic=R_basic.shape[0]; N_enh=R_enh.shape[0]
    log(f"  Features: {N_basic} → {N_enh}")

    gr={}
    for feat_name, R_use in [('Basic(1D)',R_basic),('Enhanced(5D)',R_enh)]:
        N_cur=R_use.shape[0]; N_out=N_basic  # always predict original stock returns
        R_norm=np.nan_to_num((R_use-np.nanmean(R_use,1,keepdims=True))/(np.nanstd(R_use,1,keepdims=True)+1e-8),0).T
        n=len(R_norm)-L-1
        X=np.lib.stride_tricks.sliding_window_view(R_norm,L,axis=0)[:n].transpose(0,2,1).astype(np.float32)
        y=R_norm[L+1:][:n,:N_out].astype(np.float32)
        tr=int(n*0.7)
        X_tr_flat=X[:tr].reshape(tr,-1); X_te_flat=X[tr:].reshape(n-tr,-1)
        np.random.seed(SEED); torch.manual_seed(SEED)
        Xt=torch.FloatTensor(X[:tr]).to(DEV); yt=torch.FloatTensor(y[:tr]).to(DEV)
        Xe=torch.FloatTensor(X[tr:]).to(DEV); ye=torch.FloatTensor(y[tr:]).to(DEV)
        ld=DataLoader(TensorDataset(Xt,yt),batch_size=min(B,64),shuffle=True)
        class BaseVarOut(Base):
            def __init__(self): super().__init__(N_cur); self.out_head=nn.Linear(128,N_out)
            def forward(self,x,store=False):
                x=self.proj(x)+self.pe[:,:x.shape[1],:]
                for i,b in enumerate(self.blocks): x=b(x,store=store,nm=str(i))
                return self.out_head(x[:,-1,:]),x
        base=BaseVarOut().to(DEV); opt=torch.optim.AdamW(base.parameters(),lr=3e-4,weight_decay=1e-5)
        for ep in range(60):
            base.train()
            for bx,by in ld: opt.zero_grad(); l=nn.HuberLoss(delta=1.0)(base(bx,False)[0],by); l.backward(); torch.nn.utils.clip_grad_norm_(base.parameters(),2.0); opt.step()
        base.eval()
        with torch.no_grad():
            rt,_=base(Xt,False); re,_=base(Xe,False)
            _,ht=base(Xt,True); pt=base.get_pats()['3'].cpu().numpy()
            _,he=base(Xe,True); p_te=base.get_pats()['3'].cpu().numpy()
        at=torch.abs(rt-yt).mean(-1).cpu().numpy(); ae=torch.abs(re-ye).mean(-1).cpu().numpy()
        abs_re=torch.abs(re).mean(-1).cpu().numpy()
        r_pred,_=spearmanr(abs_re, ae)
        p_tr_flat=pt.reshape(len(pt),-1); p_te_flat=p_te.reshape(len(p_te),-1)
        pca_r=PCA(16, random_state=SEED)
        r2_recon=r2_score(p_te_flat, pca_r.inverse_transform(Ridge(1.0).fit(X_tr_flat, pca_r.fit_transform(p_tr_flat)).predict(X_te_flat)))
        pca_e=PCA(64, random_state=SEED)
        ptr_e=pca_e.fit_transform(p_tr_flat); pte_e=pca_e.transform(p_te_flat)
        s1,s2,a1,a2=train_test_split(ptr_e, at, test_size=0.2, random_state=SEED)
        r_pat,_=spearmanr(Ridge(1.0).fit(s1,a1).predict(pte_e), ae)
        pred_m=Ridge(1.0).fit(torch.abs(rt).mean(-1).cpu().numpy().reshape(-1,1), at)
        resid=ae-pred_m.predict(abs_re.reshape(-1,1))
        s1,s2,r1,r2=train_test_split(pte_e, resid, test_size=0.2, random_state=SEED)
        r_resid,_=spearmanr(Ridge(1.0).fit(s1,r1).predict(pte_e), resid)
        del base; torch.xpu.empty_cache()
        gr[feat_name]={'|Pred|':r_pred,'Recon':r2_recon,'Pat→Err':r_pat,'Resid':r_resid}
        log(f"  {feat_name:<15} |Pred|={r_pred:+.3f} Recon={r2_recon:+.3f} Pat→Err={r_pat:+.3f} Resid={r_resid:+.3f}")

    all_results[gname]=gr
    for m in ['|Pred|','Pat→Err','Resid']:
        d=gr['Enhanced(5D)'][m]-gr['Basic(1D)'][m]
        better='▲ Enhanced' if d>0.02 else ('▼ Basic' if d<-0.02 else '≈ Tie')

log(f"\n{'='*70}")
log("ENHANCED vs BASIC — ALL 6 GROUPS")
log(f"{'='*70}")
for metric in ['|Pred|','Pat→Err','Resid']:
    log(f"\n  {metric}:")
    for gname in all_results:
        b=all_results[gname]['Basic(1D)'][metric]; e=all_results[gname]['Enhanced(5D)'][metric]
        d=e-b; better='▲' if d>0.02 else ('▼' if d<-0.02 else '≈')
        log(f"    {gname:<16} Basic={b:+.3f}  Enhanced={e:+.3f}  Δ={d:+.3f}  {better}")
