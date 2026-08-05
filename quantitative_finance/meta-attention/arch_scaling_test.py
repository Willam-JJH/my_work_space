"""
Architecture scaling: Tiny(2L) → Simple(4L) → PatchTST(4L+patch).
Tests whether findings generalize across architecture complexity levels.
ALL data groups (300 stocks for US/CN, all available for others).
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

DEV='xpu'; B=128; L=30; SEED=42; PATCH=5

def log(*a): print(' '.join(str(x) for x in a), flush=True)

# ============================================================
# Shared modules
# ============================================================
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

# Tiny (2 layers)
class TinyBase(nn.Module):
    def __init__(self,ns,d=128,h=8,ff=256,dp=.1):
        super().__init__(); self.h,self.nl=h,2
        self.proj=nn.Linear(ns,d); self.pe=nn.Parameter(torch.randn(1,L,d)*0.02)
        self.blocks=nn.ModuleList([Block(d,h,ff,dp) for _ in range(2)])
        self.head=nn.Sequential(nn.Linear(d,d//2),nn.GELU(),nn.Linear(d//2,ns))
    def forward(self,x,store=False):
        x=self.proj(x)+self.pe[:,:x.shape[1],:]
        for i,b in enumerate(self.blocks): x=b(x,store=store,nm=str(i))
        return self.head(x[:,-1,:]),x
    def get_pats(self): return {k:v for blk in self.blocks for k,v in blk.attn.pats.items()}

# Simple (4 layers) — current default
class SimpleBase(nn.Module):
    def __init__(self,ns,d=128,h=8,ff=256,dp=.1):
        super().__init__(); self.h,self.nl=h,4
        self.proj=nn.Linear(ns,d); self.pe=nn.Parameter(torch.randn(1,L,d)*0.02)
        self.blocks=nn.ModuleList([Block(d,h,ff,dp) for _ in range(4)])
        self.head=nn.Sequential(nn.Linear(d,d//2),nn.GELU(),nn.Linear(d//2,ns))
    def forward(self,x,store=False):
        x=self.proj(x)+self.pe[:,:x.shape[1],:]
        for i,b in enumerate(self.blocks): x=b(x,store=store,nm=str(i))
        return self.head(x[:,-1,:]),x
    def get_pats(self): return {k:v for blk in self.blocks for k,v in blk.attn.pats.items()}

# PatchTST (4 layers, 5-day patches → 6 patches)
class PatchEmbed(nn.Module):
    def __init__(self,ns,ps=5,d=128):
        super().__init__(); self.ps=ps; self.np=L//ps; self.proj=nn.Linear(ns*ps,d)
    def forward(self,x):
        B,S,N=x.shape; x=x[:,:self.np*self.ps,:]; x=x.reshape(B,self.np,N*self.ps); return self.proj(x)

class PatchTST(nn.Module):
    def __init__(self,ns,d=128,h=4,ff=256,dp=.1,ps=5):
        super().__init__(); self.h,self.nl=h,4; self.np=L//ps
        self.embed=PatchEmbed(ns,ps,d); self.pe=nn.Parameter(torch.randn(1,self.np,d)*0.02)
        self.blocks=nn.ModuleList([Block(d,h,ff,dp) for _ in range(4)])
        self.head=nn.Sequential(nn.Linear(d,d//2),nn.GELU(),nn.Linear(d//2,ns))
    def forward(self,x,store=False):
        x=self.embed(x)+self.pe
        for i,b in enumerate(self.blocks): x=b(x,store=store,nm=str(i))
        return self.head(x[:,-1,:]),x
    def get_pats(self): return {k:v for blk in self.blocks for k,v in blk.attn.pats.items()}

# ============================================================
# Data — ALL available
# ============================================================
us=pd.read_parquet('D:/code/data/us_market/log_returns.parquet')
ns=pd.read_parquet('D:/code/data/log_returns_nonstock.parquet')
cn_df=pd.read_parquet('D:/code/data/cn_market/daily_returns.parquet')
cn=cn_df.pivot(index='trddt',columns='stkcd',values='dretwd').dropna(axis=1,thresh=int(len(cn_df['trddt'].unique())*0.6)).ffill()
fx=[c for c in ns.columns if '=X' in c]; crypto=[c for c in ns.columns if '-USD' in c]
fut=[c for c in ns.columns if '=F' in c]; idx=[c for c in ns.columns if c.startswith('^')]
groups={
    'US Stocks':(us,list(us.columns[:300])),
    'CN A-Share':(cn,list(cn.columns[:300])),
    'Forex':(ns,fx),
    'Crypto':(ns,crypto),
    'Commodities':(ns,fut),
    'Indices':(ns,idx),
}
groups={k:v for k,v in groups.items() if len(v[1])>=10}

log("ARCHITECTURE SCALING: Tiny(2L) vs Simple(4L) vs PatchTST(4L+patch)")
log(f"{'='*60}  ALL GROUPS (full data)")

archs={'Tiny(2L)':TinyBase,'Simple(4L)':SimpleBase,'PatchTST(4L+5d)':PatchTST}
all_results={}

for gi,(gname,(src,tickers)) in enumerate(groups.items()):
    log(f"\n[{'='*50}]\n[{gi+1}/{len(groups)}] {gname} ({len(tickers)} assets)\n[{'='*50}]")
    sub=src[tickers].ffill().dropna(axis=0); R=sub.values.astype(np.float32); N=R.shape[1]
    R=np.nan_to_num((R-np.nanmean(R,0,keepdims=True))/(np.nanstd(R,0,keepdims=True)+1e-8),0)
    n=len(R)-L-1
    X=np.lib.stride_tricks.sliding_window_view(R,L,axis=0)[:n].transpose(0,2,1).astype(np.float32)
    y=R[L+1:][:n].astype(np.float32); tr=int(n*0.7)
    X_tr_flat=X[:tr].reshape(tr,-1); X_te_flat=X[tr:].reshape(n-tr,-1)

    gr={}
    for aname,ACls in archs.items():
        np.random.seed(SEED); torch.manual_seed(SEED)
        Xt=torch.FloatTensor(X[:tr]).to(DEV); yt=torch.FloatTensor(y[:tr]).to(DEV)
        Xe=torch.FloatTensor(X[tr:]).to(DEV); ye=torch.FloatTensor(y[tr:]).to(DEV)
        ld=DataLoader(TensorDataset(Xt,yt),batch_size=B,shuffle=True)
        base=ACls(N).to(DEV); opt=torch.optim.AdamW(base.parameters(),lr=3e-4,weight_decay=1e-5)
        for ep in range(60):
            base.train()
            for bx,by in ld: opt.zero_grad(); l=nn.HuberLoss(delta=1.0)(base(bx,False)[0],by); l.backward(); torch.nn.utils.clip_grad_norm_(base.parameters(),2.0); opt.step()
        base.eval()
        with torch.no_grad():
            rt,_=base(Xt,False); re,_=base(Xe,False)
            _,ht=base(Xt,True); pt=base.get_pats()[str(base.nl-1)].cpu().numpy()  # last layer
            _,he=base(Xe,True); p_te=base.get_pats()[str(base.nl-1)].cpu().numpy()
        at=torch.abs(rt-yt).mean(-1).cpu().numpy(); ae=torch.abs(re-ye).mean(-1).cpu().numpy()
        abs_re=torch.abs(re).mean(-1).cpu().numpy()
        r_pred,_=spearmanr(abs_re, ae)
        # Pattern reconstruction
        p_tr_flat=pt.reshape(len(pt),-1); p_te_flat=p_te.reshape(len(p_te),-1)
        pca_r=PCA(16, random_state=SEED)
        r2_recon=r2_score(p_te_flat, pca_r.inverse_transform(Ridge(1.0).fit(X_tr_flat, pca_r.fit_transform(p_tr_flat)).predict(X_te_flat)))
        # Pattern→error
        pca_e=PCA(64, random_state=SEED)
        ptr_e=pca_e.fit_transform(p_tr_flat); pte_e=pca_e.transform(p_te_flat)
        s1,s2,a1,a2=train_test_split(ptr_e, at, test_size=0.2, random_state=SEED)
        r_pat,_=spearmanr(Ridge(1.0).fit(s1,a1).predict(pte_e), ae)
        # Residual
        pred_m=Ridge(1.0).fit(torch.abs(rt).mean(-1).cpu().numpy().reshape(-1,1), at)
        resid=ae-pred_m.predict(abs_re.reshape(-1,1))
        s1,s2,r1,r2=train_test_split(pte_e, resid, test_size=0.2, random_state=SEED)
        r_resid,_=spearmanr(Ridge(1.0).fit(s1,r1).predict(pte_e), resid)
        del base; torch.xpu.empty_cache()
        gr[aname]={'|Pred|':r_pred,'Recon R²':r2_recon,'Pat→Err':r_pat,'Resid':r_resid}
    all_results[gname]=gr
    for aname in archs:
        r=gr[aname]
        log(f"  {aname:<20} |Pred|={r['|Pred|']:+.3f} Recon={r['Recon R²']:+.3f} Pat→Err={r['Pat→Err']:+.3f} Resid={r['Resid']:+.3f}")

log(f"\n{'='*70}")
log("ARCHITECTURE SCALING SUMMARY")
log(f"{'='*70}")
for metric in ['|Pred|','Recon R²','Pat→Err','Resid']:
    log(f"\n  {metric}:")
    for aname in archs:
        vals=[all_results[g][aname][metric] for g in all_results]
        log(f"    {aname:<20} mean={np.mean(vals):+.4f}  [{', '.join([f'{v:+.3f}' for v in vals])}]")
