"""
Public benchmark validation: ETTh1, ETTm1, Electricity, Weather.
5 seeds each. Core metrics: |Pred| r, Recon R², Pat→Err r, Resid r.
"""
import numpy as np, pandas as pd, warnings, urllib.request, os
warnings.filterwarnings('ignore')
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import sys
DEV='cuda' if torch.cuda.is_available() else ('xpu' if hasattr(torch,'xpu') and torch.xpu.is_available() else 'cpu')
B=128; L=48; SEEDS=[42,123,456,789,1024]
print(f"Device: {DEV}")

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

# Datasets
datasets={
    'ETTh1': 'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv',
    'ETTm1': 'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv',
}

# Electricity and Weather from UCI (local download needed, or skip)
# Actually use ETT datasets + electricity if available

# Check for electricity dataset
elec_path='/tmp/electricity.csv'
if not os.path.exists(elec_path):
    log("Note: Electricity dataset not found locally, testing ETT only")
else:
    datasets['Electricity']=elec_path

all_results={}

for dname, dpath in datasets.items():
    log(f"\n{'='*60}")
    log(f"DATASET: {dname}")
    log(f"{'='*60}")

    if dpath.startswith('http'):
        local_path=f'/tmp/{dname}.csv'
        if not os.path.exists(local_path):
            urllib.request.urlretrieve(dpath, local_path)
        df=pd.read_csv(local_path)
    else:
        df=pd.read_csv(dpath)

    data=df.select_dtypes(include=[np.number]).values.astype(np.float32)
    data=(data-data.mean(0))/(data.std(0)+1e-8)
    N=data.shape[1]; n=len(data)-L-1
    X_all=np.lib.stride_tricks.sliding_window_view(data,L,axis=0)[:n].transpose(0,2,1).astype(np.float32)
    y_all=data[L+1:][:n].astype(np.float32); tr=int(n*0.7)
    log(f"  {N} features, {len(data)} samples, Train:{tr} Test:{n-tr}")

    ds_results=[]
    for si,seed in enumerate(SEEDS):
        np.random.seed(seed); torch.manual_seed(seed)
        Xt=torch.FloatTensor(X_all[:tr]).to(DEV); yt=torch.FloatTensor(y_all[:tr]).to(DEV)
        Xe=torch.FloatTensor(X_all[tr:]).to(DEV); ye=torch.FloatTensor(y_all[tr:]).to(DEV)
        ld=DataLoader(TensorDataset(Xt,yt),batch_size=B,shuffle=True)
        X_tr_flat=X_all[:tr].reshape(tr,-1); X_te_flat=X_all[tr:].reshape(n-tr,-1)

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
        abs_re=torch.abs(re).mean(-1).cpu().numpy()
        r_pred,_=spearmanr(abs_re, ae)
        # Recon R²
        p_tr_flat=pt.reshape(len(pt),-1); p_te_flat=p_te.reshape(len(p_te),-1)
        pca_r=PCA(16, random_state=seed)
        r2_recon=r2_score(p_te_flat, pca_r.inverse_transform(Ridge(1.0).fit(X_tr_flat, pca_r.fit_transform(p_tr_flat)).predict(X_te_flat)))
        # Pat→Err
        pca_e=PCA(64, random_state=seed)
        ptr_e=pca_e.fit_transform(p_tr_flat); pte_e=pca_e.transform(p_te_flat)
        s1,s2,a1,a2=train_test_split(ptr_e, at, test_size=0.2, random_state=seed)
        r_pat,_=spearmanr(Ridge(1.0).fit(s1,a1).predict(pte_e), ae)
        # Resid
        pred_m=Ridge(1.0).fit(torch.abs(rt).mean(-1).cpu().numpy().reshape(-1,1), at)
        resid=ae-pred_m.predict(abs_re.reshape(-1,1))
        s1,s2,r1,r2=train_test_split(pte_e, resid, test_size=0.2, random_state=seed)
        r_resid,_=spearmanr(Ridge(1.0).fit(s1,r1).predict(pte_e), resid)
        del base; torch.xpu.empty_cache()
        ds_results.append({'|Pred|':r_pred,'Recon':r2_recon,'Pat→Err':r_pat,'Resid':r_resid})
        log(f"    Seed {si+1}/{len(SEEDS)}: |Pred|={r_pred:+.3f} Recon={r2_recon:+.3f} Pat→Err={r_pat:+.3f} Resid={r_resid:+.3f}")

    all_results[dname]=ds_results

# Report
log(f"\n{'='*70}")
log("PUBLIC BENCHMARKS — FINAL RESULTS (mean ± std, 5 seeds)")
log(f"{'='*70}")
for metric in ['|Pred|','Recon','Pat→Err','Resid']:
    log(f"\n  {metric}:")
    for dname, res in all_results.items():
        vals=[r[metric] for r in res]; m=np.mean(vals); s=np.std(vals)
        pat_wins=sum(1 for r in res if r['Pat→Err']>r['|Pred|']+0.01)
        log(f"    {dname:<15} {m:+.4f} ± {s:.4f}  (Pat>|Pred| in {pat_wins}/{len(SEEDS)} seeds)")
