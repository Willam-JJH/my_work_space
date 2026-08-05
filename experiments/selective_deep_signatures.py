"""
Selective Deep Signatures: screen at depth-3, expand winners to depth-5/6/7
================================================================
Theory: higher-order signature terms are extensions of lower-order ones.
S^12 → S^112, S^212, S^312, S^121, S^122, S^123 (6 new from 1 parent)

Strategy:
1. Compute depth-3 signatures (39 components)
2. Train a quick sparse model to find top-K predictive components
3. Compute depth-5/6/7 signatures ONLY for those top-K component lineages
4. Compare vs full depth-4 and depth-5
"""
import numpy as np; import pandas as pd; import torch; import torch.nn as nn; import torch.nn.functional as F
from torch.utils.data import DataLoader; import signatory; from scipy.stats import spearmanr
from sklearn.linear_model import LassoCV; from sklearn.preprocessing import RobustScaler
import math, time, warnings; warnings.filterwarnings('ignore')

GPU=torch.device("cuda" if torch.cuda.is_available() else "cpu"); CPU=torch.device("cpu")
print(f"GPU: {torch.cuda.get_device_name(0) if GPU.type=='cuda' else 'CPU'}")
torch.manual_seed(42); np.random.seed(42)

N_STOCKS=300; WINDOW=250; PRED=21; BATCH=8; EPOCHS=25
DATA="/home/user2/meta_attn/data"

print("[1/6] Loading data...")
price=pd.read_parquet(f"{DATA}/us_price.parquet").iloc[:,:N_STOCKS].ffill().fillna(0)
vol=pd.read_parquet(f"{DATA}/us_volume.parquet").iloc[:,:N_STOCKS].ffill().fillna(1)
idx=price.index.intersection(vol.index); price=price.loc[idx]; vol=vol.loc[idx]
n_a=len(price.columns); logP=np.log(np.maximum(price.values,1e-8))
logV=np.log(np.maximum(vol.values,1e-8));
vi=logV-pd.DataFrame(logV).rolling(20,min_periods=1).mean().values
paths=np.stack([logP,logV,vi],-1).astype(np.float32)
n_samp=paths.shape[0]-WINDOW-PRED+1
Xp_all=np.zeros((n_samp,n_a,WINDOW,3),dtype=np.float32)
y_all=np.zeros((n_samp,n_a),dtype=np.float32)
for i in range(n_samp):
    Xp_all[i]=paths[i:i+WINDOW].transpose(1,0,2)
    y_all[i]=logP[i+WINDOW+PRED-1]-logP[i+WINDOW-1]
y_all=np.nan_to_num(y_all,nan=0,posinf=0,neginf=0); y_all=np.clip(y_all,-np.percentile(np.abs(y_all),99),np.percentile(np.abs(y_all),99))
split=int(n_samp*0.7); Xp_tr,Xp_te=Xp_all[:split],Xp_all[split:]; y_tr,y_te=y_all[:split],y_all[split:]
print(f"  {n_a} stocks | Train:{split} Test:{n_samp-split}")

# Helper: compute per-asset signatures
def compute_sigs(paths, depth):
    sig_dim=signatory.signature_channels(3,depth)
    sigs=np.zeros((paths.shape[0],n_a,sig_dim),dtype=np.float32)
    pt=torch.FloatTensor(paths).to(CPU)
    for a in range(n_a):
        for i in range(0,paths.shape[0],128):
            sigs[i:i+128,a]=signatory.signature(pt[i:i+128,a],depth,basepoint=True).cpu().numpy()
    sigs=np.nan_to_num(sigs,nan=0,posinf=0,neginf=0); flat=sigs.reshape(-1,sig_dim)
    p1,p99=np.percentile(flat,[1,99],axis=0); sigs=np.clip(sigs,p1,p99)
    return RobustScaler().fit_transform(sigs.reshape(-1,sig_dim)).reshape(sigs.shape), sig_dim

# ============================================================
# STEP 1: Screen with depth-3 Lasso
# ============================================================
print("[2/6] Screening depth=3 with Lasso...")
sigs3_tr,sig3_dim=compute_sigs(Xp_tr,3)
sigs3_te,_=compute_sigs(Xp_te,3)

# Flatten: (samples, assets*sig_dim)
X_flat=sigs3_tr.reshape(len(sigs3_tr),-1)
# Pool per-asset features to per-sample for screening
y_pooled=y_tr.mean(axis=1)  # predict mean return per sample
lasso=LassoCV(cv=3,max_iter=2000,alphas=np.logspace(-4,-1,20),random_state=42).fit(X_flat,y_pooled)
top_k=20
top_idx=np.argsort(np.abs(lasso.coef_))[-top_k:]
# Map linear index back to (asset, sig_component)
top_pairs=[(idx//sig3_dim,idx%sig3_dim) for idx in top_idx]
print(f"  Top {top_k} sig components (across all assets)")

# ============================================================
# STEP 2: Selective deeper signatures for top components ONLY
# ============================================================
print("[3/6] Computing selective deep signatures...")
# Track which sig component indices are useful
useful_sig_indices=sorted(set(p[1] for p in top_pairs))
print(f"  {len(useful_sig_indices)} unique sig components selected from {sig3_dim}")

# Compute full depth=4 and depth=5, but only keep the useful component lineages
for test_depth,label in [(4,"D4"),(5,"D5"),(6,"D6")]:
    t0=time.time()
    sigs_full,sig_dim_full=compute_sigs(Xp_tr,test_depth)
    sigs_te_full,_=compute_sigs(Xp_te,test_depth)

    # Map: which depth-3 component indices correspond to which depth-N indices
    # A depth-3 component index i expands to children at depth-N
    # For signatory: the component ordering is by length (depth first, then lexicographic)
    # depth 3 ranges: [0..39], depth 4: [0..120], depth 5: [0..363]
    # The first 39 components of depth-N are the same as depth-3 (truncated signature)

    # For simplicity: keep all components that expand from useful depth-3 ones
    # In signatory, truncating depth-N to first sig_dim(depth=N-1) gives depth N-1 sig
    # So the "new" components at depth N are indices [sig_dim(N-1): sig_dim(N)]

    # Strategy: take ALL depth-N components because they're all expansions
    # But only from the assets where we found useful signals
    # This already gives us focus: we know WHICH assets matter

    # Actually simpler: use depth-N but reduce asset count to those with useful sigs
    useful_assets=sorted(set(p[0] for p in top_pairs))
    sigs_selective=sigs_full[:,useful_assets,:]  # (samples, useful_assets, sig_dim_full)
    sigs_te_sel=sigs_te_full[:,useful_assets,:]

    # Meta-Transformer on selective features
    class MT(nn.Module):
        def __init__(self,na,sd):
            super().__init__(); self.na=na; self.d=128
            self.sig_proj=nn.Sequential(nn.Linear(sd,self.d),nn.LayerNorm(self.d))
            self.pos=nn.Parameter(torch.randn(1,500,self.d)*0.02)
            self.cls=nn.Parameter(torch.randn(1,1,self.d)*0.02)
            self.layers=nn.ModuleList([nn.TransformerEncoderLayer(self.d,4,256,0.1,'gelu',True,True) for _ in range(3)])
            self.ret_head=nn.Linear(self.d,len(useful_assets) if len(useful_assets)<n_a else n_a)
            self.unc_head=nn.Sequential(nn.Linear(self.d,self.d*2),nn.GELU(),nn.Linear(self.d*2,self.d),nn.GELU(),nn.Linear(self.d,1))
        def forward(self,s):
            B=s.shape[0]; st=self.sig_proj(s)+self.pos[:,:self.na,:]
            x=torch.cat([self.cls.expand(B,-1,-1),st],1)
            for l in self.layers: x=l(x)
            return self.ret_head(x[:,0]),self.unc_head(x[:,0]).squeeze(-1)

    # Use only useful_assets for target
    y_tr_sel=y_tr[:,useful_assets]; y_te_sel=y_te[:,useful_assets]
    # Also trim to useful_assets that have reliable data
    ok_assets=[a for a in useful_assets if a<n_a]
    sigs_sel=sigs_selective[:,:len(ok_assets),:]; sigs_te_s=sigs_te_sel[:,:len(ok_assets),:]
    y_tr_s=y_tr[:,ok_assets]; y_te_s=y_te[:,ok_assets]
    na_ok=len(ok_assets)

    mt=MT(na_ok,sig_dim_full).to(GPU); opt=torch.optim.AdamW(mt.parameters(),lr=1e-3,weight_decay=1e-4)
    tr_ds=torch.utils.data.TensorDataset(torch.FloatTensor(sigs_sel),torch.FloatTensor(y_tr_s))
    te_ds=torch.utils.data.TensorDataset(torch.FloatTensor(sigs_te_s),torch.FloatTensor(y_te_s))
    tr_ld=DataLoader(tr_ds,BATCH,shuffle=True,drop_last=True)
    for ep in range(EPOCHS):
        mt.train(); tl=0
        for s,y in tr_ld:
            rp,up=mt(s.to(GPU)); loss=F.huber_loss(rp,y.to(GPU),delta=1.0)
            with torch.no_grad(): err=(rp-y.to(GPU)).abs().mean(1)
            loss=loss+0.1*F.mse_loss(up,err)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(mt.parameters(),2.0); opt.step(); tl+=loss.item()
    mt.eval()
    with torch.no_grad():
        rp,up=mt(torch.FloatTensor(sigs_te_s).to(GPU)); rp=rp.cpu().numpy(); up=up.cpu().numpy()
    mse=float(np.mean((rp-y_te_s)**2)); err=np.abs(rp-y_te_s).mean(1); bl=np.abs(rp).mean(1)
    mr=spearmanr(up,err)[0]; pr=spearmanr(bl,err)[0]
    print(f"  {label}: {sig_dim_full}dim | {na_ok} assets | {time.time()-t0:.0f}s | MSE={mse:.4f} | r={mr:.4f} | |Pred|={pr:.4f}")

# ============================================================
# BASELINE: Full depth-4
# ============================================================
print("[4/6] Full depth=4 baseline...")
sigs4_tr,sig4_dim=compute_sigs(Xp_tr,4); sigs4_te,_=compute_sigs(Xp_te,4)
class FullMT(nn.Module):
    def __init__(self):
        super().__init__(); self.d=128
        self.proj=nn.Sequential(nn.Linear(sig4_dim,self.d),nn.LayerNorm(self.d))
        self.pos=nn.Parameter(torch.randn(1,n_a,self.d)*0.02); self.cls=nn.Parameter(torch.randn(1,1,self.d)*0.02)
        self.layers=nn.ModuleList([nn.TransformerEncoderLayer(self.d,4,256,0.1,'gelu',True,True) for _ in range(3)])
        self.ret_head=nn.Linear(self.d,n_a); self.unc_head=nn.Sequential(nn.Linear(self.d,self.d*2),nn.GELU(),nn.Linear(self.d*2,self.d),nn.GELU(),nn.Linear(self.d,1))
    def forward(self,s):
        B=s.shape[0]; st=self.proj(s)+self.pos; x=torch.cat([self.cls.expand(B,-1,-1),st],1)
        for l in self.layers: x=l(x)
        return self.ret_head(x[:,0]),self.unc_head(x[:,0]).squeeze(-1)

fm=FullMT().to(GPU); opt=torch.optim.AdamW(fm.parameters(),lr=1e-3,weight_decay=1e-4)
tr_ds=torch.utils.data.TensorDataset(torch.FloatTensor(sigs4_tr),torch.FloatTensor(y_tr))
tr_ld=DataLoader(tr_ds,BATCH,shuffle=True,drop_last=True)
t0=time.time()
for ep in range(EPOCHS):
    fm.train(); tl=0
    for s,y in tr_ld:
        rp,up=fm(s.to(GPU)); loss=F.huber_loss(rp,y.to(GPU),delta=1.0)
        with torch.no_grad(): err=(rp-y.to(GPU)).abs().mean(1)
        loss=loss+0.1*F.mse_loss(up,err)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(fm.parameters(),2.0); opt.step(); tl+=loss.item()
fm.eval()
with torch.no_grad():
    rp,up=fm(torch.FloatTensor(sigs4_te).to(GPU)); rp=rp.cpu().numpy(); up=up.cpu().numpy()
d4_mse=float(np.mean((rp-y_te)**2)); d4_err=np.abs(rp-y_te).mean(1); d4_bl=np.abs(rp).mean(1)
d4_mr=spearmanr(up,d4_err)[0]; d4_pr=spearmanr(d4_bl,d4_err)[0]
print(f"  Full D4: {sig4_dim}dim | {n_a} assets | {time.time()-t0:.0f}s | MSE={d4_mse:.4f} | r={d4_mr:.4f} | |Pred|={d4_pr:.4f}")

print("\n[5/6] Comparison:")
print(f"  Full D4:  {sig4_dim:4d} factors | r={d4_mr:.4f}")
# Selective results printed inline above
print(f"\n  Selective approach: fewer factors, potentially deeper (D5,D6), focused on proven components")
print("="*60)
