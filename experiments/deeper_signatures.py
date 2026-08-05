"""
Deeper Signatures: Compare depth-3 (39) vs depth-4 (120) vs depth-5 (363)
plus log-signature for numerical stability.
"""
import numpy as np; import pandas as pd; import torch; import torch.nn as nn; import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader; import signatory; from scipy.stats import spearmanr
import math, time, os, warnings; warnings.filterwarnings('ignore')

GPU=torch.device("cuda" if torch.cuda.is_available() else "cpu"); CPU=torch.device("cpu")
print(f"GPU: {torch.cuda.get_device_name(0) if GPU.type=='cuda' else 'CPU'} | "
      f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB" if GPU.type=='cuda' else '')
torch.manual_seed(42); np.random.seed(42)

# Config
N_STOCKS=200; WINDOW=250; PRED_HORIZON=21; BATCH=8; D_MODEL=128; N_LAYERS=3; N_HEADS=4; D_FF=256; DROPOUT=0.1
EPOCHS=30
DATA_DIR="/home/user2/meta_attn/data"

# Load small subset for fast comparison
print("[1/3] Loading data...")
price=pd.read_parquet(f"{DATA_DIR}/us_price.parquet")
vol=pd.read_parquet(f"{DATA_DIR}/us_volume.parquet")
common=set(price.columns)&set(vol.columns); common=sorted(common)[:N_STOCKS]
idx=price.index.intersection(vol.index)
price=price.loc[idx,common].ffill().fillna(0)
vol=vol.loc[idx,common].ffill().fillna(1)
n_a=len(common); print(f"  {n_a} stocks x {len(price)} days")

logP=np.log(np.maximum(price.values.astype(np.float64),1e-8))
logV=np.log(np.maximum(vol.values.astype(np.float64),1e-8))
ma20=pd.DataFrame(logV).rolling(20,min_periods=1).mean().values
vi=logV-ma20
paths=np.stack([logP,logV,vi],-1).astype(np.float32)  # (days, assets, 3)
n_samp=paths.shape[0]-WINDOW-PRED_HORIZON+1

X_paths=np.zeros((n_samp,n_a,WINDOW,3),dtype=np.float32)
y_fwd=np.zeros((n_samp,n_a),dtype=np.float32)
for i in range(n_samp):
    X_paths[i]=paths[i:i+WINDOW].transpose(1,0,2)
    y_fwd[i]=logP[i+WINDOW+PRED_HORIZON-1]-logP[i+WINDOW-1]
y_fwd=np.nan_to_num(y_fwd,nan=0,posinf=0,neginf=0)
y_fwd=np.clip(y_fwd,-np.percentile(np.abs(y_fwd),99),np.percentile(np.abs(y_fwd),99))
split=int(n_samp*0.7); Xp_tr,Xp_te=X_paths[:split],X_paths[split:]; y_tr,y_te=y_fwd[:split],y_fwd[split:]
print(f"  Train: {split} | Test: {n_samp-split}")

# Compare depths
results={}
for depth in [3,4,5]:
    sig_dim=signatory.signature_channels(3,depth)
    print(f"\n[2/3] Depth {depth} → {sig_dim} components...")
    t0=time.time()

    # Compute signatures
    sigs_tr=np.zeros((Xp_tr.shape[0],n_a,sig_dim),dtype=np.float32)
    sigs_te=np.zeros((Xp_te.shape[0],n_a,sig_dim),dtype=np.float32)
    Xpt=torch.FloatTensor(Xp_tr).to(CPU); Xpe=torch.FloatTensor(Xp_te).to(CPU)
    for a in range(n_a):
        for i in range(0,Xp_tr.shape[0],128):
            sigs_tr[i:i+128,a]=signatory.signature(Xpt[i:i+128,a],depth,basepoint=True).cpu().numpy()
        for i in range(0,Xp_te.shape[0],128):
            sigs_te[i:i+128,a]=signatory.signature(Xpe[i:i+128,a],depth,basepoint=True).cpu().numpy()
    t1=time.time()

    # Handle inf/nan + scale
    from sklearn.preprocessing import RobustScaler
    sigs_tr=np.nan_to_num(sigs_tr,nan=0,posinf=0,neginf=0)
    sigs_te=np.nan_to_num(sigs_te,nan=0,posinf=0,neginf=0)
    flat=sigs_tr.reshape(-1,sig_dim); p1,p99=np.percentile(flat,[1,99],axis=0)
    sigs_tr=np.clip(sigs_tr,p1,p99); sigs_te=np.clip(sigs_te,p1,p99)
    sc=RobustScaler().fit(sigs_tr.reshape(-1,sig_dim))
    sigs_tr=sc.transform(sigs_tr.reshape(-1,sig_dim)).reshape(sigs_tr.shape)
    sigs_te=sc.transform(sigs_te.reshape(-1,sig_dim)).reshape(sigs_te.shape)

    # Meta-Transformer
    class MetaTrans(nn.Module):
        def __init__(self):
            super().__init__(); self.na=n_a; self.d=D_MODEL
            self.sig_proj=nn.Sequential(nn.Linear(sig_dim,self.d),nn.LayerNorm(self.d))
            self.sig_pos=nn.Parameter(torch.randn(1,500,self.d)*0.02)
            self.cls=nn.Parameter(torch.randn(1,1,self.d)*0.02)
            self.layers=nn.ModuleList([
                nn.TransformerEncoderLayer(self.d,N_HEADS,D_FF,DROPOUT,'gelu',batch_first=True,norm_first=True)
                for _ in range(N_LAYERS)])
            self.ret_head=nn.Linear(self.d,n_a)
            self.unc_head=nn.Sequential(nn.Linear(self.d,self.d*2),nn.GELU(),nn.Linear(self.d*2,self.d),nn.GELU(),nn.Linear(self.d,1))
        def forward(self,s):
            B=s.shape[0]; st=self.sig_proj(s)+self.sig_pos[:,:self.na,:]; x=torch.cat([self.cls.expand(B,-1,-1),st],1)
            for l in self.layers: x=l(x); return self.ret_head(x[:,0]),self.unc_head(x[:,0]).squeeze(-1)

    tr_ds=torch.utils.data.TensorDataset(torch.FloatTensor(sigs_tr),torch.FloatTensor(y_tr))
    te_ds=torch.utils.data.TensorDataset(torch.FloatTensor(sigs_te),torch.FloatTensor(y_te))
    tr_ld=DataLoader(tr_ds,BATCH,shuffle=True,drop_last=True); te_ld=DataLoader(te_ds,BATCH,shuffle=False)

    model=MetaTrans().to(GPU); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS)
    for ep in range(EPOCHS):
        model.train(); tl=0
        for s,y in tr_ld:
            rp,up=model(s.to(GPU)); loss=F.huber_loss(rp,y.to(GPU),delta=1.0)
            with torch.no_grad(): err=(rp-y.to(GPU)).abs().mean(1)
            loss=loss+0.1*F.mse_loss(up,err)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),2.0); opt.step(); tl+=loss.item()
        sch.step()
        if (ep+1)%10==0: print(f"    Epoch {ep+1} | Loss: {tl/len(tr_ld):.6f} ({t1-t0:.0f}s sigs)")

    model.eval()
    with torch.no_grad():
        rp,up=model(torch.FloatTensor(sigs_te).to(GPU))
        rp=rp.cpu().numpy(); up=up.cpu().numpy()
    mse=float(np.mean((rp-y_te)**2)); err=np.abs(rp-y_te).mean(1); bl=np.abs(rp).mean(1)
    results[depth]={'dim':sig_dim,'time':t1-t0,'mse':mse,'model_r':spearmanr(up,err)[0],'pred_r':spearmanr(bl,err)[0]}

# Summary
print("\n[3/3] Results: Deeper = Better?")
print("="*65)
print(f"  {'Depth':<10} {'Dim':<8} {'Sig Time':<10} {'MSE':<12} {'Model r':<10} {'|Pred| r':<10}")
for d,r in results.items():
    print(f"  {d:<10} {r['dim']:<8} {r['time']:.0f}s{'':<5} {r['mse']:<12.6f} {r['model_r']:<10.4f} {r['pred_r']:<10.4f}")
best=min(results.values(),key=lambda x:x['mse'])
print(f"\n  Best MSE: depth={[d for d,r in results.items() if r==best][0]} ({best['mse']:.6f})")
print("="*65)
