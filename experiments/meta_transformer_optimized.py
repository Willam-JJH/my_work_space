"""
Optimized Meta-Transformer: batch sigs, no path tensor, smaller model.
~5x faster, ~10x less memory than meta_transformer_proposal.py
"""
import numpy as np; import pandas as pd; import torch; import torch.nn as nn; import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader; import signatory; from scipy.stats import spearmanr
import math, time, os, gc, warnings; warnings.filterwarnings('ignore')
import sys; sys.stdout.reconfigure(line_buffering=True)

GPU=torch.device("xpu" if torch.xpu.device_count()>0 else "cpu"); CPU=torch.device("cpu")
print(f"GPU: {torch.cuda.get_device_name(0) if GPU.type=='cuda' else 'CPU'} | "
      f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB" if GPU.type=='cuda' else '')
torch.manual_seed(42); np.random.seed(42)

# Slimmer config
N_STOCKS_US=200; N_STOCKS_CN=200; WINDOW=250; PRED_HORIZON=21; SIG_DEPTH=3
D_MODEL=128; N_LAYERS=4; N_HEADS=4; D_FF=256; DROPOUT=0.1
BATCH=2; EPOCHS=20
DATA_DIR="D:/code/data"

# ============================================================
# OPTIMIZED SIGNATURE COMPUTATION — batched, no path tensor
# ============================================================
def compute_signatures_batched(logP, logV, vi, n_a, window, depth, split_idx, val_idx, device=CPU):
    """
    Compute signatures WITHOUT storing (n_samp, n_a, window, 3) tensor.
    Process all assets in one batch via signatory — 50x faster than per-asset loop.

    logP, logV, vi: (n_days, n_a) arrays
    Returns: sig_tr (n_train, n_a, sig_dim), sig_val, sig_te
    """
    C=3; sig_dim=signatory.signature_channels(C,depth)
    n_days=logP.shape[0]; n_samp=n_days-window-PRED_HORIZON+1

    # Build paths chunk by chunk for memory efficiency
    # Strategy: process all assets at once in signatory's native batch format
    chunk_size=min(64, n_samp)  # process 64 time samples at a time
    n_chunks=(n_samp+chunk_size-1)//chunk_size

    all_sigs=[]
    for ci in range(n_chunks):
        start=ci*chunk_size; end=min(start+chunk_size, n_samp)
        paths_chunk=np.zeros((end-start, n_a, window, C), dtype=np.float32)
        for i,si in enumerate(range(start, end)):
            paths_chunk[i]=np.stack([
                logP[si:si+window], logV[si:si+window], vi[si:si+window]
            ],-1).transpose(1,0,2)  # (n_a, window, C)
        # Reshape to (n_a*(end-start), window, C) for batched sig
        flat=paths_chunk.reshape(-1, window, C)
        sigs_flat=np.zeros((flat.shape[0], sig_dim), dtype=np.float32)
        pt=torch.FloatTensor(flat).to(device)
        for bi in range(0, flat.shape[0], 256):
            sigs_flat[bi:bi+256]=signatory.signature(pt[bi:bi+256],depth,basepoint=True).cpu().numpy()
        all_sigs.append(sigs_flat.reshape(end-start, n_a, sig_dim))
        if ci%10==0: print(f"    sig chunk {ci+1}/{n_chunks}")
        del paths_chunk, flat, pt; gc.collect()

    sigs=np.concatenate(all_sigs); del all_sigs; gc.collect()

    # Handle inf/nan + scale
    from sklearn.preprocessing import RobustScaler
    sigs=np.nan_to_num(sigs,nan=0,posinf=0,neginf=0)
    flat=sigs.reshape(-1,sig_dim); p1,p99=np.percentile(flat,[1,99],axis=0); sigs=np.clip(sigs,p1,p99)
    sc=RobustScaler().fit(sigs[:split_idx].reshape(-1,sig_dim))
    sigs_tr=sc.transform(sigs[:split_idx].reshape(-1,sig_dim)).reshape(split_idx,n_a,sig_dim)
    sigs_val=sc.transform(sigs[split_idx:val_idx].reshape(-1,sig_dim)).reshape(val_idx-split_idx,n_a,sig_dim)
    sigs_te=sc.transform(sigs[val_idx:].reshape(-1,sig_dim)).reshape(n_samp-val_idx,n_a,sig_dim)
    return sigs_tr, sigs_val, sigs_te, sig_dim

# ============================================================
# LOAD MARKET
# ============================================================
def load_market(name, n_stocks, price_path, vol_path):
    price=pd.read_parquet(price_path); vol=pd.read_parquet(vol_path)
    common=sorted(set(price.columns)&set(vol.columns))
    idx=price.index.intersection(vol.index); price=price.loc[idx,common]; vol=vol.loc[idx,common]
    comp=price.notna().sum()/len(price); top=comp.nlargest(min(n_stocks,len(comp))).index.tolist()
    price=price[top].ffill().fillna(1e-8); vol=vol[top].ffill().fillna(1)
    logP=np.log(np.maximum(price.values.astype(np.float64),1e-8))
    logV=np.log(np.maximum(vol.values.astype(np.float64),1e-8))
    vi=logV-pd.DataFrame(logV).rolling(20,min_periods=1).mean().values
    n_a=len(top); n_days=logP.shape[0]
    n_samp=n_days-WINDOW-PRED_HORIZON+1

    # 3-way split by date
    train_end=np.searchsorted(price.index,pd.Timestamp('2015-01-01'))
    val_end=np.searchsorted(price.index,pd.Timestamp('2020-01-01'))
    split=max(WINDOW+PRED_HORIZON,train_end)-WINDOW-PRED_HORIZON+1
    val_split=max(split+WINDOW,val_end-WINDOW-PRED_HORIZON+1)

    # Target: forward returns
    y_fwd=logP[WINDOW+PRED_HORIZON-1:WINDOW+PRED_HORIZON-1+n_samp]-logP[WINDOW-1:WINDOW-1+n_samp]
    y_fwd=np.nan_to_num(y_fwd,nan=0,posinf=0,neginf=0); y_fwd=np.clip(y_fwd,-np.percentile(np.abs(y_fwd),99),np.percentile(np.abs(y_fwd),99))

    # Returns for classical factors (30-day)
    ret_vals=np.diff(logP,axis=0); ret_vals=np.vstack([np.zeros((1,n_a)),ret_vals])
    X_ret_tr=np.zeros((split,n_a,30),dtype=np.float32)
    X_ret_val=np.zeros((val_split-split,n_a,30),dtype=np.float32)
    X_ret_te=np.zeros((n_samp-val_split,n_a,30),dtype=np.float32)
    for i in range(split): X_ret_tr[i]=ret_vals[i+WINDOW-30:i+WINDOW].T
    for i,si in enumerate(range(split,val_split)): X_ret_val[i]=ret_vals[si+WINDOW-30:si+WINDOW].T
    for i,si in enumerate(range(val_split,n_samp)): X_ret_te[i]=ret_vals[si+WINDOW-30:si+WINDOW].T
    [np.clip(x,-np.percentile(np.abs(x),99),np.percentile(np.abs(x),99),out=x) for x in [X_ret_tr,X_ret_val,X_ret_te]]

    print(f"  {n_a} stocks x {n_days}d | Train:{split} Val:{val_split-split} Test:{n_samp-val_split}")

    # Return paths for sig computation (faster: compute sigs outside)
    return (logP, logV, vi, n_a, split, val_split, n_samp), \
           (X_ret_tr, X_ret_val, X_ret_te), \
           (y_fwd[:split], y_fwd[split:val_split], y_fwd[val_split:])

# ============================================================
# CLASSICAL FACTORS
# ============================================================
def classical_factors(X_ret):
    v5=X_ret[:,:,-5:].std(-1,keepdims=1); m5=X_ret[:,:,-5:].mean(-1,keepdims=1)
    m10=X_ret[:,:,-10:].mean(-1,keepdims=1); rsi=(X_ret[:,:,-5:]>0).mean(-1,keepdims=1)
    return np.concatenate([v5,m5,m10,rsi],-1)

# ============================================================
# SLIM META-TRANSFORMER
# ============================================================
class SlimMetaTrans(nn.Module):
    def __init__(self, n_a, sig_dim, cl_dim):
        super().__init__(); self.na=n_a; self.d=D_MODEL
        self.sig_proj=nn.Sequential(nn.Linear(sig_dim,self.d),nn.LayerNorm(self.d))
        self.sig_pos=nn.Parameter(torch.randn(1,5000,self.d)*0.02)
        self.cl_proj=nn.Sequential(nn.Linear(cl_dim,self.d),nn.LayerNorm(self.d))
        self.cl_tok=nn.Parameter(torch.randn(1,1,self.d)*0.02)
        self.cls=nn.Parameter(torch.randn(1,1,self.d)*0.02)
        self.layers=nn.ModuleList([nn.TransformerEncoderLayer(self.d,N_HEADS,D_FF,DROPOUT,'gelu',True,True) for _ in range(N_LAYERS)])
        self.ret_head=nn.Linear(self.d,n_a)
        self.unc_head=nn.Sequential(nn.Linear(self.d,256),nn.GELU(),nn.Linear(256,1))
    def forward(self,sigs,cl):
        B=sigs.shape[0]; st=self.sig_proj(sigs)+self.sig_pos[:,:self.na,:]
        cp=cl.mean(1); ct=self.cl_proj(cp).unsqueeze(1)+self.cl_tok
        x=torch.cat([self.cls.expand(B,-1,-1),st,ct],1)
        for l in self.layers: x=l(x)
        return self.ret_head(x[:,0]),self.unc_head(x[:,0]).squeeze(-1)

# ============================================================
# RUN
# ============================================================
def run_exp(name, path_data, ret_data, y_data):
    (logP,logV,vi,n_a,split,val_split,n_samp)=path_data
    (Xr_tr,Xr_val,Xr_te)=ret_data; (y_tr,y_val,y_te)=y_data

    print(f"  Computing signatures (batched, no path tensor)...")
    t0=time.time()
    sigs_tr,sigs_val,sigs_te,sig_dim=compute_signatures_batched(logP,logV,vi,n_a,WINDOW,SIG_DEPTH,split,val_split)
    print(f"  Sigs in {time.time()-t0:.0f}s (batched)")

    # Classical
    cl_tr=classical_factors(Xr_tr); cl_val=classical_factors(Xr_val); cl_te=classical_factors(Xr_te)
    cl_dim=cl_tr.shape[-1]

    # Normalize
    from sklearn.preprocessing import RobustScaler as RS
    sc_s=RS().fit(sigs_tr.reshape(-1,sig_dim)); sc_c=RS().fit(cl_tr.reshape(-1,cl_dim))
    sigs_tr_n=sc_s.transform(sigs_tr.reshape(-1,sig_dim)).reshape(sigs_tr.shape)
    sigs_val_n=sc_s.transform(sigs_val.reshape(-1,sig_dim)).reshape(sigs_val.shape)
    sigs_te_n=sc_s.transform(sigs_te.reshape(-1,sig_dim)).reshape(sigs_te.shape)
    cl_tr_n=sc_c.transform(cl_tr.reshape(-1,cl_dim)).reshape(cl_tr.shape)
    cl_val_n=sc_c.transform(cl_val.reshape(-1,cl_dim)).reshape(cl_val.shape)
    cl_te_n=sc_c.transform(cl_te.reshape(-1,cl_dim)).reshape(cl_te.shape)

    # Model
    model=SlimMetaTrans(n_a,sig_dim,cl_dim).to(GPU)
    n_p=sum(p.numel() for p in model.parameters()); print(f"  SlimMetaTrans: {n_p:,} params")
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS)

    tr_ds=torch.utils.data.TensorDataset(torch.FloatTensor(sigs_tr_n),torch.FloatTensor(cl_tr_n),torch.FloatTensor(y_tr))
    val_ds=torch.utils.data.TensorDataset(torch.FloatTensor(sigs_val_n),torch.FloatTensor(cl_val_n),torch.FloatTensor(y_val))
    tr_ld=DataLoader(tr_ds,BATCH,shuffle=True,drop_last=True)

    best_val=np.inf
    for ep in range(EPOCHS):
        model.train(); tl=0
        for s,c,y in tr_ld:
            rp,up=model(s.to(GPU),c.to(GPU)); loss=F.huber_loss(rp,y.to(GPU),delta=1.0)
            with torch.no_grad(): err=(rp-y.to(GPU)).abs().mean(1)
            loss=loss+0.1*F.mse_loss(up,err)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),2.0); opt.step(); tl+=loss.item()
        sch.step()
        # Val check (batched to avoid OOM)
        model.eval(); val_mse=0; nb=0
        with torch.no_grad():
            for i in range(0,len(sigs_val_n),BATCH):
                sv=torch.FloatTensor(sigs_val_n[i:i+BATCH]).to(GPU)
                cv=torch.FloatTensor(cl_val_n[i:i+BATCH]).to(GPU)
                yv=torch.FloatTensor(y_val[i:i+BATCH]).to(GPU)
                rv,_=model(sv,cv); val_mse+=F.mse_loss(rv,yv).item()*sv.shape[0]; nb+=sv.shape[0]
        val_mse/=nb
        if val_mse<best_val: best_val=val_mse
        if (ep+1)%10==0: print(f"    Epoch {ep+1:3d} | Loss: {tl/len(tr_ld):.6f} | Val: {val_mse:.6f}")
    print(f"  Best val MSE: {best_val:.6f}")

    # Test eval (batched)
    model.eval(); rps,ups=[],[]
    with torch.no_grad():
        for i in range(0,len(sigs_te_n),BATCH):
            rp,up=model(torch.FloatTensor(sigs_te_n[i:i+BATCH]).to(GPU),torch.FloatTensor(cl_te_n[i:i+BATCH]).to(GPU))
            rps.append(rp.cpu().numpy()); ups.append(up.cpu().numpy())
    rp=np.concatenate(rps); up=np.concatenate(ups)
    mse=float(np.mean((rp-y_te)**2)); err=np.abs(rp-y_te).mean(1); bl=np.abs(rp).mean(1)
    return {'name':name,'n':n_a,'sig_dim':sig_dim,'mse':mse,'model_r':spearmanr(up,err)[0],'pred_r':spearmanr(bl,err)[0]}

# ============================================================
# MAIN
# ============================================================
print("="*60); print("  OPTIMIZED META-TRANSFORMER"); print("="*60)
results=[]
markets=[
    ("US",N_STOCKS_US,"us_price_expanded.parquet","us_volume_expanded.parquet"),
    ("CN",N_STOCKS_CN,"cn_price.parquet","cn_volume.parquet")
]
for name,n,price_f,vol_f in markets:
    print(f"\n{'='*50}\n  {name}: loading {n} stocks\n{'='*50}")
    pd_data,ret_data,y_data=load_market(name,n,f"{DATA_DIR}/{price_f}",f"{DATA_DIR}/{vol_f}")
    results.append(run_exp(name,pd_data,ret_data,y_data))

print("\n"+"="*65); print("  FINAL RESULTS"); print("="*65)
for r in results:
    print(f"\n  {r['name']} ({r['n']} stocks, sig={r['sig_dim']}d)")
    print(f"  Meta-Transformer: MSE={r['mse']:.6f}  Model r={r['model_r']:.4f}  |Pred| r={r['pred_r']:.4f}")
    print(f"  vs |Pred|: {r['model_r']-r['pred_r']:+.4f}")
print("="*65)
