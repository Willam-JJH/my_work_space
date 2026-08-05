"""
Pattern information vs training epochs.
Shows attention pattern structure emerges during learning, not random noise.
Key measurements at each epoch checkpoint:
  - Pattern reconstruction R² (from input)
  - Pattern-error Meta r (predictive power)
  - Pattern entropy (concentration vs dispersion)
  - |Prediction| baseline r (reference)
"""
import numpy as np, pandas as pd, time, warnings
warnings.filterwarnings('ignore')
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DEV='xpu'; B=128; L=30; SEED=42; N_STOCKS=200
CHECKPOINTS=[1,2,3,5,10,20,30,40,50,60]

# Data
df=pd.read_parquet('D:/code/data/cn_market/daily_returns.parquet')
rm=df.pivot(index='trddt',columns='stkcd',values='dretwd')
rm=rm.dropna(axis=1,thresh=int(len(rm)*0.8)).ffill().dropna(axis=0)
R=rm.values.astype(np.float32)[:,:N_STOCKS]
R=np.nan_to_num((R-np.nanmean(R,0,keepdims=True))/(np.nanstd(R,0,keepdims=True)+1e-8),0)
n=len(R)-L-1
X=np.lib.stride_tricks.sliding_window_view(R,L,axis=0)[:n].transpose(0,2,1).astype(np.float32)
y=R[L+1:][:n].astype(np.float32); tr=int(n*0.7)

# Base Transformer
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

print("Pattern Information vs Training Epochs")
print("="*60)
np.random.seed(SEED); torch.manual_seed(SEED)
Xt=torch.FloatTensor(X[:tr]).to(DEV); yt=torch.FloatTensor(y[:tr]).to(DEV)
Xe=torch.FloatTensor(X[tr:]).to(DEV); ye=torch.FloatTensor(y[tr:]).to(DEV)
ld=DataLoader(TensorDataset(Xt,yt),batch_size=B,shuffle=True)
X_tr_flat=X[:tr].reshape(tr,-1); X_te_flat=X[tr:].reshape(n-tr,-1)

records=[]
for ep in range(1,61):
    if ep==1:
        base=Base(N_STOCKS).to(DEV); opt=torch.optim.AdamW(base.parameters(),lr=3e-4,weight_decay=1e-5)

    base.train()
    for bx,by in ld: opt.zero_grad(); l=nn.HuberLoss(delta=1.0)(base(bx,False)[0],by); l.backward(); torch.nn.utils.clip_grad_norm_(base.parameters(),2.0); opt.step()

    if ep not in CHECKPOINTS: continue

    base.eval()
    with torch.no_grad():
        rt,_=base(Xt,False); re,_=base(Xe,False)
        _,ht=base(Xt,True); pt=base.get_pats()['3'].cpu().numpy()
        _,he=base(Xe,True); p_te=base.get_pats()['3'].cpu().numpy()
    at_np=torch.abs(rt-yt).mean(-1).cpu().numpy()
    ae_np=torch.abs(re-ye).mean(-1).cpu().numpy()
    abs_re=torch.abs(re).mean(-1).cpu().numpy()

    # 1. |Prediction| baseline
    r_pred,_=spearmanr(abs_re, ae_np)

    # 2. Pattern reconstruction R² (PCA 16 → Ridge on input X)
    p_tr_flat=pt.reshape(len(pt),-1); p_te_flat=p_te.reshape(len(p_te),-1)
    pca=PCA(16, random_state=SEED)
    p_tr_pca=pca.fit_transform(p_tr_flat); p_te_pca=pca.transform(p_te_flat)
    r2=Ridge(1.0).fit(p_tr_pca, at_np).score(p_te_pca, ae_np)
    # Reconstruction from input data
    Xr_tr,Xr_v,pr_tr,pr_v=train_test_split(X_tr_flat, p_tr_pca, test_size=0.2, random_state=SEED)
    recon_model=Ridge(1.0).fit(Xr_tr, pr_tr)
    r2_recon=r2_score(p_te_flat, pca.inverse_transform(recon_model.predict(X_te_flat)))

    # 3. Pattern entropy (concentration)
    entropies=[]
    for h in range(8):
        w=pt[:,h].reshape(len(pt),-1); w=w/np.sum(w,1,keepdims=True)
        entropies.append(-np.sum(w*np.log(w+1e-10),1).mean())
    entropy=np.mean(entropies)

    # 4. Pattern sparsity (fraction of near-zero weights)
    sparsity=(np.abs(pt)<0.01).mean()

    # 5. Base prediction MSE
    base_mse=F.mse_loss(re, ye).item()

    records.append({'epoch':ep, 'r_pred':r_pred, 'r2_recon':r2_recon, 'r2_pattern_error':r2,
                    'entropy':entropy, 'sparsity':sparsity, 'base_mse':base_mse})
    print(f"  Ep {ep:3d} | |Pred| r={r_pred:+.3f} | Recon R²={r2_recon:+.3f} | Pat→Error R²={r2:+.3f} | Entropy={entropy:.2f} | Sparsity={sparsity:.4f} | Base MSE={base_mse:.4f}")

# Report
print(f"\n{'='*60}")
print("FINAL TABLE: Pattern Information vs Training")
print(f"{'='*60}")
print(f"{'Ep':>4} {'|Pred| r':>8} {'Recon R²':>8} {'Pat→Err R²':>10} {'Entropy':>8} {'Sparsity':>8} {'Base MSE':>8}")
print("-"*60)
for r in records:
    print(f"{r['epoch']:4d} {r['r_pred']:>+8.4f} {r['r2_recon']:>+8.4f} {r['r2_pattern_error']:>+10.4f} {r['entropy']:>8.2f} {r['sparsity']:>8.4f} {r['base_mse']:>8.4f}")

# Key insight
early_r2=records[0]['r2_recon']
late_r2=records[-1]['r2_recon']
early_pred=records[0]['r_pred']
late_pred=records[-1]['r_pred']
print(f"\nKey Insight:")
print(f"  Recon R²: {early_r2:+.4f} (ep1) → {late_r2:+.4f} (ep60) = {'Pattern becomes LESS reconstructible with training' if late_r2<early_r2 else 'Pattern becomes MORE reconstructible'}")
print(f"  |Pred| r:  {early_pred:+.4f} (ep1) → {late_pred:+.4f} (ep60) = {'Prediction heuristic improves with training' if late_pred>early_pred else 'Prediction heuristic unchanged'}")
print(f"  If Recon R² decreases while |Pred| improves, Pattern information is learned, not pre-existing in data.")
