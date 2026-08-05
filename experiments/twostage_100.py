"""Two-stage: return prediction → frozen features → uncertainty MLP"""
import numpy as np; import pandas as pd; import torch; import torch.nn as nn; import torch.nn.functional as F
from torch.utils.data import DataLoader; import signatory; from scipy.stats import spearmanr
import warnings; warnings.filterwarnings('ignore')
GPU=torch.device('xpu'); CPU=torch.device('cpu')
torch.manual_seed(42); np.random.seed(42)
print(f'GPU: {torch.xpu.get_device_name(0)}')

ret=pd.read_parquet('D:/code/data/us_returns.parquet')
comp=ret.notna().sum()/len(ret); top100=comp.nlargest(100).index.tolist()
ret=ret[top100].ffill().fillna(0); ret_vals=ret.values.astype(np.float32); n_assets=len(top100)
n_samp=ret_vals.shape[0]-30
X=np.zeros((n_samp,n_assets,30),dtype=np.float32); y=np.zeros((n_samp,n_assets),dtype=np.float32)
for i in range(n_samp): X[i]=ret_vals[i:i+30].T; y[i]=ret_vals[i+30]
mu=X.mean(-1,keepdims=1); st=X.std(-1,keepdims=1)+1e-8; X=(X-mu)/st
y=np.clip(y,-np.percentile(np.abs(y),99),np.percentile(np.abs(y),99))
split=int(n_samp*0.7)

sig_full_dim=signatory.signature_channels(n_assets,1)
X_sig=torch.FloatTensor(X).transpose(1,2).to(CPU)
sig_full=np.zeros((n_samp,sig_full_dim),dtype=np.float32)
for i in range(0,n_samp,64): sig_full[i:i+64]=signatory.signature(X_sig[i:i+64],1,basepoint=True).cpu().numpy()
from sklearn.preprocessing import StandardScaler; from sklearn.decomposition import PCA
sig_full=StandardScaler().fit_transform(sig_full); sig=PCA(64).fit_transform(sig_full)

v5=X[:,:,-5:].std(-1,keepdims=1); v10=X[:,:,-10:].std(-1,keepdims=1)
m5=X[:,:,-5:].mean(-1,keepdims=1); m10=X[:,:,-10:].mean(-1,keepdims=1)
rsi=(X[:,:,-5:]>0).mean(-1,keepdims=1); tech=np.concatenate([v5,v10,m5,m10,rsi],-1)
X_tr,X_te=X[:split],X[split:]; y_tr,y_te=y[:split],y[split:]
sig_tr,sig_te=sig[:split],sig[split:]; tech_tr,tech_te=tech[:split],tech[split:]

class AssetEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(35,64),nn.GELU(),nn.Linear(64,32),nn.LayerNorm(32))
    def forward(self,x): return self.net(x)

class RetPredictor(nn.Module):
    def __init__(self):
        super().__init__(); self.d=128
        self.aenc=AssetEnc(); self.pos=nn.Parameter(torch.randn(1,n_assets,32)*0.02)
        self.sproj=nn.Sequential(nn.Linear(64,self.d),nn.LayerNorm(self.d))
        self.agg=nn.Sequential(nn.Linear(n_assets*32+self.d,self.d),nn.GELU(),nn.LayerNorm(self.d))
        self.extra=nn.Parameter(torch.randn(1,4,self.d)*0.02); self.cls=nn.Parameter(torch.randn(1,1,self.d)*0.02)
        self.trans=nn.TransformerEncoder(nn.TransformerEncoderLayer(self.d,4,256,dropout=0.1,activation='gelu',batch_first=True,norm_first=True),num_layers=3)
        self.head=nn.Linear(self.d,n_assets)
    def forward(self,r,t,s):
        B,N,L=r.shape; rf=r.reshape(B*N,L); tf=t.reshape(B*N,5)
        a=self.aenc(torch.cat([rf,tf],-1)).view(B,N,32)+self.pos[:,:N,:]
        ag=a.reshape(B,N*32); se=self.sproj(s)
        ag2=self.agg(torch.cat([ag,se],-1)).unsqueeze(1)
        tk=torch.cat([self.cls.expand(B,-1,-1),ag2,self.extra.expand(B,-1,-1)],1)
        return self.head(self.trans(tk)[:,0])
    def extract_features(self,r,t,s):
        B,N,L=r.shape; rf=r.reshape(B*N,L); tf=t.reshape(B*N,5)
        a=self.aenc(torch.cat([rf,tf],-1)).view(B,N,32)+self.pos[:,:N,:]
        ag=a.reshape(B,N*32); se=self.sproj(s)
        return self.agg(torch.cat([ag,se],-1))

print(f'Stage 1: Return predictor, {n_assets} stocks')
rpred=RetPredictor().to(GPU); opt=torch.optim.AdamW(rpred.parameters(),lr=1e-3,weight_decay=1e-4)
sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,80)
tr_ds=torch.utils.data.TensorDataset(torch.FloatTensor(X_tr),torch.FloatTensor(tech_tr),torch.FloatTensor(sig_tr),torch.FloatTensor(y_tr))
tr_ld=DataLoader(tr_ds,32,shuffle=True,drop_last=True)
for ep in range(80):
    rpred.train(); tl=0
    for x,tech,sig,y in tr_ld:
        pred=rpred(x.to(GPU),tech.to(GPU),sig.to(GPU)); loss=F.huber_loss(pred,y.to(GPU),delta=1.0)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(rpred.parameters(),2.0); opt.step(); tl+=loss.item()
    sch.step()
    if (ep+1)%20==0: print(f'  Epoch {ep+1} | Loss: {tl/len(tr_ld):.6f}')

rpred.eval()
with torch.no_grad(): pred_r=rpred(torch.FloatTensor(X_te).to(GPU),torch.FloatTensor(tech_te).to(GPU),torch.FloatTensor(sig_te).to(GPU)).detach().cpu().numpy()
mse=float(np.mean((pred_r-y_te)**2)); err_te=np.abs(pred_r-y_te).mean(1); abs_bl=np.abs(pred_r).mean(1)
print(f'  MSE: {mse:.6f} | |Pred| r: {spearmanr(abs_bl,err_te)[0]:.4f}')

print('Stage 2: Uncertainty predictor')
rpred.eval()
feat_tr_list,err_tr_list=[],[]
with torch.no_grad():
    for x,tech,sig,y in tr_ld:
        f=rpred.extract_features(x.to(GPU),tech.to(GPU),sig.to(GPU))
        p=rpred(x.to(GPU),tech.to(GPU),sig.to(GPU))
        feat_tr_list.append(f.detach().cpu().numpy())
        err_tr_list.append((p-y.to(GPU)).abs().mean(1).detach().cpu().numpy())
feat_tr=np.concatenate(feat_tr_list); err_tr=np.concatenate(err_tr_list)

feat_te_list=[]
te_ld=DataLoader(torch.utils.data.TensorDataset(torch.FloatTensor(X_te),torch.FloatTensor(tech_te),torch.FloatTensor(sig_te)),32,shuffle=False)
with torch.no_grad():
    for x,tech,sig in te_ld:
        f=rpred.extract_features(x.to(GPU),tech.to(GPU),sig.to(GPU))
        feat_te_list.append(f.detach().cpu().numpy())
feat_te=np.concatenate(feat_te_list)

unc=nn.Sequential(nn.Linear(128,512),nn.GELU(),nn.Dropout(0.25),
                  nn.Linear(512,256),nn.GELU(),nn.Dropout(0.25),
                  nn.Linear(256,128),nn.GELU(),nn.Linear(128,1)).to(GPU)
opt_u=torch.optim.AdamW(unc.parameters(),lr=1e-3,weight_decay=1e-3)
for ep in range(100):
    unc.train()
    for i in range(0,len(feat_tr),32):
        fb=torch.FloatTensor(feat_tr[i:i+32]).to(GPU); eb=torch.FloatTensor(err_tr[i:i+32]).to(GPU)
        loss=F.mse_loss(unc(fb).squeeze(),eb); opt_u.zero_grad(); loss.backward(); opt_u.step()
    if (ep+1)%25==0: print(f'  Epoch {ep+1} | Loss: {loss.item():.6f}')

unc.eval()
with torch.no_grad(): pred_u=unc(torch.FloatTensor(feat_te).to(GPU)).detach().cpu().numpy().squeeze()
def sr(a,b): return float(spearmanr(a,b)[0])
print('='*55)
print(f'  TWO-STAGE — {n_assets} Stocks')
print('='*55)
print(f'  Return MSE:  {mse:.6f}')
print(f'  |Pred| r:    {sr(abs_bl,err_te):.4f}')
print(f'  Model Unc r: {sr(pred_u,err_te):.4f}')
print(f'  Delta:       {sr(pred_u,err_te)-sr(abs_bl,err_te):+.4f}')
print('='*55)
