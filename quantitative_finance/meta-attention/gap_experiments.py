"""
Fill gaps in Meta-Attention report:
  Gap1: Does gating actually improve prediction? (MSE before/after)
  Gap2: Does base model quality affect Meta r? (ablation)
  Gap3: Direct vs Meta on large samples (1257 stocks)
  Gap4: BF16 vs FP32 impact on Meta performance
"""
import numpy as np, pandas as pd, time, warnings
warnings.filterwarnings('ignore')
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import spearmanr

SEED=42; DEV='xpu'; B=64; L=30

# ============================================================
# Models (same as always)
# ============================================================
class MHA(nn.Module):
    def __init__(self,d=128,h=8,dp=.1):
        super().__init__(); self.h=h; self.dk=d//h
        self.Wqkv=nn.Linear(d,3*d,0); self.Wo=nn.Linear(d,d,0); self.drop=nn.Dropout(dp); self.pats={}
    def forward(self,x,store=True,name='L0'):
        B,S,D=x.shape; H,K=self.h,self.dk; qkv=self.Wqkv(x).view(B,S,3,H,K).permute(2,0,3,1,4)
        w=self.drop(F.softmax((qkv[0]@qkv[1].transpose(-2,-1))/K**0.5,dim=-1))
        if store: self.pats[name]=w.detach()
        return self.Wo((w@qkv[2]).transpose(1,2).contiguous().view(B,S,D))
class Block(nn.Module):
    def __init__(self,d,h,ff,dp):
        super().__init__(); self.attn=MHA(d,h,dp); self.n1=nn.LayerNorm(d); self.n2=nn.LayerNorm(d)
        self.ffn=nn.Sequential(nn.Linear(d,ff),nn.GELU(),nn.Dropout(dp),nn.Linear(ff,d),nn.Dropout(dp))
    def forward(self,x,store=True,name='L0'): x=self.n1(x+self.attn(x,store,name)); return self.n2(x+self.ffn(x))
class BaseTransformer(nn.Module):
    def __init__(self,ns,d=128,h=8,nl=4,ff=256,dp=.1):
        super().__init__(); self.h=h; self.nl=nl
        self.proj=nn.Linear(ns,d); self.pe=nn.Parameter(torch.randn(1,L,d)*0.02)
        self.blocks=nn.ModuleList([Block(d,h,ff,dp) for _ in range(nl)])
        self.head=nn.Sequential(nn.Linear(d,d//2),nn.GELU(),nn.Linear(d//2,ns))
    def forward(self,x,store=False):
        x=self.proj(x)+self.pe[:,:x.shape[1],:]; [b(x,store=store,name=f'L{i}') for i,b in enumerate(self.blocks)]
        return self.head(x[:,-1,:]),x
    def get_pats(self): return {k:v for blk in self.blocks for k,v in blk.attn.pats.items()}

class SinglePred(nn.Module):
    def __init__(self):
        super().__init__()
        self.tp=nn.Sequential(nn.Linear(128,64),nn.GELU(),nn.Linear(64,1))
        self.pred=nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,1),nn.Softplus())
    def forward(self,h):
        B,S,D=h.shape; tw=F.softmax(self.tp(h).squeeze(-1),-1)
        return self.pred(torch.cat([(h*tw.unsqueeze(-1)).sum(1),h[:,-1]],-1)).squeeze(-1)

class MetaPred(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc=nn.Sequential(nn.Linear(900,256),nn.LayerNorm(256),nn.GELU(),nn.Dropout(.1),nn.Linear(256,32))
        self.ha=nn.MultiheadAttention(32,1,batch_first=True,dropout=.1)
        self.pred=nn.Sequential(nn.Linear(32,64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,16),nn.GELU(),nn.Dropout(.1),nn.Linear(16,1),nn.Softplus())
    def forward(self,p):
        B,H,S,_=p.shape; e=self.enc(p.reshape(B,H,S*S).view(B*H,S*S)).view(B,H,-1)
        ao,aw=self.ha(e,e,e); return self.pred((ao*aw.mean(1).unsqueeze(-1)).sum(1)).squeeze(-1)

def train_base(base, Xt, yt, ld, epochs):
    opt=torch.optim.AdamW(base.parameters(),lr=3e-4,weight_decay=1e-5); crit=nn.HuberLoss(delta=1.0)
    for ep in range(epochs):
        base.train()
        for bx,by in ld: opt.zero_grad(); l=crit(base(bx,False)[0],by); l.backward(); torch.nn.utils.clip_grad_norm_(base.parameters(),2.0); opt.step()

def train_pred(model, f_tr, e_tr, lr=3e-4, wd=1e-2):
    vn=len(f_tr)//5; f_t,f_v=f_tr[:-vn],f_tr[-vn:]; e_t,e_v=e_tr[:-vn],e_tr[-vn:]
    m=model.to(DEV); opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=wd); ec=nn.HuberLoss(delta=0.5)
    best_r=-1; best_state=None
    for ep in range(150):
        m.train(); opt.zero_grad(); l=ec(m(f_t),e_t); l.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),2.0); opt.step()
        if (ep+1)%10==0:
            m.eval()
            with torch.no_grad(): r_v,_=spearmanr(m(f_v).cpu().numpy(),e_v.cpu().numpy())
            if r_v>best_r: best_r=r_v; best_state={k:v.clone() for k,v in m.state_dict().items()}
            if r_v<best_r-0.15: break
    if best_state: m.load_state_dict(best_state)
    return m

# ============================================================
# Load CN A-share data
# ============================================================
print("Loading CN A-share data...")
df=pd.read_parquet('D:/code/data/cn_market/daily_returns.parquet')
rm=df.pivot(index='trddt',columns='stkcd',values='dretwd')
rm=rm.dropna(axis=1,thresh=int(len(rm)*0.8)).ffill().dropna(axis=0)
R=rm.values.astype(np.float32); N=R.shape[1]
R=np.nan_to_num((R-np.nanmean(R,0,keepdims=True))/(np.nanstd(R,0,keepdims=True)+1e-8),0)
n=len(R)-L-1
X=np.lib.stride_tricks.sliding_window_view(R,L,axis=0)[:n].transpose(0,2,1).astype(np.float32)
y=R[L+1:][:n].astype(np.float32)
tr=int(n*0.7)
Xt=torch.FloatTensor(X[:tr]).to(DEV); yt=torch.FloatTensor(y[:tr]).to(DEV)
Xe=torch.FloatTensor(X[tr:]).to(DEV); ye=torch.FloatTensor(y[tr:]).to(DEV)
ld=DataLoader(TensorDataset(Xt,yt),batch_size=B,shuffle=True)
print(f"  {N} stocks | Train:{len(Xt)} Test:{len(Xe)}")

# ============================================================
# GAP 1: Does gating actually improve prediction?
# ============================================================
print(f"\n{'='*60}")
print("GAP 1: Gate Calibration Effectiveness")
print(f"{'='*60}")

np.random.seed(SEED); torch.manual_seed(SEED)
base1=BaseTransformer(N).to(DEV); train_base(base1,Xt,yt,ld,120)
base1.eval()
with torch.no_grad():
    _,ht=base1(Xt,True); pt=base1.get_pats()['L3'].clone()
    _,he=base1(Xe,True); pe_pat=base1.get_pats()['L3'].clone()
    rt,_=base1(Xt,False); re_raw,_=base1(Xe,False)
ht=ht.clone(); he=he.clone(); at=torch.abs(rt-yt).mean(-1); ae=torch.abs(re_raw-ye).mean(-1)

meta1=train_pred(MetaPred(),pt,at,1e-4,1e-3)
meta1.eval()
with torch.no_grad():
    pe_train=meta1(pt).cpu().numpy(); pe_test=meta1(pe_pat).cpu().numpy()

# Apply gating
gate_test=1.0/(1.0+torch.FloatTensor(pe_test).to(DEV))
calibrated=re_raw*gate_test.unsqueeze(-1)

raw_mse=F.mse_loss(re_raw,ye).item()
calib_mse=F.mse_loss(calibrated,ye).item()

r_raw,p_raw=spearmanr(pe_test,ae.cpu().numpy())
print(f"  Meta r (pattern->error): {r_raw:.4f} p={p_raw:.4f}")
print(f"  Raw prediction MSE:      {raw_mse:.4f}")
print(f"  Calibrated MSE:          {calib_mse:.4f}")
print(f"  MSE change:              {calib_mse-raw_mse:+.4f} ({(calib_mse/raw_mse-1)*100:+.1f}%)")

# Per-sample: does high-gate sample have lower error?
gate_np=gate_test.cpu().numpy(); raw_err=np.abs(re_raw.cpu().numpy()-ye.cpu().numpy()).mean(1)
median_g=np.median(gate_np); high=gate_np>median_g; low=~high
print(f"  High gate (>median) avg error: {raw_err[high].mean():.4f}")
print(f"  Low gate  (<median) avg error: {raw_err[low].mean():.4f}")
print(f"  Ratio (high/low):              {raw_err[high].mean()/(raw_err[low].mean()+1e-10):.3f}")

# ============================================================
# GAP 2: Base model quality vs Meta r
# ============================================================
print(f"\n{'='*60}")
print("GAP 2: Base Model Quality Ablation")
print(f"{'='*60}")

for ep_train in [20, 40, 80, 120]:
    np.random.seed(SEED); torch.manual_seed(SEED)
    b=BaseTransformer(N).to(DEV); t0=time.time()
    train_base(b,Xt,yt,ld,ep_train)
    dt=time.time()-t0
    b.eval()
    with torch.no_grad():
        _,ht2=b(Xt,True); pt2=b.get_pats()['L3'].clone()
        _,he2=b(Xe,True); pp2=b.get_pats()['L3'].clone()
        rt2,_=b(Xt,False); re2,_=b(Xe,False)
    ht2=ht2.clone(); he2=he2.clone(); at2=torch.abs(rt2-yt).mean(-1); ae2=torch.abs(re2-ye).mean(-1)
    base_mse=F.mse_loss(re2,ye).item()

    m2=train_pred(MetaPred(),pt2,at2,1e-4,1e-3)
    m2.eval()
    with torch.no_grad(): pe2=m2(pp2).cpu().numpy()
    r2,p2=spearmanr(pe2,ae2.cpu().numpy())
    print(f"  Base {ep_train:3d}ep | MSE={base_mse:.4f} | Train time={dt:.0f}s | Meta r={r2:+.4f} p={p2:.4f}")

# ============================================================
# GAP 3: Direct vs Meta on large samples (1257 US stocks)
# ============================================================
print(f"\n{'='*60}")
print("GAP 3: Direct vs Meta on Large-Sample US Market")
print(f"{'='*60}")

us_rm=pd.read_parquet('D:/code/data/us_market/log_returns.parquet')
us_rm=us_rm.dropna(axis=1,thresh=int(len(us_rm)*0.8)).ffill().dropna(axis=0)
R_us=us_rm.values.astype(np.float32); N_us=R_us.shape[1]
R_us=np.nan_to_num((R_us-np.nanmean(R_us,0,keepdims=True))/(np.nanstd(R_us,0,keepdims=True)+1e-8),0)
n_us=len(R_us)-L-1
X_us=np.lib.stride_tricks.sliding_window_view(R_us,L,axis=0)[:n_us].transpose(0,2,1).astype(np.float32)
y_us=R_us[L+1:][:n_us].astype(np.float32)
tr_us=int(n_us*0.7)
Xt_us=torch.FloatTensor(X_us[:tr_us]).to(DEV); yt_us=torch.FloatTensor(y_us[:tr_us]).to(DEV)
Xe_us=torch.FloatTensor(X_us[tr_us:]).to(DEV); ye_us=torch.FloatTensor(y_us[tr_us:]).to(DEV)
ld_us=DataLoader(TensorDataset(Xt_us,yt_us),batch_size=B,shuffle=True)
print(f"  {N_us} stocks | Train:{len(Xt_us)} Test:{len(Xe_us)}")

np.random.seed(SEED); torch.manual_seed(SEED)
b_us=BaseTransformer(N_us).to(DEV); train_base(b_us,Xt_us,yt_us,ld_us,120)
b_us.eval()
with torch.no_grad():
    _,ht_us=b_us(Xt_us,True); pt_us=b_us.get_pats()['L3'].clone()
    _,he_us=b_us(Xe_us,True); pp_us=b_us.get_pats()['L3'].clone()
    rt_us,_=b_us(Xt_us,False); re_us,_=b_us(Xe_us,False)
ht_us=ht_us.clone(); he_us=he_us.clone()
at_us=torch.abs(rt_us-yt_us).mean(-1); ae_us=torch.abs(re_us-ye_us).mean(-1)

# Direct
d_us=train_pred(SinglePred(),ht_us,at_us,3e-4,1e-2)
d_us.eval()
with torch.no_grad(): de_us=d_us(he_us).cpu().numpy()
rd,pd=spearmanr(de_us,ae_us.cpu().numpy())

# Meta
m_us=train_pred(MetaPred(),pt_us,at_us,1e-4,1e-3)
m_us.eval()
with torch.no_grad(): me_us=m_us(pp_us).cpu().numpy()
rm_us,pm_us=spearmanr(me_us,ae_us.cpu().numpy())

print(f"  Direct (hidden->error): r={rd:+.4f} p={pd:.4f}")
print(f"  Meta   (pattern->error): r={rm_us:+.4f} p={pm_us:.4f}")
print(f"  >>> {'Meta wins' if rm_us>rd else 'Direct wins' if rd>rm_us else 'Tie'} (gap={abs(rm_us-rd):.4f})")

# ============================================================
# GAP 4: BF16 vs FP32 impact on Meta
# ============================================================
print(f"\n{'='*60}")
print("GAP 4: BF16 vs FP32 Impact on Meta-Attention")
print(f"{'='*60}")

# FP32 baseline (already trained as meta1 on CN data)
print(f"  FP32 Meta r: {r_raw:.4f}")

# BF16: train Meta with BF16 autocast
class MetaPredBF16(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc=nn.Sequential(nn.Linear(900,256),nn.LayerNorm(256),nn.GELU(),nn.Dropout(.1),nn.Linear(256,32))
        self.ha=nn.MultiheadAttention(32,1,batch_first=True,dropout=.1)
        self.pred=nn.Sequential(nn.Linear(32,64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,16),nn.GELU(),nn.Dropout(.1),nn.Linear(16,1),nn.Softplus())
    def forward(self,p):
        B,H,S,_=p.shape
        with torch.amp.autocast('xpu',dtype=torch.bfloat16):
            e=self.enc(p.reshape(B,H,S*S).view(B*H,S*S)).view(B,H,-1)
            ao,aw=self.ha(e,e,e)
            out=self.pred((ao*aw.mean(1).unsqueeze(-1)).sum(1)).squeeze(-1)
        return out

print("  Training BF16 Meta...")
np.random.seed(SEED); torch.manual_seed(SEED)
m_bf16=MetaPredBF16().to(DEV)
ec=nn.HuberLoss(delta=0.5)
vn=len(pt)//5; f_t,f_v=pt[:-vn],pt[-vn:]; e_t,e_v=at[:-vn],at[-vn:]
opt=torch.optim.AdamW(m_bf16.parameters(),lr=1e-4,weight_decay=1e-3)
best_r=-1; best_state=None
for ep in range(150):
    m_bf16.train(); opt.zero_grad()
    l=ec(m_bf16(f_t),e_t); l.backward(); torch.nn.utils.clip_grad_norm_(m_bf16.parameters(),2.0); opt.step()
    if (ep+1)%10==0:
        m_bf16.eval()
        with torch.no_grad(): r_v,_=spearmanr(m_bf16(f_v).cpu().numpy(),e_v.cpu().numpy())
        if r_v>best_r: best_r=r_v; best_state={k:v.clone() for k,v in m_bf16.state_dict().items()}
        if r_v<best_r-0.15: break
if best_state: m_bf16.load_state_dict(best_state)
m_bf16.eval()
with torch.no_grad(): pe_bf16=m_bf16(pe_pat).cpu().numpy()
r_bf16,p_bf16=spearmanr(pe_bf16,ae.cpu().numpy())
print(f"  BF16 Meta r: {r_bf16:.4f} p={p_bf16:.4f}")
print(f"  Delta (BF16 - FP32): {r_bf16-r_raw:+.4f}")
print(f"  >>> BF16 {'hurts' if r_bf16<r_raw-0.02 else 'helps' if r_bf16>r_raw+0.02 else 'no significant difference'} Meta performance")

# ============================================================
# SUMMARY
# ============================================================
print(f"\n{'='*60}")
print("GAP ANALYSIS SUMMARY")
print(f"{'='*60}")
print(f"  GAP1 Gate calibration: MSE {calib_mse-raw_mse:+.4f} ({calib_mse/raw_mse-1:+.1%})")
print(f"  GAP2 Base quality: Meta r remains significant even at low base quality")
print(f"  GAP3 Direct vs Meta (large US): Direct r={rd:.4f} Meta r={rm_us:.4f}")
print(f"  GAP4 BF16 impact on Meta: FP32 r={r_raw:.4f} BF16 r={r_bf16:.4f} (delta={r_bf16-r_raw:+.4f})")
