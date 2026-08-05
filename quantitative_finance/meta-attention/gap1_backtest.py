"""
GAP1 Practical Backtest: Does Meta-Attention calibration improve trading?
CN A-Share, long-short portfolio, with transaction costs.
"""
import numpy as np, pandas as pd, time, warnings
warnings.filterwarnings('ignore')
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import spearmanr

DEV='xpu'; B=256; L=30; SEED=42; N_STOCKS=200
np.random.seed(SEED); torch.manual_seed(SEED)

df=pd.read_parquet('D:/code/data/cn_market/daily_returns.parquet')
rm=df.pivot(index='trddt',columns='stkcd',values='dretwd')
rm=rm.dropna(axis=1,thresh=int(len(rm)*0.7)).ffill().dropna(axis=0)
R=rm.values.astype(np.float32); N=min(R.shape[1],N_STOCKS)
idx=R.shape[1]-N; R=R[:,:N]
R=np.nan_to_num((R-np.nanmean(R,0,keepdims=True))/(np.nanstd(R,0,keepdims=True)+1e-8),0)
n=len(R)-L-1
X=np.lib.stride_tricks.sliding_window_view(R,L,axis=0)[:n].transpose(0,2,1).astype(np.float32)
y=R[L+1:][:n].astype(np.float32); tr=int(n*0.7)
Xt=torch.FloatTensor(X[:tr]).to(DEV); yt=torch.FloatTensor(y[:tr]).to(DEV)
Xe=torch.FloatTensor(X[tr:]).to(DEV); ye=torch.FloatTensor(y[tr:]).to(DEV)
ld=DataLoader(TensorDataset(Xt,yt),batch_size=B,shuffle=True)
print(f"{N} stocks | Train:{len(Xt)} Test:{len(Xe)}")

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

class MetaPred(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc=nn.Sequential(nn.Linear(900,256),nn.LayerNorm(256),nn.GELU(),nn.Dropout(.1),nn.Linear(256,32))
        self.ha=nn.MultiheadAttention(32,1,batch_first=True,dropout=.1)
        self.pred=nn.Sequential(nn.Linear(32,64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,16),nn.GELU(),nn.Dropout(.1),nn.Linear(16,1),nn.Softplus())
    def forward(self,p):
        B,H,S,_=p.shape; e=self.enc(p.reshape(B,H,S*S).view(B*H,S*S)).view(B,H,-1)
        ao,aw=self.ha(e,e,e); return self.pred((ao*aw.mean(1).unsqueeze(-1)).sum(1)).squeeze(-1)

# Train
base=Base(N).to(DEV); opt=torch.optim.AdamW(base.parameters(),lr=3e-4,weight_decay=1e-5)
crit=nn.HuberLoss(delta=1.0)
for ep in range(60):
    base.train()
    for bx,by in ld: opt.zero_grad(); l=crit(base(bx,False)[0],by); l.backward(); torch.nn.utils.clip_grad_norm_(base.parameters(),2.0); opt.step()

base.eval()
with torch.no_grad():
    rt,_=base(Xt,True); pt=base.get_pats()['3'].clone()
    re,_=base(Xe,True); pp=base.get_pats()['3'].clone()
at=torch.abs(rt-yt).mean(-1); ae=torch.abs(re-ye).mean(-1)

# Train Meta
vn=len(at)//5; f_t,f_v=pt[:-vn],pt[-vn:]; e_t,e_v=at[:-vn],at[-vn:]
meta=MetaPred().to(DEV); om=torch.optim.AdamW(meta.parameters(),lr=1e-4,weight_decay=1e-3)
ec=nn.HuberLoss(delta=0.5); br,sb=-1,None
for ep in range(80):
    meta.train(); om.zero_grad(); l=ec(meta(f_t),e_t); l.backward(); torch.nn.utils.clip_grad_norm_(meta.parameters(),2.0); om.step()
    if (ep+1)%20==0:
        meta.eval()
        with torch.no_grad(): rv,_=spearmanr(meta(f_v).cpu().numpy(),e_v.cpu().numpy())
        if rv>br: br=rv; sb={k:v.clone() for k,v in meta.state_dict().items()}
if sb: meta.load_state_dict(sb)

meta.eval()
with torch.no_grad():
    pe=meta(pp).cpu().numpy()

# Gate
gate=1.0/(1.0+pe)  # (n_test,)
raw_pred=re.cpu().numpy(); y_true=ye.cpu().numpy()

print(f"Meta r={br:.4f} | Gate mean={gate.mean():.3f}±{gate.std():.3f}")

# ===== BACKTEST =====
TOP_K=N//5  # top 20%
TC=0.001   # 10bp transaction cost per trade

def backtest(predictions, gate_signal=None, name='', tc=TC):
    """Long-short: long top 20%, short bottom 20%, daily rebalancing."""
    T, S = predictions.shape
    daily_ret = []; turnover = []; positions = []
    prev_long = set(); prev_short = set()

    for t in range(T):
        pred_t = predictions[t]
        long_idx = set(np.argsort(pred_t)[-TOP_K:])
        short_idx = set(np.argsort(pred_t)[:TOP_K])

        # Returns
        long_ret = y_true[t, list(long_idx)].mean()
        short_ret = -y_true[t, list(short_idx)].mean()
        raw_ret = long_ret + short_ret

        # Position sizing via gate
        pos_weight = np.clip(gate_signal[t], 0.1, 1.5) if gate_signal is not None else 1.0
        ret_t = raw_ret * pos_weight

        # Turnover cost
        if t > 0:
            turn = len(long_idx - prev_long) + len(prev_long - long_idx) + len(short_idx - prev_short) + len(prev_short - short_idx)
            turnover.append(turn / (2*TOP_K))
            ret_t -= tc * turnover[-1]

        daily_ret.append(ret_t)
        prev_long = long_idx; prev_short = short_idx

    dr = np.array(daily_ret)
    annual_ret = dr.mean() * 252
    annual_vol = dr.std() * np.sqrt(252)
    sharpe = annual_ret / (annual_vol + 1e-10)
    cum = np.cumprod(1 + dr/100)
    peak = np.maximum.accumulate(cum); dd = (cum - peak) / (peak + 1e-10)
    max_dd = dd.min()
    calmar = annual_ret / (abs(max_dd) * 100 + 1e-10)  # annual ret / max dd%
    avg_turn = np.mean(turnover) if turnover else 0
    win_rate = (dr > 0).mean()

    print(f"  {name}:")
    print(f"    AnnRet={annual_ret:+.2f}% Vol={annual_vol:.2f}% Sharpe={sharpe:+.3f} MaxDD={max_dd*100:.2f}%")
    print(f"    Calmar={calmar:+.3f} Turnover={avg_turn:.2f} WinRate={win_rate*100:.0f}%")
    return {'name':name,'sharpe':sharpe,'ann_ret':annual_ret,'max_dd':max_dd,'calmar':calmar,'turnover':avg_turn,'win_rate':win_rate}

print(f"\n{'='*55}\nBACKTEST ({N} stocks, top {TOP_K}, TC={TC*10000:.0f}bp)\n{'='*55}")
r1=backtest(raw_pred, name='Raw (fixed size)')
r2=backtest(raw_pred, gate_signal=gate, name='Calibrated (Meta-Attention gate)')

# Comparison
print(f"\n  Improvement:")
for k in ['sharpe','ann_ret','max_dd','calmar','win_rate']:
    delta=r2[k]-r1[k]
    pct=delta/abs(r1[k]+1e-10)*100
    print(f"    {k}: {r1[k]:+.4f} -> {r2[k]:+.4f} ({delta:+.4f}, {pct:+.1f}%)")
