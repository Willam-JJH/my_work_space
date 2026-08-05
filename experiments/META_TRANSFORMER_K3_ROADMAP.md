# Meta-Transformer 改进路线图：基于 Kimi K3 技术栈

> **基线项目**: `experiments/meta_transformer_optimized.py`（SlimMetaTrans）
> **参考来源**: Kimi K3 研究报告（`Desktop/Kimi_K3_技术实现方案深度研究.md`）
> **日期**: 2026-07-18

---

## 基线架构速览

```
Token序列: [CLS] + [SIG_1 ... SIG_N] + [FACTOR_pooled]
           ↓ 4 × TransformerEncoderLayer(d=128, h=4, ff=256, gelu, 标准残差)
           ↓ CLS token
           ├─→ Linear(d, n_a)          → return_pred (每个资产的预期收益)
           └─→ Linear→256→GELU→1       → unc_pred    (不确定性, 对标 |error|)
           
配置:  200 stocks × 2市场 | WINDOW=250天 | SIG_DEPTH=3 → 39维 | PRED_HORIZON=21天
优化:  AdamW(lr=1e-3) + CosineAnnealing + Huber(δ=1) + 0.1×MSE(unc)
评估:  Spearman r(unc_pred, |true_error|) vs |Pred| baseline
```

---

## 优先级总览

| # | 改进项 | K3 来源 | 解决什么 | 工作量 | 风险 | 预期收益 |
|---|--------|---------|---------|--------|------|---------|
| P0 | Block AttnRes 替代标准残差 | AttnRes | token组之间无结构化融合 | 1-2周 | 低 | 训练加速+可解释融合 |
| P1a | 多维度 Attention Pattern 提取 | - | 只有一种Pattern信号,信息单一 | 3-5天 | 低 | 新增诊断通道 |
| P1b | P0+P1a 融合：AttnRes权重 + Multi-Pattern → 更强的unc预测 | AttnRes | Pattern读取方式和融合同时改进 | 1周 | 中 | 可能突破|Pred|基线 |
| P2 | KDA 混合注意力 (Linear+Full) | KDA | 3000→200 stocks被迫降采样, 无法长窗口 | 2-3周 | 中 | 全量训练+Pattern多样性 |
| P3 | 轻量 MoE Router 市场状态自适应 | LatentMoE | 所有市场状态同一套参数 | 2-3周 | 中-高 | 不同regime的差异表现 |
| P4a | Per-Head Muon 优化器 | Muon | AdamW可能非最优 | 3-5天 | 低 | 训练更稳定 |
| P4b | SiTU 激活替代 GELU | SiTU | 激活函数 | 1天 | 极低 | 边际提升 |
| P4c | 训练稳定性：梯度-输出范数监控 + QAT | 训练方案 | 训练过程不透明, 推理未量化 | 3-5天 | 极低 | 工程质量 |

---

## P0: Block AttnRes — 结构化 token 组融合

### 问题

当前架构把 `[CLS]` + `[SIG tokens]` + `[FACTOR token]` 拼接后扔进标准 Transformer，模型通过 self-attention **无约束地**混合三类信息。这有两个后果：

1. **融合不透明**：无法知道模型在做预测时，从 SIG（路径签名）和 FACTOR（经典因子）各读取了多少
2. **梯度集中**：4 层标准残差，梯度天然集中在 Layer 0-1，深层退化

### K3 的答案

AttnRes 把层分组成 Block，Block 内部标准残差，Block 之间用**可学习的注意力权重**做选择性检索。恰好对准了 Meta-Transformer 天然的三段 token 结构。

### 改造方案

#### 阶段 A: 两等分 Block（最小验证, 2-3天）

```
当前:                             阶段A:
L0 ─→ L1 ─→ L2 ─→ L3             Block 0 [L0, L1] ──→ AttnRes Query₀
  (4层标准残差)                        │                        ↓
                                  Block 1 [L2, L3] ──→ AttnRes Query₁
                                                       ├→ 读 Block 0
                                                       └→ 读 Block 1
                                                    各层输出加权求和 → CLS
```

**为什么先做两等分**：验证 AttnRes 在金融小模型上是否有效，不引入 token 分组假设。如果两等分都没收益，token 分组大概率也不行。

#### 阶段 B: 语义 Block（核心改进, 1周）

```
Block 0: [CLS token]    ─→ L0 → L1 (任务表示的自我演化)
Block 1: [SIG tokens]   ─→ L0 → L1 (路径签名跨资产交互)
Block 2: [FACTOR token] ─→ L0 → L1 (经典因子上下文)

Block 间 AttnRes:
  Query₀ (from Block 0) → 读 Block 0 100%, Block 1 α%, Block 2 β%
  Query₁ (from Block 1) → 读 Block 0 γ%, Block 1 100%, Block 2 δ%
  Query₂ (from Block 2) → 读 Block 0 ε%, Block 1 ζ%, Block 2 100%

最后: 三个 Block 输出 pooling → CLS → return_pred + unc_pred
```

**AttnRes 权重矩阵是核心诊断工具**：

```
AttnRes权重:        → Block 0    → Block 1    → Block 2
  Query₀ (CLS块)      1.0          0.37         0.52    ← CLS 块做了预测,从 SIG 读 37%,从 FACTOR 读 52%
  Query₁ (SIG块)      0.12         1.0          0.08    ← SIG 块几乎不读 FACTOR —— 信号独立!
  Query₂ (FACTOR块)   0.05         0.23         1.0     ← FACTOR 块少量读 SIG,信息交叉验证
```

#### 实现细节

```python
# 核心类: BlockAttnRes —— 替代标准残差
class BlockAttnRes(nn.Module):
    def __init__(self, d_model, n_blocks):
        # 每个 Block 一个可学习 Query
        self.queries = nn.Parameter(torch.zeros(n_blocks, d_model))  
        self.norms = nn.ModuleList([nn.RMSNorm(d_model) for _ in range(n_blocks)])
    
    def forward(self, block_outputs):  # block_outputs: list of (B, N, d)
        # block_outputs[i] = Block i 的最终输出
        keys = torch.stack([self.norms[i](o.mean(1)) for i, o in enumerate(block_outputs)], 1)
        # → (B, n_blocks, d)
        
        attn_weights = F.softmax(
            torch.einsum('bd,bkd->bk', self.queries, keys) / math.sqrt(d), 
            dim=-1
        )  # (n_blocks, n_blocks)
        
        # 加权融合
        fused = sum(attn_weights[i, j] * block_outputs[j] for i, j in ...)
        return fused, attn_weights  # ← 返回权重供诊断
```

改动量：约 80-120 行，主要替换 `SlimMetaTrans.forward()` 中的层循环。

#### 评估指标

| 指标 | 含义 |
|------|------|
| MSE (return) | 预测精度是否改善 |
| Spearman r(unc, \|error\|) | 不确定性预测是否改善 |
| vs \|Pred\| baseline | 是否缩小/超越 |
| 收敛 Epoch 数 | AttnRes 论文声称 ~25% 加速 |
| Block 间 AttnRes 权重 | **定性诊断**: 各信息源的利用比例 |

---

## P1a: 多维度 Attention Pattern 提取

### 问题

当前模型只输出 return_pred + unc_pred，没有利用 Transformer 中间层产出的 attention pattern 信息。Meta-Attention 项目（`quantitative_finance/meta-attention/`）已证明 attention pattern 包含与预测误差相关的信息——只是目前所有简单读取方式都未能超越 |Pred| 基线。

### 改进

在 `SlimMetaTrans` 的 forward 中增加 pattern 提取，不必立刻用于训练——先收集、分析、作为诊断工具：

```python
# 在 SlimMetaTrans.forward() 中新增:
def forward(self, sigs, cl, return_patterns=False):
    ...
    patterns = []
    for i, l in enumerate(self.layers):
        x = l(x)
        if return_patterns:
            # 提取 self-attention weights: (B, n_heads, 1+N+1, 1+N+1)
            pat = l.self_attn._w  # 需要 hook 或修改 TransformerEncoderLayer
            patterns.append(pat)
    ...
    if return_patterns:
        return ret_pred, unc_pred, patterns  # (n_layers, B, n_heads, n_tokens, n_tokens)
```

新增的分析维度：

| Pattern 维度 | 诊断问题 |
|-------------|---------|
| SIG→SIG attention | 哪些资产在关注哪些资产？（行业聚类、风险传染） |
| SIG→CLS attention | 哪些资产的路径签名被模型认为最重要？ |
| FACTOR→CLS attention | 经典因子 vs 路径签名：哪个被 CLS 更关注？ |
| 层间 Pattern 变化 | 低层和高层关注的东西是否不同？ |
| 头间多样性 | 多头是否学到了互补的注意力模式？ |

**价值**：即使不直接提升性能，这套 pattern 分析也是理解模型行为的核心工具——类似于 fMRI 对神经科学的作用。

---

## P1b: AttnRes 权重 + Pattern 联合融合

### 问题

这是 P0 和 P1a 的交叉点——也是 Kimi K3 报告第 15 章定位的"核心矛盾"：

> Meta-Attention 项目发现：Pattern 信号**存在**（残差化 r=0.53-0.59），但所有融合方式（Ridge、MLP、门控、Flow）都**无法超越** |Pred| 基线。瓶颈不是"找不到信息"，而是"不知道怎么融合"。

### K3 的启发

AttnRes 的核心洞察是：**用注意力机制在"信息源"维度上做选择性检索**。这正是当前项目缺失的——当前只有一种融合方式：所有信息扔进 Transformer 任其自生自灭。

### 方案

```python
class FusedMetaTrans(nn.Module):
    """P0 + P1a 联合: AttnRes权重 + Layer Attention Patterns → 联合unc预测"""
    
    def __init__(self, n_a, sig_dim, cl_dim):
        # ... 基础模块同 SlimMetaTrans ...
        
        # 新增: Pattern → Error 专用路径 (不参与主预测, 只输出unc)
        self.pattern_encoder = nn.Sequential(
            nn.Linear(n_heads * n_layers * n_blocks * n_blocks, 128),  # 展平所有pattern
            nn.GELU(), nn.Linear(128, 32)
        )
        # AttnRes 权重 → 编码
        self.attnres_encoder = nn.Sequential(
            nn.Linear(n_blocks * n_blocks, 32),  # Block间注意力权重
            nn.GELU(), nn.Linear(32, 16)
        )
        # 多源unc融合: [|Pred||, Pattern_enc, AttnRes_enc, CLS_hidden] → unc
        self.fusion_unc = nn.Sequential(
            nn.Linear(1 + 32 + 16 + d_model, 128),
            nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, sigs, cl):
        # 主预测路径 (不变)
        x = self.build_tokens(sigs, cl)
        block_outs, attnres_w = self.block_attnres_layers(x)
        ret_pred = self.ret_head(block_outs[-1][:, 0])
        
        # Pattern 路径
        patterns = self.extract_patterns()  # (n_layers, B, n_heads, n_tok, n_tok)
        pat_emb = self.pattern_encoder(patterns.flatten(2))
        
        # AttnRes 路径  
        ar_emb = self.attnres_encoder(attnres_w.flatten(1))
        
        # |Pred| 基线信号
        pred_signal = ret_pred.abs().mean(1, keepdim=True)
        
        # 三源融合 → uncertainty
        unc_pred = self.fusion_unc(torch.cat([
            pred_signal, pat_emb, ar_emb, block_outs[-1][:, 0]
        ], dim=-1)).squeeze(-1)
        
        return ret_pred, unc_pred
```

#### 关键实验矩阵

```
                                         unc r vs |Pred|
Pattern only ─────────────────────────  ?
AttnRes weights only ─────────────────  ?
|Pred| only ──────────────────────────  基线
|Pred| + Pattern ─────────────────────  (meta-attention: 失败)
|Pred| + AttnRes ─────────────────────  **待验证**
|Pred| + Pattern + AttnRes ───────────  **待验证**
```

---

## P2: KDA 混合注意力 — 全量 + 长窗口

### 问题

- `meta_transformer_optimized.py` 被迫把 3000 stocks 砍到 200，因为 O(N²) attention
- WINDOW=250 天固定，无法探索更长的历史窗口
- 只有一种 attention 类型（Full softmax），未知 Linear Attention 的 Pattern 是否不同

### KDA 方案

3 层 Linear Attention + 1 层 Full Attention 交错排列：

```
当前 (4 × Full):                       KDA:
L0: Full (O(N²·d))                    L0: Linear (O(N·d²), KV-cache -75%)
L1: Full                              L1: Linear
L2: Full                              L2: Linear
L3: Full                              L3: Full   (保留全局交互,但仅最后一层)
```

### 实现路线

#### 阶段 A: Linear Attention 实现 (3-5天)

```python
class LinearAttention(nn.Module):
    """KDA-style linear attention: φ(Q)·(φ(K)ᵀ·V) 而非 softmax(QKᵀ)·V"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        # KDA 的特征映射: φ(x) = elu(x) + 1 (确保非负)
        
    def forward(self, x):
        B, N, E = x.shape
        q = self.q(x).view(B, N, self.n_heads, self.d_k)  # (B, N, h, dk)
        k = self.k(x).view(B, N, self.n_heads, self.d_k)
        v = self.v(x).view(B, N, self.n_heads, self.d_k)
        
        # φ 映射
        q = F.elu(q) + 1
        k = F.elu(k) + 1
        
        # Linear attention: 先算 KV (O(N·d²)), 再乘 Q
        # kv = Σ k_j ᵀ · v_j  →  (B, h, dk, dk)
        kv = torch.einsum('bnhd,bnhe->bhde', k, v)
        # z = Σ k_j  →  (B, h, dk)  用于归一化
        z = k.sum(dim=1)  # (B, h, dk)
        # out = φ(q) · kv / (φ(q) · z)
        num = torch.einsum('bnhd,bhde->bnhe', q, kv)
        den = torch.einsum('bnhd,bhd->bnh', q, z).unsqueeze(-1) + 1e-8
        
        out = (num / den).contiguous().view(B, N, E)
        return self.drop(self.out(out))
```

#### 阶段 B: KDA 配置实验 (1-2周)

| 实验 | N_STOCKS | WINDOW | Attention | 对比 |
|------|----------|--------|-----------|------|
| 基线 | 200 | 250 | 4×Full | - |
| KDA-同规模 | 200 | 250 | 3×Linear+1×Full | 纯 attention 类型影响 |
| KDA-全量 | 3000 | 250 | 3×Linear+1×Full | VRAM 允许全量后的增益 |
| KDA-长窗口 | 200 | 500 | 3×Linear+1×Full | 更长历史的增量信息 |
| KDA-全量+长 | 3000 | 500 | 3×Linear+1×Full | 上限测试 |

#### 阶段 C: Pattern 对比分析 (3-5天)

分别提取 Linear 层和 Full 层的 attention pattern，分析：
- Linear Pattern → Error r vs Full Pattern → Error r
- 两种 Pattern 残差化 → 检验信息互补性
- 长窗口的 Pattern 是否包含更强的误差预测信号

---

## P3: 轻量 MoE Router — 市场状态自适应

### 问题

当前模型在所有市场状态下用同一组 FFN 参数。但：
- A 股 2015 年暴涨暴跌 vs 美股慢牛 —— 统计规律不同
- 高波动期的预测策略应区别于低波动期
- 当前模型只能学到一个"折中"的映射

### K3 方案

K3 用 896 个专家 (16 激活) + Quantile Balancing。我们不需要这么多——3-5 个足矣。

### 实现

#### 阶段 A: Router + 3 Experts (1-2周)

```python
class MarketStateRouter(nn.Module):
    """轻量 Router: 根据市场特征选择专家"""
    def __init__(self, d_model, n_experts=3):
        super().__init__()
        self.router = nn.Sequential(
            nn.Linear(4, d_model),  # 4维市场特征 → 路由
            nn.GELU(),
            nn.Linear(d_model, n_experts)  # → 3个专家的权重
        )
    
    def forward(self, market_features):
        # market_features: (B, 4) — VIX代理, 截面离散度, 涨跌比, 成交量比
        logits = self.router(market_features)
        return F.softmax(logits, dim=-1)  # (B, n_experts)


class MoELayer(nn.Module):
    """带 Router 的 MoE FFN 层"""
    def __init__(self, d_model, d_ff, n_experts=3):
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), 
                         nn.Dropout(0.1), nn.Linear(d_ff, d_model))
            for _ in range(n_experts)
        ])
        self.router = MarketStateRouter(d_model, n_experts)
    
    def forward(self, x, market_features):
        weights = self.router(market_features)  # (B, n_experts)
        expert_outs = torch.stack([e(x) for e in self.experts], dim=1)  # (B, n_exp, N, d)
        # 加权融合
        out = (expert_outs * weights[:, :, None, None]).sum(dim=1)  # (B, N, d)
        return out, weights  # 返回权重供诊断
```

#### 市场特征设计

```python
def extract_market_features(X_ret):
    """X_ret: (B, n_a, 30) 过去30天标准化收益"""
    # 1. 截面波动率: 横截面收益的标准差
    cross_sectional_vol = X_ret[:, :, -1].std(dim=1)  # (B,)
    # 2. 涨跌比: 最近5天正收益的资产比例
    up_ratio = (X_ret[:, :, -5:].mean(-1) > 0).float().mean(dim=1)  # (B,)
    # 3. 尾部风险: 最低5%资产的收益
    tail = X_ret[:, :, -5:].mean(-1).quantile(0.05, dim=1)  # (B,)
    # 4. 成交量变化率
    vol_change = ...  # 从 volume 数据提取
    return torch.stack([cross_sectional_vol, up_ratio, tail, vol_change], dim=1)
```

#### Quantile Balancing（来自 K3 的 Stable LatentMoE）

```python
def quantile_balance_loss(router_logits, expert_assignments, n_experts):
    """防止 Router 坍缩到单一专家"""
    # 每个专家的使用频率应该接近 1/n_experts
    usage = expert_assignments.float().mean(0)  # (n_experts,)
    target = 1.0 / n_experts
    return F.kl_div(usage.log(), torch.full_like(usage, target), reduction='batchmean')
```

### 评估

| 指标 | 全样本 | 高波动期 | 低波动期 | 危机期 |
|------|--------|---------|---------|--------|
| MSE | - | - | - | - |
| unc r | - | - | - | - |
| vs \|Pred\| | - | - | - | - |
| Router 使用分布 | - | Expert_0使用率 | Expert_1使用率 | Expert_2使用率 |

关键是：**高波动期的性能提升**——这正是当前统一参数模型最吃亏的地方。

---

## P4: 工程优化（低风险、快速验证）

### P4a: Per-Head Muon 优化器

```python
# 替换 AdamW
# Muon 的核心: 对权重矩阵做 Newton-Schulz 正交化后更新
# Per-Head Muon: 对每个 attention head 的参数独立做 Muon 更新

# 实现可以用 PyTorch 社区已有的 muon 包, 或手写核心逻辑:
class PerHeadMuon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.95, nesterov=True, 
                 ns_steps=5, weight_decay=1e-4):
        ...
        # 核心: step() 中识别 attention 权重矩阵 (shape [d_model, d_model])
        # 对其做 Newton-Schulz 正交化 → 再应用更新
```

**为什么可能有用**: K3 报告称 Per-Head Muon 在万亿参数规模上表现优于 AdamW。金融数据的高噪声特性可能让 Muon 的正交化约束带来更强的泛化。

**验证方式**: 同种子、同数据、换优化器 → 对比收敛曲线和最终 MSE。

### P4b: SiTU 激活替代 GELU

```python
# 当前: nn.GELU()
# 改为: SiTU (Sigmoid Tanh Unit)
class SiTU(nn.Module):
    """Sigmoid Tanh Unit — K3 采用的激活函数"""
    def forward(self, x):
        return x * torch.sigmoid(x) + torch.tanh(x)
```

**为什么可能有用**: SiTU 结合了 sigmoid 的门控特性（SiLU/Swish 的优势）和 tanh 的对称饱和特性。在金融数据中，尾部极端值常见——tanh 提供的对称性可能改善极端行情下的梯度传播。

**验证方式**: 在 FFN 中替换 GELU → SiTU，对比 MSE 和训练稳定性。

### P4c: 训练稳定性增强

```python
# 1. 梯度-输出范数监控 (来自 K3 训练方案)
def log_training_diagnostics(model, step):
    """每个 epoch 记录: 各层梯度范数, 各层输出范数, 注意力熵"""
    diag = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            diag[f'grad_norm/{name}'] = param.grad.norm().item()
    # 各层输出范数 (通过 forward hook)
    # AttnRes 论文发现: 标准残差下输出范数单调递增, AttnRes 周期重置
    return diag

# 2. QAT (量化感知训练)
# 当前模型用 FP32 训练；在 SFT 后期引入伪量化 (fake quantization)
# → 推理时可用 INT8/FP16 加速（回测 3000 stocks × 20年 × 逐日滚动时很重要）
model.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')
model_prepared = torch.ao.quantization.prepare_qat(model, inplace=False)
# 继续训练少量 epoch → convert 到量化模型
```

---

## 执行顺序与依赖关系

```
                      P0: Block AttnRes
                      /                \
                     /                  \
              P1a: Multi-Pattern      P2: KDA (可并行)
                     \                  /
                      \                /
                       P1b: 联合融合
                            |
                      P4a,b,c: 工程优化 (随时可插)
                            |
                       P3: MoE Router
```

- **P0 → P1a/P2**: P0 的 Block 结构为后续 pattern 提取和 KDA 提供自然的分组框架，先做
- **P1a 和 P2 可并行**: 一个侧重信息提取，一个侧重规模扩展
- **P1b 在 P0+P1a 之后**: 需要 AttnRes 权重和 Pattern 都有了才能做联合融合
- **P4 随时可插**: 不依赖架构改动，任何阶段都可以做 A/B 测试
- **P3 放在最后**: MoE 改动最深层，等 P0-P2 稳定后再加

---

## 每个改进的"失败"定义与 fallback

| 改进 | 什么算失败 | Fallback |
|------|-----------|----------|
| P0 | MSE 不降 + unc r 不涨 + 收敛不加速 (三者全无) | 回退标准残差，但保留 Block 间权重作为诊断通道 |
| P1a | Pattern 分析得不到有意义的聚类/诊断 | 不影响主模型，纯诊断工具，不存在"失败" |
| P1b | 三源融合 unc r 仍然 ≤ |Pred\| | 退回只用 |Pred|，但 block 权重+pattern 分析的知识已产出 |
| P2 | Linear Attention 在金融数据上数值不稳定 | 退回 Full Attention，但保留了 Linear/Full pattern 对比的经验 |
| P3 | Router 坍缩到单一专家或性能下降 | 回退无 MoE，但 Router 权重分布分析本身有价值 |
| P4a | Muon 收敛慢于 AdamW | 回退 AdamW |
| P4b | SiTU 无差异 | 回退 GELU |
| P4c | 监控本身无失败可能；QAT 精度损失过大 | QAT 精度阈值 0.5% MSE |

---

## 附录: 与 Meta-Attention 项目的关系

Meta-Attention（`quantitative_finance/meta-attention/`）研究的是 **attention pattern → 预测误差** 的信息提取问题。该项目的核心矛盾是：Pattern 包含增量信息（残差化证实），但所有融合方式都失败了。

Meta-Transformer 是**独立的金融预测 pipeline**，结构与 Meta-Attention 不同。但两者的 P1b（联合融合）可以在 Meta-Transformer 上直接验证 Meta-Attention 项目的核心假设——用 AttnRes 权重作为新的融合信号，看能否突破 |Pred| 基线。如果成功，这是一个跨项目的贡献。

Kimi K3 报告的第 15 章（`Kimi K3 × Meta-Attention 结合分析`）更详细地讨论了这个问题。
