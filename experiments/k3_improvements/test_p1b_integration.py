"""
Minimal integration test for P1b JointFusionTransformer.

Verifies:
  1. Model can be imported and constructed.
  2. Forward pass returns correct shapes and fusion_info keys.
  3. Training loop runs 2 epochs with synthetic data.
  4. Loss decreases over training.
  5. Gradient flows through all parameters.
"""

import sys
import os

# Ensure k3_improvements/ is on sys.path for sibling imports
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def main():
    success = True
    reasons = []

    # ----------------------------------------------------------------
    # 1. Import and construct the model
    # ----------------------------------------------------------------
    try:
        from p1b_fusion import JointFusionTransformer
    except Exception as e:
        print(f"P1b integration test FAILED — import error: {e}")
        sys.exit(1)

    try:
        model = JointFusionTransformer(n_a=20, sig_dim=39, cl_dim=4)
    except Exception as e:
        print(f"P1b integration test FAILED — construction error: {e}")
        sys.exit(1)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model constructed: {n_params:,} parameters")

    # ----------------------------------------------------------------
    # 2. Forward pass sanity check
    # ----------------------------------------------------------------
    torch.manual_seed(42)
    np.random.seed(42)

    B = 4
    sigs = torch.randn(B, 20, 39)
    cl   = torch.randn(B, 20, 4)

    model.eval()
    with torch.no_grad():
        ret_pred, unc_pred, fusion_info = model(sigs, cl)

    if ret_pred.shape != (B, 20):
        reasons.append(f"ret_pred shape mismatch: expected (4,20) got {tuple(ret_pred.shape)}")
        success = False
    else:
        print(f"  ret_pred     shape OK: {tuple(ret_pred.shape)}")

    if unc_pred.shape != (B,):
        reasons.append(f"unc_pred shape mismatch: expected (4,) got {tuple(unc_pred.shape)}")
        success = False
    else:
        print(f"  unc_pred     shape OK: {tuple(unc_pred.shape)}")

    # Check fusion_info keys
    expected_keys = {'cb_weights', 'block_patterns', 'pred_signal', 'cls_hidden'}
    actual_keys = set(fusion_info.keys())
    if actual_keys != expected_keys:
        missing = expected_keys - actual_keys
        extra = actual_keys - expected_keys
        reasons.append(f"fusion_info key mismatch: missing={missing}, extra={extra}")
        success = False
    else:
        print(f"  fusion_info  keys OK: {sorted(expected_keys)}")

    # Check sub-structures
    if len(fusion_info['cb_weights']) != 2:
        reasons.append(f"cb_weights length expected 2, got {len(fusion_info['cb_weights'])}")
        success = False
    else:
        print(f"  cb_weights:  {len(fusion_info['cb_weights'])} blocks, shape {tuple(fusion_info['cb_weights'][0].shape)}")

    if len(fusion_info['block_patterns']) != 6:
        reasons.append(f"block_patterns length expected 6, got {len(fusion_info['block_patterns'])}")
        success = False
    else:
        print(f"  block_patterns: {len(fusion_info['block_patterns'])} patterns")
        for i, p in enumerate(fusion_info['block_patterns']):
            print(f"    [{i}] shape={tuple(p.shape)}")

    # ----------------------------------------------------------------
    # 3. Training loop with 2 epochs of synthetic data
    # ----------------------------------------------------------------
    # Generate synthetic data: 16 samples, 20 assets, sig_dim=39, cl_dim=4
    N = 16
    n_a = 20
    sig_dim = 39
    cl_dim = 4

    # Targets: simple linear combination of sigs + noise
    X_sigs = torch.randn(N, n_a, sig_dim)
    X_cl   = torch.randn(N, n_a, cl_dim)
    y = X_sigs[:, :, :n_a].sum(dim=-1) * 0.01  # (N, n_a) — a learnable signal

    ds = TensorDataset(X_sigs, X_cl, y)
    dl = DataLoader(ds, batch_size=4, shuffle=True, drop_last=False)

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    losses = []
    for epoch in range(2):
        epoch_loss = 0.0
        for s_b, c_b, y_b in dl:
            ret_pred, unc_pred, _ = model(s_b, c_b)

            loss_r = F.huber_loss(ret_pred, y_b, delta=1.0)
            with torch.no_grad():
                err = (ret_pred - y_b).abs().mean(dim=1)
            loss_u = 0.1 * F.mse_loss(unc_pred, err)
            loss = loss_r + loss_u

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(len(dl), 1)
        losses.append(avg_loss)
        print(f"  Epoch {epoch + 1}/2  |  Avg Loss: {avg_loss:.6f}")

    # Verify loss decreases from epoch 0 to epoch 1
    if losses[1] >= losses[0]:
        reasons.append(f"Loss did not decrease: epoch0={losses[0]:.6f} epoch1={losses[1]:.6f}")
        success = False

    # ----------------------------------------------------------------
    # 4. Gradient flow check — no nil or zero grads after backward
    # ----------------------------------------------------------------
    model.train()
    s_batch = X_sigs[:4]
    c_batch = X_cl[:4]
    y_batch = y[:4]

    ret_pred, unc_pred, _ = model(s_batch, c_batch)
    loss = F.huber_loss(ret_pred, y_batch, delta=1.0)
    with torch.no_grad():
        err = (ret_pred - y_batch).abs().mean(dim=1)
    loss = loss + 0.1 * F.mse_loss(unc_pred, err)
    model.zero_grad(set_to_none=True)
    loss.backward()

    nil_grads = [n for n, p in model.named_parameters()
                 if p.requires_grad and p.grad is None]
    zero_grads = [n for n, p in model.named_parameters()
                  if p.requires_grad and p.grad is not None
                  and p.grad.norm().item() == 0.0]

    if nil_grads:
        reasons.append(f"Nil gradients: {nil_grads[:5]}")
        success = False
    if zero_grads:
        reasons.append(f"Zero gradients: {zero_grads[:5]}")
        success = False

    print(f"  Gradient flow: nil_grads={len(nil_grads)} zero_grads={len(zero_grads)}")

    # ----------------------------------------------------------------
    # Report
    # ----------------------------------------------------------------
    if success:
        print("\nP1b integration test PASSED")
    else:
        print("\nP1b integration test FAILED")
        for r in reasons:
            print(f"  - {r}")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
