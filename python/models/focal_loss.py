"""
Focal Loss with label smoothing and optional class weights.

公式:
  FL = -Σ_c y_c_smoothed * (1 - p_c)^γ * log(p_c)

其中:
  p_c = softmax(logits)_c
  y_c_smoothed = (1-ε) * 1[c=t] + ε/(C-1) * 1[c≠t]
  γ = focusing parameter (default 2.0)

References:
  Lin et al. "Focal Loss for Dense Object Detection", ICCV 2017
  Szegedy et al. "Rethinking the Inception Architecture for Computer Vision", CVPR 2016 (Label Smoothing)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss with optional class weights and label smoothing.

    Args:
        weight:          (C,) class weights (sqrt-smoothed inverse frequency recommended)
        gamma:           focusing parameter (default 2.0)
        label_smoothing: label smoothing factor (default 0.1)
        reduction:       'mean' | 'sum'
    """

    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.1, reduction='mean'):
        super().__init__()
        self.register_buffer('weight', weight)  # (C,) or None
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits:  (N, C) — raw logits from model
            targets: (N,)   — integer class indices

        Returns:
            scalar loss
        """
        n_classes = logits.size(1)
        log_probs = F.log_softmax(logits, dim=1)   # (N, C)
        probs = torch.exp(log_probs)                # (N, C)

        # ---- Build smoothed targets ----
        with torch.no_grad():
            if self.label_smoothing > 0:
                smooth = torch.full_like(
                    log_probs,
                    self.label_smoothing / (n_classes - 1)
                )
                smooth.scatter_(
                    1, targets.unsqueeze(1),
                    1.0 - self.label_smoothing
                )
            else:
                smooth = F.one_hot(targets, n_classes).float()

        # ---- Focal modulation: (1 - p_c)^γ ----
        focal_weight = (1.0 - probs) ** self.gamma   # (N, C)

        # ---- Per-class weighted NLL ----
        loss = -smooth * focal_weight * log_probs     # (N, C)

        # ---- Apply class-wise alpha weights ----
        if self.weight is not None:
            loss = loss * self.weight.view(1, -1)     # (N, C)

        loss = loss.sum(dim=1)  # (N,)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# ====== 单元测试 ======

if __name__ == "__main__":
    import numpy as np

    print("=" * 60)
    print("Testing FocalLoss")
    print("=" * 60)

    batch, n_class = 4, 10
    torch.manual_seed(42)

    logits = torch.randn(batch, n_class)
    targets = torch.randint(0, n_class, (batch,))

    # Test 1: Basic Focal Loss (no weights, no smoothing)
    fl = FocalLoss(gamma=2.0, label_smoothing=0.0)
    loss = fl(logits, targets)
    print(f"  [PASS] FocalLoss(gamma=2.0, no smoothing): {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss should not be NaN"

    # Test 2: With label smoothing
    fl_smooth = FocalLoss(gamma=2.0, label_smoothing=0.1)
    loss_smooth = fl_smooth(logits, targets)
    print(f"  [PASS] FocalLoss(gamma=2.0, smoothing=0.1): {loss_smooth.item():.4f}")
    assert not torch.isnan(loss_smooth), "Loss should not be NaN"

    # Test 3: With sqrt-smoothed class weights
    train_counts = np.array([800, 1525, 517, 215, 192, 906, 2427, 93, 453, 1185])
    total = train_counts.sum()
    weights_sqrt = np.sqrt(total / (n_class * train_counts))
    weights_tensor = torch.FloatTensor(weights_sqrt)

    fl_weighted = FocalLoss(weight=weights_tensor, gamma=2.0, label_smoothing=0.1)
    loss_w = fl_weighted(logits, targets)
    print(f"  [PASS] FocalLoss(weighted, gamma=2.0, smoothing=0.1): {loss_w.item():.4f}")
    assert not torch.isnan(loss_w), "Loss should not be NaN"

    # Test 4: Gradient check
    fl_weighted.train()
    logits_grad = torch.randn(batch, n_class, requires_grad=True)
    loss_g = fl_weighted(logits_grad, targets)
    loss_g.backward()
    print(f"  [PASS] Backward: grad shape={logits_grad.grad.shape}, "
          f"grad norm={logits_grad.grad.norm().item():.4f}")

    # Test 5: Verify per-class weights range
    print(f"\n  Smoothed class weights: {weights_sqrt}")
    print(f"  Max/Min ratio: {weights_sqrt.max() / weights_sqrt.min():.2f}x "
          f"(cf. inverse freq {np.max(total / (n_class * train_counts)) / np.min(total / (n_class * train_counts)):.1f}x)")

    print("\n  All tests PASSED!")
