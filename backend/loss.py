import torch
import torch.nn as nn
from backend.config import POS_WEIGHT, DEVICE


def ccc_loss(pred: torch.Tensor, target: torch.Tensor):
    """
    Lin's Concordance Correlation Coefficient (CCC) Loss: 1 - CCC(pred, target).
    Maximizes ranking correlation and scale alignment for continuous emotion scores.
    """
    if pred.numel() <= 1:
        return torch.tensor(0.0, device=pred.device)

    pred_mean = pred.mean()
    target_mean = target.mean()

    pred_var = pred.var(unbiased=False)
    target_var = target.var(unbiased=False)

    covariance = ((pred - pred_mean) * (target - target_mean)).mean()

    ccc = (2 * covariance) / (
        pred_var + target_var + (pred_mean - target_mean) ** 2 + 1e-8
    )
    return 1.0 - ccc


def compute_span_loss(outputs: dict, batch: dict, pos_weight: float = POS_WEIGHT):
    """
    Computes weighted BCE loss for sparse aspect and opinion span detection matrices.
    """
    device = outputs["aspect_scores"].device
    pos_weight_tensor = torch.tensor([pos_weight], device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    aspect_target = batch["aspect_matrix"].to(device)
    opinion_target = batch["opinion_matrix"].to(device)

    aspect_loss = bce(outputs["aspect_scores"], aspect_target)
    opinion_loss = bce(outputs["opinion_scores"], opinion_target)

    return aspect_loss + opinion_loss


def compute_regression_loss(outputs: dict, batch: dict):
    """
    Computes Smooth L1 (Huber) regression loss for Valence and Arousal scores.
    """
    device = outputs["valence"].device
    smooth_l1 = nn.SmoothL1Loss()

    pred_val = outputs["valence"]
    pred_aro = outputs["arousal"]

    target_val = []
    target_aro = []

    for v, a in zip(batch["valence"], batch["arousal"]):
        if not torch.is_tensor(v):
            v = torch.tensor(v, dtype=torch.float)
        if not torch.is_tensor(a):
            a = torch.tensor(a, dtype=torch.float)

        if v.numel() == 0:
            target_val.append(0.5)
        else:
            target_val.append(float(v.mean()))

        if a.numel() == 0:
            target_aro.append(0.5)
        else:
            target_aro.append(float(a.mean()))

    target_val = torch.tensor(target_val, dtype=torch.float, device=device)
    target_aro = torch.tensor(target_aro, dtype=torch.float, device=device)

    val_loss = smooth_l1(pred_val, target_val)
    aro_loss = smooth_l1(pred_aro, target_aro)

    return val_loss + aro_loss, target_val, target_aro


def compute_total_loss(outputs: dict, batch: dict, pos_weight: float = POS_WEIGHT):
    """
    Calculates unified multi-task DimABSA objective:
      Loss = Span_Loss + 0.5 * Regression_Loss + 0.5 * CCC_Valence + 0.5 * CCC_Arousal
    """
    span_loss = compute_span_loss(outputs, batch, pos_weight=pos_weight)
    reg_loss, target_val, target_aro = compute_regression_loss(outputs, batch)

    val_ccc = ccc_loss(outputs["valence"], target_val)
    aro_ccc = ccc_loss(outputs["arousal"], target_aro)

    total_loss = span_loss + 0.5 * reg_loss + 0.5 * val_ccc + 0.5 * aro_ccc

    return {
        "loss": total_loss,
        "span_loss": span_loss,
        "regression_loss": reg_loss,
        "val_ccc": val_ccc,
        "aro_ccc": aro_ccc,
    }
