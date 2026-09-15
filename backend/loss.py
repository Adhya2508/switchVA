import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from backend.config import POS_WEIGHT, LOSS_MSE_WEIGHT, LOSS_HUBER_WEIGHT, LOSS_CCC_WEIGHT


def lin_ccc(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Lin's Concordance Correlation Coefficient (numpy).
    """
    if len(y_true) < 2:
        return 1.0
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    ccc = (2 * cov) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-8)
    return float(ccc)


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


def compute_aspect_regression_loss(
    pred_val: torch.Tensor,
    pred_aro: torch.Tensor,
    target_val: torch.Tensor,
    target_aro: torch.Tensor,
    mse_w: float = LOSS_MSE_WEIGHT,
    huber_w: float = LOSS_HUBER_WEIGHT,
    ccc_w: float = LOSS_CCC_WEIGHT,
):
    """
    Calculates multi-objective loss for aspect-level Valence and Arousal.
    Combines MSE, Huber (Smooth L1), and Lin's CCC loss to directly minimize RMSE.
    """
    mse_loss_fn = nn.MSELoss()
    huber_loss_fn = nn.SmoothL1Loss(beta=0.05)

    val_mse = mse_loss_fn(pred_val, target_val)
    aro_mse = mse_loss_fn(pred_aro, target_aro)

    val_huber = huber_loss_fn(pred_val, target_val)
    aro_huber = huber_loss_fn(pred_aro, target_aro)

    val_ccc = ccc_loss(pred_val, target_val)
    aro_ccc = ccc_loss(pred_aro, target_aro)

    total_loss = (
        mse_w * (val_mse + aro_mse)
        + huber_w * (val_huber + aro_huber)
        + ccc_w * (val_ccc + aro_ccc)
    )

    return {
        "loss": total_loss,
        "val_mse": val_mse,
        "aro_mse": aro_mse,
        "val_huber": val_huber,
        "aro_huber": aro_huber,
        "val_ccc": val_ccc,
        "aro_ccc": aro_ccc,
    }


def compute_span_loss(
    aspect_scores: torch.Tensor,
    opinion_scores: torch.Tensor,
    aspect_target: torch.Tensor,
    opinion_target: torch.Tensor,
    pos_weight: float = POS_WEIGHT,
):
    """
    Computes weighted BCE loss for sparse aspect and opinion span detection matrices.
    """
    device = aspect_scores.device
    pos_weight_tensor = torch.tensor([pos_weight], device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    aspect_loss = bce(aspect_scores, aspect_target)
    opinion_loss = bce(opinion_scores, opinion_target)

    return aspect_loss + opinion_loss


def calculate_comprehensive_metrics(
    true_vals: list,
    pred_vals: list,
    true_aros: list,
    pred_aros: list,
):
    """
    Computes complete statistical evaluation metrics:
    RMSE, MAE, R², CCC, and Pearson Correlation.
    """
    y_true_v = np.array(true_vals, dtype=np.float64)
    y_pred_v = np.array(pred_vals, dtype=np.float64)
    y_true_a = np.array(true_aros, dtype=np.float64)
    y_pred_a = np.array(pred_aros, dtype=np.float64)

    if len(y_true_v) == 0:
        return {
            "valence_rmse": 0.0,
            "arousal_rmse": 0.0,
            "overall_rmse": 0.0,
            "valence_mae": 0.0,
            "arousal_mae": 0.0,
            "valence_r2": 0.0,
            "arousal_r2": 0.0,
            "valence_ccc": 0.0,
            "arousal_ccc": 0.0,
            "valence_pearson": 0.0,
            "arousal_pearson": 0.0,
        }

    val_rmse = float(np.sqrt(mean_squared_error(y_true_v, y_pred_v)))
    aro_rmse = float(np.sqrt(mean_squared_error(y_true_a, y_pred_a)))
    overall_rmse = float(np.sqrt(0.5 * (val_rmse**2 + aro_rmse**2)))

    val_mae = float(mean_absolute_error(y_true_v, y_pred_v))
    aro_mae = float(mean_absolute_error(y_true_a, y_pred_a))

    val_r2 = float(r2_score(y_true_v, y_pred_v))
    aro_r2 = float(r2_score(y_true_a, y_pred_a))

    val_ccc = lin_ccc(y_true_v, y_pred_v)
    aro_ccc = lin_ccc(y_true_a, y_pred_a)

    # Pearson r
    def pearson_r(a, b):
        if np.std(a) == 0 or np.std(b) == 0:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    val_pearson = pearson_r(y_true_v, y_pred_v)
    aro_pearson = pearson_r(y_true_a, y_pred_a)

    return {
        "valence_rmse": round(val_rmse, 4),
        "arousal_rmse": round(aro_rmse, 4),
        "overall_rmse": round(overall_rmse, 4),
        "valence_mae": round(val_mae, 4),
        "arousal_mae": round(aro_mae, 4),
        "valence_r2": round(val_r2, 4),
        "arousal_r2": round(aro_r2, 4),
        "valence_ccc": round(val_ccc, 4),
        "arousal_ccc": round(aro_ccc, 4),
        "valence_pearson": round(val_pearson, 4),
        "arousal_pearson": round(aro_pearson, 4),
    }
