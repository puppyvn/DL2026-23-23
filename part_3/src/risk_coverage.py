"""
Selective classification & Risk-Coverage curve computation:
- Risk-Coverage curve calculation
- Risk@Coverage operating points (e.g., Risk@80%, Risk@60%)
- AURC (Area Under Risk-Coverage Curve)
"""

import numpy as np


def compute_risk_coverage_curve(
    confidences: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    num_eval_points: int = 1000
) -> dict:
    """
    Compute selective classification Risk vs Coverage curve.
    
    Samples are sorted by confidence in descending order.
    Coverage is the fraction of top-confidence samples retained.
    Risk is the empirical error rate on the retained samples.
    
    Args:
        confidences: (N,) array of maximum softmax probabilities
        predictions: (N,) array of predicted class labels
        targets: (N,) array of ground-truth class labels
        num_eval_points: Number of subsampled points for smooth plotting
        
    Returns:
        dict containing:
            coverages: np.ndarray of shape (num_points,)
            risks: np.ndarray of shape (num_points,)
            aurc: float (Area Under Risk-Coverage Curve)
            risk_at_80: float (Risk at 80% coverage)
            risk_at_60: float (Risk at 60% coverage)
    """
    n = len(confidences)
    if n == 0:
        return {"coverages": np.array([]), "risks": np.array([]), "aurc": 0.0, "risk_at_80": 0.0, "risk_at_60": 0.0}

    # Sort in descending order of confidence
    order = np.argsort(-confidences)
    sorted_preds = predictions[order]
    sorted_targets = targets[order]
    
    # 1 if error, 0 if correct
    errors = (sorted_preds != sorted_targets).astype(np.float64)
    cum_errors = np.cumsum(errors)
    counts = np.arange(1, n + 1, dtype=np.float64)
    
    all_coverages = counts / n
    all_risks = cum_errors / counts
    
    # Calculate Risk at specific operating points
    idx_80 = int(np.round(0.80 * n)) - 1
    idx_80 = max(0, min(n - 1, idx_80))
    risk_at_80 = float(all_risks[idx_80])
    
    idx_60 = int(np.round(0.60 * n)) - 1
    idx_60 = max(0, min(n - 1, idx_60))
    risk_at_60 = float(all_risks[idx_60])
    
    # Compute AURC (numerical integration using trapezoidal rule)
    if hasattr(np, "trapezoid"):
        aurc = float(np.trapezoid(all_risks, all_coverages))
    else:
        aurc = float(np.trapz(all_risks, all_coverages))
    
    # Subsample for smooth visualization if n is large
    if n > num_eval_points:
        indices = np.linspace(0, n - 1, num_eval_points, dtype=int)
        plot_coverages = all_coverages[indices]
        plot_risks = all_risks[indices]
    else:
        plot_coverages = all_coverages
        plot_risks = all_risks
        
    return {
        "coverages": plot_coverages,
        "risks": plot_risks,
        "aurc": aurc,
        "risk_at_80": risk_at_80,
        "risk_at_60": risk_at_60
    }
