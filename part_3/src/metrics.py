"""
Calibration and evaluation metrics:
- Top-1 Accuracy
- ECE (Expected Calibration Error) - Equal-width binning (M=15)
- ACE (Adaptive Calibration Error) - Equal-frequency/quantile binning (M=15)
"""

import numpy as np


def compute_accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute Top-1 classification accuracy."""
    return float(np.mean(predictions == targets))


def compute_ece(
    confidences: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    num_bins: int = 15
) -> dict:
    """
    Compute Expected Calibration Error (ECE) using equal-width binning.
    
    Args:
        confidences: (N,) array of maximum softmax probabilities
        predictions: (N,) array of predicted class labels
        targets: (N,) array of ground-truth class labels
        num_bins: Number of bins (default: 15)
        
    Returns:
        dict containing:
            ece: float
            bin_accuracies: list of float
            bin_confidences: list of float
            bin_counts: list of int
            bin_edges: np.ndarray of shape (num_bins + 1,)
    """
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_indices = np.digitize(confidences, bin_edges[1:-1])  # 0 to num_bins - 1
    
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    total_samples = len(confidences)
    ece = 0.0
    
    for i in range(num_bins):
        mask = (bin_indices == i)
        count = int(np.sum(mask))
        bin_counts.append(count)
        
        if count > 0:
            acc = float(np.mean(predictions[mask] == targets[mask]))
            conf = float(np.mean(confidences[mask]))
            bin_accuracies.append(acc)
            bin_confidences.append(conf)
            ece += (count / total_samples) * abs(acc - conf)
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(0.5 * (bin_edges[i] + bin_edges[i+1]))
            
    return {
        "ece": float(ece),
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
        "bin_counts": bin_counts,
        "bin_edges": bin_edges
    }


def compute_ace(
    confidences: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    num_bins: int = 15
) -> float:
    """
    Compute Adaptive Calibration Error (ACE) using equal-frequency (quantile) binning
    as proposed by Nixon et al. (CVPRW 2019).
    
    Args:
        confidences: (N,) array of maximum softmax probabilities
        predictions: (N,) array of predicted class labels
        targets: (N,) array of ground-truth class labels
        num_bins: Number of adaptive bins (default: 15)
        
    Returns:
        ace: float
    """
    total_samples = len(confidences)
    if total_samples == 0:
        return 0.0
        
    # Sort samples by confidence
    sorted_indices = np.argsort(confidences)
    split_indices = np.array_split(sorted_indices, num_bins)
    
    ace = 0.0
    for split in split_indices:
        if len(split) == 0:
            continue
        acc = float(np.mean(predictions[split] == targets[split]))
        conf = float(np.mean(confidences[split]))
        ace += (len(split) / total_samples) * abs(acc - conf)
        
    return float(ace)
