"""
Main experiment runner for PART 3:
"Có thể tin model mãi mãi được không? (Can we trust the model forever?)"

Evaluates ResNet-18 raw confidence (T=1.0) under progressive distribution shift (CIFAR-10-C Severity 0 to 5):
1. Computes Top-1 Accuracy, ECE, ACE, Risk@80%, Risk@60%, AURC
2. Generates Dual Boxplot: Sample Confidence vs 500-sample Batch Accuracy
3. Generates 6-panel Reliability Diagrams
4. Generates Multi-curve Risk-Coverage Plot
5. Saves summary table to CSV and figures to outputs/figures/
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add workspace root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model_loader import load_cifar10_model
from src.data_loader import get_cifar10_test_loader
from src.metrics import compute_accuracy, compute_ece, compute_ace
from src.risk_coverage import compute_risk_coverage_curve
from src.visualizer import (
    plot_dual_boxplot,
    plot_reliability_diagrams_shift,
    plot_risk_coverage_shift
)


def run_inference_for_severity(model, dataloader, device):
    """Run model inference, returning confidences, predictions, targets."""
    all_confs = []
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Inference", leave=False):
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=-1)
            confs, preds = torch.max(probs, dim=-1)
            
            all_confs.append(confs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.numpy())
            
    return (
        np.concatenate(all_confs),
        np.concatenate(all_preds),
        np.concatenate(all_targets)
    )


def compute_500_sample_batch_accuracies(predictions, targets, batch_size=500):
    """
    Split 10,000 predictions into 20 non-overlapping 500-sample batches
    and compute accuracy on each batch.
    """
    n = len(predictions)
    batch_accs = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        acc = np.mean(predictions[start:end] == targets[start:end])
        batch_accs.append(float(acc))
    return np.array(batch_accs)


def main():
    parser = argparse.ArgumentParser(description="Run Part 3 Experiments: Confidence under Distribution Shift")
    parser.add_argument("--model", type=str, default="edadaltocg/resnet18_cifar10", help="Hugging Face model checkpoint")
    parser.add_argument("--corruption", type=str, default="gaussian_noise", choices=["gaussian_noise", "blur", "brightness", "contrast", "pixelation"], help="Representative corruption type")
    parser.add_argument("--batch_size", type=int, default=256, help="Inference batch size")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cpu/cuda)")
    parser.add_argument("--replot_only", action="store_true", help="Re-generate figures from cached results without running inference")
    args = parser.parse_args()

    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/tables", exist_ok=True)
    cache_path = "outputs/tables/part3_experiment_cache.pkl"

    if args.replot_only and os.path.exists(cache_path):
        import pickle
        print(f"[*] Loading cached experiment results from: {cache_path}...")
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        confidences_per_severity = cached["confidences_per_severity"]
        batch_accuracies_per_severity = cached["batch_accuracies_per_severity"]
        ece_results_per_severity = cached["ece_results_per_severity"]
        ace_per_severity = cached["ace_per_severity"]
        accuracy_per_severity = cached["accuracy_per_severity"]
        rc_results_per_severity = cached["rc_results_per_severity"]
    else:
        print("=" * 80)
        print(" PART 3: CAN WE TRUST NEURAL NETWORK CONFIDENCE UNDER DISTRIBUTION SHIFT? ")
        print("=" * 80)
        print(f"[*] Checkpoint : {args.model}")
        print(f"[*] Corruption : {args.corruption} (Severities 1 -> 5)")
        print(f"[*] Device     : {args.device}")
        print("-" * 80)

        # 1. Load Model
        model = load_cifar10_model(args.model, device=args.device)

        # Containers for results across severities
        confidences_per_severity = {}
        batch_accuracies_per_severity = {}
        ece_results_per_severity = {}
        ace_per_severity = {}
        accuracy_per_severity = {}
        rc_results_per_severity = {}
        summary_rows = []
        severities = [0, 1, 2, 3, 4, 5]

        for s in severities:
            c_name = "clean" if s == 0 else args.corruption
            desc = "Clean CIFAR-10 (Severity 0)" if s == 0 else f"{args.corruption.replace('_', ' ').title()} (Severity {s})"
            print(f"\n[+] Evaluating: {desc}...")

            loader = get_cifar10_test_loader(
                root="data/raw",
                corruption_name=c_name,
                severity=s,
                batch_size=args.batch_size
            )

            confs, preds, targets = run_inference_for_severity(model, loader, args.device)
            
            # 1. Top-1 Accuracy
            acc = compute_accuracy(preds, targets)
            accuracy_per_severity[s] = acc
            
            # 2. ECE & ACE (num_bins=15)
            ece_res = compute_ece(confs, preds, targets, num_bins=15)
            ece_results_per_severity[s] = ece_res
            ace_val = compute_ace(confs, preds, targets, num_bins=15)
            ace_per_severity[s] = ace_val
            
            # 3. 500-sample batch accuracy (20 batches)
            batch_accs = compute_500_sample_batch_accuracies(preds, targets, batch_size=500)
            confidences_per_severity[s] = confs
            batch_accuracies_per_severity[s] = batch_accs
            
            # 4. Risk-Coverage curve & operating points
            rc_res = compute_risk_coverage_curve(confs, preds, targets)
            rc_results_per_severity[s] = rc_res
            
            # Record summary
            summary_rows.append({
                "Severity": s,
                "Condition": "Clean" if s == 0 else f"{args.corruption}_sev{s}",
                "Accuracy (%)": round(acc * 100, 2),
                "ECE (%)": round(ece_res["ece"] * 100, 2),
                "ACE (%)": round(ace_val * 100, 2),
                "Median Conf (%)": round(float(np.median(confs)) * 100, 2),
                "Median Batch Acc (%)": round(float(np.median(batch_accs)) * 100, 2),
                "Risk@80% Cov (%)": round(rc_res["risk_at_80"] * 100, 2),
                "Risk@60% Cov (%)": round(rc_res["risk_at_60"] * 100, 2),
                "AURC": round(rc_res["aurc"], 4)
            })
            
            print(f"    -> Acc: {acc*100:.2f}% | ECE: {ece_res['ece']*100:.2f}% | ACE: {ace_val*100:.2f}% | Median Conf: {np.median(confs)*100:.2f}% | Risk@80%: {rc_res['risk_at_80']*100:.2f}%")

        # Save experiment cache for fast re-plotting
        import pickle
        with open(cache_path, "wb") as f:
            pickle.dump({
                "confidences_per_severity": confidences_per_severity,
                "batch_accuracies_per_severity": batch_accuracies_per_severity,
                "ece_results_per_severity": ece_results_per_severity,
                "ace_per_severity": ace_per_severity,
                "accuracy_per_severity": accuracy_per_severity,
                "rc_results_per_severity": rc_results_per_severity,
            }, f)
        print(f"[*] Experiment cache saved to: {cache_path}")

        # Save summary table
        df_summary = pd.DataFrame(summary_rows)
        table_path = "outputs/tables/table_part3_summary.csv"
        df_summary.to_csv(table_path, index=False)
        print(f"\n[*] Summary table saved to: {table_path}")
        print("\n" + df_summary.to_string(index=False))

    # Generate the 3 required publication-quality figures
    print("\n[*] Generating Part 3 Visualizations...")
    
    # Figure 1: Dual Boxplot
    plot_dual_boxplot(
        confidences_per_severity,
        batch_accuracies_per_severity,
        save_path="outputs/figures/fig4_part3_dual_boxplot_confidence_vs_acc.png"
    )
    
    # Figure 2: Reliability Diagrams across Shift
    plot_reliability_diagrams_shift(
        ece_results_per_severity,
        accuracy_per_severity,
        ace_per_severity,
        save_path="outputs/figures/fig5_part3_shift_reliability_diagrams.png"
    )
    
    # Figure 3: Risk-Coverage Curves
    plot_risk_coverage_shift(
        rc_results_per_severity,
        save_path="outputs/figures/fig6_part3_shift_risk_coverage.png"
    )

    print("\n" + "=" * 80)
    print(" PART 3 EXECUTION COMPLETED SUCCESSFULLY! ")
    print(" All figures and tables generated in outputs/ ")
    print("=" * 80)


if __name__ == "__main__":
    main()
