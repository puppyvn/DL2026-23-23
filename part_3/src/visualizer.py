"""
Visualization module for Part 3:
1. plot_dual_boxplot: Sample-level Confidence vs 500-sample Batch Accuracy across severity 0->5
2. plot_reliability_diagrams_shift: 6-panel Reliability Diagram across severity 0->5 with ECE/ACE
3. plot_risk_coverage_shift: Multi-curve Risk-Coverage across severity 0->5 with Risk@80% & Risk@60%
"""

import matplotlib.pyplot as plt
import numpy as np


def setup_matplotlib_style():
    """Set publication-quality plotting styles."""
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'lines.linewidth': 2.0
    })


def plot_dual_boxplot(
    confidences_per_severity: dict,
    batch_accuracies_per_severity: dict,
    save_path: str = "outputs/figures/fig4_part3_dual_boxplot_confidence_vs_acc.png"
):
    """
    Plot Dual Boxplot:
    - Left/Blue: Sample-level confidence distribution (N=10,000 per severity)
    - Right/Orange: 500-sample Batch Accuracy distribution (20 batches per severity)
    Across severity 0 (Clean) to 5.
    """
    setup_matplotlib_style()
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    severities = sorted(confidences_per_severity.keys())
    x_positions = np.arange(len(severities))
    width = 0.35
    
    conf_data = [confidences_per_severity[s] for s in severities]
    acc_data = [batch_accuracies_per_severity[s] for s in severities]
    
    bp1 = ax.boxplot(
        conf_data,
        positions=x_positions - width/2,
        widths=width * 0.85,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor='#4C72B0', color='#1f4068', alpha=0.85),
        medianprops=dict(color='#0d1b2a', linewidth=2.2),
        whiskerprops=dict(color='#1f4068', linewidth=1.4),
        capprops=dict(color='#1f4068', linewidth=1.4)
    )
    
    bp2 = ax.boxplot(
        acc_data,
        positions=x_positions + width/2,
        widths=width * 0.85,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor='#DD8452', color='#a34816', alpha=0.85),
        medianprops=dict(color='#5c2406', linewidth=2.2),
        whiskerprops=dict(color='#a34816', linewidth=1.4),
        capprops=dict(color='#a34816', linewidth=1.4)
    )
    
    labels = ["Clean (Sev 0)"] + [f"Sev {s}" for s in severities[1:]]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, fontweight='bold')
    ax.set_xlabel("Distribution Shift Intensity (Severity Level)", fontweight='bold')
    ax.set_ylabel("Value (Probability / Batch Accuracy)", fontweight='bold')
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Part 3.1: Sample Confidence vs 500-Sample Batch Accuracy under Shift\n(ResNet-18 on CIFAR-10 & CIFAR-10-C)", fontweight='bold', pad=12)
    
    # Legend
    ax.legend(
        [bp1["boxes"][0], bp2["boxes"][0]],
        ["Sample-level Confidence (N=10,000)", "Batch-level Accuracy (20 batches × 500 samples)"],
        loc="lower left",
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc"
    )
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[*] Saved Dual Boxplot to: {save_path}")


def plot_reliability_diagrams_shift(
    ece_results_per_severity: dict,
    accuracy_per_severity: dict,
    ace_per_severity: dict,
    save_path: str = "outputs/figures/fig5_part3_shift_reliability_diagrams.png"
):
    """
    Plot 2x3 grid of Reliability Diagrams for Severity 0 (Clean) to 5.
    """
    setup_matplotlib_style()
    severities = sorted(ece_results_per_severity.keys())
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=300, sharex=True, sharey=True)
    axes = axes.flatten()
    
    for idx, s in enumerate(severities):
        ax = axes[idx]
        res = ece_results_per_severity[s]
        acc = accuracy_per_severity[s]
        ace = ace_per_severity[s]
        ece = res["ece"]
        
        bin_confs = res["bin_confidences"]
        bin_accs = res["bin_accuracies"]
        bin_edges = res["bin_edges"]
        widths = np.diff(bin_edges)
        bin_centers = bin_edges[:-1] + widths / 2.0
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.8, label="Ideal (y=x)")
        
        # Bar chart
        for b in range(len(bin_centers)):
            c_val = bin_confs[b]
            a_val = bin_accs[b]
            w = widths[b] * 0.9
            
            # Draw outputs
            ax.bar(bin_centers[b], a_val, width=w, color="#55A868", alpha=0.75, edgecolor="#2b5c36", label="Accuracy" if b == 0 else "")
            
            # Gap
            if c_val > a_val:
                gap_height = c_val - a_val
                ax.bar(bin_centers[b], gap_height, bottom=a_val, width=w, color="#C44E52", alpha=0.7, hatch="//", edgecolor="#822024", label="Overconf. Gap" if b == 0 else "")
            elif a_val > c_val:
                gap_height = a_val - c_val
                ax.bar(bin_centers[b], gap_height, bottom=c_val, width=w, color="#4C72B0", alpha=0.5, hatch="\\\\", edgecolor="#1f4068", label="Underconf. Gap" if b == 0 else "")
                
        title = "Clean (Severity 0)" if s == 0 else f"CIFAR-10-C Severity {s}"
        ax.set_title(f"{title}\nAcc: {acc*100:.1f}% | ECE: {ece*100:.2f}% | ACE: {ace*100:.2f}%", fontsize=11, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        if idx == 0:
            ax.legend(loc="upper left", fontsize=8)
            
    fig.suptitle("Part 3.2: Reliability Diagrams under Distribution Shift (Severity 0 → 5)", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.91)
    plt.savefig(save_path)
    plt.close()
    print(f"[*] Saved Reliability Diagrams to: {save_path}")


def plot_risk_coverage_shift(
    rc_results_per_severity: dict,
    save_path: str = "outputs/figures/fig6_part3_shift_risk_coverage.png"
):
    """
    Plot Risk-Coverage curves for Severity 0 (Clean) to 5 on a single figure.
    Highlights Risk@80% and Risk@60% operating points.
    Places the legend/stats inside the plot in the upper-left region.
    The Y-axis is extended to 145% so that the legend has full vertical clearance
    above the highest curve (Severity 5 ~90%), completely avoiding any overlap.
    """
    setup_matplotlib_style()
    fig, ax = plt.subplots(figsize=(10.5, 7.0), dpi=300)
    
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd', '#8c564b']
    severities = sorted(rc_results_per_severity.keys())
    
    for idx, s in enumerate(severities):
        rc = rc_results_per_severity[s]
        cov = rc["coverages"]
        risk = rc["risks"]
        name = "Clean (Sev 0)" if s == 0 else f"Severity {s}"
        label = f"{name} (AURC: {rc['aurc']:.3f} | R@80%: {rc['risk_at_80']*100:.1f}%)"
        ax.plot(cov * 100, risk * 100, label=label, color=colors[idx % len(colors)], linewidth=2.3)
        
        # Mark operating points on the curves
        ax.scatter([80.0], [rc['risk_at_80'] * 100], color=colors[idx % len(colors)], s=45, zorder=5)
        ax.scatter([60.0], [rc['risk_at_60'] * 100], color=colors[idx % len(colors)], s=45, zorder=5)

    # Reference vertical lines for operating points
    ax.axvline(x=80.0, color='#444444', linestyle=':', linewidth=1.6, alpha=0.85, label="Operating Point: 80% Coverage")
    ax.axvline(x=60.0, color='#777777', linestyle='--', linewidth=1.6, alpha=0.85, label="Operating Point: 60% Coverage")
    
    # 100% Risk reference line to clarify theoretical maximum risk
    ax.axhline(y=100.0, color='#aaaaaa', linestyle='--', linewidth=1.2, alpha=0.7)
    ax.text(100.2, 100.0, " 100% Risk", va='center', ha='left', fontsize=8.5, color='#666666', style='italic')

    ax.set_xlabel("Coverage (%) - Proportion of Accepted Predictions", fontweight='bold', fontsize=12)
    ax.set_ylabel("Risk (%) - Error Rate on Accepted Set", fontweight='bold', fontsize=12)
    ax.set_title("Part 3.3: Risk–Coverage Curves under Progressive Distribution Shift\n(Evaluating Confidence as a Selective Classification Signal)", fontweight='bold', fontsize=13, pad=12)
    
    ax.set_xlim(0, 100.5)
    # The highest curve (Severity 5) reaches ~90% risk.
    # Increasing Y-axis to 145 gives ample room in upper-left for the legend box
    ax.set_ylim(-2, 145)
    ax.set_yticks([0, 20, 40, 60, 80, 100, 120, 140])
    
    # Place legend inside plot at upper-left with clear styling
    legend = ax.legend(
        loc="upper left",
        frameon=True,
        facecolor="#ffffff",
        edgecolor="#cccccc",
        framealpha=0.95,
        fontsize=9.2,
        borderpad=0.7,
        labelspacing=0.45
    )
    legend.get_frame().set_boxstyle("square,pad=0.5")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[*] Saved Risk-Coverage plot to: {save_path}")

