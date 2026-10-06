"""Run every supporting experiment in order (about one to two minutes on a laptop CPU).

    python experiments/run_all.py
"""
import bootstrap_ci
import positive_control
import reverse_control
import robustness_10_models
import thresholds

for name, module in [("bootstrap confidence intervals", bootstrap_ci), ("positive control", positive_control),
                     ("reverse control", reverse_control), ("fixed confidence thresholds", thresholds),
                     ("ten checkpoints", robustness_10_models)]:
    print(f"\n===== {name} =====")
    module.main()
