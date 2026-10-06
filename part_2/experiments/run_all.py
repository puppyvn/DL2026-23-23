"""Run every supporting experiment in order (about one to two minutes on a laptop CPU).

    python experiments/run_all.py
"""
import bootstrap_ci
import calibration_profile
import depth_architecture
import positive_control
import reverse_control
import robustness_10_models
import thresholds

for name, module in [("bootstrap confidence intervals", bootstrap_ci), ("positive control", positive_control),
                     ("reverse control", reverse_control), ("fixed confidence thresholds", thresholds),
                     ("ten checkpoints", robustness_10_models),
                     ("depth and architecture (report Experiments 3-4)", depth_architecture),
                     ("calibration profile of the raw ResNet-18 (RQ1)", calibration_profile)]:
    print(f"\n===== {name} =====")
    module.main()
