"""Run every supporting experiment used in the report (about one minute on a laptop CPU).

    python experiments/run_all.py
"""
import bootstrap_ci
import calibration_profile
import depth_architecture
import positive_control
import reverse_control

for name, module in [("bootstrap confidence intervals", bootstrap_ci), ("positive control", positive_control),
                     ("reverse control", reverse_control),
                     ("depth and architecture (report Experiments 3-4)", depth_architecture),
                     ("calibration profile of the raw ResNet-18", calibration_profile)]:
    print(f"\n===== {name} =====")
    module.main()
