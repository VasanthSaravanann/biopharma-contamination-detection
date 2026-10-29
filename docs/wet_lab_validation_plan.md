# Wet-Lab Validation Plan

This repository currently contains computational evidence only. Before industrial use, the method needs validation on real fermentation experiments.

## Objectives

- Verify whether the ensemble detects contamination in real fermentation samples.
- Compare performance across multiple organisms and inoculum levels.
- Check robustness against process variation, instrument drift, and batch-to-batch shift.
- Confirm whether expanded features improve generalization beyond UV-Vis plus scalar process variables.

## Minimum study design

- At least three organism classes, including Gram-negative, Gram-positive, and yeast or mold.
- Multiple contamination levels, including the 10 CFU/mL target.
- Clean controls from the same bioreactor system and operating window.
- Replicate batches to estimate variance.
- Capture process metadata, spectra, and any expanded channels such as dissolved CO2 or metabolite profiles.

## Acceptance criteria

- Pre-specified ROC-AUC, sensitivity, specificity, and detection-limit targets.
- A stability metric for drift acceptance on repeated batches.
- Transparent separation of training, tuning, and final evaluation data.
- Explicit reporting of failures and negative results.

## Reporting note

If no wet-lab data are available, the paper should state that the results are computational and simulation-backed rather than experimentally validated.
