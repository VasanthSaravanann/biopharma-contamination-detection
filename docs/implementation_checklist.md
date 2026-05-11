# Peak Novelty Implementation Checklist

Branch: `feature/fused-validation`

This checklist turns the current novelty story into branch-local tasks that can be executed and verified in the repository.

## 1. Lock the core claim

- [ ] Make the project narrative explicitly center on real-data UV-Vis contamination detection.
- [ ] Treat literature-based simulation as augmentation and feasibility support, not the main claim.
- [ ] Ensure all public-facing docs distinguish computational proof-of-concept from wet-lab validation.

## 2. Align the validation pipeline

- [ ] Verify `run_literature_validation.py` writes a complete report and reproducible outputs.
- [ ] Confirm validation targets are reported with honest labels for computational versus experimental evidence.
- [ ] Add a short result summary artifact that can be reused in docs and notebooks.

## 3. Tighten multimodal fusion

- [ ] Replace or clearly flag pseudo-sensor features in `notebooks/07_multimodal_fusion.ipynb`.
- [ ] Wire the notebook to actual process-data inputs when available in `data/processed/`.
- [ ] Keep a UV-only baseline, process-only baseline, and fused model comparison.

## 4. Improve explainability

- [ ] Make wavelength-level explanation outputs easy to trace back to biologically meaningful markers.
- [ ] Ensure the explanation notebook or pipeline exports figures and tabular summaries.
- [ ] Add a concise narrative for why the top wavelengths matter biologically.

## 5. Stabilize real-data paths

- [ ] Confirm the real dataset path used by notebooks matches the workspace layout.
- [ ] Remove hardcoded external paths where they block reproducibility.
- [ ] Keep synthetic and real-data workflows separated but interoperable.

## 6. Refresh project documentation

- [ ] Update `README.md` so the novelty description matches the current branch.
- [ ] Update `QUICK_REFERENCE.md` so the recommended run path matches the actual executable entrypoints.
- [ ] Add a short branch note describing what is complete, what is in progress, and what remains.

## 7. Validate branch health

- [ ] Re-check the git status after edits and keep unrelated worktree changes untouched.
- [ ] Run the narrowest available validation for any edited pipeline or notebook metadata.
- [ ] Capture the final state in a brief completion note for future reference.
