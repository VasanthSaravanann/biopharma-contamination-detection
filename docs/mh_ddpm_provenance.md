**MH-DDPM Provenance and Role**

Summary
-------
The MH-DDPM (Multi-Head Denoising Diffusion Probabilistic Model) implemented in this project is a physics-aware synthetic data generator used exclusively for ablation experiments and sensitivity analyses. It creates augmented UV-Vis spectra and turbidity overlays by sampling from conditioned diffusion chains seeded with physics-derived priors (Beer–Lambert attenuation + Rayleigh–Mie scattering components).

Provenance
----------
- Source: Internal research code under `src/mh_ddpm.py` (model implementation) and `notebooks/06_mh_ddpm_finetune.ipynb` (development traces).
- Training data used: sterile spectral baselines extracted from `data/Bacteria Contamination Work` and synthetic contamination kernels parameterized by CFU, particle size distribution, and refractive index perturbations.
- Checkpoints: Saved under `models/mh_ddpm/` with date-stamped filenames. Each checkpoint includes metadata with hyperparameters and seed.

Role in the project
-------------------
- Ablation-only: MH-DDPM augmentations were added/removed to measure the ensemble's sensitivity to synthetic diversity; they did not contribute to the final scoring pipeline used for reported ROC-AUC/LOD metrics.
- Documentation requirement: this file clarifies MH-DDPM's experimental (not deployed) role and lists artifacts for reproducibility.

Reproducibility notes
---------------------
- To reproduce MH-DDPM ablations: run `scripts/run_mh_ddpm_ablation.sh` (created in this commit). The script accepts `--seed`, `--n-samples`, and `--augment-ratio` flags and writes outputs to `output/augmented/mh_ddpm/`.
