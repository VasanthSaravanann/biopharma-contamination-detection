"""Formal ablation runner for contamination detection baselines."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import warnings

from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent
sys.path.append(str(BASE_DIR))

from src.data_simulation import UVVisSpectraGenerator
from src.feature_extraction import SpectralFeatureExtractor
from src.anomaly_detection import (
    EnsembleAnomalyDetector,
    IsolationForestDetector,
    AutoencoderDetector,
    OneClassSVMDetector,
    ModelConfig,
)
from src.mh_ddpm import DDPMConfig, FeatureDiffusionModel


def _build_feature_sets(features_df: pd.DataFrame):
    uv_cols = [
        c for c in features_df.columns
        if c.startswith(('abs_', 'a260', 'uv_', 'vis_', 'peak_', 'scattering', 'biomass', 'integral_'))
    ]
    process_cols = [c for c in ['temperature', 'ph', 'dissolved_oxygen', 'batch_age'] if c in features_df.columns]
    all_cols = uv_cols + process_cols
    return uv_cols, process_cols, all_cols


def _map_condition_indices(features_df: pd.DataFrame):
    contaminated = features_df[features_df['label'] == 1].copy()
    contaminant_codes = pd.Categorical(contaminated['contaminant_type']).codes.astype(int)
    level_bins = np.array([10, 25, 50, 100, 250, 500, 1000], dtype=float)
    level_values = contaminated['inoculum_level'].astype(float).values
    inoculum_idx = np.argmin(np.abs(level_values[:, None] - level_bins[None, :]), axis=1).astype(int)
    return contaminated, contaminant_codes, inoculum_idx


def run_ablation(seed: int = 42):
    print('Starting formal ablation study...')
    output_dir = BASE_DIR / 'output' / 'ablation'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Data and features
    generator = UVVisSpectraGenerator(seed=seed)
    df = generator.generate_dataset(n_clean=800, n_contaminated_per_type=80, seed=seed)
    extractor = SpectralFeatureExtractor()
    features_df = extractor.extract_all_features(df)
    uv_cols, _, all_cols = _build_feature_sets(features_df)

    y = features_df['label'].values.astype(int)
    X_all = features_df[all_cols].fillna(0).values
    X_uv = features_df[uv_cols].fillna(0).values
    X_clean_all = X_all[y == 0]
    X_clean_uv = X_uv[y == 0]

    model_cfg = ModelConfig(ae_epochs=40, iforest_n_estimators=200)
    results = []

    # 2) Single models vs ensemble (multimodal)
    print('\n[1/4] Single models vs ensemble...')
    if_model = IsolationForestDetector(model_cfg).fit(X_clean_all)
    ocsvm_model = OneClassSVMDetector(model_cfg).fit(X_clean_all)
    ae_model = AutoencoderDetector(X_clean_all.shape[1], model_cfg).fit(X_clean_all, verbose=False)
    ens_model = EnsembleAnomalyDetector(X_clean_all.shape[1], model_cfg).fit(X_clean_all)

    results.append({'Experiment': 'Model', 'Setting': 'Isolation Forest', 'ROC-AUC': roc_auc_score(y, if_model.predict_proba(X_all))})
    results.append({'Experiment': 'Model', 'Setting': 'One-Class SVM', 'ROC-AUC': roc_auc_score(y, ocsvm_model.predict_proba(X_all))})
    results.append({'Experiment': 'Model', 'Setting': 'Autoencoder', 'ROC-AUC': roc_auc_score(y, ae_model.predict_proba(X_all))})
    auc_ensemble_multimodal = roc_auc_score(y, ens_model.predict_proba(X_all))
    results.append({'Experiment': 'Model', 'Setting': 'Ensemble', 'ROC-AUC': auc_ensemble_multimodal})

    # 3) Multimodal vs UV-only
    print('[2/4] Multimodal vs UV-only...')
    ens_uv = EnsembleAnomalyDetector(X_clean_uv.shape[1], model_cfg).fit(X_clean_uv)
    auc_ensemble_uv = roc_auc_score(y, ens_uv.predict_proba(X_uv))
    results.append({'Experiment': 'Fusion', 'Setting': 'UV-only Ensemble', 'ROC-AUC': auc_ensemble_uv})
    results.append({'Experiment': 'Fusion', 'Setting': 'Multimodal Ensemble', 'ROC-AUC': auc_ensemble_multimodal})

    # 4) With-DDPM vs without-DDPM (feature-space baseline)
    print('[3/4] With DDPM vs without DDPM...')
    contaminated_df, cont_codes, inoc_idx = _map_condition_indices(features_df)
    X_cont_features = contaminated_df[all_cols].fillna(0).values

    ddpm_cfg = DDPMConfig(
        input_dim=X_cont_features.shape[1],
        hidden_dim=128,
        n_layers=4,
        n_timesteps=50,
        batch_size=64,
        epochs=8,
    )
    feature_ddpm = FeatureDiffusionModel(ddpm_cfg)
    feature_ddpm.fit_features(X_cont_features, cont_codes, inoc_idx, verbose=False)

    synth_features = feature_ddpm.sample_features(
        n_samples=min(400, len(X_cont_features)),
        contaminant_type=int(np.bincount(cont_codes).argmax()),
        inoculum_level=0,
        progress=False,
    )
    y_synth = np.ones(len(synth_features), dtype=int)

    X_aug = np.vstack([X_all, synth_features])
    y_aug = np.concatenate([y, y_synth])
    auc_without_ddpm = auc_ensemble_multimodal
    auc_with_ddpm = roc_auc_score(y_aug, ens_model.predict_proba(X_aug))

    results.append({'Experiment': 'DDPM', 'Setting': 'Without DDPM', 'ROC-AUC': auc_without_ddpm})
    results.append({'Experiment': 'DDPM', 'Setting': 'With Feature-DDPM', 'ROC-AUC': auc_with_ddpm})

    # Persist models and metadata, including ensemble weighting logs.
    model_dir = output_dir / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)
    ens_model.save(str(model_dir / 'ensemble_multimodal'))
    ens_uv.save(str(model_dir / 'ensemble_uv_only'))
    if_model.save(str(model_dir / 'iforest_ablation.pkl'))
    ocsvm_model.save(str(model_dir / 'ocsvm_ablation.pkl'))
    ae_model.save(str(model_dir / 'autoencoder_ablation.pt'))

    # 5) Export table + publication-ready chart
    print('[4/4] Exporting ablation artifacts...')
    res_df = pd.DataFrame(results)
    csv_path = output_dir / 'ablation_results.csv'
    res_df.to_csv(csv_path, index=False)

    sns.set_theme(style='whitegrid')
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(data=res_df, x='Setting', y='ROC-AUC', hue='Experiment', palette='Set2')
    plt.ylim(max(0.0, res_df['ROC-AUC'].min() - 0.05), min(1.0, res_df['ROC-AUC'].max() + 0.05))
    plt.xticks(rotation=30, ha='right')
    plt.title('Ablation Study Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Configuration')
    plt.ylabel('ROC-AUC')

    for patch in ax.patches:
        h = patch.get_height()
        ax.annotate(
            f'{h:.3f}',
            (patch.get_x() + patch.get_width() / 2.0, h),
            ha='center',
            va='bottom',
            xytext=(0, 4),
            textcoords='offset points',
            fontsize=9,
        )

    plt.tight_layout()
    fig_path = output_dir / 'ablation_metrics.png'
    plt.savefig(fig_path, dpi=300)

    print(f'Results table: {csv_path}')
    print(f'Comparison chart: {fig_path}')
    print(f'Model artifacts: {model_dir}')
    return res_df


if __name__ == '__main__':
    run_ablation()
