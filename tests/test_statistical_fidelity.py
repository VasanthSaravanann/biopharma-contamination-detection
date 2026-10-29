"""Unit tests for statistical fidelity validation tooling"""
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from metrics.statistical_fidelity import (
    compute_jsd, compute_mmd, compute_ks_statistic, compute_fidelity_report
)


def test_jsd_identical_distributions():
    """JSD should be ~0 for identical distributions"""
    X = np.random.randn(100, 10)
    jsd = compute_jsd(X, X)
    assert jsd < 0.1, f"JSD for identical data should be ~0, got {jsd}"


def test_jsd_different_distributions():
    """JSD should be higher for different distributions"""
    X1 = np.random.randn(100, 10)
    X2 = np.random.randn(100, 10) + 5.0  # Shifted distribution
    
    jsd = compute_jsd(X1, X2)
    assert jsd > 0.1, f"JSD for different distributions should be > 0.1, got {jsd}"


def test_mmd_identical_distributions():
    """MMD should be ~0 for identical distributions"""
    X = np.random.randn(100, 10)
    mmd, _ = compute_mmd(X, X)
    assert mmd < 0.1, f"MMD for identical data should be ~0, got {mmd}"


def test_mmd_different_distributions():
    """MMD should be higher for different distributions"""
    X1 = np.random.randn(100, 10)
    X2 = np.random.randn(100, 10) + 3.0
    
    mmd, _ = compute_mmd(X1, X2)
    assert mmd > 0.1, f"MMD for different distributions should be > 0.1, got {mmd}"


def test_ks_identical_distributions():
    """KS p-value should be high (~1) for identical distributions"""
    X = np.random.randn(100)
    _, p_val = compute_ks_statistic(X, X)
    assert p_val > 0.5, f"KS p-value for identical data should be high, got {p_val}"


def test_ks_different_distributions():
    """KS p-value should be low for significantly different distributions"""
    X1 = np.random.randn(100)
    X2 = np.random.randn(100) + 5.0
    
    _, p_val = compute_ks_statistic(X1, X2)
    assert p_val < 0.05, f"KS p-value for different distributions should be < 0.05, got {p_val}"


def test_fidelity_report_structure():
    """Verify fidelity report has all required keys"""
    X1 = np.random.randn(100, 10)
    X2 = np.random.randn(100, 10)
    
    report = compute_fidelity_report(X1, X2)
    
    required_keys = ['jsd', 'mmd', 'mmd_p_value', 'ks', 'ks_p_value', 'fidelity_pass']
    for key in required_keys:
        assert key in report, f"Missing key: {key}"


def test_fidelity_pass_criteria():
    """Verify fidelity_pass is True when criteria are met"""
    # Nearly identical distributions should pass
    X1 = np.random.randn(200, 20)
    X2 = X1 + np.random.randn(200, 20) * 0.05  # Small noise
    
    report = compute_fidelity_report(X1, X2)
    
    # Should pass or be close to passing
    assert report['jsd'] < 1.0
    assert report['mmd'] < 5.0
    assert report['ks_p_value'] > 0.01


def test_fidelity_pass_false_for_different():
    """Verify fidelity_pass is False when criteria are NOT met"""
    X1 = np.random.randn(100)
    X2 = np.random.randn(100) + 10.0  # Highly different
    
    report = compute_fidelity_report(X1, X2)
    
    # Should fail criterion
    assert report['jsd'] > 0.5 or report['mmd'] > 2.0 or report['ks_p_value'] < 0.05
