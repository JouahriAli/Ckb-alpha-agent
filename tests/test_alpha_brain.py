"""
Tests for alpha_brain.py — hand-verifiable values.

Each test uses small arrays so you can check the arithmetic on paper.
Comments show the derivation step by step.
"""

import math

import numpy as np
import pytest
from scipy import stats

from tools.alpha_brain import (
	AlphaReport,
	Trade,
	apply_jb_weight,
	bootstrap_kurtosis_ci,
	compute_distribution_stats,
	compute_liquidity_thinness,
	compute_signal_momentum,
	compute_volume_zscore,
	analyze,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_trade(volume: float, side: str = "buy") -> Trade:
	"""Shortcut to create a Trade with only the fields we care about."""
	return Trade(
		timestamp=0,
		price=1.0,
		volume_usd=volume,
		side=side,
		tx_hash="0x00",
	)


# ---------------------------------------------------------------------------
# Step 1: Volume Z-Score
# ---------------------------------------------------------------------------

class TestVolumeZScore:
	def test_exact_mean_returns_zero(self):
		"""If current volume equals the baseline mean, z = 0."""
		baseline = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
		# μ = 100, but σ = 0 → should raise.
		# Use a baseline with some variance instead.
		baseline = np.array([90.0, 100.0, 110.0, 100.0, 100.0])
		# μ = 100, σ_sample = std([90,100,110,100,100], ddof=1)
		# Σ(x-μ)² = 100+0+100+0+0 = 200, σ = √(200/4) = √50 ≈ 7.071
		z = compute_volume_zscore(baseline, 100.0)
		assert z == pytest.approx(0.0, abs=1e-10)

	def test_one_sigma_above(self):
		"""Current volume is exactly 1σ above mean → z = 1."""
		baseline = np.array([90.0, 100.0, 110.0, 100.0, 100.0])
		sigma = np.std(baseline, ddof=1)  # √50 ≈ 7.071
		z = compute_volume_zscore(baseline, 100.0 + sigma)
		assert z == pytest.approx(1.0, abs=1e-10)

	def test_negative_zscore(self):
		"""Volume below mean → negative z."""
		baseline = np.array([90.0, 100.0, 110.0, 100.0, 100.0])
		z = compute_volume_zscore(baseline, 80.0)
		assert z < 0.0

	def test_dead_market_raises(self):
		"""Zero variance baseline → ValueError (can't divide by 0)."""
		baseline = np.array([100.0, 100.0, 100.0])
		with pytest.raises(ValueError, match="zero variance"):
			compute_volume_zscore(baseline, 200.0)

	def test_large_spike(self):
		"""
		Hand-calculated: baseline = [10, 20, 30, 20, 20], current = 100.
		μ = 20, Σ(x-μ)² = 100+0+100+0+0 = 200, σ = √(200/4) = √50
		z = (100 - 20) / √50 = 80/7.071 ≈ 11.314
		"""
		baseline = np.array([10.0, 20.0, 30.0, 20.0, 20.0])
		z = compute_volume_zscore(baseline, 100.0)
		expected = 80.0 / math.sqrt(50.0)
		assert z == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Step 2: Distribution Shape
# ---------------------------------------------------------------------------

class TestDistributionStats:
	def test_normal_sample_low_kurtosis(self):
		"""
		A large sample from N(0,1) should have:
		- Excess kurtosis ≈ 0
		- Skewness ≈ 0
		- JB p-value > 0.05 (can't reject normality)
		"""
		rng = np.random.default_rng(42)
		data = rng.normal(0, 1, size=5000)
		kappa, gamma, jb_p = compute_distribution_stats(data)

		assert kappa == pytest.approx(0.0, abs=0.15)
		assert gamma == pytest.approx(0.0, abs=0.1)
		assert jb_p > 0.05

	def test_heavy_tailed_high_kurtosis(self):
		"""
		A t-distribution with low df has heavy tails → high excess kurtosis.
		t(df=4) has theoretical κ = 6/(4-4) → undefined, but t(df=5) has κ = 6.
		"""
		rng = np.random.default_rng(42)
		data = rng.standard_t(df=5, size=10000)
		kappa, gamma, jb_p = compute_distribution_stats(data)

		# Theoretical κ for t(df=5) = 6/(5-4) = 6.
		assert kappa > 3.0  # Clearly leptokurtic.
		assert jb_p < 0.05  # Normality rejected.

	def test_skewed_distribution(self):
		"""
		Exponential distribution is right-skewed.
		Theoretical skewness = 2, kurtosis = 6.
		"""
		rng = np.random.default_rng(42)
		data = rng.exponential(scale=1.0, size=10000)
		kappa, gamma, jb_p = compute_distribution_stats(data)

		assert gamma > 1.0  # Clearly positive skew.
		assert kappa > 3.0  # Heavy right tail.
		assert jb_p < 0.001  # Very non-normal.

	def test_uniform_platykurtic(self):
		"""
		Uniform distribution has negative excess kurtosis.
		Theoretical κ = -6/5 = -1.2.
		"""
		rng = np.random.default_rng(42)
		data = rng.uniform(0, 1, size=10000)
		kappa, gamma, jb_p = compute_distribution_stats(data)

		assert kappa < 0.0  # Platykurtic.
		assert kappa == pytest.approx(-1.2, abs=0.15)


# ---------------------------------------------------------------------------
# JB Weight
# ---------------------------------------------------------------------------

class TestJBWeight:
	def test_high_pvalue_dampens_to_zero(self):
		"""p_JB = 1.0 → w = 0 → effective moments = 0."""
		k_eff, g_eff = apply_jb_weight(5.0, 2.0, 1.0)
		assert k_eff == 0.0
		assert g_eff == 0.0

	def test_low_pvalue_preserves(self):
		"""p_JB = 0.0 → w = 1 → effective moments unchanged."""
		k_eff, g_eff = apply_jb_weight(5.0, 2.0, 0.0)
		assert k_eff == 5.0
		assert g_eff == 2.0

	def test_partial_dampening(self):
		"""
		p_JB = 0.3 → w = 0.7.
		κ_eff = 5.0 × 0.7 = 3.5
		γ_eff = 2.0 × 0.7 = 1.4
		"""
		k_eff, g_eff = apply_jb_weight(5.0, 2.0, 0.3)
		assert k_eff == pytest.approx(3.5)
		assert g_eff == pytest.approx(1.4)


# ---------------------------------------------------------------------------
# Liquidity Thinness
# ---------------------------------------------------------------------------

class TestLiquidityThinness:
	def test_basic(self):
		"""
		trades = [100, 200, 300], median = 200, TVL = 10000.
		L_thin = 200/10000 = 0.02
		"""
		vols = np.array([100.0, 200.0, 300.0])
		assert compute_liquidity_thinness(vols, 10_000.0) == pytest.approx(0.02)

	def test_deep_pool(self):
		"""Large TVL → small L_thin."""
		vols = np.array([100.0, 200.0, 300.0])
		l = compute_liquidity_thinness(vols, 50_000_000.0)
		assert l < 0.0001

	def test_zero_liquidity_raises(self):
		"""Can't divide by zero TVL."""
		vols = np.array([100.0])
		with pytest.raises(ValueError, match="positive"):
			compute_liquidity_thinness(vols, 0.0)


# ---------------------------------------------------------------------------
# Signal Momentum (S_m)
# ---------------------------------------------------------------------------

class TestSignalMomentum:
	def test_low_z_returns_zero(self):
		"""If |z| < 1, there's no signal regardless of other factors."""
		assert compute_signal_momentum(0.5, 10.0, 5.0) == 0.0
		assert compute_signal_momentum(-0.9, 10.0, 5.0) == 0.0

	def test_hand_calculated(self):
		"""
		z=3, κ_eff=2, γ_eff=1.

		log(S_m) = 0.40·log(3) + 0.30·log(1+2) + 0.15·log(1+1)
		         = 0.40·1.0986 + 0.30·1.0986 + 0.15·0.6931
		         = 0.4395 + 0.3296 + 0.1040
		         = 0.8730
		S_m = exp(0.8730) ≈ 2.3938
		"""
		sm = compute_signal_momentum(3.0, 2.0, 1.0)
		expected = math.exp(
			0.40 * math.log(3.0)
			+ 0.30 * math.log(3.0)
			+ 0.15 * math.log(2.0)
		)
		assert sm == pytest.approx(expected, rel=1e-6)

	def test_all_factors_needed(self):
		"""
		κ_eff=0 and γ_eff=0 should produce a lower S_m than when both are high,
		even if z is the same.
		"""
		sm_flat = compute_signal_momentum(5.0, 0.0, 0.0)
		sm_rich = compute_signal_momentum(5.0, 5.0, 3.0)
		assert sm_rich > sm_flat

	def test_monotone_in_z(self):
		"""Higher z → higher S_m, all else equal."""
		sm_low = compute_signal_momentum(2.0, 2.0, 1.0)
		sm_high = compute_signal_momentum(5.0, 2.0, 1.0)
		assert sm_high > sm_low


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

class TestBootstrapCI:
	def test_ci_contains_true_value(self):
		"""
		For a normal sample, the true excess kurtosis is 0.
		The 95% CI should contain 0 most of the time.
		"""
		rng = np.random.default_rng(42)
		data = rng.normal(0, 1, size=300)
		lower, upper = bootstrap_kurtosis_ci(data, seed=42)
		# CI should bracket 0 for a normal sample.
		assert lower < 0.0 < upper

	def test_ci_ordering(self):
		"""Lower bound < upper bound."""
		data = np.random.default_rng(42).exponential(1.0, size=200)
		lower, upper = bootstrap_kurtosis_ci(data, seed=42)
		assert lower < upper


# ---------------------------------------------------------------------------
# Full pipeline (integration)
# ---------------------------------------------------------------------------

class TestAnalyze:
	def test_tier3_insufficient_data(self):
		"""< 100 trades → tier 3, only z-score computed."""
		trades = [make_trade(v) for v in range(50)]
		baseline = np.array([100.0] * 167 + [120.0])  # 168 hours.
		report = analyze(
			token_id="test:TOKEN",
			trades=trades,
			hourly_volumes=baseline,
			current_volume=500.0,
			liquidity_usd=100_000.0,
			timestamp=1000,
		)

		assert report.confidence_tier == 3
		assert report.volume_zscore != 0.0
		assert report.excess_kurtosis is None
		assert report.skewness is None
		assert report.signal_momentum is None
		# Liquidity thinness is always computed.
		assert report.liquidity_thinness > 0.0

	def test_tier1_full_pipeline(self):
		"""≥ 500 trades → tier 1, all stats computed."""
		rng = np.random.default_rng(42)
		# Simulate whale accumulation: mostly small trades + a few big buys.
		small = rng.exponential(scale=100.0, size=480)
		big = rng.exponential(scale=5000.0, size=20)
		volumes = np.concatenate([small, big])
		rng.shuffle(volumes)

		trades = [make_trade(v, "buy") for v in volumes]
		baseline = rng.normal(5000.0, 500.0, size=168)  # Realistic noisy baseline.

		report = analyze(
			token_id="test:WHALE",
			trades=trades,
			hourly_volumes=baseline,
			current_volume=50_000.0,  # 10x above mean → high z.
			liquidity_usd=200_000.0,
			timestamp=2000,
		)

		assert report.confidence_tier == 1
		assert report.volume_zscore > 2.0  # Clearly anomalous.
		assert report.excess_kurtosis is not None
		assert report.excess_kurtosis > 0.0  # Heavy tails from whale trades.
		assert report.skewness is not None
		assert report.skewness > 0.0  # Right-skewed (big buys).
		assert report.signal_momentum is not None
		assert report.signal_momentum > 0.0
		assert report.bootstrap_ci is None  # Only for tier 2.

	def test_tier2_has_bootstrap(self):
		"""100-499 trades → tier 2, bootstrap CI computed."""
		trades = [make_trade(float(i)) for i in range(1, 201)]
		rng = np.random.default_rng(99)
		baseline = rng.normal(100.0, 15.0, size=168)

		report = analyze(
			token_id="test:MID",
			trades=trades,
			hourly_volumes=baseline,
			current_volume=300.0,
			liquidity_usd=50_000.0,
			timestamp=3000,
		)

		assert report.confidence_tier == 2
		assert report.bootstrap_ci is not None
		lower, upper = report.bootstrap_ci
		assert lower < upper
