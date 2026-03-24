"""
Alpha Brain — Signal-to-Noise Momentum statistical pipeline.

All computations are standard moment-based statistics:
	- Volume Z-score: (V - μ) / σ on hourly aggregated volumes.
	- Excess kurtosis: m₄/m₂² - 3 (Fisher definition, normal → 0).
	- Skewness: m₃/m₂^(3/2).
	- Jarque-Bera: (n/6)(γ² + κ²/4), χ²(2) under H₀.
	- S_m: log-additive composite of z, κ_eff, γ_eff.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Trade:
	"""A single trade from a DEX."""
	timestamp: int		# unix ms
	price: float
	volume_usd: float
	side: str			# "buy" | "sell"
	tx_hash: str


@dataclass
class AlphaReport:
	"""Output of the statistical pipeline."""
	token_id: str
	timestamp: int
	sample_n: int
	confidence_tier: int				# 1, 2, or 3

	# Core statistics
	volume_zscore: float
	excess_kurtosis: float | None		# None if tier 3
	skewness: float | None				# None if tier 3
	jarque_bera_pvalue: float | None	# None if tier 3

	# JB-weighted effective values
	kurtosis_effective: float | None	# κ × (1 - p_JB)
	skewness_effective: float | None	# γ × (1 - p_JB)

	# Composite signal
	signal_momentum: float | None		# None if tier 3

	# Risk context (separate axis, not part of S_m)
	liquidity_thinness: float
	liquidity_usd: float

	# For proof blob
	trade_data: list[Trade]
	bootstrap_ci: tuple[float, float] | None	# 95% CI for κ (tier 2)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# S_m weights — log-additive composite.
# Volume anomaly dominates; distribution shape is secondary.
W_Z = 0.40
W_KAPPA = 0.30
W_GAMMA = 0.15
# Remaining 0.15 is reserved for future factors; currently S_m uses 3 axes.

# Confidence tier thresholds (trade count).
TIER1_MIN = 500
TIER2_MIN = 100

# Bootstrap resamples for tier 2 kurtosis CI.
BOOTSTRAP_N = 1000


# ---------------------------------------------------------------------------
# Step 1: Volume Z-Score
# ---------------------------------------------------------------------------

def compute_volume_zscore(
	hourly_volumes: np.ndarray,
	current_volume: float,
) -> float:
	"""
	Z-score of current hour's volume against the baseline distribution.

		z = (V_current - μ_baseline) / σ_baseline

	Args:
		hourly_volumes: Array of hourly aggregate volumes (the 7-day baseline).
		current_volume: Aggregate volume for the current hour.

	Returns:
		The z-score. Positive = above average, negative = below.

	Raises:
		ValueError: If baseline has zero standard deviation (dead market).
	"""
	mu = np.mean(hourly_volumes)
	sigma = np.std(hourly_volumes, ddof=1)  # Sample std (Bessel's correction).

	if sigma == 0.0:
		raise ValueError(
			"Baseline volume has zero variance — market is completely dead."
		)

	return float((current_volume - mu) / sigma)


# ---------------------------------------------------------------------------
# Step 2: Distribution shape (kurtosis, skewness, Jarque-Bera)
# ---------------------------------------------------------------------------

def compute_distribution_stats(
	trade_volumes: np.ndarray,
) -> tuple[float, float, float]:
	"""
	Compute excess kurtosis, skewness, and Jarque-Bera p-value
	on individual trade sizes.

		κ = m₄/m₂² - 3        (Fisher, normal → 0)
		γ = m₃/m₂^(3/2)
		JB = (n/6)(γ² + κ²/4)  ~ χ²(2) under H₀

	Args:
		trade_volumes: Array of individual trade sizes (USD).

	Returns:
		(excess_kurtosis, skewness, jarque_bera_pvalue)
	"""
	kurtosis = float(stats.kurtosis(trade_volumes, fisher=True))
	skewness = float(stats.skew(trade_volumes))
	jb_stat, jb_pvalue = stats.jarque_bera(trade_volumes)

	return kurtosis, skewness, float(jb_pvalue)


def apply_jb_weight(
	kurtosis: float,
	skewness: float,
	jb_pvalue: float,
) -> tuple[float, float]:
	"""
	Dampen higher moments by JB confidence.

		κ_eff = κ × (1 - p_JB)
		γ_eff = γ × (1 - p_JB)

	When p_JB → 1 (normality not rejected), moments shrink to zero.
	When p_JB → 0 (strong non-normality), full weight preserved.
	"""
	w = 1.0 - jb_pvalue
	return kurtosis * w, skewness * w


# ---------------------------------------------------------------------------
# Step 2b: Bootstrap CI for kurtosis (tier 2 only)
# ---------------------------------------------------------------------------

def bootstrap_kurtosis_ci(
	trade_volumes: np.ndarray,
	n_resamples: int = BOOTSTRAP_N,
	ci: float = 0.95,
	seed: int | None = None,
) -> tuple[float, float]:
	"""
	Non-parametric bootstrap 95% CI for excess kurtosis.

	Resamples trade_volumes with replacement n_resamples times,
	computes kurtosis on each resample, returns percentile interval.

	Args:
		trade_volumes: Original trade size array.
		n_resamples: Number of bootstrap resamples.
		ci: Confidence level (default 0.95).
		seed: Optional RNG seed for reproducibility.

	Returns:
		(lower_bound, upper_bound) of the CI.
	"""
	rng = np.random.default_rng(seed)
	n = len(trade_volumes)
	kurtosis_samples = np.empty(n_resamples)

	for i in range(n_resamples):
		resample = rng.choice(trade_volumes, size=n, replace=True)
		kurtosis_samples[i] = stats.kurtosis(resample, fisher=True)

	alpha = (1.0 - ci) / 2.0
	lower = float(np.percentile(kurtosis_samples, 100 * alpha))
	upper = float(np.percentile(kurtosis_samples, 100 * (1.0 - alpha)))

	return lower, upper


# ---------------------------------------------------------------------------
# Step 3: Liquidity thinness (risk context, NOT part of S_m)
# ---------------------------------------------------------------------------

def compute_liquidity_thinness(
	trade_volumes: np.ndarray,
	liquidity_usd: float,
) -> float:
	"""
	L_thin = median(trade_sizes) / pool_TVL

	High → trades are large relative to pool → high slippage risk.
	Low → deep pool → low slippage.

	This is a risk descriptor, not a signal amplifier.
	"""
	if liquidity_usd <= 0.0:
		raise ValueError("Liquidity must be positive.")

	return float(np.median(trade_volumes) / liquidity_usd)


# ---------------------------------------------------------------------------
# Composite: Signal-to-Noise Momentum (S_m)
# ---------------------------------------------------------------------------

def compute_signal_momentum(
	z: float,
	kappa_eff: float,
	gamma_eff: float,
) -> float:
	"""
	Log-additive composite — all factors must be elevated for a high score.

		log(S_m) = W_Z·log(|z|) + W_KAPPA·log(1 + |κ_eff|) + W_GAMMA·log(1 + |γ_eff|)
		S_m = exp(log(S_m))

	The sign information (direction of anomaly, tail type, skew direction)
	is preserved in the AlphaReport fields, not collapsed into S_m.
	S_m is a magnitude — "how strong is the signal, ignoring direction?"

	Returns 0.0 if z is negligible (|z| < 1, i.e. within 1 std dev).
	"""
	abs_z = abs(z)

	# If volume isn't even 1 sigma above baseline, there's no signal.
	if abs_z < 1.0:
		return 0.0

	log_sm = (
		W_Z * math.log(abs_z)
		+ W_KAPPA * math.log(1.0 + abs(kappa_eff))
		+ W_GAMMA * math.log(1.0 + abs(gamma_eff))
	)

	return math.exp(log_sm)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def analyze(
	token_id: str,
	trades: list[Trade],
	hourly_volumes: np.ndarray,
	current_volume: float,
	liquidity_usd: float,
	timestamp: int | None = None,
) -> AlphaReport:
	"""
	Run the full statistical pipeline on a token.

	Args:
		token_id: Identifier for the token being analyzed.
		trades: List of Trade objects in the current analysis window.
		hourly_volumes: 7-day baseline of hourly aggregate volumes.
		current_volume: Current hour's aggregate volume.
		liquidity_usd: Pool TVL in USD.
		timestamp: Unix timestamp for the report (defaults to now-ish).

	Returns:
		A complete AlphaReport.
	"""
	import time

	if timestamp is None:
		timestamp = int(time.time())

	n = len(trades)
	trade_volumes = np.array([t.volume_usd for t in trades])

	# --- Step 1: Volume Z-score (always computed) ---
	z = compute_volume_zscore(hourly_volumes, current_volume)

	# --- Determine confidence tier ---
	if n >= TIER1_MIN:
		tier = 1
	elif n >= TIER2_MIN:
		tier = 2
	else:
		tier = 3

	# --- Step 2: Distribution shape (tiers 1 & 2 only) ---
	kurtosis = None
	skewness = None
	jb_pvalue = None
	kappa_eff = None
	gamma_eff = None
	sm = None
	bootstrap = None

	if tier <= 2:
		kurtosis, skewness, jb_pvalue = compute_distribution_stats(trade_volumes)
		kappa_eff, gamma_eff = apply_jb_weight(kurtosis, skewness, jb_pvalue)
		sm = compute_signal_momentum(z, kappa_eff, gamma_eff)

		# Bootstrap CI for tier 2 (sample size is moderate, quantify uncertainty).
		if tier == 2:
			bootstrap = bootstrap_kurtosis_ci(trade_volumes)

	# --- Step 3: Liquidity context (always computed) ---
	l_thin = compute_liquidity_thinness(trade_volumes, liquidity_usd)

	return AlphaReport(
		token_id=token_id,
		timestamp=timestamp,
		sample_n=n,
		confidence_tier=tier,
		volume_zscore=z,
		excess_kurtosis=kurtosis,
		skewness=skewness,
		jarque_bera_pvalue=jb_pvalue,
		kurtosis_effective=kappa_eff,
		skewness_effective=gamma_eff,
		signal_momentum=sm,
		liquidity_thinness=l_thin,
		liquidity_usd=liquidity_usd,
		trade_data=trades,
		bootstrap_ci=bootstrap,
	)
