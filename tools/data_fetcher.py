"""
Data Fetcher — pulls trade-level and aggregate data for statistical analysis.

Two data sources:
	- Birdeye API: Individual trades (volume_usd, side, timestamp per trade).
	- DEXScreener API: Pair metadata, aggregated volume, liquidity (TVL).

Birdeye provides the micro view (trade-level distribution).
DEXScreener provides the macro view (baseline volume + liquidity context).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import httpx
import numpy as np

from tools.alpha_brain import Trade


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BIRDEYE_BASE = "https://public-api.birdeye.so"
DEXSCREENER_BASE = "https://api.dexscreener.com"

# Birdeye V3 token trades endpoint.
# Docs: https://docs.birdeye.so/reference
# Fields we use: volume_usd, side, block_unix_time, tx_hash, source.
BIRDEYE_TRADES_PATH = "/defi/v3/token/txs"

# DEXScreener token pairs endpoint.
# Returns: volume (h1/h6/h24), liquidity.usd, txns counts.
DEXSCREENER_PAIRS_PATH = "/token-pairs/v1"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PairContext:
	"""Aggregated context from DEXScreener for a token pair."""
	chain_id: str
	pair_address: str
	base_symbol: str
	quote_symbol: str
	price_usd: float | None
	volume_h1: float
	volume_h6: float
	volume_h24: float
	liquidity_usd: float
	txns_h1_buys: int
	txns_h1_sells: int
	pair_url: str


# ---------------------------------------------------------------------------
# Birdeye: Individual trades
# ---------------------------------------------------------------------------

async def fetch_trades_birdeye(
	token_address: str,
	api_key: str,
	chain: str = "solana",
	max_trades: int = 500,
	after_time: int | None = None,
	before_time: int | None = None,
) -> list[Trade]:
	"""
	Fetch individual trades from Birdeye V3 API.

	Uses time-windowed pagination: fetches pages of 100 trades,
	advancing the before_time cursor to walk backwards in time.

	Args:
		token_address: On-chain token address.
		api_key: Birdeye API key.
		chain: Blockchain name (default: solana).
		max_trades: Maximum trades to fetch (default: 500 for tier 1).
		after_time: Only fetch trades after this unix timestamp (seconds).
		before_time: Only fetch trades before this unix timestamp (seconds).

	Returns:
		List of Trade objects, most recent first.
	"""
	headers = {
		"X-API-KEY": api_key,
		"x-chain": chain,
	}

	trades: list[Trade] = []
	cursor_before = before_time or int(time.time())
	page_size = 100

	async with httpx.AsyncClient(timeout=30.0) as client:
		while len(trades) < max_trades:
			params: dict = {
				"address": token_address,
				"limit": page_size,
				"sort_by": "block_unix_time",
				"sort_type": "desc",
				"tx_type": "swap",
				"before_time": cursor_before,
			}
			if after_time is not None:
				params["after_time"] = after_time

			resp = await client.get(
				f"{BIRDEYE_BASE}{BIRDEYE_TRADES_PATH}",
				headers=headers,
				params=params,
			)
			resp.raise_for_status()
			data = resp.json()

			if not data.get("success"):
				raise RuntimeError(f"Birdeye API error: {data}")

			items = data["data"]["items"]
			if not items:
				break

			for item in items:
				trades.append(Trade(
					timestamp=item["block_unix_time"] * 1000,  # Convert to ms.
					price=float(item.get("from", {}).get("price", 0) or 0),
					volume_usd=float(item["volume_usd"]),
					side=item["side"],
					tx_hash=item["tx_hash"],
				))

			# Advance cursor to the oldest trade in this page.
			cursor_before = items[-1]["block_unix_time"]

			# Stop if no more pages.
			if not data["data"].get("hasNext", False):
				break

	return trades[:max_trades]


# ---------------------------------------------------------------------------
# DEXScreener: Pair metadata + aggregated volume + liquidity
# ---------------------------------------------------------------------------

async def fetch_pair_context(
	token_address: str,
	chain_id: str = "solana",
) -> PairContext | None:
	"""
	Fetch the highest-liquidity pair for a token from DEXScreener.

	Returns aggregated volume, liquidity, and transaction counts.
	No API key required — DEXScreener public API.

	Args:
		token_address: On-chain token address.
		chain_id: Chain identifier (e.g., "solana", "ethereum").

	Returns:
		PairContext for the most liquid pair, or None if no pairs found.
	"""
	url = f"{DEXSCREENER_BASE}{DEXSCREENER_PAIRS_PATH}/{chain_id}/{token_address}"

	async with httpx.AsyncClient(timeout=15.0) as client:
		resp = await client.get(url)
		resp.raise_for_status()
		pairs = resp.json()

	if not pairs:
		return None

	# Pick the pair with the highest liquidity.
	best = max(pairs, key=lambda p: (p.get("liquidity") or {}).get("usd", 0) or 0)

	liquidity = best.get("liquidity") or {}
	volume = best.get("volume") or {}
	txns = best.get("txns") or {}
	txns_h1 = txns.get("h1") or {}

	return PairContext(
		chain_id=best.get("chainId", chain_id),
		pair_address=best.get("pairAddress", ""),
		base_symbol=best.get("baseToken", {}).get("symbol", "???"),
		quote_symbol=best.get("quoteToken", {}).get("symbol", "???"),
		price_usd=float(best["priceUsd"]) if best.get("priceUsd") else None,
		volume_h1=float(volume.get("h1", 0) or 0),
		volume_h6=float(volume.get("h6", 0) or 0),
		volume_h24=float(volume.get("h24", 0) or 0),
		liquidity_usd=float(liquidity.get("usd", 0) or 0),
		txns_h1_buys=int(txns_h1.get("buys", 0) or 0),
		txns_h1_sells=int(txns_h1.get("sells", 0) or 0),
		pair_url=best.get("url", ""),
	)


# ---------------------------------------------------------------------------
# Combined: Build volume baseline from DEXScreener aggregates
# ---------------------------------------------------------------------------

def estimate_hourly_baseline(pair_ctx: PairContext) -> np.ndarray:
	"""
	Estimate a 7-day hourly volume baseline from DEXScreener aggregates.

	DEXScreener gives us h1, h6, h24 volumes. We construct a rough
	hourly baseline by assuming the 24h volume distributes evenly
	across hours (with noise from the h1/h6 ratios).

	This is an approximation — the alternative is fetching 7 days of
	trade-level data from Birdeye, which costs ~168 API calls.

	Returns:
		numpy array of 168 estimated hourly volumes (7 days × 24 hours).
	"""
	# Hourly rate from 24h volume.
	avg_hourly = pair_ctx.volume_h24 / 24.0

	if avg_hourly <= 0:
		return np.ones(168)  # Avoid zero baseline; analyze will handle edge cases.

	# Use h1/h6 ratio to estimate recent vs. baseline activity level.
	# This adds some realistic variance to the baseline.
	h6_hourly = pair_ctx.volume_h6 / 6.0 if pair_ctx.volume_h6 > 0 else avg_hourly

	# Generate baseline with some variance around the 24h average.
	# The ratio between h6 and h24 hourly rates gives us a variance signal.
	ratio = h6_hourly / avg_hourly if avg_hourly > 0 else 1.0
	std_estimate = avg_hourly * abs(ratio - 1.0) * 0.5

	# Ensure minimum variance so Z-score doesn't blow up.
	std_estimate = max(std_estimate, avg_hourly * 0.1)

	rng = np.random.default_rng(42)  # Deterministic for reproducibility.
	baseline = rng.normal(avg_hourly, std_estimate, size=168)

	# Volumes can't be negative.
	baseline = np.maximum(baseline, 0.0)

	return baseline
