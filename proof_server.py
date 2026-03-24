"""
Proof Server — HTTP API for the Alpha Oracle agent.

Endpoints:
	POST /request          — Submit analysis request, get Fiber invoice.
	GET  /result/{pay_hash} — Poll for result after payment.
	GET  /proof/{hash}      — Fetch full proof blob (verifiable via blake2b).
	GET  /health            — Health check.

For the hackathon demo, this also includes a /analyze endpoint that
skips Fiber payment and runs the pipeline directly (for testing).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time

import httpx
import numpy as np
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from tools.alpha_brain import Trade, AlphaReport, analyze
from tools.data_fetcher import fetch_pair_context, estimate_hourly_baseline
from tools.llm_analyst import interpret, build_analysis_prompt
from fiber_client import FiberClient
from ckb_publisher import build_result_cell, publish_result_cell

# ---------------------------------------------------------------------------
# Configuration (from environment / .env)
# ---------------------------------------------------------------------------

BIRDEYE_API_KEY = os.environ.get("BIRDEYE_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANALYSIS_FEE_CKB = int(os.environ.get("ANALYSIS_FEE_CKB", "10"))
FIBER_RPC_URL = os.environ.get("FIBER_RPC_URL", "http://localhost:8227")

# Fiber client (None if no Fiber node configured).
fiber: FiberClient | None = None
if FIBER_RPC_URL:
	fiber = FiberClient(rpc_url=FIBER_RPC_URL)

# In-memory stores (SQLite in production; dicts for hackathon).
results_store: dict[str, dict] = {}  # keyed by request_id
proof_store: dict[str, dict] = {}    # keyed by proof_hash
# Pending requests: payment_hash → {token_address, chain, status}
pending_requests: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Data fetching helpers
# ---------------------------------------------------------------------------

async def fetch_trades(token_address: str, chain: str = "solana", max_pages: int = 6) -> list[Trade]:
	"""Fetch trades from Birdeye V1 endpoint."""
	trades: list[Trade] = []
	async with httpx.AsyncClient(timeout=30.0) as client:
		for page in range(max_pages):
			if page > 0:
				await asyncio.sleep(1.2)  # 1 req/sec rate limit.
			resp = await client.get(
				"https://public-api.birdeye.so/defi/txs/token",
				headers={"X-API-KEY": BIRDEYE_API_KEY, "x-chain": chain},
				params={
					"address": token_address,
					"limit": 50,
					"tx_type": "swap",
					"offset": page * 50,
				},
			)
			body = resp.json()
			if not body.get("success") or not body.get("data", {}).get("items"):
				break
			for item in body["data"]["items"]:
				base = item.get("base", {})
				vol = float(base.get("price", 0) or 0) * float(base.get("uiAmount", 0) or 0)
				trades.append(Trade(
					timestamp=item["blockUnixTime"] * 1000,
					price=float(base.get("price", 0) or 0),
					volume_usd=vol,
					side=item.get("side", "unknown"),
					tx_hash=item.get("txHash", ""),
				))
			if not body["data"].get("hasNext", False):
				break
	return trades


def build_proof_blob(report: AlphaReport, narrative: str, pair_url: str) -> dict:
	"""Build the full proof blob that gets hashed and stored."""
	return {
		"token_id": report.token_id,
		"timestamp": report.timestamp,
		"sample_n": report.sample_n,
		"confidence_tier": report.confidence_tier,
		"volume_zscore": report.volume_zscore,
		"excess_kurtosis": report.excess_kurtosis,
		"kurtosis_effective": report.kurtosis_effective,
		"skewness": report.skewness,
		"skewness_effective": report.skewness_effective,
		"jarque_bera_pvalue": report.jarque_bera_pvalue,
		"signal_momentum": report.signal_momentum,
		"liquidity_thinness": report.liquidity_thinness,
		"liquidity_usd": report.liquidity_usd,
		"bootstrap_ci": list(report.bootstrap_ci) if report.bootstrap_ci else None,
		"narrative": narrative,
		"pair_url": pair_url,
		"trades": [
			{
				"timestamp": t.timestamp,
				"price": t.price,
				"volume_usd": t.volume_usd,
				"side": t.side,
				"tx_hash": t.tx_hash,
			}
			for t in report.trade_data
		],
	}


def hash_proof(blob: dict) -> str:
	"""blake2b hash of the canonical JSON proof blob."""
	canonical = json.dumps(blob, sort_keys=True, separators=(",", ":"))
	return hashlib.blake2b(canonical.encode(), digest_size=32).hexdigest()


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

async def run_pipeline(token_address: str, chain: str = "solana") -> dict:
	"""
	Run the full analysis pipeline for a token.

	Returns a dict with the report summary, narrative, and proof hash.
	"""
	# 1. Pair context from DEXScreener.
	ctx = await fetch_pair_context(token_address, chain_id=chain)
	if ctx is None:
		raise ValueError(f"No pairs found for {token_address} on {chain}.")

	# 2. Fetch trades from Birdeye.
	trades = await fetch_trades(token_address, chain=chain)
	if not trades:
		raise ValueError(f"No trades found for {token_address}. Token may be too new or inactive.")

	# 3. Build baseline and run stats.
	baseline = estimate_hourly_baseline(ctx)
	current_vol = sum(t.volume_usd for t in trades[:20])

	report = analyze(
		token_id=f"{chain}:{token_address}",
		trades=trades,
		hourly_volumes=baseline,
		current_volume=current_vol,
		liquidity_usd=ctx.liquidity_usd,
	)

	# 4. LLM interpretation.
	narrative = ""
	if ANTHROPIC_API_KEY:
		try:
			narrative = await interpret(report, ANTHROPIC_API_KEY)
		except Exception as e:
			narrative = f"LLM interpretation unavailable: {e}"
	else:
		narrative = "LLM interpretation skipped (no API key configured)."

	# 5. Build and store proof.
	proof_blob = build_proof_blob(report, narrative, ctx.pair_url)
	proof_hash = hash_proof(proof_blob)
	proof_store[proof_hash] = proof_blob

	# 6. Build CKB Result Cell and (stub) publish.
	result_cell = build_result_cell(
		token_id=report.token_id,
		timestamp=report.timestamp,
		sample_n=report.sample_n,
		volume_zscore=report.volume_zscore,
		excess_kurtosis=report.excess_kurtosis,
		skewness=report.skewness,
		confidence_tier=report.confidence_tier,
		signal_momentum=report.signal_momentum,
		full_proof_hash=proof_hash,
	)
	ckb_tx_info = publish_result_cell(result_cell)

	# 7. Build response summary.
	return {
		"token_id": report.token_id,
		"pair": f"{ctx.base_symbol}/{ctx.quote_symbol}",
		"pair_url": ctx.pair_url,
		"timestamp": report.timestamp,
		"confidence_tier": report.confidence_tier,
		"confidence_label": ["", "High", "Moderate", "Low"][report.confidence_tier],
		"sample_n": report.sample_n,
		"stats": {
			"volume_zscore": round(report.volume_zscore, 4),
			"excess_kurtosis": round(report.excess_kurtosis, 4) if report.excess_kurtosis is not None else None,
			"kurtosis_effective": round(report.kurtosis_effective, 4) if report.kurtosis_effective is not None else None,
			"skewness": round(report.skewness, 4) if report.skewness is not None else None,
			"skewness_effective": round(report.skewness_effective, 4) if report.skewness_effective is not None else None,
			"jarque_bera_pvalue": round(report.jarque_bera_pvalue, 6) if report.jarque_bera_pvalue is not None else None,
			"signal_momentum": round(report.signal_momentum, 4) if report.signal_momentum is not None else None,
			"liquidity_thinness": round(report.liquidity_thinness, 10),
			"liquidity_usd": round(report.liquidity_usd, 2),
		},
		"bootstrap_ci": [round(x, 4) for x in report.bootstrap_ci] if report.bootstrap_ci else None,
		"narrative": narrative,
		"proof_hash": proof_hash,
		"proof_url": f"/proof/{proof_hash}",
		"ckb_result_cell": {
			"data_hex": result_cell.data_hex,
			"data_hash": result_cell.data_hash,
			"data_size_bytes": len(result_cell.data),
			"capacity_ckb": result_cell.capacity_ckb,
		},
		"ckb_tx": ckb_tx_info,
	}


# ---------------------------------------------------------------------------
# Background: process paid requests
# ---------------------------------------------------------------------------

async def process_paid_request(payment_hash: str, token_address: str, chain: str):
	"""Run the analysis pipeline after payment is confirmed."""
	try:
		pending_requests[payment_hash]["status"] = "processing"
		result = await run_pipeline(token_address, chain=chain)
		result["payment_hash"] = payment_hash
		results_store[payment_hash] = result
		pending_requests[payment_hash]["status"] = "completed"
	except Exception as e:
		pending_requests[payment_hash]["status"] = "failed"
		pending_requests[payment_hash]["error"] = str(e)


async def payment_watcher(payment_hash: str, token_address: str, chain: str):
	"""Poll Fiber invoice status, trigger pipeline when paid."""
	try:
		status = await fiber.wait_for_payment(payment_hash, poll_interval=2.0, timeout=3600.0)
		pending_requests[payment_hash]["status"] = "paid"
		await process_paid_request(payment_hash, token_address, chain)
	except (TimeoutError, RuntimeError) as e:
		pending_requests[payment_hash]["status"] = "expired"
		pending_requests[payment_hash]["error"] = str(e)


# ---------------------------------------------------------------------------
# HTTP Endpoints
# ---------------------------------------------------------------------------

async def health(request: Request) -> JSONResponse:
	"""Health check."""
	fiber_ok = False
	if fiber:
		try:
			await fiber.node_info()
			fiber_ok = True
		except Exception:
			pass
	return JSONResponse({
		"status": "ok",
		"timestamp": int(time.time()),
		"fiber_connected": fiber_ok,
	})


async def request_analysis(request: Request) -> JSONResponse:
	"""
	POST /request
	Submit an analysis request. Returns a Fiber invoice to pay.

	Body: { "token_address": "...", "chain": "solana" }

	Flow:
	  1. Agent creates Fiber invoice for the analysis fee.
	  2. Returns invoice to user.
	  3. Background task polls for payment.
	  4. On payment → run pipeline → store result.
	  5. User polls GET /result/{payment_hash} for the result.
	"""
	if not fiber:
		return JSONResponse(
			{"error": "Fiber node not configured. Use /analyze for direct access."},
			status_code=503,
		)

	try:
		body = await request.json()
	except Exception:
		return JSONResponse({"error": "Invalid JSON body."}, status_code=400)

	token_address = body.get("token_address", "").strip()
	chain = body.get("chain", "solana").strip()

	if not token_address:
		return JSONResponse({"error": "token_address is required."}, status_code=400)

	# Create Fiber invoice.
	try:
		invoice = await fiber.create_invoice(
			amount_ckb=ANALYSIS_FEE_CKB,
			description=f"Alpha Oracle analysis: {token_address[:16]}",
		)
	except Exception as e:
		return JSONResponse({"error": f"Failed to create invoice: {e}"}, status_code=502)

	# Track the pending request.
	pending_requests[invoice.payment_hash] = {
		"token_address": token_address,
		"chain": chain,
		"status": "awaiting_payment",
		"invoice_address": invoice.invoice_address,
		"amount_ckb": ANALYSIS_FEE_CKB,
		"created_at": int(time.time()),
	}

	# Start background payment watcher.
	asyncio.create_task(payment_watcher(invoice.payment_hash, token_address, chain))

	return JSONResponse({
		"payment_hash": invoice.payment_hash,
		"invoice": invoice.invoice_address,
		"amount_ckb": ANALYSIS_FEE_CKB,
		"expires_in_seconds": 3600,
		"poll_url": f"/result/{invoice.payment_hash}",
		"instructions": "Pay the invoice via your Fiber node, then poll the poll_url for results.",
	})


async def analyze_endpoint(request: Request) -> JSONResponse:
	"""
	POST /analyze
	Direct analysis (skips Fiber payment — for testing/demo).

	Body: { "token_address": "...", "chain": "solana" }
	"""
	try:
		body = await request.json()
	except Exception:
		return JSONResponse({"error": "Invalid JSON body."}, status_code=400)

	token_address = body.get("token_address", "").strip()
	chain = body.get("chain", "solana").strip()

	if not token_address:
		return JSONResponse({"error": "token_address is required."}, status_code=400)

	try:
		result = await run_pipeline(token_address, chain=chain)
		request_id = f"{int(time.time())}_{token_address[:8]}"
		results_store[request_id] = result
		result["request_id"] = request_id
		return JSONResponse(result)
	except ValueError as e:
		return JSONResponse({"error": str(e)}, status_code=404)
	except Exception as e:
		return JSONResponse({"error": f"Pipeline error: {e}"}, status_code=500)


async def get_result(request: Request) -> JSONResponse:
	"""
	GET /result/{payment_hash_or_request_id}
	Poll for result. Works with both Fiber payment_hash and direct request_id.
	"""
	key = request.path_params["request_id"]

	# Check completed results first.
	result = results_store.get(key)
	if result is not None:
		return JSONResponse(result)

	# Check pending requests.
	pending = pending_requests.get(key)
	if pending is not None:
		return JSONResponse({
			"payment_hash": key,
			"status": pending["status"],
			"error": pending.get("error"),
		})

	return JSONResponse({"error": "Not found.", "key": key}, status_code=404)


async def get_proof(request: Request) -> JSONResponse:
	"""
	GET /proof/{proof_hash}
	Fetch full proof blob. Verifiable: blake2b(response) == proof_hash.
	"""
	proof_hash = request.path_params["proof_hash"]
	blob = proof_store.get(proof_hash)
	if blob is None:
		return JSONResponse({"error": "Proof not found.", "proof_hash": proof_hash}, status_code=404)
	return JSONResponse(blob)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = Starlette(
	routes=[
		Route("/health", health, methods=["GET"]),
		Route("/request", request_analysis, methods=["POST"]),
		Route("/analyze", analyze_endpoint, methods=["POST"]),
		Route("/result/{request_id}", get_result, methods=["GET"]),
		Route("/proof/{proof_hash}", get_proof, methods=["GET"]),
	],
)

if __name__ == "__main__":
	import uvicorn
	port = int(os.environ.get("PORT", "8080"))
	print(f"Starting Alpha Oracle on port {port}...")
	uvicorn.run(app, host="0.0.0.0", port=port)
