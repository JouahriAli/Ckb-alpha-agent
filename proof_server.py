"""
Proof Server — HTTP API for the Alpha Oracle agent.

Trustless Fiber payment flow:
  1. POST /request          — Agent runs analysis, returns Fiber invoice
                              (preimage = proof_hash, locked in TLC).
  2. User pays invoice      — TLC settles, revealing proof_hash to user.
  3. GET  /proof/{hash}     — User fetches full results with the proof_hash.

Other endpoints:
  GET  /result/{pay_hash}   — Poll payment settlement status.
  GET  /health              — Health check.
  POST /analyze             — Direct analysis (skips Fiber, for testing).
  POST /demo/pay            — Demo: pay invoice from local user Fiber node.
  GET  /demo/preimage/{ph}  — Demo: simulate preimage reveal after settlement.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time

import httpx
import numpy as np
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, HTMLResponse
from starlette.routing import Route, Mount
from starlette.staticfiles import StaticFiles

from tools.alpha_brain import Trade, AlphaReport, analyze
from tools.data_fetcher import fetch_pair_context, estimate_hourly_baseline
from tools.llm_analyst import interpret, build_analysis_prompt
from fiber_client import FiberClient, SHANNON_PER_CKB
from ckb_publisher import build_result_cell, publish_result_cell

# ---------------------------------------------------------------------------
# Configuration (from environment / .env)
# ---------------------------------------------------------------------------

BIRDEYE_API_KEY = os.environ.get("BIRDEYE_API_KEY", "")
LLM_BACKEND = os.environ.get("LLM_BACKEND", "anthropic")
LLM_API_KEY = os.environ.get("OPENAI_API_KEY", "") or os.environ.get("ANTHROPIC_API_KEY", "")
ANALYSIS_FEE_CKB = int(os.environ.get("ANALYSIS_FEE_CKB", "10"))
FIBER_RPC_URL = os.environ.get("FIBER_RPC_URL", "http://localhost:8227")
AGENT_PEER_ID = os.environ.get("AGENT_PEER_ID", "")
AGENT_P2P_HOST = os.environ.get("AGENT_P2P_HOST", "127.0.0.1")
AGENT_P2P_PORT = os.environ.get("AGENT_P2P_PORT", "8228")

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
	if LLM_API_KEY or LLM_BACKEND == "ollama":
		try:
			narrative = await interpret(report, LLM_API_KEY)
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
# Background: payment watcher (trustless delivery)
# ---------------------------------------------------------------------------

async def payment_watcher(payment_hash: str, proof_hash: str):
	"""
	Poll Fiber invoice status. When paid, the user has received the
	proof_hash as the TLC preimage — they can now fetch /proof/{hash}.

	Trustless flow:
	  - Agent runs analysis FIRST, gets proof_hash.
	  - Agent creates invoice with preimage = proof_hash.
	  - User pays → TLC settles → user learns proof_hash (the preimage).
	  - User calls GET /proof/{proof_hash} to retrieve full results.
	  - Agent can't get paid without revealing proof_hash.
	  - User can't get proof_hash without paying.
	"""
	try:
		status = await fiber.wait_for_payment(payment_hash, poll_interval=2.0, timeout=3600.0)
		pending_requests[payment_hash]["status"] = "settled"
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
	Trustless analysis request. Runs pipeline FIRST, then creates a Fiber
	invoice whose TLC preimage IS the proof_hash. Payment and data delivery
	are atomic: the user can't learn the proof_hash without paying, and
	the agent can't collect without revealing it.

	Body: { "token_address": "...", "chain": "solana" }

	Flow:
	  1. Agent runs the full analysis pipeline → gets proof_hash.
	  2. Agent creates Fiber invoice with preimage = proof_hash.
	  3. Returns invoice to user (WITHOUT results or proof_hash).
	  4. User pays → TLC settles → user learns proof_hash (the preimage).
	  5. User calls GET /proof/{proof_hash} to retrieve full results.
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

	# 1. Run pipeline FIRST to get the proof_hash.
	try:
		result = await run_pipeline(token_address, chain=chain)
	except ValueError as e:
		return JSONResponse({"error": str(e)}, status_code=404)
	except Exception as e:
		return JSONResponse({"error": f"Pipeline error: {e}"}, status_code=500)

	proof_hash = result["proof_hash"]

	# Store full result keyed by proof_hash (already in proof_store from run_pipeline).
	results_store[proof_hash] = result

	# 2. Create Fiber invoice with preimage = proof_hash.
	#    payment_hash = sha256(proof_hash). When the TLC settles,
	#    the user's node learns the preimage (= proof_hash).
	try:
		invoice = await fiber.create_invoice(
			amount_ckb=ANALYSIS_FEE_CKB,
			description=f"Alpha Oracle analysis: {token_address[:16]}",
			preimage_hex=proof_hash,
		)
	except Exception as e:
		return JSONResponse({"error": f"Failed to create invoice: {e}"}, status_code=502)

	# 3. Track the pending request.
	pending_requests[invoice.payment_hash] = {
		"token_address": token_address,
		"chain": chain,
		"status": "awaiting_payment",
		"proof_hash": proof_hash,
		"invoice_address": invoice.invoice_address,
		"amount_ckb": ANALYSIS_FEE_CKB,
		"created_at": int(time.time()),
	}

	# 4. Start background payment watcher.
	asyncio.create_task(payment_watcher(invoice.payment_hash, proof_hash))

	# 5. Return invoice — NOT the results or proof_hash.
	return JSONResponse({
		"payment_hash": invoice.payment_hash,
		"invoice": invoice.invoice_address,
		"amount_ckb": ANALYSIS_FEE_CKB,
		"expires_in_seconds": 3600,
		"status": "awaiting_payment",
		"instructions": (
			"Pay the invoice via your Fiber node. "
			"When the payment settles, your node will receive the preimage — "
			"that IS the proof_hash. Use it to fetch results at GET /proof/{proof_hash}."
		),
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
	GET /result/{payment_hash}
	Poll for payment status. In the trustless flow, the actual results
	are fetched via GET /proof/{proof_hash} once the user learns the
	preimage from the TLC settlement.
	"""
	key = request.path_params["request_id"]

	# Check completed results (direct /analyze requests keyed by request_id).
	result = results_store.get(key)
	if result is not None:
		return JSONResponse(result)

	# Check pending Fiber requests (keyed by payment_hash).
	pending = pending_requests.get(key)
	if pending is not None:
		resp = {
			"payment_hash": key,
			"status": pending["status"],
		}
		if pending.get("error"):
			resp["error"] = pending["error"]
		# Once settled, tell the user to check their node for the preimage.
		if pending["status"] == "settled":
			resp["message"] = (
				"Payment settled. The TLC preimage revealed to your node "
				"is the proof_hash. Use GET /proof/{proof_hash} to fetch results."
			)
		return JSONResponse(resp)

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
# Demo: simulate user payment from local Fiber node
# ---------------------------------------------------------------------------

USER_FIBER_RPC = os.environ.get("USER_FIBER_RPC_URL", "http://localhost:8237")


async def agent_info(request: Request) -> JSONResponse:
	"""
	GET /agent/info
	Return the agent Fiber node's connection details so users can
	connect their own node and open a channel.
	"""
	if not fiber:
		return JSONResponse({"error": "Fiber not configured."}, status_code=503)
	try:
		info = await fiber.node_info()
	except Exception as e:
		return JSONResponse({"error": f"Failed to get node info: {e}"}, status_code=502)

	# Peer ID from node addresses or environment override
	peer_id = AGENT_PEER_ID
	addresses = info.get("addresses", [])
	for addr in addresses:
		if "/p2p/" in addr:
			peer_id = addr.split("/p2p/")[-1]
			break

	node_id = info.get("node_id", "")
	p2p_addr = f"/ip4/{AGENT_P2P_HOST}/tcp/{AGENT_P2P_PORT}/p2p/{peer_id}" if peer_id else None
	return JSONResponse({
		"node_id": node_id,
		"peer_id": peer_id or None,
		"fiber_version": info.get("version", ""),
		"chain": "testnet",
		"p2p_addr": p2p_addr,
		"rpc_note": "Agent RPC is not publicly exposed. Connect via p2p_addr.",
		"analysis_fee_ckb": ANALYSIS_FEE_CKB,
		"auto_accept_min_ckb": int(info.get("open_channel_auto_accept_min_ckb_funding_amount", "0x0"), 16) / SHANNON_PER_CKB if info.get("open_channel_auto_accept_min_ckb_funding_amount") else None,
		"instructions": [
			"1. Run your own Fiber node (fnn) on CKB testnet.",
			"2. Connect to the agent: connect_peer with the p2p_addr above.",
			"3. Open a channel with enough CKB to cover the analysis fee.",
			"4. POST /request with your token — you'll get a Fiber invoice.",
			"5. Pay the invoice from your node (send_payment).",
			"6. Your node reveals the TLC preimage (= proof_hash) on settlement.",
			"7. GET /proof/{proof_hash} to retrieve your results.",
		],
	})


async def _user_rpc(method: str, params: list | None = None) -> dict:
	"""Make a JSON-RPC call to the demo USER Fiber node."""
	async with httpx.AsyncClient(timeout=30.0) as client:
		resp = await client.post(
			USER_FIBER_RPC,
			json={"id": "demo", "jsonrpc": "2.0", "method": method, "params": params or []},
			headers={"Content-Type": "application/json"},
		)
		body = resp.json()
	if "error" in body and body["error"] is not None:
		raise RuntimeError(f"User Fiber RPC error ({method}): {body['error']}")
	return body.get("result")


async def demo_node_status(request: Request) -> JSONResponse:
	"""
	GET /demo/node-status
	Return the demo user Fiber node's status: node info, peers, channels.
	"""
	try:
		node = await _user_rpc("node_info")
		peers_result = await _user_rpc("list_peers")
		channels_result = await _user_rpc("list_channels", [{}])
	except Exception as e:
		return JSONResponse({"error": f"User node unavailable: {e}"}, status_code=502)

	peers = peers_result.get("peers", [])
	channels = []
	agent_channel = None
	for ch in channels_result.get("channels", []):
		info = {
			"channel_id": ch["channel_id"],
			"peer_id": ch["peer_id"],
			"state": ch["state"]["state_name"],
			"local_balance_ckb": round(int(ch["local_balance"], 16) / SHANNON_PER_CKB, 1),
			"remote_balance_ckb": round(int(ch["remote_balance"], 16) / SHANNON_PER_CKB, 1),
		}
		channels.append(info)
		if ch["peer_id"] == AGENT_PEER_ID and ch["state"]["state_name"] == "CHANNEL_READY":
			agent_channel = info

	return JSONResponse({
		"node_id": node.get("node_id", ""),
		"peer_count": len(peers),
		"peers": [{"peer_id": p["peer_id"], "address": p.get("address", "")} for p in peers],
		"channel_count": len(channels),
		"channels": channels,
		"agent_peer_id": AGENT_PEER_ID,
		"connected_to_agent": any(p["peer_id"] == AGENT_PEER_ID for p in peers),
		"agent_channel_ready": agent_channel is not None,
		"agent_channel": agent_channel,
	})


async def demo_connect(request: Request) -> JSONResponse:
	"""
	POST /demo/connect
	Connect the demo user node to the agent node via connect_peer.
	"""
	if not AGENT_PEER_ID:
		return JSONResponse({"error": "Agent peer ID not configured."}, status_code=500)

	multiaddr = f"/ip4/{AGENT_P2P_HOST}/tcp/{AGENT_P2P_PORT}/p2p/{AGENT_PEER_ID}"
	try:
		await _user_rpc("connect_peer", [{"address": multiaddr}])
		return JSONResponse({"status": "connected", "address": multiaddr})
	except RuntimeError as e:
		# "already connected" is fine
		if "already" in str(e).lower():
			return JSONResponse({"status": "already_connected", "address": multiaddr})
		return JSONResponse({"error": str(e)}, status_code=502)
	except Exception as e:
		return JSONResponse({"error": f"Connect failed: {e}"}, status_code=502)


async def demo_open_channel(request: Request) -> JSONResponse:
	"""
	POST /demo/open-channel
	Open a payment channel from the demo user node to the agent.
	Body (optional): { "funding_ckb": 200 }
	"""
	try:
		body = await request.json()
	except Exception:
		body = {}

	funding_ckb = body.get("funding_ckb", 200)
	funding_hex = hex(int(funding_ckb * SHANNON_PER_CKB))

	if not AGENT_PEER_ID:
		return JSONResponse({"error": "Agent peer ID not configured."}, status_code=500)

	try:
		result = await _user_rpc("open_channel", [{
			"peer_id": AGENT_PEER_ID,
			"funding_amount": funding_hex,
		}])
		return JSONResponse({
			"status": "channel_opening",
			"temporary_channel_id": result.get("temporary_channel_id", ""),
			"funding_ckb": funding_ckb,
		})
	except RuntimeError as e:
		return JSONResponse({"error": str(e)}, status_code=502)
	except Exception as e:
		return JSONResponse({"error": f"Open channel failed: {e}"}, status_code=502)


async def demo_pay(request: Request) -> JSONResponse:
	"""
	POST /demo/pay
	Simulate user paying a Fiber invoice from the local demo user node.
	For hackathon demo only — in production the user pays from their own node.

	Body: { "invoice": "fibt1..." }
	"""
	try:
		body = await request.json()
	except Exception:
		return JSONResponse({"error": "Invalid JSON body."}, status_code=400)

	invoice = body.get("invoice", "").strip()
	if not invoice:
		return JSONResponse({"error": "invoice is required."}, status_code=400)

	try:
		async with httpx.AsyncClient(timeout=15.0) as client:
			resp = await client.post(
				USER_FIBER_RPC,
				json={
					"id": "demo-pay",
					"jsonrpc": "2.0",
					"method": "send_payment",
					"params": [{"invoice": invoice}],
				},
				headers={"Content-Type": "application/json"},
			)
			result = resp.json()

		if "error" in result and result["error"] is not None:
			return JSONResponse({"error": f"Payment failed: {result['error']}"}, status_code=502)

		return JSONResponse({
			"status": "payment_sent",
			"payment_hash": result["result"]["payment_hash"],
		})
	except Exception as e:
		return JSONResponse({"error": f"Payment error: {e}"}, status_code=502)


async def demo_preimage(request: Request) -> JSONResponse:
	"""
	GET /demo/preimage/{payment_hash}
	Demo-only: return the proof_hash (TLC preimage) for a settled payment.

	In production, the user's Fiber node learns the preimage automatically
	when the TLC settles. This endpoint simulates that for the web demo.
	"""
	payment_hash = request.path_params["payment_hash"]
	pending = pending_requests.get(payment_hash)
	if not pending:
		return JSONResponse({"error": "Not found."}, status_code=404)
	if pending["status"] != "settled":
		return JSONResponse({"error": "Payment not yet settled.", "status": pending["status"]}, status_code=409)
	return JSONResponse({"proof_hash": pending["proof_hash"]})


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

STATIC_DIR = Path(__file__).parent / "static"


async def index(request: Request) -> HTMLResponse:
	"""Serve the frontend dashboard."""
	return HTMLResponse((STATIC_DIR / "index.html").read_text())


app = Starlette(
	routes=[
		Route("/", index, methods=["GET"]),
		Route("/health", health, methods=["GET"]),
		Route("/request", request_analysis, methods=["POST"]),
		Route("/analyze", analyze_endpoint, methods=["POST"]),
		Route("/agent/info", agent_info, methods=["GET"]),
		Route("/demo/node-status", demo_node_status, methods=["GET"]),
		Route("/demo/connect", demo_connect, methods=["POST"]),
		Route("/demo/open-channel", demo_open_channel, methods=["POST"]),
		Route("/demo/pay", demo_pay, methods=["POST"]),
		Route("/demo/preimage/{payment_hash}", demo_preimage, methods=["GET"]),
		Route("/result/{request_id}", get_result, methods=["GET"]),
		Route("/proof/{proof_hash}", get_proof, methods=["GET"]),
		Mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static"),
	],
)

if __name__ == "__main__":
	import uvicorn
	port = int(os.environ.get("PORT", "8080"))
	print(f"Starting Alpha Oracle on port {port}...")
	uvicorn.run(app, host="0.0.0.0", port=port)
