"""
Fiber RPC Client — wraps Fiber Network JSON-RPC for payment channel operations.

The agent uses this to:
	1. Create invoices for analysis requests (new_invoice).
	2. Poll for payment confirmation (get_invoice → status == "Received").
	3. Check channel health (list_channels).

All amounts are in hex-encoded shannon (1 CKB = 10^8 shannon = 0x5f5e100).
"""

from __future__ import annotations

import asyncio
import secrets
from dataclasses import dataclass

import httpx


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# 1 CKB = 100_000_000 shannon
SHANNON_PER_CKB = 100_000_000


def ckb_to_hex_shannon(ckb: int | float) -> str:
	"""Convert CKB amount to hex-encoded shannon string."""
	return hex(int(ckb * SHANNON_PER_CKB))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Invoice:
	"""A Fiber payment invoice."""
	invoice_address: str	# The encoded invoice string (fibt1...)
	payment_hash: str		# Hex hash — used to poll status
	amount_shannon: int
	description: str
	currency: str


@dataclass
class InvoiceStatus:
	"""Result of polling an invoice."""
	payment_hash: str
	status: str				# "Open" | "Received" | "Cancelled"
	invoice_address: str


@dataclass
class ChannelInfo:
	"""Summary of a payment channel."""
	channel_id: str
	peer_id: str
	state: str				# e.g., "CHANNEL_READY"
	local_balance: int		# shannon
	remote_balance: int		# shannon
	is_public: bool


# ---------------------------------------------------------------------------
# RPC Client
# ---------------------------------------------------------------------------

class FiberClient:
	"""Thin async wrapper around Fiber Network JSON-RPC."""

	def __init__(self, rpc_url: str = "http://localhost:8227"):
		self.rpc_url = rpc_url
		self._request_id = 0

	async def _call(self, method: str, params: list | None = None) -> dict:
		"""Make a JSON-RPC call to the Fiber node."""
		self._request_id += 1
		payload = {
			"id": str(self._request_id),
			"jsonrpc": "2.0",
			"method": method,
			"params": params or [],
		}

		async with httpx.AsyncClient(timeout=15.0) as client:
			resp = await client.post(
				self.rpc_url,
				json=payload,
				headers={"Content-Type": "application/json"},
			)
			resp.raise_for_status()
			body = resp.json()

		if "error" in body and body["error"] is not None:
			raise RuntimeError(f"Fiber RPC error ({method}): {body['error']}")

		return body.get("result")

	# -------------------------------------------------------------------
	# Invoice operations
	# -------------------------------------------------------------------

	async def create_invoice(
		self,
		amount_ckb: int | float,
		description: str = "Alpha Oracle analysis",
		currency: str = "Fibt",
		expiry_seconds: int = 3600,
		preimage_hex: str | None = None,
	) -> Invoice:
		"""
		Create a new invoice on our Fiber node.

		Args:
			amount_ckb: Amount in CKB (e.g., 10 for 10 CKB).
			description: Human-readable description.
			currency: "Fibt" (testnet), "Fibb" (mainnet), "Fibd" (devnet).
			expiry_seconds: Invoice validity period.
			preimage_hex: Optional 32-byte hex preimage. If provided, the
				invoice's payment_hash = sha256(preimage). This enables
				trustless data delivery: revealing the preimage (= the data
				hash) settles the payment atomically.

		Returns:
			Invoice object with address and payment_hash for polling.
		"""
		preimage = "0x" + (preimage_hex or secrets.token_hex(32))
		amount_hex = ckb_to_hex_shannon(amount_ckb)

		result = await self._call("new_invoice", [{
			"amount": amount_hex,
			"currency": currency,
			"description": description,
			"payment_preimage": preimage,
			"expiry": hex(expiry_seconds),
			"hash_algorithm": "sha256",
		}])

		payment_hash = result["invoice"]["data"]["payment_hash"]

		return Invoice(
			invoice_address=result["invoice_address"],
			payment_hash=payment_hash,
			amount_shannon=int(amount_ckb * SHANNON_PER_CKB),
			description=description,
			currency=currency,
		)

	async def get_invoice_status(self, payment_hash: str) -> InvoiceStatus:
		"""
		Check the status of an invoice.

		Returns:
			InvoiceStatus with status field:
				"Open"      — waiting for payment.
				"Received"  — payment received.
				"Cancelled" — expired or cancelled.
		"""
		result = await self._call("get_invoice", [{"payment_hash": payment_hash}])

		return InvoiceStatus(
			payment_hash=payment_hash,
			status=result["status"],
			invoice_address=result["invoice_address"],
		)

	async def wait_for_payment(
		self,
		payment_hash: str,
		poll_interval: float = 2.0,
		timeout: float = 3600.0,
	) -> InvoiceStatus:
		"""
		Poll until an invoice is paid or times out.

		Args:
			payment_hash: The invoice's payment hash.
			poll_interval: Seconds between polls.
			timeout: Maximum wait time in seconds.

		Returns:
			InvoiceStatus with final status.

		Raises:
			TimeoutError: If payment not received within timeout.
		"""
		import time
		start = time.time()

		while time.time() - start < timeout:
			status = await self.get_invoice_status(payment_hash)

			if status.status in ("Received", "Paid"):
				return status
			if status.status == "Cancelled":
				raise RuntimeError(
					f"Invoice {payment_hash} was cancelled."
				)

			await asyncio.sleep(poll_interval)

		raise TimeoutError(
			f"Payment not received within {timeout}s for invoice {payment_hash}."
		)

	# -------------------------------------------------------------------
	# Channel operations
	# -------------------------------------------------------------------

	async def list_channels(self, peer_id: str | None = None) -> list[ChannelInfo]:
		"""List payment channels, optionally filtered by peer."""
		params = {}
		if peer_id:
			params["peer_id"] = peer_id

		result = await self._call("list_channels", [params])

		channels = []
		for ch in result.get("channels", []):
			channels.append(ChannelInfo(
				channel_id=ch["channel_id"],
				peer_id=ch["peer_id"],
				state=ch["state"]["state_name"],
				local_balance=int(ch["local_balance"], 16),
				remote_balance=int(ch["remote_balance"], 16),
				is_public=ch.get("is_public", False),
			))
		return channels

	async def has_ready_channel(self) -> bool:
		"""Check if we have at least one CHANNEL_READY channel."""
		channels = await self.list_channels()
		return any(ch.state == "CHANNEL_READY" for ch in channels)

	# -------------------------------------------------------------------
	# Node info
	# -------------------------------------------------------------------

	async def node_info(self) -> dict:
		"""Get node identity and status."""
		return await self._call("node_info")
