"""
CKB L1 Publisher — builds the 109-byte Result Cell binary blob.

Per SPEC.md §3.6, the Result Cell data layout is:

    Offset  Size   Field
    0       32     token_id_hash (blake2b of token identifier)
    32       8     timestamp (u64 LE, unix seconds)
    40       4     sample_n (u32 LE)
    44       8     volume_zscore (f64 LE)
    52       8     excess_kurtosis (f64 LE, 0.0 if unavailable)
    60       8     skewness (f64 LE, 0.0 if unavailable)
    68       1     confidence_tier (u8: 1, 2, or 3)
    69       8     signal_momentum (f64 LE, 0.0 if unavailable)
    77      32     full_proof_hash (blake2b of JSON proof blob)
    Total: 109 bytes

This module is a STUB for the hackathon — it constructs the binary blob
and logs what it would publish, but does not submit an actual CKB transaction
(no funded testnet wallet is configured).
"""

from __future__ import annotations

import hashlib
import logging
import struct
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ResultCell:
    """Represents a CKB Result Cell ready for on-chain publication."""
    data: bytes          # 109-byte binary blob
    data_hex: str        # hex-encoded blob
    data_hash: str       # blake2b hash of the blob
    capacity_ckb: int    # CKB capacity needed (~200 CKB)


def build_result_cell(
    token_id: str,
    timestamp: int,
    sample_n: int,
    volume_zscore: float,
    excess_kurtosis: float | None,
    skewness: float | None,
    confidence_tier: int,
    signal_momentum: float | None,
    full_proof_hash: str,
) -> ResultCell:
    """
    Build the 109-byte Result Cell binary blob per SPEC.md §3.6.

    Args:
        token_id: Token identifier string (e.g. "solana:So111...").
        timestamp: Unix timestamp in seconds.
        sample_n: Number of trades in the sample.
        volume_zscore: Volume Z-score.
        excess_kurtosis: Excess kurtosis (None → 0.0).
        skewness: Skewness (None → 0.0).
        confidence_tier: 1 (High), 2 (Moderate), or 3 (Low).
        signal_momentum: S_m value (None → 0.0).
        full_proof_hash: Hex-encoded blake2b hash of the full JSON proof blob.

    Returns:
        ResultCell with the packed binary data.
    """
    # Field 1: token_id_hash — blake2b(token_id), 32 bytes
    token_id_hash = hashlib.blake2b(token_id.encode(), digest_size=32).digest()

    # Field 2: timestamp — u64 LE, 8 bytes
    ts_bytes = struct.pack("<Q", timestamp)

    # Field 3: sample_n — u32 LE, 4 bytes
    n_bytes = struct.pack("<I", sample_n)

    # Field 4: volume_zscore — f64 LE, 8 bytes
    vz_bytes = struct.pack("<d", volume_zscore)

    # Field 5: excess_kurtosis — f64 LE, 8 bytes
    ek_bytes = struct.pack("<d", excess_kurtosis if excess_kurtosis is not None else 0.0)

    # Field 6: skewness — f64 LE, 8 bytes
    sk_bytes = struct.pack("<d", skewness if skewness is not None else 0.0)

    # Field 7: confidence_tier — u8, 1 byte
    ct_bytes = struct.pack("<B", confidence_tier)

    # Field 8: signal_momentum — f64 LE, 8 bytes
    sm_bytes = struct.pack("<d", signal_momentum if signal_momentum is not None else 0.0)

    # Field 9: full_proof_hash — 32 bytes from hex
    proof_hash_bytes = bytes.fromhex(full_proof_hash)
    assert len(proof_hash_bytes) == 32, f"proof hash must be 32 bytes, got {len(proof_hash_bytes)}"

    # Concatenate all fields
    blob = token_id_hash + ts_bytes + n_bytes + vz_bytes + ek_bytes + sk_bytes + ct_bytes + sm_bytes + proof_hash_bytes
    assert len(blob) == 109, f"Result Cell blob must be 109 bytes, got {len(blob)}"

    data_hash = hashlib.blake2b(blob, digest_size=32).hexdigest()

    return ResultCell(
        data=blob,
        data_hex=blob.hex(),
        data_hash=data_hash,
        capacity_ckb=200,
    )


def publish_result_cell(cell: ResultCell) -> dict:
    """
    Publish the Result Cell to CKB L1.

    STUB: Logs the would-be transaction details. In production, this would:
    1. Find live cells owned by the agent for capacity.
    2. Build a transaction with Output 0 = Result Cell, Output 1 = change.
    3. Sign with the agent's secp256k1 key.
    4. Submit via CKB RPC.

    Returns:
        Dict with stub transaction info.
    """
    tx_info = {
        "status": "stub",
        "message": "Result Cell built but not submitted (no funded testnet wallet).",
        "result_cell": {
            "data_hex": cell.data_hex,
            "data_hash": cell.data_hash,
            "data_size_bytes": len(cell.data),
            "capacity_ckb": cell.capacity_ckb,
        },
        "would_be_tx": {
            "outputs": [
                {
                    "capacity": f"{cell.capacity_ckb} CKB",
                    "lock": "agent secp256k1 lock (not configured)",
                    "type": None,
                    "data": f"0x{cell.data_hex}",
                },
            ],
            "note": "Output 1 (change cell) omitted from stub.",
        },
    }

    logger.info(
        "Result Cell ready for CKB L1: hash=%s size=%d bytes capacity=%d CKB",
        cell.data_hash,
        len(cell.data),
        cell.capacity_ckb,
    )
    logger.info("Data hex: 0x%s", cell.data_hex)

    return tx_info
