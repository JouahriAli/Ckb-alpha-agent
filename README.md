# CKB Alpha Agent — Verifiable Alpha Oracle

An autonomous AI agent that monitors token momentum via statistical analysis, accepts micropayments over the **Fiber Network (CKB L2)**, and delivers results trustlessly via **TLC preimage exchange**.

**Analyze → Pay → Prove.** No trust required.

## Demo

[![Watch the Demo](https://img.shields.io/badge/Watch_Demo-Streamable-blue?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik04IDV2MTRsMTEtN3oiLz48L3N2Zz4=)](https://streamable.com/jq8wof)

> Fiber node setup → Token analysis → TLC payment → Preimage reveal → Verifiable proof

## How It Works (Trustless Flow)

1. **Agent runs analysis first** — fetches trades, computes statistical moments, generates `proof_hash`
2. **Invoice created** — `proof_hash` becomes the TLC preimage in the Fiber invoice
3. **User pays** — TLC settles on CKB Fiber Network, revealing `proof_hash` to the user's node
4. **User fetches results** — `GET /proof/{proof_hash}` returns the full analysis

The agent can't collect payment without revealing the proof. The user can't get the proof without paying. **Atomic exchange.**

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER                                    │
│  (Fiber node + CLI/web interface)                               │
└──────────┬──────────────────────────────────────────────────────┘
           │ 1. Request (token address via API)
           │ 2. Receive Fiber invoice
           │ 3. Pay invoice
           │ 6. Receive result summary + proof hash + CKB cell
           │
┌──────────▼──────────────────────────────────────────────────────┐
│                      AGENT SERVER                                │
│                                                                  │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────────────┐  │
│  │ Fiber RPC   │   │ Agent Core   │   │ CKB L1 Publisher     │  │
│  │ Client      │   │ (Python)     │   │                      │  │
│  │             │   │              │   │ - Build Result Cell   │  │
│  │ - Invoice   │   │ - Orchestr.  │   │   (109-byte blob)    │  │
│  │ - Payment   │   │ - Dispatch   │   │ - Sign & Submit TX   │  │
│  │   verify    │   │              │   │                      │  │
│  └─────────────┘   └──────┬───────┘   └──────────────────────┘  │
│                           │                                      │
│                    ┌──────▼───────┐                               │
│                    │   Tools      │                               │
│                    │              │                               │
│                    │ Data Fetcher │  → Birdeye / DEXScreener     │
│                    │ Alpha Brain  │  → Statistical pipeline       │
│                    │ LLM Analyst  │  → Claude interpretation      │
│                    └──────────────┘                               │
└──────────────────────────────────────────────────────────────────┘
```

## Quickstart

### 1. Install

```bash
pip install -e .
```

### 2. Set environment variables

```bash
export BIRDEYE_API_KEY="your-birdeye-api-key"

# LLM backend — pick one:
# Option A: Anthropic (default)
export LLM_BACKEND="anthropic"
export ANTHROPIC_API_KEY="your-key"
# export ANTHROPIC_BASE_URL="https://share-ai.ckbdev.com"  # optional proxy

# Option B: OpenAI or any OpenAI-compatible endpoint
export LLM_BACKEND="openai"
export OPENAI_API_KEY="your-key"
# export OPENAI_BASE_URL="https://api.openai.com/v1"       # default
# export LLM_MODEL="gpt-4o-mini"                           # default

# Option C: Ollama (local, no API key needed)
export LLM_BACKEND="ollama"
# export OLLAMA_BASE_URL="http://localhost:11434"           # default
# export LLM_MODEL="llama3.2"                              # default

# Fiber (optional, for payment flow)
export FIBER_RPC_URL="http://localhost:8227"
```

### 3. Run the server

```bash
python proof_server.py
# → Starting Alpha Oracle on port 8080...
```

### 4. Analyze a token (demo endpoint, skips payment)

```bash
curl -X POST http://localhost:8080/analyze \
  -H 'Content-Type: application/json' \
  -d '{"token_address": "So11111111111111111111111111111111111111112", "chain": "solana"}'
```

### 5. Verify a proof

```bash
# Use the proof_hash from the response:
curl http://localhost:8080/proof/{proof_hash}
```

### Example response (abbreviated)

```json
{
  "token_id": "solana:So111...",
  "pair": "SOL/USDC",
  "confidence_tier": 1,
  "confidence_label": "High",
  "sample_n": 247,
  "stats": {
    "volume_zscore": 3.1415,
    "excess_kurtosis": 5.221,
    "skewness": 1.87,
    "signal_momentum": 2.4102,
    "liquidity_thinness": 0.0000034,
    "liquidity_usd": 12500000.0
  },
  "narrative": "Strong accumulation signal...",
  "proof_hash": "a1b2c3...",
  "proof_url": "/proof/a1b2c3...",
  "ckb_result_cell": {
    "data_hex": "0x...",
    "data_hash": "d4e5f6...",
    "data_size_bytes": 109,
    "capacity_ckb": 200
  }
}
```

## The Math: Signal-to-Noise Momentum (S_m)

The core metric uses a log-additive (geometric) structure — all factors must be elevated for a high score:

```
log(S_m) = 0.4·log(|z|) + 0.3·log(1 + |κ_eff|) + 0.15·log(1 + |γ_eff|)
```

Where:
- **z** — Volume Z-score (anomaly detection vs. 7-day hourly baseline)
- **κ_eff** — Excess kurtosis × Jarque-Bera weight (heavy tails = whale activity)
- **γ_eff** — Skewness × Jarque-Bera weight (directional bias)

The Jarque-Bera p-value acts as a continuous damper: when the distribution is normal (p → 1), higher moments shrink gracefully to zero. No information is discarded.

Liquidity thinness (`median(trade_size) / pool_TVL`) is reported alongside S_m, not baked in — a whale accumulating in a deep pool is *more* significant, not less.

## Payment Flow (Fiber Network)

For the full paid flow (not just the demo shortcut):

1. `POST /request` → Agent creates a Fiber invoice, returns it
2. User pays the invoice via their Fiber node
3. Agent detects payment → runs pipeline → publishes CKB Result Cell
4. `GET /result/{payment_hash}` → Poll for results

## CKB L1 Result Cell

The agent builds a compact 109-byte binary blob per the [spec](SPEC.md):

| Offset | Size | Field |
|--------|------|-------|
| 0 | 32 | token_id_hash (blake2b) |
| 32 | 8 | timestamp (u64 LE) |
| 40 | 4 | sample_n (u32 LE) |
| 44 | 8 | volume_zscore (f64 LE) |
| 52 | 8 | excess_kurtosis (f64 LE) |
| 60 | 8 | skewness (f64 LE) |
| 68 | 1 | confidence_tier (u8) |
| 69 | 8 | signal_momentum (f64 LE) |
| 77 | 32 | full_proof_hash (blake2b) |

Verifiable: `blake2b(proof_blob_json) == full_proof_hash` on-chain.

## Full Spec

See [SPEC.md](SPEC.md) for complete technical details.
