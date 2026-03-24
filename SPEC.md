# CKB Alpha Agent — Technical Specification

## 1. Overview

An autonomous AI agent that sells on-demand statistical analysis of token momentum. Users pay via Fiber Network micropayments, the agent runs a Signal-to-Noise statistical pipeline, an LLM interprets the results, and a verifiable proof is published on CKB L1.

**One-liner:** Pay → Analyze → Prove.

### Scope Boundaries

| In Scope (Hackathon) | Out of Scope |
|---|---|
| Single Fiber channel (agent ↔ demo user) | Multi-user channel management |
| DEXScreener API as data source | Raw blockchain DEX indexing |
| Local proof endpoint (`/proof/{hash}`) | IPFS / decentralized storage |
| Testnet deployment | Mainnet |
| Always-success Type Script for result cells | Full on-chain data validation |

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER                                    │
│  (Fiber node + CLI/web interface)                               │
└──────────┬──────────────────────────────────────────────────────┘
           │ 1. Request (token_id via Fiber message or API)
           │ 2. Receive invoice
           │ 3. Pay invoice
           │ 6. Receive result summary + L1 tx hash
           │
┌──────────▼──────────────────────────────────────────────────────┐
│                      AGENT SERVER                                │
│                                                                  │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────────────┐  │
│  │ Fiber RPC   │   │ Agent Core   │   │ CKB L1 Publisher     │  │
│  │ Client      │   │ (Python)     │   │ (CCC / ckb-cli)      │  │
│  │             │   │              │   │                      │  │
│  │ - Invoice   │   │ - Orchestr.  │   │ - Build Result Cell  │  │
│  │ - Payment   │   │ - Dispatch   │   │ - Sign & Submit TX   │  │
│  │   verify    │   │              │   │                      │  │
│  └──────┬──────┘   └──────┬───────┘   └──────────┬───────────┘  │
│         │                 │                      │               │
│         │          ┌──────▼───────┐               │               │
│         │          │  FastMCP     │               │               │
│         │          │  Tools       │               │               │
│         │          │              │               │               │
│         │          │ ┌──────────┐ │               │               │
│         │          │ │ Data     │ │               │               │
│         │          │ │ Fetcher  │ │               │               │
│         │          │ └──────────┘ │               │               │
│         │          │ ┌──────────┐ │               │               │
│         │          │ │ Alpha    │ │               │               │
│         │          │ │ Brain    │ │               │               │
│         │          │ │ (Stats)  │ │               │               │
│         │          │ └──────────┘ │               │               │
│         │          │ ┌──────────┐ │               │               │
│         │          │ │ LLM      │ │               │               │
│         │          │ │ Analyst  │ │               │               │
│         │          │ └──────────┘ │               │               │
│         │          └──────────────┘               │               │
│         │                                        │               │
│  ┌──────▼────────────────────────────────────────▼───────────┐  │
│  │                    Local State (SQLite)                    │  │
│  │  - Processed request log (idempotency)                    │  │
│  │  - Full proof blobs (served at /proof/{hash})             │  │
│  └───────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Specifications

### 3.1 Agent Core (`agent_core.py`)

The main event loop. Deterministic Python — no LLM here.

```python
# Pseudocode
async def main_loop():
    while True:
        # Poll for pending requests (Fiber invoices marked as paid)
        paid_invoices = await fiber_client.list_paid_invoices()

        for invoice in paid_invoices:
            if db.already_processed(invoice.payment_hash):
                continue

            token_id = extract_token_id(invoice)

            # Dispatch async pipeline
            asyncio.create_task(process_request(token_id, invoice))

        await asyncio.sleep(12)  # CKB block time
```

**Request flow:**
1. User hits an HTTP endpoint (e.g., `POST /request`) with `{ "token_id": "..." }`.
2. Agent creates a Fiber invoice for the analysis fee → returns invoice string.
3. User pays the invoice via their Fiber node.
4. Agent polls `list_channels` or invoice status to detect payment.
5. On payment confirmation → run pipeline → publish result.

### 3.2 Data Fetcher (`tools/data_fetcher.py`)

FastMCP tool. Pulls trade-level data from DEXScreener.

**API endpoints used:**
- `GET /tokens/{token_id}` — token metadata, current price.
- `GET /orders/v1/{chain}/{pair}` — recent trades (the atomic unit for our stats).

**Output:** A structured list of trades:
```python
@dataclass
class Trade:
    timestamp: int      # unix ms
    price: float
    volume_usd: float
    side: str           # "buy" | "sell"
    tx_hash: str
```

**Sample size gate:** Returns trade count alongside data. The Alpha Brain decides whether to proceed based on tier thresholds.

### 3.3 Alpha Brain (`tools/alpha_brain.py`)

FastMCP tool. Pure statistics — no LLM, no network calls.

**Input:** List of `Trade` objects + lookback window config.

**Computations:**

#### Volume Z-Score (Anomaly Detection)
```
μ_v = mean(volumes over baseline window)
σ_v = std(volumes over baseline window)
z = (current_window_volume - μ_v) / σ_v
```
- Baseline window: 7 days of hourly volume.
- Current window: last 1 hour.
- Flag if |z| > 2.0.

#### Excess Kurtosis (Distribution Shape)
```
κ = (1/n) Σ((x_i - μ) / σ)^4 - 3
```
- Computed on individual trade sizes (volume per trade).
- κ > 3 → heavy tails → whale activity or wash trading.
- κ ≈ 0 → normal-ish distribution → organic.
- Only computed at Tier 1/2 sample sizes.

#### Skewness (Directional Bias)
```
γ = (1/n) Σ((x_i - μ) / σ)^3
```
- Positive skew → a few very large buy trades (accumulation).
- Negative skew → a few very large sell trades (distribution).

#### Jarque-Bera Normality Check (Continuous Weight)
```
JB = (n/6) × (γ² + (κ²/4))
w_JB = 1 - p_JB
```
Rather than a hard gate, the JB p-value acts as a continuous damper:
- `κ_eff = κ × w_JB` and `γ_eff = γ × w_JB`.
- When p_JB → 1 (can't reject normality), higher moments shrink to zero gracefully.
- When p_JB → 0 (strong non-normality), full weight. No information discarded.

#### Liquidity Thinness (Risk Context — NOT part of S_m)
```
L_thin = median(v_i) / liquidity_usd
```
Where `liquidity_usd` = total pool TVL from DEXScreener.

**Key design decision:** L_thin is reported *alongside* S_m, not multiplied into it.
A whale accumulating in a deep pool is *more* significant, not less — they're deploying
serious capital. L_thin tells the user about slippage risk, not signal strength.

| L_thin | Interpretation |
|---|---|
| High | Thin liquidity — high slippage, volatile entry. |
| Low | Deep liquidity — low slippage, favorable conditions. |

#### Signal-to-Noise Momentum (S_m)
The composite metric uses a log-additive (geometric) structure so all factors must
be elevated for a high score, while preventing scale explosion:
```
log(S_m) = 0.4·log(|z|) + 0.3·log(1 + |κ_eff|) + 0.15·log(1 + |γ_eff|)

S_m = exp(log(S_m))
Direction signal: sign(z), sign(κ_eff), sign(γ_eff) reported separately.
```
Weights are tunable. Starting point: volume anomaly dominates (0.4), distribution
shape is secondary (0.3 + 0.15), reflecting that volume is the gatekeeper.

**Interpretation matrix (what the LLM synthesizes):**

| Volume Z | Kurtosis | Skewness | Reading |
|---|---|---|---|
| High (+) | High (+) | Strong positive | Whale accumulation — few very large buys |
| High (+) | High (+) | Strong negative | Whale distribution — few very large sells |
| High (+) | High (+) | ~0 | Large trades both directions — OTC or arbitrage |
| High (+) | ~0 | Any | Organic crowd, no dominant actor |
| High (+) | Negative | ~0 | Suspiciously uniform trades — wash trading signal |
| Low | Any | Any | No significant activity |

**Output:**
```python
@dataclass
class AlphaReport:
    token_id: str
    timestamp: int
    sample_n: int
    confidence_tier: int            # 1, 2, or 3
    volume_zscore: float
    excess_kurtosis: float | None   # None if tier 3
    kurtosis_effective: float | None
    skewness: float | None          # None if tier 3
    skewness_effective: float | None
    jarque_bera_pvalue: float | None
    signal_momentum: float | None   # None if tier 3
    liquidity_thinness: float       # Risk context, always computed
    liquidity_usd: float            # Raw pool TVL
    trade_data: list[Trade]         # Full dataset for proof
    bootstrap_ci: tuple | None      # 95% CI for kurtosis (tier 2)
```

**Confidence tiers:**
| Tier | Sample Size | Stats Computed | Confidence Label |
|------|------------|----------------|-----------------|
| 1 | n ≥ 500 | All | High |
| 2 | 100 ≤ n < 500 | All + bootstrap CI | Moderate |
| 3 | n < 100 | Z-score only | Low (insufficient data) |

### 3.4 LLM Analyst (`tools/llm_analyst.py`)

FastMCP tool. This is where the LLM earns its keep.

**Input:** `AlphaReport` dataclass.

**Prompt template:**
```
You are a quantitative analyst. Given the following statistical report
for token {token_id}, provide a 2-3 sentence interpretation:

- Volume Z-Score: {z} (baseline: 7d hourly)
- Excess Kurtosis: {κ} (trade-size distribution)
- Skewness: {γ}
- Signal Momentum (S_m): {s_m}
- Sample size: {n} trades (Confidence: {tier})
- Jarque-Bera p-value: {jb_p}

Focus on: Is this organic momentum or artificial? What does the
trade-size distribution suggest about participant behavior?
Do NOT give financial advice. State observations only.
```

**Output:** A `str` — the human-readable interpretation appended to the on-chain summary.

### 3.5 Fiber RPC Client (`fiber_client.py`)

Wraps Fiber node JSON-RPC. Thin HTTP client.

**Methods used:**
| RPC Method | Purpose |
|---|---|
| `new_invoice` | Create payment request for analysis fee |
| `send_payment` | (Not needed for agent — agent receives, not sends) |
| `list_channels` | Monitor channel state and balances |
| `parse_invoice` | Decode incoming invoice for verification |

**Invoice parameters:**
- `amount`: Configurable fee (e.g., 10 CKB per analysis).
- `currency`: `"Fibt"` (testnet) or `"Fibb"` (mainnet).
- `expiry`: 3600 seconds (1 hour).
- `description`: `"Alpha analysis: {token_id}"`.
- `payment_preimage`: Random 32 bytes (generated per request).

**Payment verification:**
Poll invoice status. When the preimage is revealed (payment complete), the agent proceeds.

### 3.6 CKB L1 Publisher (`ckb_publisher.py`)

Constructs and submits the Result Cell transaction.

**Result Cell layout (data field, ~120 bytes):**
```
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
```
Total: **109 bytes** of data.

**Cell structure:**
- **Capacity:** ~200 CKB (covers cell size overhead).
- **Lock Script:** Agent's secp256k1 lock (agent can reclaim capacity later).
- **Type Script:** None for hackathon (always-success not needed if no type).
- **Data:** Binary blob above.

**Transaction construction:**
- Use CKB RPC via the CKB AI MCP tools or `ckb-cli`.
- Input: Agent's live cells (capacity source).
- Output 0: Result Cell.
- Output 1: Change cell back to agent.

### 3.7 Proof Server (`proof_server.py`)

Minimal HTTP endpoint serving full proof blobs.

- `GET /proof/{blake2b_hash}` → Returns JSON blob.
- Proof blob contains: full trade dataset, all computed statistics, bootstrap samples, LLM interpretation, metadata.
- Verification: `blake2b(response_body) == on_chain_full_proof_hash`.

Runs as part of the agent's HTTP server (same process as the request endpoint).

### 3.8 Local State (SQLite)

**Tables:**

```sql
CREATE TABLE processed_requests (
    payment_hash TEXT PRIMARY KEY,
    token_id TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    result_tx_hash TEXT,
    proof_hash TEXT,
    status TEXT NOT NULL  -- 'processing' | 'completed' | 'failed'
);

CREATE TABLE proof_blobs (
    proof_hash TEXT PRIMARY KEY,
    blob JSON NOT NULL,
    created_at INTEGER NOT NULL
);
```

---

## 4. Request Lifecycle (End-to-End)

```
 Time ──►

 User                    Agent HTTP         Agent Core        Fiber       Alpha Brain      LLM         CKB L1
  │                         │                   │               │             │             │            │
  │  POST /request          │                   │               │             │             │            │
  │  { token: "XXX" }      │                   │               │             │             │            │
  │────────────────────────►│                   │               │             │             │            │
  │                         │  new_invoice()    │               │             │             │            │
  │                         │──────────────────►│──────────────►│             │             │            │
  │                         │◄──────────────────│◄──────────────│             │             │            │
  │  invoice: "fibt..."     │                   │               │             │             │            │
  │◄────────────────────────│                   │               │             │             │            │
  │                         │                   │               │             │             │            │
  │  (User pays invoice     │                   │               │             │             │            │
  │   via their Fiber node) │                   │  poll status  │             │             │            │
  │                         │                   │──────────────►│             │             │            │
  │                         │                   │  paid=true    │             │             │            │
  │                         │                   │◄──────────────│             │             │            │
  │                         │                   │               │             │             │            │
  │                         │                   │  fetch_data(token)          │             │            │
  │                         │                   │──────────────────────────►  │             │            │
  │                         │                   │  trades[]                   │             │            │
  │                         │                   │◄──────────────────────────  │             │            │
  │                         │                   │                             │             │            │
  │                         │                   │  compute_alpha(trades)      │             │            │
  │                         │                   │──────────────────────────►  │             │            │
  │                         │                   │  AlphaReport                │             │            │
  │                         │                   │◄──────────────────────────  │             │            │
  │                         │                   │                             │             │            │
  │                         │                   │  interpret(report)          │             │            │
  │                         │                   │────────────────────────────────────────►  │            │
  │                         │                   │  narrative                  │             │            │
  │                         │                   │◄────────────────────────────────────────  │            │
  │                         │                   │                             │             │            │
  │                         │                   │  store proof blob (SQLite)  │             │            │
  │                         │                   │  publish_result_cell()      │             │            │
  │                         │                   │──────────────────────────────────────────────────────► │
  │                         │                   │  tx_hash                    │             │            │
  │                         │                   │◄────────────────────────────────────────────────────── │
  │                         │                   │               │             │             │            │
  │  GET /result/{pay_hash} │                   │               │             │             │            │
  │────────────────────────►│                   │               │             │             │            │
  │  { report, tx_hash,     │                   │               │             │             │            │
  │    proof_url }          │                   │               │             │             │            │
  │◄────────────────────────│                   │               │             │             │            │
```

---

## 5. File Structure

```
ckb-alpha-agent/
├── CLAUDE.md
├── SPEC.md                      # This file
├── pyproject.toml               # Python project config (uv/pip)
├── config.yml                   # Agent configuration
│
├── agent_core.py                # Main loop, orchestration
├── fiber_client.py              # Fiber RPC wrapper
├── ckb_publisher.py             # CKB L1 transaction builder
├── proof_server.py              # HTTP server (requests + proof serving)
├── db.py                        # SQLite helpers
│
├── tools/                       # FastMCP tool modules
│   ├── __init__.py
│   ├── data_fetcher.py          # DEXScreener API client
│   ├── alpha_brain.py           # Statistical computations
│   └── llm_analyst.py           # LLM interpretation
│
├── tests/
│   ├── test_alpha_brain.py      # Unit tests for stats
│   ├── test_data_fetcher.py     # Mock API response tests
│   └── test_integration.py      # End-to-end with mock Fiber
│
└── scripts/
    ├── setup_fiber_nodes.sh     # Spin up 2 local Fiber nodes
    └── fund_channel.sh          # Open & fund a test channel
```

---

## 6. Configuration (`config.yml`)

```yaml
agent:
  analysis_fee_ckb: 10            # CKB charged per analysis
  poll_interval_seconds: 12       # CKB block time
  max_concurrent_requests: 5

fiber:
  node_url: "http://localhost:8227"
  currency: "Fibt"               # Testnet
  invoice_expiry_seconds: 3600

ckb:
  rpc_url: "https://testnet.ckb.dev"
  agent_private_key_path: "./ckb/key"

data:
  dexscreener_base_url: "https://api.dexscreener.com"
  baseline_window_hours: 168      # 7 days
  current_window_hours: 1

stats:
  tier1_min_trades: 500
  tier2_min_trades: 100
  zscore_threshold: 2.0
  bootstrap_resamples: 1000

server:
  host: "0.0.0.0"
  port: 8080
```

---

## 7. Dependencies

```
python >= 3.11
fastmcp
httpx                  # Async HTTP (DEXScreener + Fiber RPC)
numpy
scipy                  # kurtosis, skewness, jarque_bera
aiosqlite              # Async SQLite
pyyaml
uvicorn                # ASGI server for proof endpoint
blake3 or hashlib      # blake2b hashing (hashlib has it built-in)
anthropic              # Claude API for LLM analyst tool
```

---

## 8. Demo Script (Hackathon Presentation)

1. Start two Fiber nodes locally (`scripts/setup_fiber_nodes.sh`).
2. Open and fund a 500 CKB channel between them.
3. Start the agent server.
4. From "user" terminal: `POST /request { "token_id": "solana:SOL" }`.
5. Agent returns invoice → user pays via Fiber CLI.
6. Agent fetches data → computes stats → LLM interprets → publishes Result Cell.
7. Show the CKB explorer with the Result Cell data.
8. Verify: `GET /proof/{hash}` → blake2b matches on-chain value.

**Total demo time: ~30 seconds end-to-end.**

---

## 9. Resolved Decisions

| # | Decision | Resolution |
|---|---|---|
| 1 | Token selection | User picks any token. Demo uses 2-3 high-volume tokens. |
| 2 | Fiber payment detection | Poll invoice status via RPC. Exact method TBD during implementation. |
| 3 | CKB transaction construction | TypeScript thin service using CCC library. Python calls it via localhost HTTP. |
| 4 | LLM model for analyst | Claude Haiku — fast, cheap, sufficient for 2-3 sentence interpretations. |
| 5 | Stack | Hybrid: Python (stats, orchestration, FastMCP) + TypeScript (CKB/Fiber via CCC). |
| 6 | L_thin in S_m? | No. L_thin is a separate risk axis reported alongside S_m, not baked into it. |
| 7 | Jarque-Bera gate | Continuous weight (1 - p_JB), not binary threshold. |
| 8 | MCP servers role | Documentation, blockchain data, dev tooling context — NOT transaction construction. |

## 10. Open Decisions

| # | Question | Current Lean |
|---|---|---|
| 1 | Fiber invoice status RPC | Need to confirm exact method for detecting paid invoices. |
| 2 | TypeScript service shape | Thin HTTP server vs. CLI scripts invoked by Python? Lean HTTP. |
| 3 | S_m weight tuning | Starting at [0.4, 0.3, 0.15]. Tune after seeing real data. |
