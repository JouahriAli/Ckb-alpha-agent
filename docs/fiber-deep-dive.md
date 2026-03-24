# Fiber Network — Technical Deep Dive

## 1. The Problem

CKB L1 processes one block every ~12 seconds. Each transaction costs fees and occupies on-chain capacity. For micropayments (e.g., paying $0.08 for an analysis), this is unacceptable: the fee and latency overhead exceed the value of the transaction itself.

**Fiber's solution:** Move the bulk of transactions off-chain, only touching L1 for opening and closing payment channels.

---

## 2. Payment Channels — The Core Primitive

### 2.1 What a Channel Is

A payment channel is a **two-party contract** where both parties lock funds into a shared on-chain cell (the "funding cell"). Once locked, they can exchange signed state updates off-chain — no blockchain interaction needed.

The on-chain cell acts as an escrow. The off-chain state updates are a sequence of signed agreements about how to split that escrow. Only the final split is ever published on-chain.

### 2.2 Opening a Channel

Opening requires **one L1 transaction** (the "funding transaction"):

```
Alice:  500 CKB  ─┐
                   ├──►  Funding Cell on CKB L1
Bob:      0 CKB  ─┘     (locked under a 2-of-2 multisig)

Initial state: Alice = 500 CKB, Bob = 0 CKB
```

The funding cell is locked by a **2-of-2 Schnorr multisig** — neither party can spend it unilaterally. Both must sign to update or close the channel.

**Protocol messages involved:**
1. `OpenChannel` — initiator proposes parameters (funding amount, fee rates, TLC limits).
2. `AcceptChannel` — responder agrees (can add their own funding, or fund 0).
3. Both collaborate to construct the funding transaction (`TxUpdate`, `TxComplete`).
4. `CommitmentSigned` — each signs the first commitment transaction.
5. `TxSignatures` — both sign the funding transaction and broadcast it.
6. `ChannelReady` — sent by each party once the funding tx is confirmed on L1.

At this point, the channel state is `CHANNEL_READY` and payments can flow.

### 2.3 Off-Chain State Updates

Each payment is a new **commitment transaction** — a pre-signed transaction that *could* be broadcast to close the channel at the current balances. It is never broadcast unless one party wants to close.

```
State 0: Alice = 500, Bob =   0   (opening)
State 1: Alice = 490, Bob =  10   (Alice pays Bob 10)
State 2: Alice = 485, Bob =  15   (Alice pays Bob 5)
State 3: Alice = 495, Bob =   5   (Bob pays Alice 10)
...
```

Each new state **invalidates** the previous one. This is enforced via a **revocation mechanism**: when Alice signs state N+1, she reveals a secret (`per_commitment_secret`) that lets Bob penalize her if she ever tries to broadcast state N. The penalty: Bob takes *all* the funds.

**Messages per state update:**
1. `AddTlc` — propose a new payment (with hash lock + time lock).
2. `CommitmentSigned` — sign the new commitment incorporating the TLC.
3. `RevokeAndAck` — reveal the previous state's revocation secret, acknowledging the new state.

### 2.4 Closing a Channel

**Cooperative close** (the normal case):
1. Either party sends `Shutdown` (with their desired payout script and fee rate).
2. Both wait for all pending TLCs (in-flight payments) to resolve.
3. Exchange `ClosingSigned` — partial Schnorr signatures on the closing transaction.
4. Broadcast the closing transaction to L1.

**Result:** One L1 transaction that sends the final balances to each party's address.

**Force close** (uncooperative): If the counterparty is unresponsive, you broadcast your latest commitment transaction. There's a `to_self_delay` (a locktime in CKB epochs) before you can claim your funds — this window allows the counterparty to submit a penalty if you broadcast a revoked state.

---

## 3. Hash/Time Locked Contracts (HTLCs → TLCs)

Fiber calls them **TLCs** (Time Locked Contracts) instead of HTLCs, because the protocol is designed to support both HTLC (hash-based) and future PTLC (point-based, more private) schemes.

### 3.1 The Mechanism

A TLC is a conditional payment: "Bob gets X CKB **if** he reveals a secret (the preimage of a hash) **before** a timeout."

```
Alice creates TLC:
  - Amount: 10 CKB
  - Payment hash: H = sha256(preimage)
  - Expiry: block timestamp T

Resolution:
  - If Bob reveals preimage before T → Bob gets 10 CKB (RemoveTlcFulfill)
  - If timeout T passes → Alice gets refund (RemoveTlcFail)
```

The preimage is chosen by the final recipient (Bob). Alice only knows the hash H. When Bob reveals the preimage, it cryptographically proves the payment was received.

### 3.2 Why This Enables Multi-Hop Routing

Alice and Bob don't need a direct channel. If channels exist along a path (Alice ↔ Charlie ↔ Bob), the payment can route through intermediaries using **chained TLCs with the same payment hash**.

```
Step 1: Bob generates preimage P, computes H = sha256(P).
Step 2: Bob creates invoice containing H.
Step 3: Alice receives invoice.

Step 4: Alice → Charlie:  TLC(10 CKB, hash=H, expiry=T+Δ)
Step 5: Charlie → Bob:    TLC(10 CKB, hash=H, expiry=T)

Step 6: Bob reveals P to Charlie → Charlie's TLC resolves.
Step 7: Charlie reveals P to Alice → Alice's TLC resolves.
```

**Key properties:**
- **Atomicity:** Either all hops complete (Bob reveals P) or none do (timeout). No intermediate state where Charlie loses money.
- **Decreasing timelocks:** Each hop has a shorter expiry, ensuring the preimage propagates backwards before any timeout fires.
- **Privacy:** Charlie knows he's routing a payment but doesn't know Alice is the sender or Bob is the receiver (onion routing hides this).

### 3.3 TLC vs. HTLC vs. PTLC

| Property | HTLC (Lightning) | TLC/PTLC (Fiber, future) |
|---|---|---|
| Lock type | Hash preimage | Elliptic curve point (adaptor signature) |
| Privacy | Same hash across hops → linkable | Different points per hop → unlinkable |
| Wormhole attack | Vulnerable | Resistant |
| Implementation | Mature | In progress (Fiber currently uses hash-based TLCs) |

---

## 4. Invoice Protocol

### 4.1 Structure

A Fiber invoice encodes a payment request. It is **not compatible** with Lightning's BOLT 11 format, though the interface is similar.

```
invoice = human_readable_prefix + bech32m(compress(molecule_encode(data))) + signature
```

**Encoding steps:**
1. Data fields serialized with **Molecule** (CKB's deterministic binary serialization).
2. Compressed with **arcode-rs** (arithmetic coding, ~50% size reduction).
3. Encoded as **bech32m** (not bech32 — note the 'm').
4. Optional secp256k1 signature appended.

**Human-readable prefixes:**
- `fibb` — CKB mainnet
- `fibt` — CKB testnet
- `fibd` — CKB devnet

### 4.2 Data Fields

| Field | Size | Required | Description |
|---|---|---|---|
| `timestamp` | 128 bits | Yes | Milliseconds since Unix epoch. |
| `payment_hash` | 256 bits | Yes | `sha256(preimage)` — unique identifier. |
| `expiry` | 32 bits | No | Seconds until invoice expires. |
| `description` | variable | No | UTF-8 payment description. |
| `final_htlc_timeout` | 32 bits | No | TLC timeout for last hop (seconds). |
| `fallback` | variable | No | CKB L1 address for on-chain fallback. |
| `feature` | 32 bits | No | Feature flags. |
| `payee_public_key` | 33 bytes | No | Payee's compressed public key. |
| `udt_script` | variable | No | Type script for UDT (stablecoin) payments. |
| `hash_algorithm` | 1 byte | No | 0 = ckb_hash, 1 = sha256. |

**Signature:** `message_hash = SHA256(human_readable_bytes || data_bytes)`, signed with secp256k1. This proves the invoice was created by the payee.

### 4.3 Amount Unit

All amounts are in **shannon**. 1 CKB = 10^8 shannon.

```
 10 CKB = 0x3B9ACA00     (  1,000,000,000 shannon)
100 CKB = 0x2540BE400     ( 10,000,000,000 shannon)
500 CKB = 0xBA43B7400     ( 50,000,000,000 shannon)
```

---

## 5. Routing and Pathfinding

### 5.1 Network Graph

Each Fiber node maintains a **gossip-updated network graph** — a directed graph where:
- **Nodes** = Fiber nodes (identified by public key).
- **Edges** = Payment channels (bidirectional, with per-direction capacity and fee info).

Gossip messages:
- `NodeAnnouncement` — node joins or updates identity/features.
- `ChannelAnnouncement` — new channel published (requires both parties' signatures).
- `ChannelUpdate` — fee rate or capacity change for one direction of a channel.

**Important:** Actual balance partitions within a channel are **private**. The graph only knows total channel capacity, not how much is on each side. This means routing is probabilistic — a path that *looks* feasible might fail because an intermediate channel doesn't have enough balance on the right side.

### 5.2 Pathfinding Algorithm

Fiber uses a **modified Dijkstra expanding backward from target to source**. The cost function combines:
- **Fee:** base fee + proportional fee (millionths of amount).
- **TLC lock time:** longer locks are penalized (capital is locked longer).
- **Success probability:** estimated from historical payment data (`eval_probability` module).

```
distance = fee_penalty + locktime_penalty + probability_penalty
```

The algorithm finds the cheapest route that is likely to succeed. If a route fails (insufficient balance at some hop), the sender retries with an alternative path.

### 5.3 Onion Routing (Privacy)

Each hop in the route only sees:
- The previous hop.
- The next hop.
- The TLC amount and expiry for its leg.

The sender constructs an **onion packet** (using `fiber-sphinx`, analogous to Lightning's Sphinx) where each hop decrypts one layer, revealing only routing instructions for the next hop. The sender's identity and total route length are hidden from intermediaries.

---

## 6. Security Model

### 6.1 What is Secure

- **Funds cannot be stolen** as long as at least one party monitors the chain. The 2-of-2 multisig ensures no unilateral spending.
- **Revoked states are penalized.** Broadcasting an old commitment transaction allows the counterparty to take all funds (they have the revocation secret).
- **Atomic multi-hop payments.** The TLC hash-lock chain guarantees either all hops resolve or none do.

### 6.2 What Requires Trust / Assumptions

- **Liveness:** You must be online (or have a watchtower) to detect and punish fraud. If your counterparty broadcasts a revoked state and you're offline for the entire `to_self_delay` period, you lose funds.
- **Watchtowers:** Third-party services that monitor the chain on your behalf. Fiber supports watchtowers. They receive revocation data and can submit penalty transactions if fraud is detected.
- **Routing privacy:** Intermediary nodes see payment amounts (per hop) and can infer timing patterns. Full sender/receiver privacy relies on onion routing, which is weaker than e.g., Tor (because the payment amount constrains the set of possible endpoints).

### 6.3 Known Attack Vectors

| Attack | Description | Mitigation |
|---|---|---|
| Bribery | Pay a miner to censor the penalty transaction | Time-sensitive monitoring, multiple watchtowers |
| Flood & Loot | Flood channels with small TLCs, then force-close during congestion | `max_tlc_number_in_flight` and `max_tlc_value_in_flight` limits |
| Wormhole | Intermediary skips hops in the HTLC chain to steal routing fees | PTLC (future) eliminates this by using per-hop adaptor signatures |
| Channel Griefing | Open channel, never cooperate on closing | Force close with `to_self_delay` |

---

## 7. Fiber vs. Lightning Network

| Dimension | Lightning (Bitcoin) | Fiber (CKB) |
|---|---|---|
| Base layer | Bitcoin | Nervos CKB |
| Lock mechanism | HTLC (hash preimage) | TLC (hash now, PTLC planned) |
| Multi-asset | Bitcoin only | CKB + UDTs (stablecoins, RGB++ assets) |
| Invoice encoding | BOLT 11 (bech32) | Molecule + arcode + bech32m |
| Serialization | Protocol Buffers | Molecule (deterministic, CKB-native) |
| Cross-chain | N/A | Planned CKB ↔ Bitcoin (via Cross Hub) |
| Maturity | Production (2018+) | Testnet (v0.6.1, 2025) |

---

## 8. How Our Agent Uses Fiber

### 8.1 The Payment Flow

```
                  User Node                              Agent Node
                     │                                       │
  1. POST /request   │                                       │
     {token: "JUP"}  │                                       │
     ────────────────►                                       │
                      │                                      │
  2. Agent creates    │        new_invoice(10 CKB)           │
     invoice          │  ◄───────────────────────────────────│
                      │                                      │
  3. Return invoice   │                                      │
     to user          │                                      │
     ◄────────────────                                       │
                      │                                      │
  4. User pays via    │        send_payment(invoice)         │
     their node       │  ───────────────────────────────────►│
                      │                                      │
  5. Agent polls      │        get_invoice(payment_hash)     │
     invoice status   │        → status: "Received"          │
                      │                                      │
  6. Agent runs       │                                      │
     pipeline         │                                      │
                      │                                      │
  7. Return result    │                                      │
     + proof hash     │                                      │
     ◄────────────────                                       │
```

### 8.2 RPC Methods Used

| Method | Called By | Purpose |
|---|---|---|
| `new_invoice` | Agent node | Create payment request (amount, description, preimage, expiry). |
| `get_invoice` | Agent node | Poll invoice status: `"Open"` → `"Received"` → proceed. |
| `send_payment` | User node | Pay the invoice (routes automatically through the network). |
| `list_channels` | Agent node | Verify channel health before accepting requests. |

### 8.3 Prerequisites for Demo

1. **Two Fiber nodes running** (agent + demo user).
2. **A funded channel between them** (e.g., user funds 500 CKB toward agent).
3. Channel state = `CHANNEL_READY`.
4. Agent node's RPC accessible at configured URL (default `localhost:8227`).

---

## 9. References

- Fiber docs: https://docs.fiber.world/docs
- Fiber source: https://github.com/nervosnetwork/fiber
- Lightning Network paper: https://lightning.network/lightning-network-paper.pdf
- CKB Cell Model: https://docs.nervos.org/docs/basics/concepts/cell-model
