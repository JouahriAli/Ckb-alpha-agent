# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# CKB Alpha Agent: Claw & Order Hackathon

## 🎯 Project Mission
Build a **Verifiable Alpha Oracle**: An autonomous AI agent that monitors the Nervos CKB blockchain for "paid requests," performs high-fidelity statistical analysis on token momentum (Volume vs. Liquidity), and returns an on-chain report. 

## 🏆 Hackathon Requirements & Goals
- **Mission Brief:** Create an agent-based application/tool on CKB or its payment channels (Fiber Network).
- **Core Technology:** Must utilize Claude Code/Opus + CKB AI (MCP) + Playwright (for scraping/data).
- **Deliverables:** Project summary, technical breakdown, GitHub repo, testable version, and a demo video.
- **Timeline:** March 11th (Start) – March 25th (Submission Deadline).

### Judging Criteria (Priority for Development)
1. **Autonomy of Design:** Can the agent handle the request-payment-execution loop without human intervention?
2. **Abstraction/UX Benefits:** Does the agent make interacting with CKB easier for the user?
3. **Soundness & Robustness:** Does the statistical math (Alpha Brain) actually provide value?
4. **Viability as a Product:** Is there a clear use case for users to pay for this intelligence?

## 🛠 Project Specifics
- **Full Technical Spec:** See `SPEC.md` for architecture, component details, data flow, and file structure.
- **Primary Tool:** `agent_brain.py` (Custom FastMCP server).
- **Statistical Metric:** **Signal-to-Noise Momentum ($S_m$)**. Identifying tokens where volume velocity is high but liquidity depth is still shallow (high slippage potential = early momentum).
- **Payment Layer:** Fiber Network (CKB L2) for instant micropayments. Invoice-based flow.
- **Settlement Layer:** CKB L1 for publishing verifiable Result Cells containing compact statistical proofs.

## 🧠 Memory & Context for Claude
- **User Background:** MS in Statistics, strong algebra/probability. Focus on mathematical elegance.
- **Workflow:** Using the **CKB AI MCP** to handle RPC calls and cell management.
- **Constraint:** Avoid high-friction scraping (like raw Twitter) in the prototype; prioritize DEX data (DEXScreener API) + Playwright for targeted social checks.
- **CNS Guardrail:** If tasks become vague or unstructured, Claude must proactively break them into 5-minute executable "Sprints."