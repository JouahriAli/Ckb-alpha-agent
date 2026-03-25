"""
LLM Analyst — interprets statistical reports into human-readable narratives.

This is where the LLM earns its keep. The statistical pipeline outputs numbers;
this tool synthesizes them into a judgment about market behavior.

Supports multiple backends via LLM_BACKEND env var:
  - "anthropic" (default) — Claude via Anthropic API or compatible proxy
  - "openai"              — OpenAI API or any OpenAI-compatible endpoint
  - "ollama"              — Local Ollama instance (no API key needed)
"""

from __future__ import annotations

import os

from tools.alpha_brain import AlphaReport

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a quantitative analyst interpreting statistical signals from DEX trading data. You speak precisely and concisely. You never give financial advice — you state observations about market microstructure.

Your audience is a sophisticated trader who understands statistics. Do not explain what kurtosis or Z-scores are. Focus on what the numbers *mean* for this specific token right now."""

def build_analysis_prompt(report: AlphaReport) -> str:
	"""Build the user prompt from an AlphaReport."""

	lines = [
		f"Token: {report.token_id}",
		f"Sample: {report.sample_n} trades (Confidence: {'High' if report.confidence_tier == 1 else 'Moderate' if report.confidence_tier == 2 else 'Low'})",
		f"Volume Z-Score: {report.volume_zscore:+.2f} (baseline: 7d hourly estimate)",
	]

	if report.excess_kurtosis is not None:
		lines.append(f"Excess Kurtosis: {report.excess_kurtosis:.2f} (effective after JB weight: {report.kurtosis_effective:.2f})")
	if report.skewness is not None:
		lines.append(f"Skewness: {report.skewness:+.2f} (effective: {report.skewness_effective:+.2f})")
	if report.jarque_bera_pvalue is not None:
		lines.append(f"Jarque-Bera p-value: {report.jarque_bera_pvalue:.6f}")
	if report.signal_momentum is not None:
		lines.append(f"Signal Momentum (S_m): {report.signal_momentum:.4f}")
	if report.bootstrap_ci is not None:
		lines.append(f"Bootstrap 95% CI for kurtosis: [{report.bootstrap_ci[0]:.2f}, {report.bootstrap_ci[1]:.2f}]")

	lines.append(f"Liquidity Thinness (L_thin): {report.liquidity_thinness:.8f}")
	lines.append(f"Pool TVL: ${report.liquidity_usd:,.0f}")

	lines.append("")
	lines.append("Provide a 2-3 sentence interpretation. Address:")
	lines.append("1. Is this organic momentum or artificial (whale/wash)?")
	lines.append("2. What does the trade-size distribution reveal about participant behavior?")
	lines.append("3. Given the liquidity context, what is the risk profile?")

	return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------

async def _interpret_anthropic(report: AlphaReport, api_key: str) -> str:
	"""Call via Anthropic SDK (Claude or compatible proxy)."""
	import anthropic

	base_url = os.environ.get("ANTHROPIC_BASE_URL", None)
	kwargs = {"api_key": api_key}
	if base_url:
		kwargs["base_url"] = base_url

	client = anthropic.AsyncAnthropic(**kwargs)

	message = await client.messages.create(
		model=os.environ.get("LLM_MODEL", "claude-haiku-4-5-20251001"),
		max_tokens=300,
		system=SYSTEM_PROMPT,
		messages=[
			{"role": "user", "content": build_analysis_prompt(report)},
		],
	)

	return message.content[0].text


async def _interpret_openai(report: AlphaReport, api_key: str) -> str:
	"""Call via OpenAI-compatible API (OpenAI, share-ai w/ OpenAI, vLLM, etc.)."""
	import httpx

	base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
	model = os.environ.get("LLM_MODEL", "gpt-4o-mini")

	async with httpx.AsyncClient(timeout=30.0) as client:
		resp = await client.post(
			f"{base_url}/chat/completions",
			headers={
				"Authorization": f"Bearer {api_key}",
				"Content-Type": "application/json",
			},
			json={
				"model": model,
				"max_tokens": 300,
				"messages": [
					{"role": "system", "content": SYSTEM_PROMPT},
					{"role": "user", "content": build_analysis_prompt(report)},
				],
			},
		)
		resp.raise_for_status()
		return resp.json()["choices"][0]["message"]["content"]


async def _interpret_ollama(report: AlphaReport, api_key: str) -> str:
	"""Call a local Ollama instance (no API key needed)."""
	import httpx

	base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
	model = os.environ.get("LLM_MODEL", "llama3.2")

	async with httpx.AsyncClient(timeout=60.0) as client:
		resp = await client.post(
			f"{base_url}/api/chat",
			json={
				"model": model,
				"stream": False,
				"messages": [
					{"role": "system", "content": SYSTEM_PROMPT},
					{"role": "user", "content": build_analysis_prompt(report)},
				],
			},
		)
		resp.raise_for_status()
		return resp.json()["message"]["content"]


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

_BACKENDS = {
	"anthropic": _interpret_anthropic,
	"openai": _interpret_openai,
	"ollama": _interpret_ollama,
}

async def interpret(report: AlphaReport, api_key: str) -> str:
	"""
	Interpret an AlphaReport using the configured LLM backend.

	Backend is selected via LLM_BACKEND env var (default: "anthropic").
	"""
	backend = os.environ.get("LLM_BACKEND", "anthropic")
	fn = _BACKENDS.get(backend)
	if fn is None:
		raise ValueError(f"Unknown LLM_BACKEND={backend!r}. Options: {list(_BACKENDS.keys())}")
	return await fn(report, api_key)
