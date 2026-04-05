from .openai import OpenAIModel
from .anthropic import AnthropicModel
from .gemini import GeminiModel
from .deepseek import DeepSeekModel
from .deepinfra import DeepInfraModel

REGISTRY = {
    # ── OpenAI ────────────────────────────────────────────────────────────────
    # GPT-5.4 series — current frontier (as of Apr 2026)
    "gpt-5.4":      lambda: OpenAIModel("gpt-5.4"),       # flagship
    "gpt-5.4-mini": lambda: OpenAIModel("gpt-5.4-mini"),  # strong + fast
    "gpt-5.4-nano": lambda: OpenAIModel("gpt-5.4-nano"),  # cheapest
    # o3 — still available, not deprecated; succeeded by gpt-5.4 per OpenAI docs
    "o3":           lambda: OpenAIModel("o3"),

    # ── Anthropic ─────────────────────────────────────────────────────────────
    # Claude 4.6 generation (Feb 2026) — current latest
    "claude-opus-4-6":   lambda: AnthropicModel("claude-opus-4-6"),
    "claude-sonnet-4-6": lambda: AnthropicModel("claude-sonnet-4-6"),
    "claude-haiku-4-5":  lambda: AnthropicModel("claude-haiku-4-5-20251001"),

    # ── Google ────────────────────────────────────────────────────────────────
    # Gemini 3.1 series — latest (preview, production-usable)
    "gemini-3.1-pro":        lambda: GeminiModel("gemini-3.1-pro-preview"),
    "gemini-3.1-flash":      lambda: GeminiModel("gemini-3-flash-preview"),
    "gemini-3.1-flash-lite": lambda: GeminiModel("gemini-3.1-flash-lite-preview"),
    # Gemini 2.5 series — last stable GA
    "gemini-2.5-pro":        lambda: GeminiModel("gemini-2.5-pro"),
    "gemini-2.5-flash":      lambda: GeminiModel("gemini-2.5-flash"),

    # ── DeepSeek ──────────────────────────────────────────────────────────────
    # Both point to DeepSeek-V3.2 (128K ctx); -chat = standard, -reasoner = thinking
    "deepseek-chat":     lambda: DeepSeekModel("deepseek-chat"),
    "deepseek-reasoner": lambda: DeepSeekModel("deepseek-reasoner"),

    # ── Open-weight via DeepInfra ─────────────────────────────────────────────
    # Llama 4 Maverick — Meta's latest multimodal MoE (Mar 2025)
    "llama-4-maverick":  lambda: DeepInfraModel("meta-llama/Llama-4-Maverick-17B-128E-Instruct"),
    # Qwen 3.5 — Alibaba's latest MoE flagship
    "qwen3.5-72b":       lambda: DeepInfraModel("Qwen/Qwen3.5-72B-A10B"),
    # DeepSeek-R1 open-weight via DeepInfra
    "deepseek-r1":       lambda: DeepInfraModel("deepseek-ai/DeepSeek-R1"),
}
