from .openai import OpenAIModel
from .anthropic import AnthropicModel
from .gemini import GeminiModel
from .deepseek import DeepSeekModel
from .deepinfra import DeepInfraModel
from .dashscope import DashScopeModel
from .zai import ZAIModel

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
    "claude-opus-4-6":        lambda: AnthropicModel("claude-opus-4-6"),
    "claude-sonnet-4-6":      lambda: AnthropicModel("claude-sonnet-4-6"),
    "claude-sonnet-4-6-xt8k": lambda: AnthropicModel("claude-sonnet-4-6", thinking_budget=8000),  # extended thinking high
    "claude-haiku-4-5":       lambda: AnthropicModel("claude-haiku-4-5-20251001"),

    # ── Google ────────────────────────────────────────────────────────────────
    # Gemini 3.1 series — latest (preview, production-usable)
    "gemini-3.1-pro":        lambda: GeminiModel("gemini-3.1-pro-preview"),
    "gemini-3.1-flash":         lambda: GeminiModel("gemini-3-flash-preview"),                       # always thinks (default = HIGH)
    "gemini-3.1-flash-minimal": lambda: GeminiModel("gemini-3-flash-preview", thinking_level="minimal"),  # minimal thinking budget
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
    # Qwen3 Max Thinking — Alibaba's frontier reasoning model (always-on thinking)
    "qwen3-max-thinking": lambda: DeepInfraModel("Qwen/Qwen3-Max-Thinking", thinking_native=True),
    # DeepSeek-R1 open-weight via DeepInfra
    "deepseek-r1":       lambda: DeepInfraModel("deepseek-ai/DeepSeek-R1"),

    # ── ZhipuAI / Z.AI (OpenAI-compatible) ───────────────────────────────────
    # GLM-5 — 744B MoE, 40B active; flagship (Feb 2026). Requires ZAI_API_KEY.
    "glm-5":          lambda: ZAIModel("glm-5"),                   # standard, thinking off
    "glm-5-thinking": lambda: ZAIModel("glm-5", thinking=True),   # optional thinking on
    # GLM-Z1-Flash — free-tier Z1 reasoning model, always-on thinking (cannot disable)
    # (Z1-Air requires separate API credits; Z1-Flash covered by GLM Coding Plan)
    "glm-z1-flash":   lambda: ZAIModel("glm-z1-flash"),

    # ── Alibaba DashScope (OpenAI-compatible) ─────────────────────────────────
    # Qwen3.5-Omni — native audio + image + text multimodal (Mar 2026)
    # Requires DASHSCOPE_API_KEY. Free during preview (90-day quota after activation).
    # Audio: 7 tokens/sec; endpoint: dashscope-intl.aliyuncs.com
    "qwen3.5-omni-plus":  lambda: DashScopeModel("qwen3.5-omni-plus"),   # 30B-A3B MoE, highest quality
    "qwen3.5-omni-flash": lambda: DashScopeModel("qwen3.5-omni-flash"),  # lower latency
}
