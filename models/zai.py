"""ZhipuAI / Z.AI GLM models via Anthropic-compatible endpoint.

Uses the same endpoint/token as the claude-glm5 shell alias:
  ANTHROPIC_BASE_URL=https://open.bigmodel.cn/api/anthropic
  ANTHROPIC_AUTH_TOKEN=<ZAI_API_KEY>

This allows GLM Coding Plan subscribers to call GLM models without
requiring separate paas/v4 credits.

Supports GLM-5, GLM-5.1, and GLM-Z1 series (text-only).

Thinking modes:
  GLM-5 / GLM-5.1 — optional thinking via thinking={budget_tokens:N}
  GLM-Z1 series   — always-on reasoning; thinking param omitted (built-in)
"""

import os
from .base import BaseModel, MediaItem

_ZAI_ANTHROPIC_BASE = "https://open.bigmodel.cn/api/anthropic"

# GLM-Z1 family: always-thinking, cannot toggle
_Z1_PREFIXES = ("glm-z1",)


def _is_z1(model_id: str) -> bool:
    return any(model_id.lower().startswith(p) for p in _Z1_PREFIXES)


class ZAIModel(BaseModel):
    """ZhipuAI GLM models via Anthropic-compatible proxy.

    Args:
        model_id:  API model name, e.g. "glm-5", "glm-5.1", "glm-z1-air".
        thinking:  For GLM-5/5.1 only — enable optional thinking mode.
                   Ignored for GLM-Z1 (always-on reasoning).
    """

    def __init__(self, model_id: str, thinking: bool = False):
        super().__init__(model_id)
        import anthropic

        self.client = anthropic.Anthropic(
            api_key=os.environ["ZAI_API_KEY"],
            base_url=_ZAI_ANTHROPIC_BASE,
        )

        self._z1 = _is_z1(model_id)
        self._thinking = self._z1 or thinking

        self.config = {
            "temperature":       None if self._thinking else 0,
            "max_output_tokens": 4096 if self._thinking else 1024,
            "thinking":          "native" if self._z1 else ("enabled" if thinking else False),
        }

    def complete(self, prompt: str, system: str | None = None,
                 media: list[MediaItem] | None = None) -> str:
        # ZAI is text-only; media is not supported
        kwargs: dict = {
            "model":      self.model_id,
            "max_tokens": 4096 if self._thinking else 1024,
            "messages":   [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        if not self._thinking:
            kwargs["temperature"] = 0

        # Thinking param for GLM-5/5.1 (Anthropic format: budget_tokens)
        # GLM-Z1 always thinks — no param needed
        if self._thinking and not self._z1:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": 2048}

        resp = self.client.messages.create(**kwargs)

        # Extract text content.
        # GLM-5/5.1 with thinking: Anthropic-style thinking blocks (block.type=="thinking")
        # GLM-Z1 series: thinking embedded as <think>...</think> in text block
        text = ""
        thinking_tokens = 0
        for block in resp.content:
            if block.type == "thinking":
                thinking_tokens += max(1, len(block.thinking) // 4)
            elif block.type == "text":
                raw = block.text
                if self._z1:
                    # Strip inline <think>...</think> block; keep only the answer after it
                    import re as _re
                    think_m = _re.search(r"<think>(.*?)</think>\s*", raw, _re.DOTALL)
                    if think_m:
                        thinking_tokens = max(1, len(think_m.group(1)) // 4)
                        raw = raw[think_m.end():]
                text += raw

        # Capture usage
        self.last_usage = None
        try:
            u = resp.usage
            self.last_usage = {
                "prompt_tokens":     u.input_tokens,
                "completion_tokens": u.output_tokens,
                "thinking_tokens":   thinking_tokens if self._thinking else None,
                "total_tokens":      u.input_tokens + u.output_tokens,
            }
        except Exception:
            pass

        return text.strip()
