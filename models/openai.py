import base64
import os
from .base import BaseModel, MediaItem

# Reasoning models don't accept temperature; they use max_completion_tokens
# and have a minimum of 1000 tokens (reasoning output included).
_REASONING_MODEL_PREFIXES = ("o1", "o3", "o4", "gpt-5")


def _is_reasoning_model(model_id: str) -> bool:
    return any(model_id.startswith(p) for p in _REASONING_MODEL_PREFIXES)


class OpenAIModel(BaseModel):
    def __init__(self, model_id: str, reasoning_effort: str | None = None):
        """
        Args:
            model_id: OpenAI model identifier.
            reasoning_effort: For reasoning models only — "low", "medium", "high", "xhigh".
                              None = model default.
        """
        super().__init__(model_id)
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self._reasoning = _is_reasoning_model(model_id)
        self.reasoning_effort = reasoning_effort

        self.config = {
            "temperature":       None if self._reasoning else 0,
            "max_output_tokens": 1024,
            "thinking":          "native" if self._reasoning else False,
            "reasoning_effort":  reasoning_effort,
        }

    def complete(self, prompt: str, system: str | None = None,
                 media: list[MediaItem] | None = None) -> str:
        messages = []
        if system:
            # Reasoning models don't support system role — inject as first user message
            if self._reasoning:
                messages.append({"role": "user", "content": system})
            else:
                messages.append({"role": "system", "content": system})

        if media:
            content: list | str = []
            for m in media:
                if m["mime_type"].startswith("image/"):
                    b64 = base64.b64encode(m["data"]).decode()
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{m['mime_type']};base64,{b64}"},
                    })
                # audio not supported by OpenAI chat completions — skip silently
            content.append({"type": "text", "text": prompt})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt})

        kwargs: dict = {
            "model": self.model_id,
            "messages": messages,
            "max_completion_tokens": self.config.get("max_output_tokens", 1024),
        }
        if not self._reasoning:
            kwargs["temperature"] = 0
        if self._reasoning and self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort

        resp = self.client.chat.completions.create(**kwargs)

        # Capture token usage, including reasoning tokens for o-series models
        self.last_usage = None
        try:
            u = resp.usage
            reasoning_tokens = None
            try:
                reasoning_tokens = u.completion_tokens_details.reasoning_tokens
            except Exception:
                pass
            self.last_usage = {
                "prompt_tokens":     u.prompt_tokens,
                "completion_tokens": u.completion_tokens,
                "thinking_tokens":   reasoning_tokens,
                "total_tokens":      u.total_tokens,
            }
        except Exception:
            pass

        return (resp.choices[0].message.content or "").strip()
