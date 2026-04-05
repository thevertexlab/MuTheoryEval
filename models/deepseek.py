import os
from .base import BaseModel, MediaItem

_DEEPSEEK_REASONING_MODELS = ("deepseek-reasoner",)


class DeepSeekModel(BaseModel):
    def __init__(self, model_id: str):
        super().__init__(model_id)
        from openai import OpenAI
        self.client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com",
        )
        self._reasoning = model_id in _DEEPSEEK_REASONING_MODELS
        self.config = {
            "temperature":       None if self._reasoning else 0,
            "max_output_tokens": 16,
            "thinking":          "native" if self._reasoning else False,
        }

    def complete(self, prompt: str, system: str | None = None,
                 media: list[MediaItem] | None = None) -> str:
        # DeepSeek API is text-only; media is ignored
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": 1024 if self._reasoning else 16,
        }
        if not self._reasoning:
            kwargs["temperature"] = 0

        resp = self.client.chat.completions.create(**kwargs)

        # Capture token usage
        self.last_usage = None
        try:
            u = resp.usage
            self.last_usage = {
                "prompt_tokens":     u.prompt_tokens,
                "completion_tokens": u.completion_tokens,
                "thinking_tokens":   None,
                "total_tokens":      u.total_tokens,
            }
        except Exception:
            pass

        return resp.choices[0].message.content.strip()
