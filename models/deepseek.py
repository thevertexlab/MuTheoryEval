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
            # ⚠️  DeepSeek-R1 (deepseek-reasoner) max_tokens semantics differ from all
            # other providers: it is a TOTAL budget covering both the internal chain-of-
            # thought (reasoning_content) AND the final answer (content).  A low value
            # like 16 is entirely consumed by the reasoning chain, leaving content="".
            #
            # run.py sets: model.config["max_output_tokens"] = max(fmt_max, model_max)
            # For MCQ fmt_max=16.  If model_max is also 16, max()=16 — reasoning model
            # gets starved.  Setting 4096 here ensures max() always preserves a sane
            # budget regardless of which format is active.
            #
            # Non-reasoning models (deepseek-chat) keep 16: they only produce a short
            # letter answer and the fmt max() guard is sufficient.
            "max_output_tokens": 4096 if self._reasoning else 16,
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
            "max_tokens": self.config.get("max_output_tokens", 1024 if self._reasoning else 16),
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
