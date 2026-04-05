import os
from .base import BaseModel, MediaItem


class DeepInfraModel(BaseModel):
    def __init__(self, model_id: str, thinking_native: bool = False):
        """
        Args:
            model_id: DeepInfra model identifier (e.g. "Qwen/Qwen3-Max-Thinking").
            thinking_native: Set True for always-on thinking models (Qwen3-Thinking, DeepSeek-R1).
                             These don't support temperature and have larger max_tokens.
        """
        super().__init__(model_id)
        from openai import OpenAI
        self.client = OpenAI(
            api_key=os.environ["DEEPINFRA_API_KEY"],
            base_url="https://api.deepinfra.com/v1/openai",
        )
        self._thinking_native = thinking_native
        self.config = {
            "temperature":       None if thinking_native else 0,
            "max_output_tokens": 1024 if thinking_native else 16,
            "thinking":          "native" if thinking_native else False,
        }

    def complete(self, prompt: str, system: str | None = None,
                 media: list[MediaItem] | None = None) -> str:
        # DeepInfra text models don't support media; ignored here
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": self.config.get("max_output_tokens", 1024 if self._thinking_native else 16),
        }
        if not self._thinking_native:
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
