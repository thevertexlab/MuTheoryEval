import base64
import os
from .base import BaseModel, MediaItem

# Reasoning models don't accept temperature; they also use max_completion_tokens
# and have minimum max_completion_tokens of 1000 (reasoning output included).
_REASONING_MODEL_PREFIXES = ("o1", "o3", "o4", "gpt-5")


def _is_reasoning_model(model_id: str) -> bool:
    return any(model_id.startswith(p) for p in _REASONING_MODEL_PREFIXES)


class OpenAIModel(BaseModel):
    def __init__(self, model_id: str):
        super().__init__(model_id)
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self._reasoning = _is_reasoning_model(model_id)

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
            content = []
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

        kwargs = {
            "model": self.model_id,
            "messages": messages,
            "max_completion_tokens": 1024,  # must be ≥1000 for reasoning models
        }
        if not self._reasoning:
            # Standard models: pin temperature for reproducibility
            kwargs["temperature"] = 0

        resp = self.client.chat.completions.create(**kwargs)
        return (resp.choices[0].message.content or "").strip()
