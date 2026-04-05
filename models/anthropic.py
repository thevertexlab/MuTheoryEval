import os
from .base import BaseModel


class AnthropicModel(BaseModel):
    def __init__(self, model_id: str):
        super().__init__(model_id)
        import anthropic
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def complete(self, prompt: str, system: str | None = None) -> str:
        kwargs = {"model": self.model_id, "max_tokens": 16, "messages": [{"role": "user", "content": prompt}]}
        if system:
            kwargs["system"] = system
        resp = self.client.messages.create(**kwargs)
        return resp.content[0].text.strip()
