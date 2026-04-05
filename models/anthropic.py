import base64
import os
from .base import BaseModel, MediaItem


class AnthropicModel(BaseModel):
    def __init__(self, model_id: str):
        super().__init__(model_id)
        import anthropic
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def complete(self, prompt: str, system: str | None = None,
                 media: list[MediaItem] | None = None) -> str:
        if media:
            content = []
            for m in media:
                if m["mime_type"].startswith("image/"):
                    b64 = base64.b64encode(m["data"]).decode()
                    content.append({
                        "type": "image",
                        "source": {"type": "base64", "media_type": m["mime_type"], "data": b64},
                    })
                # audio not natively supported by Anthropic messages API — skip
            content.append({"type": "text", "text": prompt})
        else:
            content = prompt

        kwargs = {
            "model": self.model_id,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": content}],
        }
        if system:
            kwargs["system"] = system
        resp = self.client.messages.create(**kwargs)
        return resp.content[0].text.strip()
