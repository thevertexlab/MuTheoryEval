import base64
import os
from .base import BaseModel, MediaItem


class OpenAIModel(BaseModel):
    def __init__(self, model_id: str):
        super().__init__(model_id)
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def complete(self, prompt: str, system: str | None = None,
                 media: list[MediaItem] | None = None) -> str:
        messages = []
        if system:
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

        resp = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_tokens=16,
            temperature=0,
        )
        return resp.choices[0].message.content.strip()
