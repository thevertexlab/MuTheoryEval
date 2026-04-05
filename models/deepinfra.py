import os
from .base import BaseModel, MediaItem


class DeepInfraModel(BaseModel):
    def __init__(self, model_id: str):
        super().__init__(model_id)
        from openai import OpenAI
        self.client = OpenAI(
            api_key=os.environ["DEEPINFRA_API_KEY"],
            base_url="https://api.deepinfra.com/v1/openai",
        )

    def complete(self, prompt: str, system: str | None = None,
                 media: list[MediaItem] | None = None) -> str:
        # DeepInfra text models don't support media; ignored here
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_tokens=16,
            temperature=0,
        )
        return resp.choices[0].message.content.strip()
