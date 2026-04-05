import os
from .base import BaseModel


class GeminiModel(BaseModel):
    def __init__(self, model_id: str):
        super().__init__(model_id)
        from google import genai
        self.client = genai.Client(api_key=os.environ["GOOGLE_GEMINI_API_KEY"])

    def complete(self, prompt: str, system: str | None = None) -> str:
        from google.genai import types
        full = f"{system}\n\n{prompt}" if system else prompt
        resp = self.client.models.generate_content(
            model=self.model_id,
            contents=full,
            config=types.GenerateContentConfig(max_output_tokens=16, temperature=0),
        )
        text = resp.text or (resp.candidates[0].content.parts[0].text if resp.candidates else "")
        return text.strip()
