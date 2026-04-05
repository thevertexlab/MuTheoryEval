import os
from .base import BaseModel


class GeminiModel(BaseModel):
    def __init__(self, model_id: str):
        super().__init__(model_id)
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GOOGLE_GEMINI_API_KEY"])
        self.model = genai.GenerativeModel(model_id)

    def complete(self, prompt: str, system: str | None = None) -> str:
        full = f"{system}\n\n{prompt}" if system else prompt
        resp = self.model.generate_content(full)
        return resp.text.strip()
