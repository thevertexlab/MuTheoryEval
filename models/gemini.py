import os
from .base import BaseModel, MediaItem


class GeminiModel(BaseModel):
    def __init__(self, model_id: str):
        super().__init__(model_id)
        from google import genai
        self.client = genai.Client(api_key=os.environ["GOOGLE_GEMINI_API_KEY"])

    def complete(self, prompt: str, system: str | None = None,
                 media: list[MediaItem] | None = None) -> str:
        from google.genai import types

        text_part = f"{system}\n\n{prompt}" if system else prompt

        if media:
            # Build a multi-part content list: [media parts..., text part]
            parts = [
                types.Part.from_bytes(data=m["data"], mime_type=m["mime_type"])
                for m in media
            ]
            parts.append(types.Part.from_text(text=text_part))
            contents = [types.Content(role="user", parts=parts)]
        else:
            contents = text_part

        resp = self.client.models.generate_content(
            model=self.model_id,
            contents=contents,
            config=types.GenerateContentConfig(max_output_tokens=16, temperature=0),
        )
        text = resp.text or (resp.candidates[0].content.parts[0].text if resp.candidates else "")
        return text.strip()
