import os
from .base import BaseModel, MediaItem

# Gemini 3-series models always think (thinking cannot be disabled).
# thinking_level controls the budget: None = model default, "high" = unlimited (-1).
_GEMINI3_THINKING_MODELS = ("gemini-3-", "gemini-3.1-")


def _is_thinking_model(model_id: str) -> bool:
    return any(model_id.startswith(p) for p in _GEMINI3_THINKING_MODELS)


class GeminiModel(BaseModel):
    def __init__(self, model_id: str, thinking_level: str | None = None):
        """
        Args:
            model_id: Gemini model identifier.
            thinking_level: One of "minimal", "low", "medium", "high", or None (model default).
                            Only applies to thinking-capable models (Gemini 3 series).
        """
        super().__init__(model_id)
        from google import genai
        self.client = genai.Client(api_key=os.environ["GOOGLE_GEMINI_API_KEY"])
        self.thinking_level = thinking_level

        native_thinking = _is_thinking_model(model_id)
        thinking_config: bool | dict | str
        if native_thinking:
            thinking_config = (
                {"level": thinking_level, "budget_tokens": -1}
                if thinking_level == "high"
                else {"level": thinking_level or "default"}
            )
        else:
            thinking_config = False

        self.config = {
            "temperature": 0,
            "max_output_tokens": 4096,
            "thinking": thinking_config,
        }

    def complete(self, prompt: str, system: str | None = None,
                 media: list[MediaItem] | None = None) -> str:
        from google.genai import types

        text_part = f"{system}\n\n{prompt}" if system else prompt

        if media:
            parts = [
                types.Part.from_bytes(data=m["data"], mime_type=m["mime_type"])
                for m in media
            ]
            parts.append(types.Part.from_text(text=text_part))
            contents = [types.Content(role="user", parts=parts)]
        else:
            contents = text_part

        config_kwargs: dict = dict(max_output_tokens=self.config.get("max_output_tokens", 4096), temperature=0)
        if self.thinking_level is not None:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=-1 if self.thinking_level == "high" else None,
                include_thoughts=False,
            )

        resp = self.client.models.generate_content(
            model=self.model_id,
            contents=contents,
            config=types.GenerateContentConfig(**config_kwargs),
        )

        # Capture token usage
        self.last_usage = None
        try:
            um = resp.usage_metadata
            if um is not None:
                self.last_usage = {
                    "prompt_tokens":     um.prompt_token_count or 0,
                    "completion_tokens": um.candidates_token_count or 0,
                    "thinking_tokens":   um.thoughts_token_count or 0,
                    "total_tokens":      um.total_token_count or 0,
                }
        except Exception:
            pass

        try:
            text = resp.text or ""
        except Exception:
            text = ""
        if not text and resp.candidates:
            try:
                text = resp.candidates[0].content.parts[0].text or ""
            except Exception:
                text = ""
        return text.strip()
