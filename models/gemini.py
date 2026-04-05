import os
from .base import BaseModel, MediaItem

# ── Gemini thinking API versions — DO NOT MIX ─────────────────────────────────
# Gemini 2.5 (gemini-2.5-*):
#   ThinkingConfig(thinking_budget=N)   — N is int token count; -1 = automatic; 0 = disabled
#
# Gemini 3+ (gemini-3-*, gemini-3.1-*):
#   ThinkingConfig(thinking_level=<ThinkingLevel enum>)  — HIGH / MEDIUM / LOW / MINIMAL
#   thinking_budget is IGNORED or undefined on Gemini 3 — never pass it to Gemini 3 models.
#   Omitting ThinkingConfig entirely = model default (= HIGH as of Gemini 3 Flash launch).
# ──────────────────────────────────────────────────────────────────────────────

_GEMINI3_THINKING_MODELS = ("gemini-3-", "gemini-3.1-")
_VALID_THINKING_LEVELS = {"minimal", "low", "medium", "high"}


def _is_thinking_model(model_id: str) -> bool:
    return any(model_id.startswith(p) for p in _GEMINI3_THINKING_MODELS)


class GeminiModel(BaseModel):
    def __init__(self, model_id: str, thinking_level: str | None = None):
        """
        Args:
            model_id: Gemini model identifier.
            thinking_level: One of "minimal", "low", "medium", "high", or None (model default).
                            Only applies to Gemini 3-series models (always-on thinking).
                            Do NOT use thinking_budget here — that is Gemini 2.5 API only.
        """
        super().__init__(model_id)
        from google import genai
        self.client = genai.Client(api_key=os.environ["GOOGLE_GEMINI_API_KEY"])

        if thinking_level is not None and thinking_level not in _VALID_THINKING_LEVELS:
            raise ValueError(
                f"Invalid thinking_level={thinking_level!r}. "
                f"Must be one of {_VALID_THINKING_LEVELS} or None. "
                f"(Note: thinking_budget is Gemini 2.5 API only — use thinking_level for Gemini 3.)"
            )
        self.thinking_level = thinking_level

        native_thinking = _is_thinking_model(model_id)
        thinking_config: bool | dict | str
        if native_thinking:
            # Gemini 3: always thinks. None = model default (HIGH). Explicit level if set.
            thinking_config = {"level": thinking_level} if thinking_level else {"level": "default"}
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
            # Gemini 3: use thinking_level enum (HIGH/MEDIUM/LOW/MINIMAL).
            # Do NOT use thinking_budget here — that is Gemini 2.5 API only.
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_level=self.thinking_level.upper(),
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
