from abc import ABC, abstractmethod
from typing import TypedDict


class MediaItem(TypedDict):
    """A media attachment to pass alongside a text prompt."""
    mime_type: str   # e.g. "image/png", "audio/mp3", "audio/wav"
    data: bytes      # raw binary content


class BaseModel(ABC):
    def __init__(self, model_id: str):
        self.model_id = model_id

    @abstractmethod
    def complete(self, prompt: str, system: str | None = None,
                 media: list[MediaItem] | None = None) -> str:
        """Return the model's text response to a prompt.

        Args:
            prompt: The user-facing text prompt.
            system: Optional system prompt.
            media: Optional list of MediaItem dicts with 'mime_type' and 'data'.
                   Pass images for VLM benchmarks, audio for audio-language benchmarks.
        """
        ...

    def extract_choice(self, response: str) -> str:
        """Extract A/B/C/D from a free-text response."""
        import re
        # Look for leading letter answer first
        m = re.search(r'\b([A-D])\b', response.strip()[:20])
        if m:
            return m.group(1).upper()
        # Fallback: first A/B/C/D anywhere
        m = re.search(r'\b([A-D])\b', response.upper())
        return m.group(1) if m else "X"
