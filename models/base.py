from abc import ABC, abstractmethod


class BaseModel(ABC):
    def __init__(self, model_id: str):
        self.model_id = model_id

    @abstractmethod
    def complete(self, prompt: str, system: str | None = None) -> str:
        """Return the model's text response to a prompt."""
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
