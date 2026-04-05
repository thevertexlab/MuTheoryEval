import base64
import os
from .base import BaseModel, MediaItem


class AnthropicModel(BaseModel):
    def __init__(self, model_id: str, thinking_budget: int | None = None):
        """
        Args:
            model_id: Anthropic model identifier.
            thinking_budget: If set, enables extended thinking with this token budget (min 1024).
                             max_tokens is automatically set to thinking_budget + 256.
                             None = standard mode (no thinking).
        """
        super().__init__(model_id)
        import anthropic
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.thinking_budget = thinking_budget

        self.config = {
            "temperature": 0,
            "max_output_tokens": (thinking_budget + 256) if thinking_budget else 16,
            "thinking": {"budget_tokens": thinking_budget} if thinking_budget else False,
        }

    def complete(self, prompt: str, system: str | None = None,
                 media: list[MediaItem] | None = None) -> str:
        if media:
            content: list | str = []
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

        if self.thinking_budget:
            # Extended thinking: max_tokens must exceed thinking_budget.
            # System prompt injected into user turn — extended thinking may not support system role.
            if system:
                if isinstance(content, str):
                    user_content: list | str = f"{system}\n\n{content}"
                else:
                    user_content = [{"type": "text", "text": system}, *content]
            else:
                user_content = content

            kwargs: dict = {
                "model": self.model_id,
                "max_tokens": self.thinking_budget + 256,
                "thinking": {"type": "enabled", "budget_tokens": self.thinking_budget},
                "messages": [{"role": "user", "content": user_content}],
            }
        else:
            kwargs = {
                "model": self.model_id,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": content}],
            }
            if system:
                kwargs["system"] = system

        resp = self.client.messages.create(**kwargs)

        # Capture token usage (Anthropic has no separate thinking_tokens field;
        # thinking content is counted within output_tokens)
        self.last_usage = None
        try:
            u = resp.usage
            self.last_usage = {
                "prompt_tokens":     u.input_tokens,
                "completion_tokens": u.output_tokens,
                "thinking_tokens":   None,   # not separately reported by Anthropic API
                "total_tokens":      u.input_tokens + u.output_tokens,
            }
        except Exception:
            pass

        # With thinking enabled, content is a list of blocks; extract only text blocks
        for block in resp.content:
            if block.type == "text":
                return block.text.strip()
        # Fallback for non-thinking responses
        return resp.content[0].text.strip()
