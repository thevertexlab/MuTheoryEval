import base64
import os
from .base import BaseModel, MediaItem

_DASHSCOPE_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


class DashScopeModel(BaseModel):
    """Alibaba DashScope models via OpenAI-compatible endpoint.

    Supports Qwen-Omni series with native audio + image input.
    Streaming is used for all requests (required by Qwen-Omni for audio).
    """

    def __init__(self, model_id: str):
        super().__init__(model_id)
        from openai import OpenAI
        self.client = OpenAI(
            api_key=os.environ["DASHSCOPE_API_KEY"],
            base_url=_DASHSCOPE_BASE_URL,
        )
        self.config = {
            "temperature": 0,
            "max_output_tokens": 1024,
            "thinking": False,
        }

    def complete(self, prompt: str, system: str | None = None,
                 media: list[MediaItem] | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})

        if media:
            content: list = []
            for m in media:
                b64 = base64.b64encode(m["data"]).decode()
                if m["mime_type"].startswith("audio/"):
                    # Qwen-Omni audio input format
                    fmt = m["mime_type"].split("/", 1)[1]  # e.g. "wav", "mp3"
                    content.append({
                        "type": "input_audio",
                        "input_audio": {
                            "data": f"data:{m['mime_type']};base64,{b64}",
                            "format": fmt,
                        },
                    })
                elif m["mime_type"].startswith("image/"):
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{m['mime_type']};base64,{b64}"},
                    })
            content.append({"type": "text", "text": prompt})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt})

        # Streaming is required for Qwen-Omni audio calls; use it universally.
        stream = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            modalities=["text"],   # text-only output (no audio speech synthesis)
            temperature=0,
            max_tokens=self.config.get("max_output_tokens", 1024),
            stream=True,
            stream_options={"include_usage": True},
        )

        chunks = []
        usage_chunk = None
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                chunks.append(chunk.choices[0].delta.content)
            if hasattr(chunk, "usage") and chunk.usage is not None:
                usage_chunk = chunk.usage

        self.last_usage = None
        if usage_chunk is not None:
            try:
                self.last_usage = {
                    "prompt_tokens":     usage_chunk.prompt_tokens,
                    "completion_tokens": usage_chunk.completion_tokens,
                    "thinking_tokens":   None,
                    "total_tokens":      usage_chunk.total_tokens,
                }
            except Exception:
                pass

        return "".join(chunks).strip()
