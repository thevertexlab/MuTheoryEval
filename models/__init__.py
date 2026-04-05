from .openai import OpenAIModel
from .anthropic import AnthropicModel
from .gemini import GeminiModel
from .deepseek import DeepSeekModel
from .deepinfra import DeepInfraModel

REGISTRY = {
    # OpenAI
    "gpt-4o": lambda: OpenAIModel("gpt-4o"),
    "gpt-4o-mini": lambda: OpenAIModel("gpt-4o-mini"),
    "gpt-4-turbo": lambda: OpenAIModel("gpt-4-turbo"),
    "gpt-3.5-turbo": lambda: OpenAIModel("gpt-3.5-turbo"),
    # Anthropic
    "claude-3-5-sonnet": lambda: AnthropicModel("claude-3-5-sonnet-20241022"),
    "claude-3-5-haiku": lambda: AnthropicModel("claude-3-5-haiku-20241022"),
    "claude-3-opus": lambda: AnthropicModel("claude-3-opus-20240229"),
    # Gemini
    "gemini-2.0-flash": lambda: GeminiModel("gemini-2.0-flash"),
    "gemini-1.5-pro": lambda: GeminiModel("gemini-1.5-pro"),
    # DeepSeek
    "deepseek-chat": lambda: DeepSeekModel("deepseek-chat"),
    "deepseek-reasoner": lambda: DeepSeekModel("deepseek-reasoner"),
    # Open-weight via DeepInfra
    "llama-3.3-70b": lambda: DeepInfraModel("meta-llama/Llama-3.3-70B-Instruct"),
    "qwen2.5-72b": lambda: DeepInfraModel("Qwen/Qwen2.5-72B-Instruct"),
    "mixtral-8x7b": lambda: DeepInfraModel("mistralai/Mixtral-8x7B-Instruct-v0.1"),
}
