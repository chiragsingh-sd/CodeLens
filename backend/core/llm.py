import os
import time

print("=" * 70)
print("LOADED LLM FROM:")
print(os.path.abspath(__file__))
print("=" * 70)

from groq import Groq          # always import — needed for fallback
from openai import OpenAI      # always import — needed for NVIDIA

from backend.core.config import settings

USE_NVIDIA = settings.llm_provider.lower() == "nvidia"

if USE_NVIDIA:
    _client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=settings.nvidia_api_key,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )
    MODEL = settings.llm_model
    print(f"[llm] using NVIDIA: {MODEL}")
else:
    _client = Groq(
        api_key=settings.groq_api_key,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )
    MODEL = settings.groq_model
    print(f"[llm] using Groq: {MODEL}")

# Fallback client created once at module level — not inside except block
# This is why the original crashed: Groq() inside except has no import
_groq_fallback = Groq(
    api_key=settings.groq_api_key,
    timeout=settings.llm_timeout_seconds,
    max_retries=settings.llm_max_retries,
)


def call_llm(system_prompt: str, user_prompt: str, json_mode: bool = False) -> str:
    print(">>>>>>>>>>>> ENTERED call_llm <<<<<<<<<<<<", flush=True)
    llm_start = time.perf_counter()
    print(f"[TIMING] llm request started: provider={settings.llm_provider} model={MODEL}")

    kwargs = {
        "model":    MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "temperature": 0.1,
        "max_tokens":  2048,
    }

    if not USE_NVIDIA and MODEL.startswith("openai/gpt-oss"):
        kwargs["max_tokens"] = 4096
        kwargs["reasoning_effort"] = "low"

    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    try:
        response = _client.chat.completions.create(**kwargs)
        elapsed = time.perf_counter() - llm_start
        print(f"[TIMING] llm response: {elapsed:.2f}s provider={settings.llm_provider} model={MODEL}")
        return response.choices[0].message.content.strip()

    except Exception as e:
        if USE_NVIDIA:
            print(f"[llm] NVIDIA failed ({e}), falling back to Groq")
            fallback_model = settings.groq_model
            fallback_kwargs = {
                "model":       fallback_model,
                "messages":    kwargs["messages"],
                "temperature": 0.1,
                "max_tokens":  4096 if fallback_model.startswith("openai/gpt-oss") else 2048,
            }
            if fallback_model.startswith("openai/gpt-oss"):
                fallback_kwargs["reasoning_effort"] = "low"

            fallback = _groq_fallback.chat.completions.create(**fallback_kwargs)
            elapsed = time.perf_counter() - llm_start
            print(f"[TIMING] llm fallback response: {elapsed:.2f}s provider=groq model={fallback_model}")
            return fallback.choices[0].message.content.strip()
        raise