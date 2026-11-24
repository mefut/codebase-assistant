import os
import sys
import time
from dataclasses import dataclass

import httpx
from dotenv import load_dotenv

load_dotenv()

MODEL = "nvidia/nemotron-3.5-super-vl:free"


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    cost_usd: float | None


class LLMError(Exception):
    pass


class RateLimitError(LLMError):
    pass


class LLMClient:
    def __init__(
        self, model: str = MODEL, *, api_key: str | None = None, timeout_s: float = 120.0
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise LLMError("OPENROUTER_API_KEY is missing. Check your .env file.")

        # It keeps the connection to OpenRouter open between calls
        self._http = httpx.Client(
            base_url="https://openrouter.ai/api/v1",
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=timeout_s,
        )
        self.model = model
        self.timeout_s = timeout_s

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> LLMResponse:
        messages = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        start = time.perf_counter()

        try:
            response = self._http.post("/chat/completions", json=body)
        except httpx.TimeoutException as e:
            raise LLMError(f"OpenRouter didn't respond in {self.timeout_s} seconds: {e}") from e
        except httpx.HTTPError as e:
            raise LLMError(f"Network Error: {e}") from e

        elapsed_ms = (time.perf_counter() - start) * 1000

        if response.status_code == 429:
            raise RateLimitError(f"HTTP 429 rate limited: {response.text[:300]}")
        if response.status_code != 200:
            raise LLMError(f"HTTP {response.status_code}: {response.text[:300]}")

        # The body might not be JSON (proxy error page, cut-off response)
        try:
            data = response.json()
        except ValueError as e:
            raise LLMError(f"Response was not JSON: {response.text[:300]}") from e

        # OpenRouter can answer 200 but put an error object in the body
        if "error" in data:
            raise LLMError(f"OpenRouter error: {data['error']}")

        # The structure might be missing, or the content empty or None
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"Unexpected response shape: {str(data)[:300]}") from e
        if not text:
            raise LLMError("Model returned empty content")

        # Did we get the model we asked for?
        returned_model = data.get("model", "")
        if not returned_model.startswith(self.model.removesuffix(":free")):
            raise LLMError(f"Asked for {self.model}, got {returned_model!r}")

        usage = data.get("usage") or {}
        return LLMResponse(
            text=text,
            model=returned_model,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            latency_ms=elapsed_ms,
            cost_usd=usage.get("cost"),
        )


if __name__ == "__main__":
    try:
        with LLMClient() as client:
            r1 = client.generate("What is FastAPI? One sentence.")
            print(f"{r1.model}: {r1.text} ({r1.latency_ms:.0f} ms)")

            r2 = client.generate('Return {"ok": true} as JSON.', json_mode=True)
            print(f"{r2.text} ({r2.latency_ms:.0f} ms)")
    except RateLimitError as e:
        print(f"Warning: {e}")
    except LLMError as e:
        sys.exit(f"Critical Error: {e}")
