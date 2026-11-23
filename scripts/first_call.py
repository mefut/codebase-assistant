import os
import sys
import time

import httpx
from dotenv import load_dotenv

URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "openrouter/free"


def main() -> None:
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("API key is missing, Check .env")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Keep your answers very short and concise"},
            {"role": "user", "content": "why I should write code by hand?"},
        ],
    }

    start = time.perf_counter()
    timeout_s = 50
    try:
        response = httpx.post(URL, headers=headers, json=body, timeout=timeout_s)
    except httpx.TimeoutException as x:
        sys.exit(f"OpenRouter didn't respond within {timeout_s} seconds !: {x}")
    except httpx.HTTPError as e:
        sys.exit(f"Network error: {e}")
    elapsed_ms = (time.perf_counter() - start) * 1000

    if response.status_code == 429:
        sys.exit("Rate limited (429). Please wait a bit before retrying.")

    if response.status_code != 200:
        sys.exit(f"Error{response.status_code}: {response.text}")

    data = response.json()
    answer = data["choices"][0]["message"]["content"]
    model_used = data.get("model", "unknown")
    prompt_tokens = data["usage"]["prompt_tokens"]
    completion_tokens = data["usage"]["completion_tokens"]

    print("\n=== API Call Succeed ===")
    print(f"Answer: {answer}")
    print(f"Model: {model_used}")
    print(f"Input Token: {prompt_tokens} | Output Token: {completion_tokens}")
    print(f"Time: {elapsed_ms:.2f} ms")


if __name__ == "__main__":
    main()
