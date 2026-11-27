import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from codebase_assistant.llm import LLMClient, LLMError, RateLimitError

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "evals" / "questions.jsonl"
RUNS = ROOT / "evals" / "runs"

PROMPT_VERSION = "v1"
CLOSED_BOOK = (
    "You answer questions about FastAPI 0.165.0. "
    "If FastAPI has no such feature, say so plainly. Answer in at most 5 sentences."
)


def load_questions(ids: list[str] | None) -> list[dict]:
    questions = []
    with open(QUESTIONS, encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            if ids is None or q["id"] in ids:
                questions.append(q)
    return questions


def generate_with_retry(client: LLMClient, prompt: str, **kwargs):
    for attempt in range(3):
        try:
            return client.generate(prompt, **kwargs)
        except RateLimitError:
            if attempt == 2:
                raise
            time.sleep(30)  # wait a bit before retrying


def run_closed_book(client: LLMClient, q: dict) -> dict:
    record = {
        "id": q["id"],
        "mode": "closed-book",
        "model": client.model,
        "prompt_version": PROMPT_VERSION,
        "question": q["question"],
        "correct": None,
    }

    try:
        response = generate_with_retry(client, q["question"], system=CLOSED_BOOK)
    except LLMError as e:
        record.update(
            {
                "answer": None,
                "error": str(e),
                "latency_s": None,
                "prompt_tokens": None,
                "completion_tokens": None,
                "cost_usd": None,
            }
        )
    else:
        record.update(
            {
                "answer": response.text,
                "error": None,
                "latency_s": round(response.latency_ms / 1000, 2),
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "cost_usd": response.cost_usd,
            }
        )
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["closed-book"], required=True)
    parser.add_argument("--ids", help="comma-separated, e.g. q13,q16")
    args = parser.parse_args()

    ids = args.ids.split(",") if args.ids else None
    questions = load_questions(ids)

    RUNS.mkdir(parents=True, exist_ok=True)
    out_path = RUNS / f"{datetime.now():%Y%m%d-%H%M}_{args.mode}.jsonl"

    with LLMClient() as client, out_path.open("%Y%m%d-%H%M%S", encoding="utf-8") as f:
        for q in questions:
            record = run_closed_book(client, q)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            print(q["id"], "ok" if record["error"] is None else f"ERROR {record['error'][:80]}")

    print(f"Saved {len(questions)} records to {out_path}")


if __name__ == "__main__":
    main()
