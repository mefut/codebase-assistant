import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from codebase_assistant.llm import LLMClient, LLMError, RateLimitError
from codebase_assistant.search import build_snippets

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "evals" / "questions.jsonl"
RUNS = ROOT / "evals" / "runs"
REPO = ROOT / "data" / "fastapi"

PROMPT_VERSION = "v1"
MAX_TOKENS = 4096
CLOSED_BOOK = (
    "You answer questions about FastAPI 0.165.0. "
    "If FastAPI has no such feature, say so plainly. Answer in at most 5 sentences."
)

PATTERNS_PROMPT = (
    "You write ripgrep search strings to find the answer to a question in the FastAPI 0.165.0 "
    "repository (source code, English docs, doc examples, pyproject.toml). "
    "Each string is matched as exact text, case-insensitive, so use words likely to appear "
    "verbatim: function names, parameter names, config keys, error messages. "
    'Most specific first. Return JSON: {{"patterns": [...]}} with 3 to 6 patterns.\n\n'
    "Question: {question}"
)
RIPGREP_SYSTEM = (
    "You answer questions about FastAPI 0.165.0 using only the snippets you are given. "
    "Cite each claim as [path:line]. If the snippets do not contain the answer, say the "
    "repository does not cover it and add nothing else. Answer in at most 5 sentences."
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
        response = generate_with_retry(
            client, q["question"], system=CLOSED_BOOK, max_tokens=MAX_TOKENS
        )
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


def parse_patterns(text: str) -> list[str]:
    data = json.loads(text)
    patterns = data["patterns"]

    if not isinstance(patterns, list) or not all(
        isinstance(p, str) and p.strip() for p in patterns
    ):
        raise ValueError(f"bad patterns: {patterns!r}")
    return patterns


def format_snippets(snippets) -> str:
    if not snippets:
        return "(no snippets found)"
    return "\n\n".join(f"--- {s.path}:{s.start_line}-{s.end_line} ---\n{s.text}" for s in snippets)


def run_ripgrep(client: LLMClient, q: dict) -> dict:
    record = {
        "id": q["id"],
        "mode": "ripgrep",
        "model": client.model,
        "prompt_version": PROMPT_VERSION,
        "question": q["question"],
        "correct": None,
        "citations_supported": None,
        "patterns": None,
        "snippets": None,
        "answer": None,
        "error": None,
        "latency_s": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "cost_usd": None,
    }

    try:
        p = generate_with_retry(
            client,
            PATTERNS_PROMPT.format(question=q["question"]),
            json_mode=True,
            max_tokens=MAX_TOKENS,
        )
        record["patterns"] = parse_patterns(p.text)

        snippets = build_snippets(record["patterns"], REPO)
        record["snippets"] = [asdict(s) for s in snippets]

        prompt = f"Question: {q['question']}\n\nSnippets:\n{format_snippets(snippets)}"
        a = generate_with_retry(client, prompt, system=RIPGREP_SYSTEM, max_tokens=MAX_TOKENS)
        record["answer"] = a.text
        record.update(
            {
                "latency_s": round((p.latency_ms + a.latency_ms) / 1000, 2),
                "prompt_tokens": p.prompt_tokens + a.prompt_tokens,
                "completion_tokens": p.completion_tokens + a.completion_tokens,
                "cost_usd": (p.cost_usd or 0) + (a.cost_usd or 0),
            }
        )
    except (LLMError, ValueError, KeyError, TypeError, RuntimeError) as e:
        record["error"] = f"{type(e).__name__}: {e}"
    return record


RUNNERS = {"closed-book": run_closed_book, "ripgrep": run_ripgrep}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=list(RUNNERS), required=True)
    parser.add_argument("--ids", help="comma-separated, e.g. q13,q16")
    args = parser.parse_args()

    ids = args.ids.split(",") if args.ids else None
    questions = load_questions(ids)

    RUNS.mkdir(parents=True, exist_ok=True)
    out_path = RUNS / f"{datetime.now():%Y%m%d-%H%M%S}_{args.mode}.jsonl"

    with LLMClient() as client, out_path.open("w", encoding="utf-8") as f:
        for q in questions:
            record = RUNNERS[args.mode](client, q)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            print(q["id"], "ok" if record["error"] is None else f"ERROR {record['error'][:80]}")

    print(f"Saved {len(questions)} records to {out_path}")


if __name__ == "__main__":
    main()
