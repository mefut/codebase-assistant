import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Snippet:
    path: str
    start_line: int
    end_line: int
    match_line: int
    text: str


CORPUS = ["fastapi", "docs/en", "docs_src", "pyproject.toml"]  # D-002


def rg_search(pattern: str, repo_root: Path) -> list[tuple[str, int]]:
    """Return (path, line_number) for every corpus line that contains pattern"""
    cmd = ["rg", "-n", "--no-heading", "-F", "-i", "--sort", "path", "-e", pattern, *CORPUS]
    result = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, encoding="utf-8")

    if result.returncode == 1:
        return []
    if result.returncode != 0:
        raise RuntimeError(f"rg failed: {result.stderr}")

    matches = []
    for line in result.stdout.splitlines():
        path, number, _ = line.split(":", 2)
        matches.append((path.replace("\\", "/"), int(number)))
    return matches


def build_snippets(
    patterns: list[str], repo_root: Path, k: int = 8, context: int = 5
) -> list[Snippet]:
    """Return up to k snippets in total, following the patterns in order"""
    snippets: list[Snippet] = []
    for pattern in patterns:
        for path, line in rg_search(pattern, repo_root):
            # skip a match that an earlier snippet already shows
            already_shown = any(
                s.path == path and s.start_line <= line <= s.end_line for s in snippets
            )
            if already_shown:
                continue

            lines = (repo_root / path).read_text(encoding="utf-8").splitlines()
            start = max(1, line - context)
            end = min(len(lines), line + context)
            text = "\n".join(f"{n}: {lines[n - 1]}" for n in range(start, end + 1))
            snippets.append(Snippet(path, start, end, line, text))
            if len(snippets) == k:
                return snippets
    return snippets
