"""Executable checks for the pilot reference answers (evals/questions.jsonl).

Runs against FastAPI installed from the pinned checkout in data/fastapi (0.165.0),
so a passing test means the pinned code really behaves as the reference answer says.
Run (from the repo root):
    uv run --no-project --python 3.12 --with ./data/fastapi --with pytest --with httpx
        --with python-multipart pytest evals/verify -q
"""

import asyncio
import re
import tomllib
from datetime import date
from pathlib import Path
from typing import Annotated

import fastapi
import httpx
import pytest
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Body,
    Depends,
    FastAPI,
    File,
    Form,
    UploadFile,
)
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

FASTAPI_ROOT = Path(__file__).resolve().parents[2] / "data" / "fastapi"


def test_pinned_version():
    assert fastapi.__version__ == "0.165.0"


def test_q01_endpoint_override_keeps_inherited_field_limit():
    router = APIRouter(form_limits={"max_files": 5, "max_fields": 1})

    @router.post("/override-files-only", form_limits={"max_files": 10})
    def override(a: Annotated[str, Form()], b: Annotated[str, Form()]):
        return "ok"

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    # Two text fields exceed the inherited max_fields=1, although the endpoint
    # only overrode max_files.
    response = client.post("/override-files-only", data={"a": "1", "b": "2"})
    assert response.status_code == 400


def test_q02_no_openapi_url_means_no_swagger_ui():
    app = FastAPI(openapi_url=None, docs_url="/docs")
    assert TestClient(app).get("/docs").status_code == 404


def test_q03_override_resolves_replacements_own_dependencies():
    def original():
        return "original"

    def sub_dependency():
        return "from-sub-dependency"

    def replacement(value: Annotated[str, Depends(sub_dependency)]):
        return value

    app = FastAPI()

    @app.get("/")
    def endpoint(value: Annotated[str, Depends(original)]):
        return value

    app.dependency_overrides[original] = replacement
    assert TestClient(app).get("/").json() == "from-sub-dependency"


def test_q04_use_cache_false_does_not_overwrite_cached_value():
    calls = {"n": 0}

    def counter():
        calls["n"] += 1
        return calls["n"]

    app = FastAPI()

    @app.get("/")
    def endpoint(
        first: Annotated[int, Depends(counter)],
        uncached: Annotated[int, Depends(counter, use_cache=False)],
        third: Annotated[int, Depends(counter)],
    ):
        return [first, uncached, third]

    # uncached re-runs the dependency (2); third still sees the original cached 1.
    assert TestClient(app).get("/").json() == [1, 2, 1]


def test_q06_invalid_json_is_json_invalid_with_body_offset():
    app = FastAPI()

    @app.post("/")
    def endpoint(payload: Annotated[dict, Body()]):
        return payload

    bad = '{"a": }'
    response = TestClient(app).post("/", content=bad, headers={"content-type": "application/json"})
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["type"] == "json_invalid"
    assert error["loc"] == ["body", bad.index("}")]


def test_q07_returned_response_skips_model_and_keeps_own_background():
    ran = []
    app = FastAPI()

    @app.get("/no-own-background", response_model=int)
    def no_own(tasks: BackgroundTasks):
        tasks.add_task(ran.append, "collected")
        return JSONResponse({"not": "an int"})

    @app.get("/own-background")
    def own(tasks: BackgroundTasks):
        tasks.add_task(ran.append, "collected")
        return JSONResponse({}, background=fastapi.BackgroundTasks([]))

    client = TestClient(app)
    response = client.get("/no-own-background")
    # response_model=int would reject this dict if serialization ran.
    assert response.status_code == 200
    assert response.json() == {"not": "an int"}
    assert ran == ["collected"]

    ran.clear()
    client.get("/own-background")
    assert ran == []


def test_q08_204_with_dict_sends_empty_body():
    app = FastAPI()

    @app.get("/", status_code=204)
    def endpoint():
        return {"a": 1}

    response = TestClient(app).get("/")
    assert response.status_code == 204
    assert response.content == b""


def test_q09_default_validation_handler_is_422_with_detail():
    app = FastAPI()

    @app.get("/")
    def endpoint(n: int):
        return n

    response = TestClient(app).get("/", params={"n": "not-a-number"})
    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)


def test_q10_tuple_becomes_list_with_items_encoded():
    result = jsonable_encoder((1, "a", date(2026, 11, 24)))
    assert type(result) is list
    assert result == [1, "a", "2026-11-24"]  # the date item was encoded too


def test_q11_chunked_upload_over_limit_gets_413():
    app = FastAPI(max_body_size=1024 * 1024)

    @app.post("/")
    def endpoint(payload: Annotated[bytes, Body(media_type="application/octet-stream")]):
        return len(payload)

    def chunks():
        for _ in range(5 * 16):  # 5 MiB in 64 KiB chunks, no Content-Length
            yield b"x" * 65536

    response = TestClient(app).post(
        "/", content=chunks(), headers={"content-type": "application/octet-stream"}
    )
    assert response.status_code == 413


def test_q12_too_many_files_gets_400():
    app = FastAPI()

    @app.post("/", form_limits={"max_files": 1})
    def endpoint(files: Annotated[list[UploadFile], File()]):
        return len(files)

    response = TestClient(app).post(
        "/", files=[("files", ("a.txt", b"a")), ("files", ("b.txt", b"b"))]
    )
    assert response.status_code == 400


def test_q13_oldest_supported_python_is_311():
    pyproject = tomllib.loads((FASTAPI_ROOT / "pyproject.toml").read_text("utf-8"))
    assert pyproject["project"]["requires-python"] == ">=3.11"


def _max_concurrent_sessions(share_one_depends: bool) -> int:
    state = {"active": 0, "peak": 0}

    async def get_session():
        state["active"] += 1
        state["peak"] = max(state["peak"], state["active"])
        try:
            yield "session"
        finally:
            state["active"] -= 1

    shared = Depends(get_session, max_concurrency=5)
    dep_a = shared if share_one_depends else Depends(get_session, max_concurrency=5)
    dep_b = shared if share_one_depends else Depends(get_session, max_concurrency=5)

    app = FastAPI()

    @app.get("/a")
    async def route_a(session: Annotated[str, dep_a]):
        await asyncio.sleep(0.2)
        return session

    @app.get("/b")
    async def route_b(session: Annotated[str, dep_b]):
        await asyncio.sleep(0.2)
        return session

    async def fire():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            await asyncio.gather(
                *[client.get("/a") for _ in range(10)],
                *[client.get("/b") for _ in range(10)],
            )

    asyncio.run(fire())
    return state["peak"]


def test_q14_separate_depends_have_separate_limits():
    assert _max_concurrent_sessions(share_one_depends=False) == 10


def test_q14_reusing_one_depends_shares_the_limit():
    assert _max_concurrent_sessions(share_one_depends=True) == 5


# Known distractors: release notes mention GitHub API rate limits hit by FastAPI's own
# CI scripts. They are not a FastAPI feature, but ripgrep mode will surface them for q16.
KNOWN_DISTRACTORS = {
    ("release-notes.md", 2725),
    ("release-notes.md", 5001),
    ("release-notes.md", 5002),
}


@pytest.mark.parametrize(
    "pattern",
    [r"\badmin\s+(panel|interface|site)\b", r"\brate[\s_-]?limit"],  # \b avoids "sepaRATE LIMIT"
)
def test_q15_q16_unanswerable_features_absent_from_corpus(pattern):
    corpus = [FASTAPI_ROOT / "fastapi", FASTAPI_ROOT / "docs" / "en", FASTAPI_ROOT / "docs_src"]
    hits = []
    for root in corpus:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".md"}:
                lines = path.read_text("utf-8", errors="ignore").splitlines()
                for number, line in enumerate(lines, start=1):
                    if re.search(pattern, line, re.IGNORECASE):
                        hits.append((path.name, number))
    assert set(hits) <= KNOWN_DISTRACTORS, hits
