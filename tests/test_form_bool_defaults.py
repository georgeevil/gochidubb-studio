"""An omitted boolean form field means what the route declares it means.

FastAPI 0.147.0 through 0.161.1 read every omitted `bool` and `Optional[bool]`
form field as an unchecked HTML checkbox, so False, whatever the route declared.
0.162.0 restored the declared defaults, and requirements.txt now requires it.

The CLI always sends these fields, so it never noticed. creator.html and go.html
do not send keep_bg, and a preset edit in index.html sends consent_attested only
when the toggle changed, so those paths lost the background bed and the consent
record. A script that left out dry_run deleted files. CI installs an unpinned
FastAPI, so these tests are also what catches a future upstream change.

The probe runs each route's real signature through FastAPI's form parsing and
never reaches the handler, because the routes that queue a dub need a lot of
stubbing that has nothing to do with parsing. Cleanup is the exception: what it
does with the parsed value is the point, so it runs for real, on a temp folder.
"""
import functools
import re

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

import server


def _submit(method, path, field, form):
    """Send `form` through the signature of a real route, to an echo handler."""
    route = next(r for r in server.app.routes
                 if isinstance(r, APIRoute) and r.path == path and method in r.methods)

    @functools.wraps(route.endpoint)
    async def echo(**kwargs):
        return {"value": kwargs[field]}

    probe = FastAPI()
    probe.add_api_route(path, echo, methods=[method])
    url = re.sub(r"\{\w+\}", "x", path)
    return TestClient(probe).request(method, url, data={"unrelated": "1", **form})


@pytest.mark.parametrize("method,path,field,form,declared", [
    ("POST", "/api/narrate", "render_video", {"script": "# Hi"}, True),
    ("POST", "/api/dub", "keep_bg", {}, True),
    ("POST", "/api/dub/batch", "keep_bg", {}, True),
    ("POST", "/api/quick_test", "keep_bg", {}, True),
    ("POST", "/api/showcase", "keep_bg", {}, True),
    ("POST", "/api/storage/cleanup", "dry_run", {}, True),
    ("POST", "/api/storage/cleanup", "include_errored", {}, True),
    ("POST", "/api/storage/cleanup", "include_cancelled", {}, True),
    # None is its own instruction here: inherit from the original job, and
    # leave the consent record alone. False would overwrite both.
    ("POST", "/api/job/{job_id}/redub", "keep_bg", {}, None),
    ("PUT", "/api/voice_presets/{preset_id}", "consent_attested", {}, None),
])
def test_an_omitted_field_takes_its_declared_default(method, path, field, form, declared):
    r = _submit(method, path, field, form)
    assert r.status_code == 200, r.text
    assert r.json()["value"] is declared


@pytest.mark.parametrize("method,path,field", [
    ("POST", "/api/admin/intake", "paused"),
    ("POST", "/api/storage/star/{job_id}", "starred"),
])
def test_an_omitted_required_field_is_refused(method, path, field):
    r = _submit(method, path, field, {})
    assert r.status_code == 422
    assert r.json()["detail"][0]["loc"] == ["body", field]


def test_a_sent_false_still_reaches_the_route():
    r = _submit("POST", "/api/dub", "keep_bg", {"keep_bg": "false"})
    assert r.json()["value"] is False


@pytest.fixture
def old_jobs(tmp_path, monkeypatch):
    """One old job in each settled status, with a file cleanup would remove."""
    monkeypatch.setattr(server, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(server, "save_job", lambda job: None)
    monkeypatch.setattr(server, "jobs", {})
    files = []
    for status in ("complete", "error", "cancelled"):
        server.jobs[status] = {"id": status, "status": status, "created": 0}
        (tmp_path / status).mkdir()
        files.append(tmp_path / status / "segment_0001.wav")
        files[-1].write_bytes(b"audio")
    return files


def test_cleanup_without_dry_run_deletes_nothing(old_jobs):
    r = TestClient(server.app).post("/api/storage/cleanup", data={"older_than_days": "0"})
    assert r.json()["dry_run"] is True
    assert r.json()["candidates"] == 3       # include_errored and include_cancelled
    assert r.json()["bytes_freed"] > 0
    assert all(f.exists() for f in old_jobs)


def test_cleanup_deletes_when_dry_run_is_turned_off(old_jobs):
    r = TestClient(server.app).post(
        "/api/storage/cleanup", data={"older_than_days": "0", "dry_run": "false"})
    assert r.json()["dry_run"] is False
    assert not any(f.exists() for f in old_jobs)
