"""Phase 10 acceptance tests: the FastAPI service (dev doc §11.2),
module M10b — real HTTP requests against the real Kurla graph (via
`fastapi.testclient.TestClient`, in-process, no live server/port
needed), no live LLM API calls (no `UDT_LLM_API_KEY` configured in this
test environment, so every episode uses the disclosed `NullLLMClient`
fallback — every decision's `llm.fallback` reads `True`, which is
itself part of what's verified below).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
REQUIRED_FILES = (
    "dependency_graph.json",
    "flood_susceptibility.tif",
    "ward_boundary.geojson",
)

pytestmark = pytest.mark.skipif(
    not all((PROCESSED_DIR / f).exists() for f in REQUIRED_FILES),
    reason="needs data/processed/{dependency_graph.json,flood_susceptibility.tif,"
    "ward_boundary.geojson} (run the 01-06 pipeline scripts)",
)


@pytest.fixture
def client() -> TestClient:
    from udt.api.main import app

    with TestClient(app) as c:
        yield c


@pytest.mark.phase10
def test_create_episode_returns_a_real_incident(client: TestClient) -> None:
    response = client.post("/episodes", json={"seed": 0})
    assert response.status_code == 200
    body = response.json()
    assert body["episode_id"].startswith("ep_")
    assert body["incident_id"]
    assert 0.5 <= body["severity"] <= 1.0  # dev doc §4.3's default severity_range


@pytest.mark.phase10
def test_twin_graph_returns_the_real_static_topology(client: TestClient) -> None:
    response = client.get("/twin/graph")
    assert response.status_code == 200
    body = response.json()
    assert len(body["assets"]) > 0
    assert len(body["edges"]) > 0


@pytest.mark.phase10
def test_twin_state_before_any_step_is_409(client: TestClient) -> None:
    episode_id = client.post("/episodes", json={"seed": 0}).json()["episode_id"]
    response = client.get("/twin/state", params={"episode_id": episode_id})
    assert response.status_code == 409


@pytest.mark.phase10
def test_step_then_twin_state_reflects_the_real_twin(client: TestClient) -> None:
    episode_id = client.post("/episodes", json={"seed": 0}).json()["episode_id"]
    step_response = client.post(f"/episodes/{episode_id}/step", params={"n": 3})
    assert step_response.status_code == 200
    body = step_response.json()
    assert body["tick"] == 3
    assert len(body["snapshots"]) == 3
    assert len(body["decisions"]) == 1  # tick 0 is a decision tick, 3 ticks -> one decision
    assert body["decisions"][0]["llm"]["fallback"] is True  # no live LLM key in this test env

    state_response = client.get("/twin/state", params={"episode_id": episode_id})
    assert state_response.status_code == 200
    assert state_response.json()["tick"] == 2  # the *last* snapshot's own tick (0-indexed)


@pytest.mark.phase10
def test_step_unknown_episode_is_404(client: TestClient) -> None:
    response = client.post("/episodes/does-not-exist/step", params={"n": 1})
    assert response.status_code == 404


@pytest.mark.phase10
def test_step_respects_n_ticks_max(client: TestClient) -> None:
    episode_id = client.post("/episodes", json={"seed": 0, "n_ticks_max": 2}).json()["episode_id"]
    response = client.post(f"/episodes/{episode_id}/step", params={"n": 10})
    body = response.json()
    assert body["tick"] == 2  # capped, not 10
    assert body["episode_ended"] is True


@pytest.mark.phase10
def test_inject_incident_replaces_the_running_episodes_incident(client: TestClient) -> None:
    episode_id = client.post("/episodes", json={"seed": 0}).json()["episode_id"]
    response = client.post(
        "/incidents", json={"episode_id": episode_id, "severity": 0.9, "incident_type": "flood"}
    )
    assert response.status_code == 200
    assert "incident_id" in response.json()


@pytest.mark.phase10
def test_inject_incident_unknown_episode_is_404(client: TestClient) -> None:
    response = client.post(
        "/incidents", json={"episode_id": "does-not-exist", "severity": 0.5}
    )
    assert response.status_code == 404


@pytest.mark.phase10
def test_list_decisions_grows_as_the_episode_steps(client: TestClient) -> None:
    episode_id = client.post("/episodes", json={"seed": 0}).json()["episode_id"]
    client.post(f"/episodes/{episode_id}/step", params={"n": 3})
    client.post(f"/episodes/{episode_id}/step", params={"n": 3})
    response = client.get("/decisions", params={"episode_id": episode_id})
    assert response.status_code == 200
    assert len(response.json()) == 2


@pytest.mark.phase10
def test_list_decisions_unknown_episode_is_404(client: TestClient) -> None:
    response = client.get("/decisions", params={"episode_id": "does-not-exist"})
    assert response.status_code == 404


@pytest.mark.phase10
def test_pending_approvals_includes_this_episodes_decisions(client: TestClient) -> None:
    episode_id = client.post("/episodes", json={"seed": 0}).json()["episode_id"]
    client.post(f"/episodes/{episode_id}/step", params={"n": 3})
    decisions = client.get("/decisions", params={"episode_id": episode_id}).json()
    pending = client.get("/approvals/pending").json()
    required_decision_ids = {d["decision_id"] for d in decisions if d["approval"]["required"]}
    pending_ids = {d["decision_id"] for d in pending}
    assert required_decision_ids <= pending_ids


@pytest.mark.phase10
def test_override_approval_updates_the_recorded_response(client: TestClient) -> None:
    episode_id = client.post("/episodes", json={"seed": 0}).json()["episode_id"]
    client.post(f"/episodes/{episode_id}/step", params={"n": 3})
    decisions = client.get("/decisions", params={"episode_id": episode_id}).json()
    decision_id = decisions[0]["decision_id"]

    response = client.post(f"/approvals/{decision_id}", json={"response": "reject"})
    assert response.status_code == 200
    assert response.json()["approval"]["response"] == "reject"

    refetched = client.get("/decisions", params={"episode_id": episode_id}).json()
    assert refetched[0]["approval"]["response"] == "reject"


@pytest.mark.phase10
def test_override_approval_unknown_decision_is_404(client: TestClient) -> None:
    response = client.post("/approvals/does-not-exist", json={"response": "approve"})
    assert response.status_code == 404


@pytest.mark.phase10
def test_websocket_receives_a_broadcast_on_step(client: TestClient) -> None:
    episode_id = client.post("/episodes", json={"seed": 0}).json()["episode_id"]
    with client.websocket_connect(f"/ws/live?episode_id={episode_id}") as ws:
        client.post(f"/episodes/{episode_id}/step", params={"n": 1})
        message = ws.receive_json()
        assert message["type"] == "step"
        assert message["tick"] == 1
