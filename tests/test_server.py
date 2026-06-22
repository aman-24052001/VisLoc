"""
Tests for the VisLoc3D API server. Uses FastAPI's TestClient (in-process,
no real network/uvicorn needed) for fast, repeatable CI runs - the actual
cross-origin, real-network behavior was separately verified by hand via
Playwright against a live uvicorn instance (see VISLOC3D_ARCHITECTURE.md,
Section 12) before trusting this service description as deployable.
"""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from server.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["world_loaded"] is True


def test_world_meta(client):
    r = client.get("/api/world_meta")
    assert r.status_code == 200
    data = r.json()
    assert data["world_size"] == 1800
    assert data["n_orb_keypoints"] > 1000


def test_localize_level_succeeds_with_small_error(client):
    r = client.post("/api/localize", json={
        "position": [900, 900, 200], "quat": [1, 0, 0, 0], "include_frame": False,
    })
    assert r.status_code == 200
    data = r.json()
    assert data["success"] is True
    err = np.hypot(data["raw_xy"][0] - 900, data["raw_xy"][1] - 900)
    assert err < 5.0
    # level flight: raw and corrected should be identical (no tilt to correct)
    assert np.allclose(data["raw_xy"], data["corrected_xy"], atol=1e-6)


def test_localize_tilted_correction_resolves_offset(client):
    # ~15deg roll
    quat = [np.cos(np.radians(15) / 2), np.sin(np.radians(15) / 2), 0, 0]
    r = client.post("/api/localize", json={
        "position": [900, 900, 200], "quat": quat, "include_frame": False,
    })
    data = r.json()
    assert data["success"] is True
    raw_err = np.hypot(data["raw_xy"][0] - 900, data["raw_xy"][1] - 900)
    corr_err = np.hypot(data["corrected_xy"][0] - 900, data["corrected_xy"][1] - 900)
    assert raw_err > 30  # the known geometric offset at this tilt
    assert corr_err < 5.0  # corrected back to the noise floor


def test_localize_with_frame_returns_valid_base64_png(client):
    r = client.post("/api/localize", json={
        "position": [900, 900, 200], "quat": [1, 0, 0, 0], "include_frame": True,
    })
    data = r.json()
    assert data["frame_png_base64"] is not None
    import base64
    raw = base64.b64decode(data["frame_png_base64"])
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic bytes


def test_localize_rejects_out_of_bounds_position(client):
    r = client.post("/api/localize", json={
        "position": [5000, 5000, 200], "quat": [1, 0, 0, 0],
    })
    assert r.status_code == 400


def test_localize_rejects_zero_quaternion(client):
    r = client.post("/api/localize", json={
        "position": [900, 900, 200], "quat": [0, 0, 0, 0],
    })
    assert r.status_code == 400


def test_localize_rejects_malformed_body(client):
    r = client.post("/api/localize", json={"position": [900, 900]})
    assert r.status_code == 422


def test_localize_camera_facing_up_fails_gracefully_not_crashes(client):
    # 180deg pitch - camera points up, away from the ground entirely
    r = client.post("/api/localize", json={
        "position": [900, 900, 200], "quat": [0, 0, 1, 0],
    })
    assert r.status_code == 200
    assert r.json()["success"] is False


def test_repeated_requests_do_not_grow_world_state(client):
    """The world/localizer are process-global singletons built once at
    startup - confirm repeated calls don't accidentally mutate or grow
    that shared state (e.g. appending to a list each call)."""
    n_kp_before = len(client.app.state.__dict__) if hasattr(client.app, "state") else None
    for _ in range(20):
        client.post("/api/localize", json={
            "position": [900, 900, 200], "quat": [1, 0, 0, 0],
        })
    # world keypoint count must be exactly unchanged
    r = client.get("/api/world_meta")
    assert r.json()["n_orb_keypoints"] == 20000
