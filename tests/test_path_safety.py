from fastapi.testclient import TestClient

import app as app_module
from app import is_path_safe


def test_dotdot_is_rejected(tmp_path):
    assert not is_path_safe(tmp_path / "..", tmp_path)


def test_good_name_passes(tmp_path):
    assert is_path_safe(tmp_path / "alice", tmp_path)


def test_slash_is_rejected(tmp_path):
    assert not is_path_safe(tmp_path / "a/b", tmp_path)


def test_empty_name_fails(tmp_path):
    assert not is_path_safe(tmp_path / "", tmp_path)


def test_slash_name_fails(tmp_path):
    assert not is_path_safe(tmp_path / "../x", tmp_path)


def test_list_route_rejects_bad_user_id(monkeypatch):
    monkeypatch.setattr(app_module, "is_single_user_mode", lambda: False)
    client = TestClient(app_module.app)
    response = client.get("/api/documents", params={"user_id": ".."})
    assert response.status_code == 400
