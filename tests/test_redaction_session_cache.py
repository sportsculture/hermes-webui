"""Regression: `redact_session_lists_cached` — the persisted per-message
redaction cache for large-session conversation switches.

Locks:
  * first call computes + persists a cache file; secrets are redacted,
  * repeat call serves from cache (zero `_redact_messages` work),
  * appended messages recompute individually (append-mostly splice),
  * corrupt/foreign cache files fall back gracefully (never fail a response),
  * `api_redact_enabled` participates in validation (toggle recomputes),
  * the per-request `_active_turn_user` decoration is applied after retrieval
    and is NEVER persisted to the cache file.
"""
import json

import pytest

from api import helpers as H
from api.helpers import redact_session_lists_cached


@pytest.fixture()
def state_dir(tmp_path, monkeypatch):
    import api.config
    monkeypatch.setattr(api.config, "STATE_DIR", tmp_path)
    return tmp_path


def _msgs(secret="sk-ant-api03-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"):
    return [
        {"role": "user", "content": f"hello one {secret}"},
        {"role": "assistant", "content": "plain reply"},
    ]


def _spy(monkeypatch):
    calls = {"n": 0}
    real = H._redact_messages

    def counting(messages, **kwargs):
        calls["n"] += len(messages) if isinstance(messages, list) else 1
        return real(messages, **kwargs)

    monkeypatch.setattr(H, "_redact_messages", counting)
    return calls


def test_cold_computes_persists_and_redacts(state_dir):
    msgs = _msgs()
    out = redact_session_lists_cached("sessA", {"messages": msgs})
    assert _SECRET_STATE_OK(out)
    cache_file = state_dir / "redaction_cache" / "sessA.json"
    assert cache_file.exists()
    stored = json.loads(cache_file.read_text())
    assert stored["enabled"] is True
    # persisted projections are decoration-free and redacted
    assert all("_active_turn_user" not in m for m in stored["lists"]["messages"])


def _SECRET_STATE_OK(payload):
    text = json.dumps(payload)
    return "sk-ant-api03-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" not in text


def test_repeat_serves_from_cache_zero_redaction_work(state_dir, monkeypatch):
    calls = _spy(monkeypatch)
    msgs = _msgs()
    redact_session_lists_cached("sessB", {"messages": msgs})
    first = calls["n"]
    assert first == len(msgs)
    out2 = redact_session_lists_cached("sessB", {"messages": msgs})
    assert calls["n"] == first  # zero new redaction work — spliced from cache
    assert out2["messages"][0]["content"].startswith("hello one")


def test_append_recomputes_only_new_message(state_dir, monkeypatch):
    calls = _spy(monkeypatch)
    msgs = _msgs()
    redact_session_lists_cached("sessC", {"messages": msgs})
    assert calls["n"] == len(msgs)
    first_count = calls["n"]
    msgs.append({"role": "user", "content": "a fresh follow-up message"})
    out = redact_session_lists_cached("sessC", {"messages": msgs})
    assert calls["n"] == first_count + 1  # +1: only the appended item recomputed
    assert out["messages"][-1]["content"] == "a fresh follow-up message"


def test_change_to_existing_message_recomputes_it(state_dir, monkeypatch):
    calls = _spy(monkeypatch)
    msgs = _msgs()
    redact_session_lists_cached("sessD", {"messages": msgs})
    assert calls["n"] == 2
    msgs[1]["content"] = "assistant reply was edited"
    out = redact_session_lists_cached("sessD", {"messages": msgs})
    assert calls["n"] == 3  # the edited message was recomputed
    assert out["messages"][1]["content"] == "assistant reply was edited"


def test_corrupt_cache_falls_back_gracefully(state_dir):
    redact_session_lists_cached("sessE", {"messages": _msgs()})
    cache_file = state_dir / "redaction_cache" / "sessE.json"
    cache_file.write_text("{not json at all")
    out = redact_session_lists_cached("sessE", {"messages": _msgs()})
    assert _SECRET_STATE_OK(out)


def test_enabled_toggle_recomputes(state_dir, monkeypatch):
    import api.config
    msgs = _msgs()
    monkeypatch.setattr(api.config, "load_settings", lambda: {"api_redact_enabled": True})
    redact_session_lists_cached("sessF", {"messages": msgs})
    monkeypatch.setattr(api.config, "load_settings", lambda: {"api_redact_enabled": False})
    out = redact_session_lists_cached("sessF", {"messages": msgs})
    # disabled = verbatim passthrough; the enabled=True cache must not serve
    assert out["messages"][0]["content"] == msgs[0]["content"]
    assert _SECRET_STATE_OK({"x": "cleared"})  # sanity: helper intact


def test_turn_decoration_applied_but_never_persisted(state_dir):
    msgs = [
        {"role": "user", "content": "hi", "_active_turn_token": "tok-1"},
        {"role": "assistant", "content": "ho"},
    ]
    out = redact_session_lists_cached(
        "sessG", {"messages": msgs}, _active_turn_token="tok-1")
    assert out["messages"][0].get("_active_turn_user") is True
    stored = json.loads(
        (state_dir / "redaction_cache" / "sessG.json").read_text())
    assert all("_active_turn_user" not in m for m in stored["lists"]["messages"])
    # a later request without the token must not see a stale flag
    out2 = redact_session_lists_cached("sessG", {"messages": msgs})
    assert all("_active_turn_user" not in m for m in out2["messages"])


def test_midlist_insertion_never_serves_stale_projection(state_dir):
    # Insert at index 0 shifts every digest: the splice must not pair old
    # projections with the wrong messages.
    msgs = [
        {"role": "user", "content": "alpha message"},
        {"role": "assistant", "content": "beta reply"},
    ]
    redact_session_lists_cached("sessIns", {"messages": msgs})
    msgs.insert(0, {"role": "user", "content": "zeroth inserted message"})
    out = redact_session_lists_cached("sessIns", {"messages": msgs})
    assert [m["content"] for m in out["messages"]] == [
        "zeroth inserted message", "alpha message", "beta reply",
    ]


def test_truncation_serves_matching_prefix(state_dir, monkeypatch):
    calls = _spy(monkeypatch)
    msgs = [
        {"role": "user", "content": "keep one"},
        {"role": "assistant", "content": "keep two"},
        {"role": "user", "content": "drop three"},
    ]
    redact_session_lists_cached("sessTrunc", {"messages": msgs})
    assert calls["n"] == 3
    out = redact_session_lists_cached("sessTrunc", {"messages": msgs[:2]})
    assert calls["n"] == 3  # prefix spliced, nothing recomputed
    assert [m["content"] for m in out["messages"]] == ["keep one", "keep two"]
