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
  * `delete_redaction_session_cache` removes the file on session deletion
    (deleted conversations must not linger in `redaction_cache/`) and is a
    safe no-op for missing/unsafe ids.
"""
import json

import pytest

from api import helpers as H
from api.helpers import delete_redaction_session_cache, redact_session_lists_cached


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


def test_delete_removes_cache_file_leaves_sibling_intact(state_dir):
    # Deleting a session must remove its redaction-cache file (a deleted
    # conversation is not recoverable from redaction_cache/), while an
    # unrelated session's cache is untouched.
    redact_session_lists_cached("sessDel", {"messages": _msgs()})
    redact_session_lists_cached("sessKeep", {"messages": _msgs()})
    doomed = state_dir / "redaction_cache" / "sessDel.json"
    sibling = state_dir / "redaction_cache" / "sessKeep.json"
    assert doomed.exists() and sibling.exists()
    assert delete_redaction_session_cache("sessDel") is True
    assert not doomed.exists()
    assert sibling.exists()
    # second delete is a no-op
    assert delete_redaction_session_cache("sessDel") is False


def test_delete_noop_on_missing_or_invalid(state_dir):
    assert delete_redaction_session_cache("nope") is False
    assert delete_redaction_session_cache("") is False
    assert delete_redaction_session_cache("../escape") is False
    assert delete_redaction_session_cache("..\\escape") is False
    assert delete_redaction_session_cache(".") is False
    # nothing escaped the cache dir
    assert list((state_dir / "redaction_cache").glob("*.json")) == []
    assert (state_dir / "escape.json").exists() is False


def _install_fake_agent_redact(monkeypatch, tmp_path, body):
    """Populate ``sys.modules['agent']`` / ``['agent.redact']`` so that
    ``import agent.redact`` resolves to a module whose ``__file__`` points at a
    file whose bytes are exactly ``body``. Returns that file path."""
    import sys
    import types
    pkg_dir = tmp_path / "agent"
    pkg_dir.mkdir(exist_ok=True)
    (pkg_dir / "__init__.py").write_text("")
    redact_path = pkg_dir / "redact.py"
    redact_path.write_text(body)
    pkg = types.ModuleType("agent")
    pkg.__path__ = [str(pkg_dir)]
    redact_mod = types.ModuleType("agent.redact")
    redact_mod.__file__ = str(redact_path)
    monkeypatch.setitem(sys.modules, "agent", pkg)
    monkeypatch.setitem(sys.modules, "agent.redact", redact_mod)
    return redact_path


def _utime(path, mtime_ns):
    import os
    os.utime(path, ns=(mtime_ns, mtime_ns))


def test_rules_key_is_content_identity_not_pathname_metadata(tmp_path, monkeypatch):
    # Regression (#7414 review): the rules key must be a CONTENT identity, not
    # pathname metadata. Two different policy byte-for-byte files that share
    # size AND mtime_ns must produce DIFFERENT keys — otherwise a stale
    # projection survives a redaction-policy change that happens to retain the
    # same file size/mtime, serving text the CURRENT redactor would remove.
    import os
    body_a = "VALUE = 1\n"
    body_b = "VALUE = 2\n"
    assert len(body_a) == len(body_b)  # same size by construction
    fixed_mtime = 1234567890123456789
    redact_path = _install_fake_agent_redact(monkeypatch, tmp_path, body_a)

    _utime(redact_path, fixed_mtime)
    key_a = H._redact_session_cache_rules_key()

    # Rewrite with different bytes, same size, same mtime_ns.
    redact_path.write_text(body_b)
    _utime(redact_path, fixed_mtime)
    st = os.stat(redact_path)
    assert st.st_mtime_ns == fixed_mtime
    key_b = H._redact_session_cache_rules_key()

    assert key_a != key_b, "content-identity key must change when policy bytes change"


def test_rules_key_independent_of_webui_version_constant(monkeypatch):
    # Regression (#7414 review): the WebUI version stamp must NOT be the
    # authoritative identity — it degrades to a constant (None / 'unknown')
    # when git and generated version files are unavailable. With the version
    # removed from the identity, a constant version cannot collapse the key and
    # hide a policy change; the key reflects policy content only.
    import api.config as C
    monkeypatch.setattr(C, "_current_webui_version", lambda: None)
    k_none = H._redact_session_cache_rules_key()
    monkeypatch.setattr(C, "_current_webui_version", lambda: "unknown")
    k_unknown = H._redact_session_cache_rules_key()
    monkeypatch.setattr(C, "_current_webui_version", lambda: "v0.52.264")
    k_real = H._redact_session_cache_rules_key()

    # Unchanged policy content => stable, deterministic key, independent of the
    # (constant or real) version stamp.
    assert k_none == k_unknown == k_real
    # Deterministic across repeated calls (the "unchanged rule identity" control).
    assert H._redact_session_cache_rules_key() == k_real


def test_rules_key_none_fail_closed_recomputes(state_dir, monkeypatch):
    # Regression (#7414 review): if no trustworthy content identity can be
    # computed (rules_key is None), the persistent cache must be SKIPPED — not
    # read-and-reused, and not freshly re-validated against an authorizable key.
    msgs = _msgs()
    redact_session_lists_cached("sessNone", {"messages": msgs})  # seed a valid cache
    calls = _spy(monkeypatch)
    monkeypatch.setattr(H, "_redact_session_cache_rules_key", lambda: None)

    out = redact_session_lists_cached("sessNone", {"messages": msgs})

    # Fail-closed: full recompute, nothing spliced from the seeded cache.
    assert calls["n"] == len(msgs)
    assert _SECRET_STATE_OK(out)
    # A None identity must NOT be persisted as an authorizable cache key.
    stored = json.loads((state_dir / "redaction_cache" / "sessNone.json").read_text())
    assert stored["rules_key"] is not None


def test_delete_clears_in_memory_redaction_memos(state_dir):
    # The in-memory decision/redactor LRUs key on ORIGINAL strings (including
    # plaintext secrets) and are retained process-wide; deleting a session must
    # drop them from RAM, not just the on-disk projection (#7414 review follow-up
    # on the adversarial review's secret-retention finding).
    secret = "sk-ant-api03-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    H._redact_text_lru.cache_clear()
    H._redact_fn_lru.cache_clear()
    H._redact_text_big_lru.cache_clear()
    sample = f"delete me and my secret {secret} must not linger"
    out = H._redact_text(sample, _enabled=True)
    assert secret not in out
    assert H._redact_text_lru.cache_info().currsize >= 1
    assert H._redact_fn_lru.cache_info().currsize >= 1
    # Deleting a valid session (even with no projection file) clears the memos.
    assert delete_redaction_session_cache("sessRam") is False  # no on-disk file
    assert H._redact_text_lru.cache_info().currsize == 0
    assert H._redact_fn_lru.cache_info().currsize == 0
    assert H._redact_text_big_lru.cache_info().currsize == 0
