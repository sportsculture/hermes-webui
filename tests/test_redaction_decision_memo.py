"""Regression: `_redact_text` decision memo must be byte-identical to the
uncached path, bounded, and safe on toggle.

Companion to the #5204 redactor memo contract: the perf(conversation-switch)
change routes `_redact_text` through a per-string memo of the entire
clean-or-redacted decision (prefilter + redactor) in two size tiers. Locks:
  * memoized results are byte-identical to `_redact_text_impl` (no behavior
    change from caching),
  * repeat calls hit the memo (that is the point),
  * strings above the big-tier ceiling stay uncached yet still redact,
  * enabled=False bypasses the memo entirely (cache only ever holds
    enabled=True results, so no staleness on toggle).
"""
from api import helpers as H

_SECRET = "sk-ant-api03-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def test_redact_text_matches_uncached_decision_all_sizes():
    small = f"note {_SECRET} tail"                                   # small tier
    big = ("x" * (H._REDACT_CACHE_MAX_TEXT_LEN + 100)) + f" {_SECRET}"   # big tier
    giant = ("y" * (H._REDACT_TEXT_BIG_CACHE_MAX + 100)) + f" {_SECRET}"  # uncached
    for s in (small, big, giant):
        assert H._redact_text(s, _enabled=True) == H._redact_text_impl(s)
        assert _SECRET not in H._redact_text(s, _enabled=True)


def test_redact_text_repeat_hits_memo():
    sample = f"session {getattr(H, '_REDACT_CACHE_MAX_TEXT_LEN', 0)} {_SECRET} repeat"
    H._redact_text_lru.cache_clear()
    first = H._redact_text(sample, _enabled=True)
    hits_before = H._redact_text_lru.cache_info().hits
    second = H._redact_text(sample, _enabled=True)
    assert first == second
    assert H._redact_text_lru.cache_info().hits == hits_before + 1


def test_redact_text_benign_string_returned_verbatim():
    benign = "plain conversation text with no credential shapes at all"
    assert H._redact_text(benign, _enabled=True) == benign
    assert H._redact_text_impl(benign) == benign


def test_redact_text_disabled_bypasses_memo():
    sample = f"disabled path {_SECRET} not memoized"
    H._redact_text_lru.cache_clear()
    hits_before = H._redact_text_lru.cache_info().hits
    misses_before = H._redact_text_lru.cache_info().misses
    out = H._redact_text(sample, _enabled=False)
    assert out == sample  # disabled = verbatim, no redaction
    info = H._redact_text_lru.cache_info()
    assert (info.hits, info.misses) == (hits_before, misses_before)


def test_redact_text_giant_above_ceiling_not_memoized():
    giant = ("z" * (H._REDACT_TEXT_BIG_CACHE_MAX + 1)) + f" {_SECRET}"
    H._redact_text_big_lru.cache_clear()
    before = H._redact_text_big_lru.cache_info().misses
    out = H._redact_text(giant, _enabled=True)
    assert _SECRET not in out
    assert H._redact_text_big_lru.cache_info().misses == before  # tier skipped
