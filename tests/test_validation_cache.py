import time

import pytest

from data.training_upload import ValidationResult
from data import validation_cache


@pytest.fixture(autouse=True)
def _clear_cache():
    """The cache is module-level, in-memory state -- reset it around every
    test so one test's tokens can never leak into another's."""
    validation_cache._BATCHES.clear()
    yield
    validation_cache._BATCHES.clear()


SAMPLE = ValidationResult(
    accepted=[{"row_number": 2, "laboratory_code": "2026-T1", "raw": {}, "derived": {}}],
    rejected=[],
)


class TestStoreAndPop:
    def test_pop_returns_the_stored_result(self):
        token = validation_cache.store(SAMPLE)
        assert validation_cache.pop(token) is SAMPLE

    def test_pop_removes_it_one_shot(self):
        """A token must not be reusable -- committing the same batch twice
        (e.g. a double-click, or a retried request) must not silently
        recommit or double-count rows."""
        token = validation_cache.store(SAMPLE)
        validation_cache.pop(token)
        assert validation_cache.pop(token) is None

    def test_unknown_token_returns_none(self):
        assert validation_cache.pop("does-not-exist") is None

    def test_each_store_call_returns_a_distinct_token(self):
        first = validation_cache.store(SAMPLE)
        second = validation_cache.store(SAMPLE)
        assert first != second
        # Both remain independently retrievable until popped.
        assert validation_cache.pop(first) is SAMPLE
        assert validation_cache.pop(second) is SAMPLE


class TestExpiry:
    def test_expired_token_returns_none(self, monkeypatch):
        monkeypatch.setattr(validation_cache, "_BATCH_TTL_SECONDS", 0.01)
        token = validation_cache.store(SAMPLE)
        time.sleep(0.05)
        assert validation_cache.pop(token) is None

    def test_expired_entries_are_evicted_from_the_dict_not_just_unreadable(self, monkeypatch):
        """A leaked/abandoned batch (validated, never committed) must not
        accumulate in memory forever -- eviction has to actually free the
        entry, not just make pop() refuse to return it."""
        monkeypatch.setattr(validation_cache, "_BATCH_TTL_SECONDS", 0.01)
        token = validation_cache.store(SAMPLE)
        time.sleep(0.05)
        # Storing something new triggers the eviction sweep.
        validation_cache.store(SAMPLE)
        assert token not in validation_cache._BATCHES
