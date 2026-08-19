import threading

import pytest

from ml import model_state


class TestSnapshotAtomicity:
    def test_snapshot_raises_before_anything_is_loaded(self, monkeypatch):
        monkeypatch.setattr(model_state, "_current", None)
        with pytest.raises(RuntimeError, match="No model artifacts loaded"):
            model_state.snapshot()

    def test_snapshot_returns_the_bundle_that_was_set(self, monkeypatch):
        monkeypatch.setattr(model_state, "_current", None)
        bundle = {"version": "v1", "preprocessor": object(),
                  "regressor": object(), "metadata": {}}
        model_state.set_bundle(bundle)
        assert model_state.snapshot()["version"] == "v1"

    def test_a_snapshot_is_never_half_swapped(self, monkeypatch):
        """The whole point of the refactor: a reader must see one coherent
        version, never the old regressor paired with the new version string."""
        monkeypatch.setattr(model_state, "_current", None)
        model_state.set_bundle({"version": "v1", "preprocessor": "p1",
                                "regressor": "r1", "metadata": {}})

        seen = []
        stop = threading.Event()

        def reader():
            while not stop.is_set():
                snapshot = model_state.snapshot()
                seen.append((snapshot["version"], snapshot["preprocessor"],
                             snapshot["regressor"]))

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        for index in range(2, 60):
            model_state.set_bundle({
                "version": f"v{index}", "preprocessor": f"p{index}",
                "regressor": f"r{index}", "metadata": {},
            })
        stop.set()
        thread.join(timeout=5)

        for version, preprocessor, regressor in seen:
            suffix = version[1:]
            assert preprocessor == f"p{suffix}"
            assert regressor == f"r{suffix}"


class TestAvailableVersions:
    def test_lists_only_directories_sorted_ascending(self, tmp_path):
        (tmp_path / "2026-08-01-000000").mkdir()
        (tmp_path / "2026-08-03-000000").mkdir()
        (tmp_path / "2026-08-02-000000").mkdir()
        (tmp_path / "notes.txt").write_text("ignored")

        assert model_state.available_versions(str(tmp_path)) == [
            "2026-08-01-000000", "2026-08-02-000000", "2026-08-03-000000",
        ]

    def test_missing_root_returns_empty(self, tmp_path):
        assert model_state.available_versions(str(tmp_path / "nope")) == []


class TestVerifyBundle:
    def test_rejects_a_bundle_whose_preprocessor_cannot_transform(self):
        class BrokenPreprocessor:
            def transform(self, _frame):
                raise ValueError("categories changed")

        bundle = {"version": "v1", "preprocessor": BrokenPreprocessor(),
                  "regressor": object(), "metadata": {}}
        with pytest.raises(ValueError, match="cannot serve"):
            model_state.verify_bundle(bundle)

    def test_accepts_the_real_currently_loaded_bundle(self):
        model_state.verify_bundle(model_state.snapshot())
