import threading

import pytest

from ml import model_state
from ml.artifacts_store import ARTIFACT_FILES


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


def make_version_dir(root, name, files=ARTIFACT_FILES):
    """A version directory only counts as available once every artifact file is
    there -- the contents are never parsed by available_versions()."""
    directory = root / name
    directory.mkdir()
    for filename in files:
        (directory / filename).write_bytes(b"x")
    return directory


class TestAvailableVersions:
    def test_lists_only_directories_sorted_ascending(self, tmp_path):
        make_version_dir(tmp_path, "2026-08-01-000000")
        make_version_dir(tmp_path, "2026-08-03-000000")
        make_version_dir(tmp_path, "2026-08-02-000000")
        (tmp_path / "notes.txt").write_text("ignored")

        assert model_state.available_versions(str(tmp_path)) == [
            "2026-08-01-000000", "2026-08-02-000000", "2026-08-03-000000",
        ]

    def test_missing_root_returns_empty(self, tmp_path):
        assert model_state.available_versions(str(tmp_path / "nope")) == []

    def test_a_half_written_directory_is_not_a_version(self, tmp_path):
        """ml/train.py makes its output directory first and writes
        metadata.json last, so a killed run leaves this behind. Counting it
        would make the next boot load a metadata.json that never existed."""
        make_version_dir(tmp_path, "2026-08-01-000000")
        (tmp_path / "2026-08-04-000000").mkdir()  # killed mid-train: empty
        make_version_dir(
            tmp_path, "2026-08-05-000000",
            files=("preprocessor.joblib", "regressor.joblib"),  # no metadata.json
        )

        assert model_state.available_versions(str(tmp_path)) == ["2026-08-01-000000"]

    def test_a_misnamed_directory_is_not_a_version(self, tmp_path):
        """Only ml/train.py's %Y-%m-%d-%H%M%S stamp names a real artifact.
        `.partial` is artifacts_store's staging directory, which survives a
        SIGKILL that its own cleanup cannot cover."""
        make_version_dir(tmp_path, "2026-08-01-000000")
        make_version_dir(tmp_path, "2026-08-02-000000.partial")
        make_version_dir(tmp_path, "backup")
        make_version_dir(tmp_path, "2026-08-03-00000")  # one digit short

        assert model_state.available_versions(str(tmp_path)) == ["2026-08-01-000000"]


class TestIsVersionName:
    @pytest.mark.parametrize("name", ["2026-08-06-055755", "1999-12-31-235959"])
    def test_accepts_the_training_stamp(self, name):
        assert model_state.is_version_name(name)

    @pytest.mark.parametrize("name", [
        "", "latest", "../../etc/passwd", "2026-08-06-055755/..",
        "2026-08-06-055755.partial", "2026-8-6-55755", "2026-08-06-055755\n",
    ])
    def test_rejects_anything_else(self, name):
        assert not model_state.is_version_name(name)


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

    def test_the_newest_artifact_s_encoder_vocabulary_matches_the_code(self):
        """Artifact/code drift, asserted directly instead of by luck.

        verify_bundle()'s probe row is built from CROPS[0] alone, so it only
        notices a vocabulary change when the crop that happens to sort first
        is the one that moved. Today that is "Alugbati", which is why the
        crop-coverage expansion's drift was caught at all -- a crop added
        further down the list would have sailed through and reached a live
        artifact, exactly the failure Task 9 of
        docs/superpowers/plans/2026-08-18-crop-coverage-expansion.md had to
        repair by hand.

        Anchored on the newest complete artifact directory rather than
        model_state.snapshot(): the newest directory is what a retrain just
        produced and what a fresh instance falls back to, so this fails the
        moment someone edits GUIDE_ROWS without retraining. snapshot() is the
        wrong anchor here -- it holds whatever the shared registry currently
        calls live, which is deliberately allowed to be an older, narrower
        artifact (promotion is a human decision), and main.py's predict path
        already refuses crops absent from a bundle's own by_crop metrics."""
        from data.nutrient_recommendation_guide import CANONICAL_CROPS
        from ml.train import ARTIFACT_ROOT

        newest = model_state.available_versions(ARTIFACT_ROOT)[-1]
        bundle = model_state.load_bundle(newest, ARTIFACT_ROOT)
        encoder = bundle["preprocessor"].named_transformers_["nom"]

        assert list(encoder.categories_[0]) == CANONICAL_CROPS
