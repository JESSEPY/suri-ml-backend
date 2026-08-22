import json

from ml import artifacts_store


class FakeBucket:
    def __init__(self, store, fail=False):
        self.store = store
        self.fail = fail

    def upload(self, path, data, file_options=None):
        if self.fail:
            raise RuntimeError("storage unavailable")
        self.store[path] = data

    def download(self, path):
        return self.store[path]

    def remove(self, paths):
        if self.fail:
            raise RuntimeError("storage unavailable")
        for path in paths:
            self.store.pop(path, None)

    def list(self, prefix=""):
        # Real Supabase Storage .list(prefix) returns one entry per
        # immediate child (folder or file) under that prefix, not every
        # nested key -- good enough here since this store is always exactly
        # two levels deep (version/filename).
        if prefix == "":
            names = {path.split("/", 1)[0] for path in self.store}
        else:
            depth = prefix.rstrip("/") + "/"
            names = {
                path[len(depth):] for path in self.store if path.startswith(depth)
            }
        return [{"name": name} for name in sorted(names)]


class FakeStorage:
    def __init__(self, store, fail=False):
        self.store = store
        self.fail = fail
        self.buckets = []

    def from_(self, bucket):
        self.buckets.append(bucket)
        return FakeBucket(self.store, self.fail)


class FakeClient:
    def __init__(self, store=None, fail=False):
        self.storage = FakeStorage(store if store is not None else {}, fail)


def write_artifacts(root, version):
    directory = root / version
    directory.mkdir(parents=True)
    (directory / "preprocessor.joblib").write_bytes(b"pre")
    (directory / "regressor.joblib").write_bytes(b"reg")
    (directory / "metadata.json").write_text(json.dumps({"version": version}))
    return directory


class TestUploadVersion:
    def test_uploads_every_artifact_file_under_the_version_prefix(self, tmp_path):
        write_artifacts(tmp_path, "2026-08-18-101112")
        client = FakeClient()

        assert artifacts_store.upload_version(
            "2026-08-18-101112", str(tmp_path), client=client
        ) is True
        assert sorted(client.storage.store) == [
            "2026-08-18-101112/metadata.json",
            "2026-08-18-101112/preprocessor.joblib",
            "2026-08-18-101112/regressor.joblib",
        ]
        assert client.storage.buckets[0] == artifacts_store.BUCKET

    def test_a_storage_outage_is_reported_not_raised(self, tmp_path):
        """The trained artifact is already on local disk and usable; losing the
        off-instance backup must not fail the retrain."""
        write_artifacts(tmp_path, "2026-08-18-101112")
        assert artifacts_store.upload_version(
            "2026-08-18-101112", str(tmp_path), client=FakeClient(fail=True)
        ) is False


class TestDownloadVersion:
    def test_restores_files_to_disk(self, tmp_path):
        store = {
            "v1/preprocessor.joblib": b"pre",
            "v1/regressor.joblib": b"reg",
            "v1/metadata.json": b'{"version": "v1"}',
        }
        assert artifacts_store.download_version(
            "v1", str(tmp_path), client=FakeClient(store=store)
        ) is True
        assert (tmp_path / "v1" / "regressor.joblib").read_bytes() == b"reg"

    def test_missing_version_returns_false(self, tmp_path):
        assert artifacts_store.download_version(
            "nope", str(tmp_path), client=FakeClient(store={})
        ) is False

    def test_a_failed_download_leaves_no_directory_behind(self, tmp_path):
        """The bug this fix closes: a failed restore used to leave an empty,
        metadata.json-less directory that boot_version()'s "newest on disk"
        fallback could mistake for a valid version on the next boot."""
        assert artifacts_store.download_version(
            "nope", str(tmp_path), client=FakeClient(store={})
        ) is False
        assert not (tmp_path / "nope").exists()
        assert not (tmp_path / "nope.partial").exists()


class TestDeleteVersion:
    def test_removes_every_artifact_file_under_the_version_prefix(self):
        store = {
            "v1/preprocessor.joblib": b"pre", "v1/regressor.joblib": b"reg",
            "v1/metadata.json": b"{}",
            "v2/preprocessor.joblib": b"pre2",  # a different version, must survive
        }
        client = FakeClient(store=store)

        assert artifacts_store.delete_version("v1", client=client) is True
        assert sorted(store) == ["v2/preprocessor.joblib"]

    def test_a_storage_outage_is_reported_not_raised(self):
        store = {"v1/preprocessor.joblib": b"pre"}
        assert artifacts_store.delete_version(
            "v1", client=FakeClient(store=store, fail=True)
        ) is False


class TestDeleteAllVersions:
    def test_removes_every_object_in_the_bucket(self):
        store = {
            "v1/preprocessor.joblib": b"pre", "v1/regressor.joblib": b"reg",
            "v1/metadata.json": b"{}",
            "v2/preprocessor.joblib": b"pre2", "v2/regressor.joblib": b"reg2",
            "v2/metadata.json": b"{}",
        }
        client = FakeClient(store=store)

        assert artifacts_store.delete_all_versions(client=client) is True
        assert store == {}

    def test_an_already_empty_bucket_is_a_no_op_success(self):
        assert artifacts_store.delete_all_versions(client=FakeClient(store={})) is True

    def test_a_storage_outage_is_reported_not_raised(self):
        store = {"v1/preprocessor.joblib": b"pre"}
        assert artifacts_store.delete_all_versions(
            client=FakeClient(store=store, fail=True)
        ) is False
