import time

import pytest

from ml import retrain_job


@pytest.fixture(autouse=True)
def _clean_jobs(monkeypatch):
    monkeypatch.setattr(retrain_job, "_JOBS", {})
    monkeypatch.setattr(retrain_job, "_ACTIVE", None)


def wait_for(job_id, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = retrain_job.get_job(job_id)
        if job and job["status"] in {"done", "failed"}:
            return job
        time.sleep(0.01)
    raise AssertionError("job did not finish in time")


METRICS = {
    "overall": {
        "target_n": {"mae": 5.0, "r2": 0.8, "dummy_mae": 20.0},
        "target_p2o5": {"mae": 4.0, "r2": 0.7, "dummy_mae": 15.0},
        "target_k2o": {"mae": 6.0, "r2": 0.6, "dummy_mae": 18.0},
    }
}


class TestBuildComparison:
    def test_flags_every_target_that_beats_the_mean_baseline(self):
        comparison = retrain_job.build_comparison(METRICS, None)
        assert comparison["beats_baseline"] is True
        assert comparison["targets"]["target_n"]["beats_baseline"] is True

    def test_flags_failure_to_beat_the_baseline(self):
        weak = {"overall": {"target_n": {"mae": 25.0, "r2": 0.1, "dummy_mae": 20.0}}}
        comparison = retrain_job.build_comparison(weak, None)
        assert comparison["beats_baseline"] is False
        assert comparison["targets"]["target_n"]["beats_baseline"] is False

    def test_reports_improvement_against_the_live_model(self):
        current = {
            "overall": {
                "target_n": {"mae": 7.0, "r2": 0.7, "dummy_mae": 20.0},
                "target_p2o5": {"mae": 3.0, "r2": 0.75, "dummy_mae": 15.0},
                "target_k2o": {"mae": 6.0, "r2": 0.6, "dummy_mae": 18.0},
            }
        }
        comparison = retrain_job.build_comparison(METRICS, current)

        assert comparison["targets"]["target_n"]["mae_delta"] == pytest.approx(-2.0)
        assert comparison["targets"]["target_n"]["improved"] is True
        assert comparison["targets"]["target_p2o5"]["improved"] is False
        assert comparison["regressed"] is True

    def test_no_current_metrics_means_no_regression_claim(self):
        comparison = retrain_job.build_comparison(METRICS, None)
        assert comparison["regressed"] is False
        assert comparison["targets"]["target_n"]["mae_delta"] is None


class TestJobLifecycle:
    def test_successful_run_records_version_metrics_and_backup(self, monkeypatch):
        calls = {}

        monkeypatch.setattr(retrain_job, "_load_training_frame", lambda: "FRAME")
        monkeypatch.setattr(retrain_job, "_train", lambda frame: {
            "version": "2026-08-18-101112", "trained_at": "2026-08-18T10:11:12+00:00",
            "n_rows": 3076, "metrics": METRICS,
        })
        monkeypatch.setattr(retrain_job, "_record_version",
                            lambda **kwargs: calls.setdefault("recorded", kwargs))
        monkeypatch.setattr(retrain_job, "_upload_version", lambda version: True)
        monkeypatch.setattr(retrain_job, "_current_metrics", lambda: None)

        job = wait_for(retrain_job.start_job())

        assert job["status"] == "done"
        assert job["version"] == "2026-08-18-101112"
        assert job["backed_up"] is True
        assert job["comparison"]["beats_baseline"] is True
        assert calls["recorded"]["row_count"] == 3076

    def test_training_failure_is_captured_not_raised(self, monkeypatch):
        monkeypatch.setattr(retrain_job, "_load_training_frame", lambda: "FRAME")

        def explode(_frame):
            raise RuntimeError("not enough rows")

        monkeypatch.setattr(retrain_job, "_train", explode)

        job = wait_for(retrain_job.start_job())
        assert job["status"] == "failed"
        assert "not enough rows" in job["error"]

    def test_a_failed_backup_still_completes_the_job(self, monkeypatch):
        monkeypatch.setattr(retrain_job, "_load_training_frame", lambda: "FRAME")
        monkeypatch.setattr(retrain_job, "_train", lambda frame: {
            "version": "v1", "trained_at": "t", "n_rows": 10, "metrics": METRICS,
        })
        monkeypatch.setattr(retrain_job, "_record_version", lambda **kwargs: None)
        monkeypatch.setattr(retrain_job, "_upload_version", lambda version: False)
        monkeypatch.setattr(retrain_job, "_current_metrics", lambda: None)

        job = wait_for(retrain_job.start_job())
        assert job["status"] == "done"
        assert job["backed_up"] is False

    def test_refuses_a_second_concurrent_run(self, monkeypatch):
        release = __import__("threading").Event()
        monkeypatch.setattr(retrain_job, "_load_training_frame", lambda: "FRAME")
        monkeypatch.setattr(retrain_job, "_train",
                            lambda frame: (release.wait(5), {
                                "version": "v1", "trained_at": "t",
                                "n_rows": 1, "metrics": METRICS,
                            })[1])
        monkeypatch.setattr(retrain_job, "_record_version", lambda **kwargs: None)
        monkeypatch.setattr(retrain_job, "_upload_version", lambda version: True)
        monkeypatch.setattr(retrain_job, "_current_metrics", lambda: None)

        first = retrain_job.start_job()
        with pytest.raises(retrain_job.RetrainInProgress):
            retrain_job.start_job()
        release.set()
        wait_for(first)

    def test_unknown_job_id_returns_none(self):
        assert retrain_job.get_job("nope") is None
