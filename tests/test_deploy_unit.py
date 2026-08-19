"""The systemd unit file is part of the contract, not just documentation.

Promotion hot-swaps the model in one process's memory (ml/model_state.py) and
retrain job state lives in a module-level dict (ml/retrain_job.py). Both are
correct only under a single uvicorn worker. Until now that requirement was
carried by a comment above ExecStart, which nothing checks -- someone adding
--workers 4 to squeeze more throughput out of the instance would get a service
that promotes a model into one worker and silently keeps serving the old one
from the other three, with no way to tell which answered a given request.
"""

import re
from pathlib import Path

import pytest

UNIT_FILE = Path(__file__).resolve().parent.parent / "deploy" / "suri-ml.service"


def exec_start() -> str:
    for line in UNIT_FILE.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("ExecStart="):
            return stripped
    raise AssertionError(f"{UNIT_FILE} has no ExecStart= line")


class TestSingleWorkerIsEnforced:
    def test_the_unit_file_exists_with_an_exec_start(self):
        assert exec_start().startswith("ExecStart=")

    def test_exec_start_never_asks_for_more_than_one_worker(self):
        command = exec_start()
        for count in re.findall(r"--workers[=\s]+(\d+)", command):
            assert int(count) == 1, (
                f"ExecStart requests {count} uvicorn workers. Promotion hot-swaps "
                "the model in a single process's memory; more than one worker "
                "would leave the others serving the old model."
            )

    def test_exec_start_does_not_enable_reload(self):
        assert "--reload" not in exec_start(), (
            "--reload restarts the process on file changes, discarding the "
            "in-memory promoted model and any running retrain job's state."
        )

    def test_the_reason_is_still_written_down_next_to_the_setting(self):
        """The test catches the violation; the comment explains it to whoever
        was about to make it."""
        assert "--workers" in UNIT_FILE.read_text()


class TestTheTestWouldActuallyCatchAViolation:
    """A guard that cannot fail is not a guard -- pin the parsing itself."""

    @pytest.mark.parametrize("command", [
        "ExecStart=/x/uvicorn main:app --workers 4",
        "ExecStart=/x/uvicorn main:app --workers=4",
        "ExecStart=/x/uvicorn main:app --port 8001 --workers 2 --host 0.0.0.0",
    ])
    def test_multi_worker_command_lines_are_detected(self, command):
        counts = [int(n) for n in re.findall(r"--workers[=\s]+(\d+)", command)]
        assert counts and max(counts) > 1

    def test_the_real_command_line_is_detected_as_single_worker(self):
        assert [int(n) for n in re.findall(r"--workers[=\s]+(\d+)", exec_start())] == []
