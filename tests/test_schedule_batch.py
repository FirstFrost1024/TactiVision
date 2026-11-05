"""Exercise daily scheduling with a fake clock; never start inference jobs."""

from datetime import datetime, time
import fcntl
import importlib.util
from pathlib import Path
import signal
import subprocess
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest


spec = importlib.util.spec_from_file_location(
    "schedule_batch", Path(__file__).resolve().parents[1] / "scripts" / "schedule_batch.py")
scheduler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scheduler)
ZONE = ZoneInfo("Asia/Shanghai")


@pytest.fixture
def conda_runtime(tmp_path, monkeypatch):
    prefix = tmp_path / "envs" / "tactivision"
    (prefix / "conda-meta").mkdir(parents=True)
    executable = prefix / "bin" / "python"
    monkeypatch.setattr(scheduler, "sys", SimpleNamespace(prefix=str(prefix), executable=str(executable)))
    return executable


def at(day, hour, minute):
    return datetime(2026, 11, day, hour, minute, tzinfo=ZONE)


@pytest.mark.parametrize("hour,minute,active,next_day", [
    (0, 0, False, 5), (8, 29, False, 5), (8, 30, True, 6),
    (16, 29, True, 6), (16, 30, False, 6), (23, 59, False, 6),
])
def test_daily_window_boundaries(hour, minute, active, next_day):
    assert scheduler.window_at(at(5, hour, minute), time(8, 30), time(16, 30)) == (
        active, at(next_day, 8, 30))


class Clock:
    def __init__(self, times):
        self.times = times
        self.index = 0

    def is_set(self):
        return self.index >= len(self.times)

    def now(self):
        return self.times[self.index]

    def wait(self, seconds):
        self.index += 1


class Process:
    pid = 123456789  # Fake process; all signal delivery is mocked.

    def __init__(self, returncode=None):
        self.returncode = returncode
        self.signals = []

    def poll(self):
        return self.returncode

    def send_signal(self, sig):
        self.signals.append(sig)
        self.returncode = -sig

    def wait(self, timeout=None):
        return self.returncode


def run_clock(tmp_path, monkeypatch, times, *, returncode=None):
    args = scheduler.parser().parse_args(["--output", str(tmp_path)])
    clock = Clock(times)
    processes, group_signals = [], []
    def launch(command, **kwargs):
        assert kwargs["start_new_session"] is True
        process = Process(returncode)
        processes.append(process)
        return process
    monkeypatch.setattr(scheduler.os, "killpg", lambda *args: group_signals.append(args))
    scheduler.run_schedule(args, ZONE, ["fake-batch"], clock, now=clock.now, launch=launch)
    return processes, group_signals


def test_stops_at_cutoff_and_resumes_next_morning(tmp_path, monkeypatch):
    processes, groups = run_clock(tmp_path, monkeypatch, [
        at(5, 8, 29), at(5, 8, 30), at(5, 16, 29), at(5, 16, 30),
        at(5, 23, 59), at(6, 8, 29), at(6, 8, 30), at(6, 16, 30),
    ])
    assert len(processes) == 2
    assert all(p.signals == [signal.SIGTERM] for p in processes)
    assert len(groups) == 2


@pytest.mark.parametrize("returncode", [0, 1])
def test_completed_or_failed_batch_waits_until_tomorrow(tmp_path, monkeypatch, returncode):
    processes, _ = run_clock(tmp_path, monkeypatch, [
        at(5, 10, 0), at(5, 10, 1), at(5, 11, 0), at(5, 16, 30), at(6, 8, 30),
    ], returncode=returncode)
    assert len(processes) == 2


def test_manual_shutdown_also_stops_batch(tmp_path, monkeypatch):
    processes, _ = run_clock(tmp_path, monkeypatch, [at(5, 10, 0)])
    assert processes[0].signals == [signal.SIGTERM]


def test_no_batch_started_outside_window(tmp_path, monkeypatch):
    processes, _ = run_clock(tmp_path, monkeypatch, [at(5, 16, 30), at(6, 8, 29)])
    assert not processes


def test_timeout_kills_entire_batch_group(monkeypatch):
    process = Process()
    calls = []
    def wait(timeout=None):
        if timeout is not None:
            raise subprocess.TimeoutExpired("fake-batch", timeout)
        calls.append("reaped")
    process.wait = wait
    monkeypatch.setattr(scheduler.os, "killpg", lambda *args: calls.append(args))
    scheduler.stop_batch(process, .1)
    assert process.signals == [signal.SIGTERM]
    assert calls == [(process.pid, signal.SIGKILL), "reaped"]


def test_dry_run_preserves_corrected_selection_and_creates_nothing(tmp_path, capsys, conda_runtime):
    output = tmp_path / "unused"
    scheduler.main(["--output", str(output), "--dry-run"])
    text = capsys.readouterr().out
    assert "08:30–16:30" in text and "Asia/Shanghai" in text
    assert "--devices 0 1" in text and "--tile-size 512" in text
    assert "--pitch-region keypoints --focus-pitch" in text
    assert "--save-candidates" in text
    assert str(conda_runtime) in text
    assert not output.exists()


def test_non_conda_python_is_rejected_without_creating_outputs(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(scheduler, "sys", SimpleNamespace(prefix=str(tmp_path), executable="/other/python"))
    output = tmp_path / "unused"
    with pytest.raises(SystemExit) as error:
        scheduler.main(["--output", str(output), "--dry-run"])
    assert error.value.code == 2
    assert "conda activate tactivision" in capsys.readouterr().err
    assert not output.exists()


def test_duplicate_scheduler_is_rejected_without_launch(tmp_path, monkeypatch, conda_runtime):
    weights = tmp_path / "fake.pt"
    weights.touch()
    def forbidden(*args, **kwargs):
        pytest.fail("Duplicate scheduler must not start a task")
    monkeypatch.setattr(scheduler, "run_schedule", forbidden)
    with (tmp_path / "schedule.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(SystemExit) as error:
            scheduler.main(["--output", str(tmp_path), "--dataset", str(tmp_path), "--weights", str(weights)])
    assert error.value.code == 2
    assert not (tmp_path / "schedule.pid").exists()


@pytest.mark.parametrize("args", [
    ["--start", "16:30", "--stop", "08:30"], ["--start", "25:00"],
    ["--timezone", "No/SuchZone"], ["--devices", "0", "0"],
    ["--stop-timeout", "nan"], ["--cpu-threads", "0"],
])
def test_invalid_schedule_arguments(args):
    with pytest.raises(SystemExit) as error:
        scheduler.main([*args, "--dry-run"])
    assert error.value.code == 2
