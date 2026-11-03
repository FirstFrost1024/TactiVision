"""Downloader integrity checks use controlled bytes and never contact the network."""

import hashlib
import importlib.util
import io
from pathlib import Path
import sys

import pytest


spec = importlib.util.spec_from_file_location(
    "download_models", Path(__file__).resolve().parents[1] / "scripts" / "download_models.py")
downloader = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = downloader
spec.loader.exec_module(downloader)


@pytest.fixture
def asset(monkeypatch):
    data = b"controlled test model bytes"
    item = downloader.ModelAsset(len(data), hashlib.sha256(data).hexdigest())
    monkeypatch.setitem(downloader.MODELS, "testmodel", item)
    return data


def test_successful_download_is_verified_and_partial_removed(tmp_path, monkeypatch, asset):
    monkeypatch.setattr(downloader, "urlopen", lambda *a, **k: io.BytesIO(asset))
    path = downloader.download("testmodel", tmp_path)
    assert path.read_bytes() == asset
    assert not list(tmp_path.glob("*.part"))


def test_existing_correct_model_is_skipped_offline(tmp_path, monkeypatch, asset):
    (tmp_path / "testmodel.pt").write_bytes(asset)
    def forbidden(*a, **k):
        pytest.fail("An existing valid model must not trigger any network access")
    monkeypatch.setattr(downloader, "urlopen", forbidden)
    assert downloader.download("testmodel", tmp_path).read_bytes() == asset


def test_wrong_digest_preserves_existing_file_even_with_force(tmp_path, monkeypatch, asset):
    target = tmp_path / "testmodel.pt"
    target.write_bytes(b"existing model")
    monkeypatch.setattr(downloader, "urlopen", lambda *a, **k: io.BytesIO(b"X" * len(asset)))
    with pytest.raises(RuntimeError, match="SHA-256"):
        downloader.download("testmodel", tmp_path, retries=1, force=True)
    assert target.read_bytes() == b"existing model"
    assert not list(tmp_path.glob("*.part"))


def test_interrupted_transfer_leaves_no_partial_model(tmp_path, monkeypatch, asset):
    class Broken(io.BytesIO):
        def read(self, *args):
            if self.tell():
                raise OSError("connection interrupted")
            return super().read(5)
    monkeypatch.setattr(downloader, "urlopen", lambda *a, **k: Broken(asset))
    with pytest.raises(RuntimeError, match="interrupted"):
        downloader.download("testmodel", tmp_path, retries=1)
    assert list(tmp_path.iterdir()) == []


def test_dry_run_has_no_network_or_file_writes(tmp_path, monkeypatch, capsys):
    def forbidden(*a, **k):
        pytest.fail("Dry run must not contact the network")
    monkeypatch.setattr(downloader, "urlopen", forbidden)
    target = tmp_path / "not-created"
    downloader.main(["--dry-run", "--output", str(target)])
    assert "yolo26x.pt" in capsys.readouterr().out
    assert not target.exists()


def test_failed_transfer_retries_before_publishing_model(tmp_path, monkeypatch, asset):
    responses = iter([asset[:4], asset])
    monkeypatch.setattr(downloader, "urlopen", lambda *a, **k: io.BytesIO(next(responses)))
    monkeypatch.setattr(downloader.time, "sleep", lambda seconds: None)
    result = downloader.download("testmodel", tmp_path, retries=2)
    assert result.read_bytes() == asset
    assert sorted(p.name for p in tmp_path.iterdir()) == ["testmodel.pt"]
