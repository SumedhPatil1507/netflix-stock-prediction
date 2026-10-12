import json
import subprocess
from types import SimpleNamespace

import pytest

from src.catalog import asset_pipeline


def test_text_asset_is_parsed_and_indexed_when_catalog_is_configured(tmp_path, monkeypatch):
    source = tmp_path / "catalog.json"
    source.write_text(json.dumps({"title": "Night Signal", "genre": "Thriller"}), encoding="utf-8")
    monkeypatch.setattr(asset_pipeline, "_index_qdrant", lambda text, asset_id, metadata: True)

    result = asset_pipeline.process_saved_upload("catalog.json", source)

    assert result["asset_type"] == "text"
    assert result["indexed"] is True
    parsed = json.loads((tmp_path / "parsed.json").read_text(encoding="utf-8"))
    assert parsed["metadata"]["title"] == "Night Signal"


def test_invalid_json_upload_fails_without_marking_asset_ready(tmp_path):
    source = tmp_path / "catalog.json"
    source.write_text("{broken", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        asset_pipeline.process_saved_upload("catalog.json", source)


def test_ffmpeg_builds_only_source_compatible_hls_renditions(tmp_path, monkeypatch):
    monkeypatch.setattr(asset_pipeline.shutil, "which", lambda _binary: "/usr/bin/tool")
    calls = []

    def run(args, **_kwargs):
        calls.append(args)
        if args[0] == "ffprobe":
            return SimpleNamespace(returncode=0, stdout=json.dumps({"streams": [
                {"codec_type": "video", "height": 1080}, {"codec_type": "audio"}]}), stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    result = asset_pipeline._ffmpeg_hls(tmp_path / "input.mp4", tmp_path / "hls")

    assert result["renditions"] == ["1080p", "720p", "360p"]
    assert result["manifest"] == "master.m3u8"
    assert "name:1080p" in calls[1][calls[1].index("-var_stream_map") + 1]
    assert (tmp_path / "hls").is_dir()


def test_ffmpeg_failure_is_reported_with_bounded_stderr(tmp_path, monkeypatch):
    monkeypatch.setattr(asset_pipeline.shutil, "which", lambda _binary: "/usr/bin/tool")

    def run(args, **_kwargs):
        if args[0] == "ffprobe":
            return SimpleNamespace(returncode=0, stdout=json.dumps({"streams": [
                {"codec_type": "video", "height": 720}]}), stderr="")
        return SimpleNamespace(returncode=1, stdout="", stderr="codec failure")

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(RuntimeError, match="FFmpeg could not transcode"):
        asset_pipeline._ffmpeg_hls(tmp_path / "input.mp4", tmp_path / "hls")
