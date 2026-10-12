"""Bounded local catalog upload processing for video and text sidecars."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any
from src.observability import span

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov"}
TEXT_EXTENSIONS = {".json", ".srt", ".vtt"}
MAX_UPLOAD_BYTES = int(os.getenv("CATALOG_MAX_UPLOAD_BYTES", str(2 * 1024**3)))
UPLOAD_ROOT = Path(os.getenv("CATALOG_UPLOAD_DIR", "data/catalog"))


def _safe_filename(filename: str) -> str:
    name = Path(filename).name
    if not name or name in {".", ".."}:
        raise ValueError("A valid filename is required")
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)[:180]


def _ffmpeg_hls(source: Path, output: Path) -> dict[str, Any]:
    ffmpeg = os.getenv("FFMPEG_BINARY", "ffmpeg")
    if shutil.which(ffmpeg) is None:
        raise RuntimeError("FFmpeg is required to process video uploads")
    ffprobe = os.getenv("FFPROBE_BINARY", "ffprobe")
    if shutil.which(ffprobe) is None:
        raise RuntimeError("FFprobe is required to inspect video uploads")
    probe = subprocess.run([ffprobe, "-v", "error", "-show_entries",
                            "stream=height,codec_type", "-of", "json", str(source)],
                           capture_output=True, text=True, timeout=30)
    if probe.returncode:
        raise RuntimeError("FFprobe could not inspect this video: " + probe.stderr[-800:])
    try:
        streams = json.loads(probe.stdout).get("streams", [])
        source_height = int(next(stream["height"] for stream in streams if stream.get("codec_type") == "video"))
        has_audio = any(stream.get("codec_type") == "audio" for stream in streams)
    except (ValueError, IndexError, KeyError, StopIteration, json.JSONDecodeError) as exc:
        raise ValueError("The uploaded file does not contain a readable video stream") from exc
    heights = [height for height in (2160, 1080, 720, 360) if height <= source_height]
    if not heights:
        heights = [source_height]
    output.mkdir(parents=True, exist_ok=True)
    # Build the source-compatible 4K/1080p/720p ladder plus a 360p fallback.
    split_labels = "".join(f"[v{index}]" for index in range(len(heights)))
    filters = [f"[0:v]split={len(heights)}{split_labels}"]
    args = [ffmpeg, "-hide_banner", "-y", "-i", str(source), "-filter_complex"]
    maps: list[str] = []
    stream_maps: list[str] = []
    for index, height in enumerate(heights):
        filters.append(f"[v{index}]scale=-2:min({height}\\,ih)[vout{index}]")
        maps.extend(["-map", f"[vout{index}]"])
        if has_audio:
            maps.extend(["-map", "0:a:0"])
            stream_maps.append(f"v:{index},a:{index},name:{height}p")
        else:
            stream_maps.append(f"v:{index},name:{height}p")
    args.extend([";".join(filters), *maps])
    args.extend(["-c:v", "libx264"])
    if has_audio:
        args.extend(["-c:a", "aac"])
    for index, height in enumerate(heights):
        bitrate = {2160: "12000k", 1080: "5000k", 720: "2800k", 360: "900k"}.get(
            height, f"{max(450, height * 3)}k")
        args.extend([f"-b:v:{index}", bitrate])
        if has_audio:
            args.extend([f"-b:a:{index}", "128k"])
    args.extend(["-f", "hls", "-hls_time", "6", "-hls_playlist_type", "vod",
                 "-master_pl_name", "master.m3u8", "-var_stream_map", " ".join(stream_maps),
                 str(output / "stream_%v.m3u8")])
    with span("catalog.ffmpeg_transcode"):
        result = subprocess.run(args, capture_output=True, text=True,
                                timeout=int(os.getenv("FFMPEG_TIMEOUT_SECONDS", "7200")))
    if result.returncode:
        raise RuntimeError("FFmpeg could not transcode this video: " + result.stderr[-1200:])
    return {"format": "hls", "manifest": "master.m3u8", "renditions": [f"{height}p" for height in heights]}


def _parse_text(path: Path) -> tuple[str, dict[str, Any]]:
    ext = path.suffix.lower()
    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    metadata: dict[str, Any] = {}
    if ext == ".json":
        data = json.loads(raw)
        if not isinstance(data, (dict, list)):
            raise ValueError("JSON catalog metadata must be an object or array")
        metadata = data if isinstance(data, dict) else {"items": data}
        text = "\n".join(str(v) for v in metadata.values() if isinstance(v, (str, int, float)))
    else:
        text = re.sub(r"<[^>]+>", " ", raw)
        text = re.sub(r"(?m)^\s*(?:\d+|\d{2}:\d{2}:\d{2}[,.]\d{3}\s+-->.*|WEBVTT)\s*$", "", text)
        text = re.sub(r"\s+", " ", text).strip()
    return text, metadata


def _index_qdrant(text: str, asset_id: str, metadata: dict[str, Any]) -> bool:
    url = os.getenv("QDRANT_URL")
    if not url:
        return False
    if not text.strip():
        return False
    try:
        from qdrant_client import QdrantClient, models
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Install Qdrant and sentence-transformers dependencies to index text uploads") from exc
    model = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    vector = model.encode(text[:20000], normalize_embeddings=True).tolist()
    collection = os.getenv("QDRANT_COLLECTION", "netflix_catalog")
    client = QdrantClient(url=url, api_key=os.getenv("QDRANT_API_KEY"))
    if not client.collection_exists(collection):
        client.create_collection(collection, vectors_config=models.VectorParams(size=len(vector), distance=models.Distance.COSINE))
    client.upsert(collection, points=[models.PointStruct(id=asset_id, vector=vector,
                 payload={"text": text[:20000], "metadata": metadata})])
    return True


def process_saved_upload(filename: str, saved_path: Path) -> dict[str, Any]:
    """Transcode a saved video or parse/index a text asset."""
    ext = Path(filename).suffix.lower()
    asset_id = saved_path.parent.name
    if ext in VIDEO_EXTENSIONS:
        return {"asset_id": asset_id, "asset_type": "video", **_ffmpeg_hls(saved_path, saved_path.parent / "hls")}
    if ext in TEXT_EXTENSIONS:
        text, metadata = _parse_text(saved_path)
        indexed = _index_qdrant(text, asset_id, metadata)
        (saved_path.parent / "parsed.json").write_text(json.dumps({"text": text, "metadata": metadata}, ensure_ascii=False), encoding="utf-8")
        return {"asset_id": asset_id, "asset_type": "text", "characters": len(text), "indexed": indexed}
    raise ValueError("Supported uploads: MP4, MKV, MOV, JSON, SRT, VTT")


def new_asset_destination(filename: str) -> tuple[str, Path]:
    asset_id = str(uuid.uuid4())
    directory = UPLOAD_ROOT / asset_id
    directory.mkdir(parents=True, exist_ok=False)
    return asset_id, directory / _safe_filename(filename)
