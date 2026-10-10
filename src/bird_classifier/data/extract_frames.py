"""Extract reproducible random still frames from a directory of videos."""

from __future__ import annotations

import csv
import hashlib
import random
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Final


VIDEO_EXTENSIONS: Final = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}
MANIFEST_FIELDS: Final = ("source_video", "timestamp_seconds", "output_image")


@dataclass(frozen=True)
class ExtractedFrame:
    """The provenance of one image extracted from a source video."""

    source_video: str
    timestamp_seconds: float
    output_image: str


def find_videos(input_dir: Path) -> list[Path]:
    """Return supported video files below ``input_dir`` in a stable order."""
    return sorted(
        (path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS),
        key=lambda path: path.as_posix().casefold(),
    )


def video_duration_seconds(video_path: Path) -> float:
    """Read a video's duration with ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    duration = float(result.stdout.strip())
    if duration <= 0:
        raise ValueError(f"Video has no positive duration: {video_path}")
    return duration


def random_timestamps(duration: float, count: int, seed: int, video_key: str) -> list[float]:
    """Choose stable, distinct timestamps while avoiding the very start and end."""
    digest = hashlib.sha256(f"{seed}:{video_key}".encode("utf-8")).digest()
    video_seed = int.from_bytes(digest[:8], byteorder="big")
    generator = random.Random(video_seed)

    # Avoid metadata/title frames at the start and truncation at the end when possible.
    margin = min(1.0, duration * 0.1)
    start, end = margin, max(margin, duration - margin)
    return sorted(generator.uniform(start, end) for _ in range(count))


def output_path_for(video: Path, input_dir: Path, output_dir: Path, timestamp: float) -> Path:
    """Create a collision-resistant, readable path for an extracted JPEG."""
    relative = video.relative_to(input_dir)
    milliseconds = round(timestamp * 1000)
    return output_dir / relative.parent / f"{relative.stem}_t{milliseconds:010d}ms.jpg"


def extract_frame(video_path: Path, timestamp: float, output_path: Path, overwrite: bool) -> None:
    """Extract one JPEG frame with ffmpeg."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y" if overwrite else "-n"]
    command.extend(["-ss", f"{timestamp:.3f}", "-i", str(video_path), "-frames:v", "1", "-q:v", "2", str(output_path)])
    subprocess.run(command, check=True)


def write_manifest(records: list[ExtractedFrame], manifest_path: Path) -> None:
    """Record the source and timestamp for every exported frame."""
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "source_video": record.source_video,
                    "timestamp_seconds": f"{record.timestamp_seconds:.3f}",
                    "output_image": record.output_image,
                }
            )


def extract_random_frames(
    input_dir: Path,
    output_dir: Path,
    frames_per_video: int = 3,
    seed: int = 42,
    overwrite: bool = False,
) -> list[ExtractedFrame]:
    """Extract a reproducible random sample of frames from every supported video."""
    if frames_per_video < 1:
        raise ValueError("frames_per_video must be at least 1")
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg and ffprobe must be installed and available on PATH")
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input video directory does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[ExtractedFrame] = []
    for video in find_videos(input_dir):
        relative_video = video.relative_to(input_dir).as_posix()
        duration = video_duration_seconds(video)
        for timestamp in random_timestamps(duration, frames_per_video, seed, relative_video):
            output_path = output_path_for(video, input_dir, output_dir, timestamp)
            if overwrite or not output_path.exists():
                extract_frame(video, timestamp, output_path, overwrite)
            records.append(
                ExtractedFrame(
                    source_video=relative_video,
                    timestamp_seconds=timestamp,
                    output_image=output_path.relative_to(output_dir).as_posix(),
                )
            )

    write_manifest(records, output_dir / "frame_manifest.csv")
    return records
