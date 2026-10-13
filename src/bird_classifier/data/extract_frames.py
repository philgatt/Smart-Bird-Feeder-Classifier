"""Extract reproducible random still frames from a directory of videos."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Final


VIDEO_EXTENSIONS: Final = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}
MANIFEST_FIELDS: Final = ("source_video", "timestamp_seconds", "output_image")
DEFAULT_MIN_FRAME_GAP: Final = 15


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


def video_metadata(video_path: Path) -> tuple[float, float]:
    """Read a video's duration and average frames per second with one ffprobe call."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "format=duration:stream=avg_frame_rate",
            "-of",
            "json",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    metadata = json.loads(result.stdout)
    duration = float(metadata["format"]["duration"])
    if duration <= 0:
        raise ValueError(f"Video has no positive duration: {video_path}")
    stream = metadata["streams"][0]
    numerator, denominator = stream["avg_frame_rate"].split("/", maxsplit=1)
    frame_rate = float(numerator) / float(denominator)
    if frame_rate <= 0:
        raise ValueError(f"Video has no positive frame rate: {video_path}")
    return duration, frame_rate


def random_frame_indices(
    frame_count: int,
    count: int,
    min_frame_gap: int,
    seed: int,
    video_key: str,
) -> list[int]:
    """Choose stable random frame indices with at least ``min_frame_gap`` between them."""
    if frame_count < 1:
        return []
    if count < 1:
        raise ValueError("count must be at least 1")
    if min_frame_gap < 1:
        raise ValueError("min_frame_gap must be at least 1")

    digest = hashlib.sha256(f"{seed}:{video_key}".encode("utf-8")).digest()
    video_seed = int.from_bytes(digest[:8], byteorder="big")
    generator = random.Random(video_seed)

    # Avoid title frames and end-of-video truncation when there are enough frames.
    margin = min(30, max(0, frame_count // 10))
    first_frame, last_frame = margin, frame_count - margin - 1
    usable_frames = last_frame - first_frame + 1
    if usable_frames < 1:
        first_frame, last_frame, usable_frames = 0, frame_count - 1, frame_count

    # For short videos, return fewer frames rather than breaking the gap guarantee.
    sample_count = min(count, 1 + (usable_frames - 1) // min_frame_gap)
    compressed_length = usable_frames - (sample_count - 1) * (min_frame_gap - 1)
    compressed_indices = sorted(generator.sample(range(compressed_length), sample_count))
    return [first_frame + index + position * (min_frame_gap - 1) for position, index in enumerate(compressed_indices)]


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
    frames_per_video: int = 8,
    min_frame_gap: int = DEFAULT_MIN_FRAME_GAP,
    seed: int = 42,
    overwrite: bool = False,
) -> list[ExtractedFrame]:
    """Extract a reproducible random sample of frames from every supported video."""
    if frames_per_video < 1:
        raise ValueError("frames_per_video must be at least 1")
    if min_frame_gap < 1:
        raise ValueError("min_frame_gap must be at least 1")
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg and ffprobe must be installed and available on PATH")
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input video directory does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[ExtractedFrame] = []
    for video in find_videos(input_dir):
        relative_video = video.relative_to(input_dir).as_posix()
        duration, frame_rate = video_metadata(video)
        frame_count = max(1, round(duration * frame_rate))
        frame_indices = random_frame_indices(
            frame_count, frames_per_video, min_frame_gap, seed, relative_video
        )
        for frame_index in frame_indices:
            timestamp = frame_index / frame_rate
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
