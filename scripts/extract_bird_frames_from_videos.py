
#python scripts\extract_bird_frames_from_videos.py --batch-size 256 --decode-workers 8
"""Extract at most one high-confidence bird frame from each video.

The binary Bird/No Bird classifier is sampled at deterministic positions: the
video midpoint, midpoint of its first half, midpoint of its latter half, and
the midpoint of its first quarter. The first frame with Bird probability at or
above the configured threshold is saved; videos without such a frame are
recorded as skipped.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import transforms  # noqa: E402

from bird_classifier.models.mobilenet_v3_small import create_mobilenet_v3_small  # noqa: E402

DEFAULT_SOURCE_ROOT = Path("F:/VogelVideos")
DEFAULT_OUTPUT_ROOT = Path("F:/video vogel")
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "synthetic_bird_presence_mobilenet_v3_small"
    / "bird_presence_frame_selection_mobilenet_v3_small.pt"
)
VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".wmv"}
SAMPLE_POSITIONS = (
    (0.50, "middle"),
    (0.25, "first_half_middle"),
    (0.75, "latter_half_middle"),
    (0.125, "first_quarter_middle"),
)
DEFAULT_THRESHOLD = 0.80
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Minimum Bird probability to save a frame (default: 0.80).",
    )
    parser.add_argument("--recursive", action="store_true", help="Also search video subfolders.")
    parser.add_argument(
        "--max-images",
        type=int,
        help="Optional limit for extracted selected_*.jpg frames; omit to process every video.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional random-order seed. Omit it for a different random order on each run.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Videos evaluated together in one GPU inference batch (default: 128).",
    )
    parser.add_argument(
        "--decode-workers",
        type=int,
        default=8,
        help="Concurrent video decoders used to prepare each batch (default: 8).",
    )
    parser.add_argument(
        "--decoder",
        choices=("ffmpeg-cuda", "opencv"),
        default="ffmpeg-cuda",
        help=(
            "Frame decoder to use. ffmpeg-cuda uses NVIDIA NVDEC when supported by the "
            "video codec (default); opencv keeps the previous CPU decoder."
        ),
    )
    parser.add_argument(
        "--ffmpeg-path",
        type=Path,
        help="Optional path to ffmpeg.exe. Normally it is found in the active Conda environment.",
    )
    parser.add_argument(
        "--ffprobe-path",
        type=Path,
        help="Optional path to ffprobe.exe. Normally it is found next to ffmpeg.exe.",
    )
    return parser.parse_args()


def load_model(
    checkpoint_path: Path,
) -> tuple[torch.nn.Module, int, torch.device, transforms.Compose]:
    """Load the binary MobileNetV3-Small checkpoint and its preprocessing settings."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Binary frame-selection checkpoint not found: {checkpoint_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_names = tuple(checkpoint["class_names"])
    if checkpoint.get("architecture") != "mobilenet-v3-small" or "Bird" not in class_names:
        raise ValueError(
            "Checkpoint is not the expected binary MobileNetV3-Small Bird/No Bird model"
        )
    image_size = int(checkpoint.get("image_size", 192))
    normalization = checkpoint.get("normalization", {})
    mean = tuple(normalization.get("mean", IMAGENET_MEAN))
    standard_deviation = tuple(normalization.get("std", IMAGENET_STD))
    model = create_mobilenet_v3_small(len(class_names), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, standard_deviation),
        ]
    )
    return model, class_names.index("Bird"), device, transform


def bird_probabilities(
    frames: list,
    model: torch.nn.Module,
    bird_index: int,
    device: torch.device,
    transform: transforms.Compose,
) -> list[float]:
    """Return Bird probabilities for many OpenCV BGR frames in one GPU call."""
    image_tensors = [
        transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))) for frame in frames
    ]
    tensor = torch.stack(image_tensors).to(device, non_blocking=True)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"),
    ):
        probabilities = model(tensor).softmax(dim=1)[:, bird_index]
    return probabilities.float().cpu().tolist()


def output_path_for(video_path: Path, source_root: Path, output_root: Path) -> Path:
    """Create a stable, collision-resistant output filename without mirroring folders."""
    relative_path = video_path.relative_to(source_root).as_posix()
    source_hash = hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:10]
    return output_root / f"selected_{video_path.stem}_{source_hash}.jpg"


def selected_frame_count(output_root: Path) -> int:
    """Count only frames produced by this selector, not unrelated images in the folder."""
    return sum(1 for _ in output_root.glob("selected_*.jpg"))


@dataclass
class VideoState:
    """One video being evaluated across the four ordered candidate positions."""

    path: Path
    destination: Path
    attempts: list[str] = field(default_factory=list)
    terminal_status: str | None = None
    duration_seconds: float | None = None
    frames_per_second: float = 0.0


@dataclass
class FrameCandidate:
    """A decoded video frame waiting for batch inference."""

    state: VideoState
    frame: object
    sample_name: str
    frame_index: int
    frames_per_second: float


def read_candidate_opencv(
    state: VideoState, fraction: float, sample_name: str
) -> FrameCandidate | None:
    """Decode one candidate frame, retaining the state for later rounds on seek failure."""
    video_path = state.path
    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        video.release()
        state.terminal_status = "unreadable_video"
        state.attempts.append(f"{sample_name}:unreadable_video")
        return None
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_per_second = float(video.get(cv2.CAP_PROP_FPS) or 0.0)
    if frame_count < 1:
        video.release()
        state.terminal_status = "missing_frame_count"
        state.attempts.append(f"{sample_name}:missing_frame_count")
        return None
    try:
        frame_index = min(frame_count - 1, round((frame_count - 1) * fraction))
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = video.read()
        if not success:
            state.attempts.append(f"{sample_name}:unreadable")
            return None
        return FrameCandidate(state, frame, sample_name, frame_index, frames_per_second)
    finally:
        video.release()


def resolve_ffmpeg_tools(
    ffmpeg_path: Path | None, ffprobe_path: Path | None
) -> tuple[Path, Path]:
    """Find the Conda-installed FFmpeg tools, without requiring them on PATH."""
    if ffmpeg_path is None:
        candidates = [
            Path(sys.prefix) / "Library" / "bin" / "ffmpeg.exe",
            Path(sys.prefix) / "bin" / "ffmpeg",
        ]
        located = shutil.which("ffmpeg")
        if located:
            candidates.append(Path(located))
        ffmpeg_path = next((path for path in candidates if path.is_file()), None)
    if ffmpeg_path is None or not ffmpeg_path.is_file():
        raise FileNotFoundError(
            "FFmpeg was not found. Install it in this environment or use --ffmpeg-path. "
            "Alternatively run with --decoder opencv."
        )
    if ffprobe_path is None:
        suffix = ".exe" if ffmpeg_path.suffix.lower() == ".exe" else ""
        ffprobe_path = ffmpeg_path.with_name(f"ffprobe{suffix}")
    if not ffprobe_path.is_file():
        raise FileNotFoundError(
            f"FFprobe was not found next to FFmpeg: {ffprobe_path}. Use --ffprobe-path."
        )
    return ffmpeg_path.resolve(), ffprobe_path.resolve()


def media_metadata(state: VideoState, ffprobe_path: Path) -> bool:
    """Read duration/FPS once; FFmpeg needs duration to seek by a fractional position."""
    if state.duration_seconds is not None:
        return True
    command = [
        str(ffprobe_path),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "format=duration:stream=avg_frame_rate",
        "-of",
        "default=noprint_wrappers=1",
        str(state.path),
    ]
    try:
        probe = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
    except (OSError, subprocess.TimeoutExpired):
        state.terminal_status = "unreadable_video"
        state.attempts.append("ffprobe:unreadable_video")
        return False
    if probe.returncode != 0:
        state.terminal_status = "unreadable_video"
        state.attempts.append("ffprobe:unreadable_video")
        return False
    values = dict(
        line.split("=", maxsplit=1) for line in probe.stdout.splitlines() if "=" in line
    )
    try:
        duration = float(values["duration"])
        numerator, denominator = values.get("avg_frame_rate", "0/1").split("/", maxsplit=1)
        frames_per_second = float(numerator) / float(denominator) if float(denominator) else 0.0
    except (KeyError, ValueError, ZeroDivisionError):
        state.terminal_status = "missing_video_metadata"
        state.attempts.append("ffprobe:missing_video_metadata")
        return False
    if duration <= 0:
        state.terminal_status = "missing_video_metadata"
        state.attempts.append("ffprobe:missing_video_metadata")
        return False
    state.duration_seconds = duration
    state.frames_per_second = frames_per_second
    return True


def read_candidate_ffmpeg_cuda(
    state: VideoState,
    fraction: float,
    sample_name: str,
    *,
    ffmpeg_path: Path,
    ffprobe_path: Path,
) -> FrameCandidate | None:
    """Decode one frame with FFmpeg/NVDEC and return it as an OpenCV BGR image."""
    if not media_metadata(state, ffprobe_path):
        return None
    assert state.duration_seconds is not None
    timestamp_seconds = max(0.0, min(state.duration_seconds - 0.001, state.duration_seconds * fraction))
    command = [
        str(ffmpeg_path),
        "-v",
        "error",
        "-nostdin",
        "-hwaccel",
        "cuda",
        "-ss",
        f"{timestamp_seconds:.6f}",
        "-i",
        str(state.path),
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-c:v",
        "mjpeg",
        "pipe:1",
    ]
    try:
        decoded = subprocess.run(command, capture_output=True, timeout=20, check=False)
    except (OSError, subprocess.TimeoutExpired):
        state.attempts.append(f"{sample_name}:ffmpeg_decode_failed")
        return None
    if decoded.returncode != 0 or not decoded.stdout:
        state.attempts.append(f"{sample_name}:ffmpeg_decode_failed")
        return None
    frame = cv2.imdecode(np.frombuffer(decoded.stdout, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        state.attempts.append(f"{sample_name}:ffmpeg_decode_failed")
        return None
    frame_index = round(timestamp_seconds * state.frames_per_second)
    return FrameCandidate(state, frame, sample_name, frame_index, state.frames_per_second)


def result_for(state: VideoState, status: str, **fields: object) -> dict[str, object]:
    """Build one manifest result while recording every attempted sample."""
    return {
        "source": str(state.path),
        "status": status,
        "attempts": "; ".join(state.attempts),
        **fields,
    }


def main() -> None:
    arguments = parse_arguments()
    if not 0 <= arguments.threshold <= 1:
        raise ValueError("--threshold must be between 0 and 1")
    if arguments.max_images is not None and arguments.max_images < 1:
        raise ValueError("--max-images must be at least 1")
    if arguments.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if arguments.decode_workers < 1:
        raise ValueError("--decode-workers must be at least 1")
    source_root = arguments.source.resolve()
    output_root = arguments.output.resolve()
    if not source_root.is_dir():
        raise NotADirectoryError(f"Video source folder not found: {source_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    video_paths = [
        path
        for path in (source_root.rglob("*") if arguments.recursive else source_root.iterdir())
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    if not video_paths:
        raise ValueError(f"No supported videos found in {source_root}")
    random.Random(arguments.seed).shuffle(video_paths)
    model, bird_index, device, transform = load_model(arguments.checkpoint.resolve())
    ffmpeg_path: Path | None = None
    ffprobe_path: Path | None = None
    if arguments.decoder == "ffmpeg-cuda":
        ffmpeg_path, ffprobe_path = resolve_ffmpeg_tools(
            arguments.ffmpeg_path, arguments.ffprobe_path
        )
        candidate_reader = partial(
            read_candidate_ffmpeg_cuda,
            ffmpeg_path=ffmpeg_path,
            ffprobe_path=ffprobe_path,
        )
        decoder_description = f"FFmpeg CUDA/NVDEC ({ffmpeg_path})"
    else:
        candidate_reader = read_candidate_opencv
        decoder_description = "OpenCV CPU"
    manifest_path = output_root / "binary_frame_selection_manifest.csv"
    fieldnames = [
        "source",
        "status",
        "output",
        "sample",
        "frame_index",
        "timestamp_seconds",
        "bird_probability",
        "attempts",
    ]
    existing_count = selected_frame_count(output_root)
    saved_count = 0
    skipped_count = 0
    pending_states = [
        VideoState(path, output_path_for(path, source_root, output_root))
        for path in video_paths
        if not output_path_for(path, source_root, output_root).is_file()
    ]
    already_sampled_count = len(video_paths) - len(pending_states)
    target_description = (
        f"{arguments.max_images:,} selected frames"
        if arguments.max_images is not None
        else "all videos"
    )
    print(
        f"Using {device.type.upper()} | {len(video_paths):,} videos in random order | "
        f"GPU batch size: {arguments.batch_size} | decode workers: {arguments.decode_workers} | "
        f"decoder: {decoder_description} | Bird threshold: {arguments.threshold:.0%}\n"
        f"Already selected: {existing_count:,} | target: {target_description} | "
        f"already sampled videos: {already_sampled_count:,}",
        flush=True,
    )
    if arguments.max_images is not None and existing_count >= arguments.max_images:
        print("Target already reached. No videos need processing.", flush=True)
        return
    append_manifest = manifest_path.is_file()
    with (
        manifest_path.open("a" if append_manifest else "w", newline="", encoding="utf-8") as handle,
        ThreadPoolExecutor(
            max_workers=arguments.decode_workers, thread_name_prefix="video-decode"
        ) as executor,
    ):
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not append_manifest:
            writer.writeheader()
        target_reached = False
        for batch_start in range(0, len(pending_states), arguments.batch_size):
            if target_reached:
                break
            active_states = pending_states[batch_start : batch_start + arguments.batch_size]
            for fraction, sample_name in SAMPLE_POSITIONS:
                if not active_states or target_reached:
                    break
                candidates: list[FrameCandidate] = []
                next_round_states: list[VideoState] = []
                decoded_candidates = executor.map(
                    lambda state: candidate_reader(state, fraction, sample_name), active_states
                )
                for state, candidate in zip(active_states, decoded_candidates, strict=True):
                    if candidate is not None:
                        candidates.append(candidate)
                    elif state.terminal_status is None:
                        next_round_states.append(state)
                    else:
                        writer.writerow(result_for(state, state.terminal_status))
                        skipped_count += 1
                if candidates:
                    probabilities = bird_probabilities(
                        [candidate.frame for candidate in candidates],
                        model,
                        bird_index,
                        device,
                        transform,
                    )
                    for candidate, probability in zip(candidates, probabilities, strict=True):
                        state = candidate.state
                        state.attempts.append(f"{candidate.sample_name}:{probability:.4f}")
                        if probability < arguments.threshold:
                            next_round_states.append(state)
                            continue
                        if (
                            arguments.max_images is not None
                            and existing_count + saved_count >= arguments.max_images
                        ):
                            target_reached = True
                            break
                        if not cv2.imwrite(str(state.destination), candidate.frame):
                            writer.writerow(result_for(state, "output_write_failed"))
                            skipped_count += 1
                            continue
                        writer.writerow(
                            result_for(
                                state,
                                "saved",
                                output=str(state.destination),
                                sample=candidate.sample_name,
                                frame_index=candidate.frame_index,
                                timestamp_seconds=(
                                    candidate.frame_index / candidate.frames_per_second
                                    if candidate.frames_per_second
                                    else None
                                ),
                                bird_probability=round(probability, 6),
                            )
                        )
                        saved_count += 1
                active_states = next_round_states
            if not target_reached:
                for state in active_states:
                    writer.writerow(result_for(state, "no_confident_bird_frame"))
                    skipped_count += 1
            handle.flush()
            videos_seen = min(batch_start + arguments.batch_size, len(pending_states))
            print(
                f"[{videos_seen:,}/{len(pending_states):,}] selected: "
                f"{existing_count + saved_count:,} | "
                f"new: {saved_count:,} | skipped: {skipped_count:,}",
                flush=True,
            )
            if (
                arguments.max_images is not None
                and existing_count + saved_count >= arguments.max_images
            ):
                target_reached = True
                print(f"Reached target of {arguments.max_images:,} selected frames.", flush=True)
    print(
        f"Finished. Selected: {existing_count + saved_count:,} | "
        f"new: {saved_count:,} | already sampled: {already_sampled_count:,} | "
        f"skipped: {skipped_count:,}.\n"
        f"Manifest: {manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
