"""Run the local ConvNeXt-Tiny prediction service for Label Studio.

Start this script while Label Studio is open, then connect the project model to
http://127.0.0.1:9090. The service reads local Label Studio image URLs directly
from ``F:\\video vogel`` and never alters Label Studio annotations.
"""

from __future__ import annotations

import os
import sys
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse
from xml.etree import ElementTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import torch
import uvicorn
from fastapi import Body, FastAPI, HTTPException
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from bird_classifier.models.ConVNeXt_tiny import create_convnext_tiny


DEFAULT_MODEL = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "combined_synthetic_real_convnext_tiny"
    / "model.pt"
)
IMAGE_ROOT = Path(os.environ.get("BIRD_IMAGE_ROOT", r"F:\video vogel"))
MODEL_PATH = Path(os.environ.get("BIRD_MODEL_PATH", str(DEFAULT_MODEL)))
MODEL_VERSION = "convnext-tiny-combined-synthetic-real-v2"
SERVER_HOST = os.environ.get("BIRD_BACKEND_HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("BIRD_BACKEND_PORT", "9090"))
IMAGE_SIZE = 224
INFERENCE_BATCH_SIZE = int(os.environ.get("BIRD_INFERENCE_BATCH_SIZE", "128"))
# JPEG decoding and torchvision transforms are CPU-bound.  Decode only one
# inference batch at a time and use a small worker pool so a 24k-task request
# does not first build 24k tensors in memory (or leave the GPU idle for ages).
DECODE_WORKERS = int(os.environ.get("BIRD_DECODE_WORKERS", "8"))
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
REQUEST_LOG = PROJECT_ROOT / "data" / "interim" / "label_studio_backend_requests.log"
TRACE_LOG = PROJECT_ROOT / "data" / "interim" / "label_studio_backend_trace.jsonl"


def find_label_tags(label_config: str) -> tuple[str, str]:
    """Read the selected Choices and Image control names from project XML."""
    try:
        root = ElementTree.fromstring(label_config)
    except ElementTree.ParseError as error:
        raise ValueError("Label Studio sent an invalid labeling configuration.") from error
    choices = next(
        (element for element in root.iter() if element.tag.lower() == "choices" and element.get("name")),
        None,
    )
    image = next(
        (element for element in root.iter() if element.tag.lower() == "image" and element.get("name")),
        None,
    )
    if choices is None or image is None:
        raise ValueError("The project needs one named Choices control and one named Image control.")
    return str(choices.get("name")), str(image.get("name"))


def filename_from_task(task: dict) -> str:
    try:
        image_url = str(task["data"]["image"])
    except (KeyError, TypeError) as error:
        raise ValueError("Task data must contain an image URL under data.image.") from error
    encoded_path = parse_qs(urlparse(image_url).query).get("d", [""])[0]
    if not encoded_path:
        raise ValueError(f"Could not resolve local-file URL: {image_url}")
    return Path(unquote(encoded_path)).name


def log_prediction_request(task_count: int, image_count: int) -> None:
    """Keep a compact audit trail for Label Studio batch-prediction requests."""
    REQUEST_LOG.parent.mkdir(parents=True, exist_ok=True)
    with REQUEST_LOG.open("a", encoding="utf-8") as log_file:
        log_file.write(
            f"{datetime.now().isoformat(timespec='seconds')} "
            f"tasks={task_count} readable_images={image_count}\n"
        )


def log_prediction_trace(event: str, **details: Any) -> None:
    """Write compact, inspectable request/response diagnostics for this backend."""
    TRACE_LOG.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": datetime.now().isoformat(timespec="seconds"), "event": event, **details}
    with TRACE_LOG.open("a", encoding="utf-8") as trace_file:
        trace_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def task_samples(tasks: list[Any], limit: int = 5) -> list[dict[str, Any]]:
    """Return a privacy-conscious, small summary rather than entire task payloads."""
    samples: list[dict[str, Any]] = []
    for task in tasks[:limit]:
        if not isinstance(task, dict):
            samples.append({"task_type": type(task).__name__})
            continue
        image_url = task.get("data", {}).get("image") if isinstance(task.get("data"), dict) else None
        samples.append({"id": task.get("id"), "image": Path(str(image_url)).name if image_url else None})
    return samples


def load_classifier() -> tuple[torch.nn.Module, tuple[str, ...], torch.device]:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    class_names = tuple(checkpoint["class_names"])
    model = create_convnext_tiny(len(class_names), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_names, device


MODEL, CLASS_NAMES, DEVICE = load_classifier()
TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)
app = FastAPI(title="Bird Classifier Label Studio Backend")
# Label Studio normally supplies the labeling configuration with `/setup` and
# `/predict`. Its connection test may send a bare JSON list of tasks, however,
# so retain the most recently validated control names for that lightweight test.
LABEL_TAGS: tuple[str, str] | None = None
# Label Studio can issue a retry while a large request is still processing.
# Serialising model access avoids two 24k-image jobs fighting for GPU/CPU RAM.
PREDICTION_LOCK = threading.Lock()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "UP", "model_version": MODEL_VERSION, "device": DEVICE.type}


@app.post("/setup")
def setup(payload: Any = Body(...)) -> dict[str, str]:
    global LABEL_TAGS
    # Label Studio's connection tester can post a bare list here. It does not
    # need setup metadata in that case; normal connection requests are objects.
    label_config = str(payload.get("label_config", "")) if isinstance(payload, dict) else ""
    if label_config:
        LABEL_TAGS = find_label_tags(label_config)
    return {"model_version": MODEL_VERSION}


def load_task_tensor(position: int, task: Any) -> tuple[int, torch.Tensor | None]:
    """Decode one task safely; unreadable files yield an empty prediction."""
    try:
        if not isinstance(task, dict):
            return position, None
        image_path = IMAGE_ROOT / filename_from_task(task)
        with Image.open(image_path) as image:
            return position, TRANSFORM(image.convert("RGB"))
    except (FileNotFoundError, UnidentifiedImageError, OSError, ValueError):
        return position, None


@app.post("/predict")
def predict(payload: Any = Body(...)) -> dict[str, list[dict]]:
    """Return one class prediction for every task in a Label Studio request."""
    global LABEL_TAGS

    # The normal endpoint format is {"tasks": [...], "label_config": "..."}.
    # The "Test request" button can send the task list itself instead.
    if isinstance(payload, list):
        tasks = payload
        label_config = ""
        request_shape = "bare_task_list"
    elif isinstance(payload, dict):
        tasks = payload.get("tasks", [])
        label_config = str(payload.get("label_config", ""))
        request_shape = "object_with_tasks"
    else:
        raise HTTPException(status_code=422, detail="Expected a task list or an object containing tasks.")
    if not isinstance(tasks, list):
        raise HTTPException(status_code=422, detail="Expected a list in tasks.")
    log_prediction_trace(
        "predict_received",
        request_shape=request_shape,
        task_count=len(tasks),
        task_samples=task_samples(tasks),
    )
    if label_config:
        try:
            LABEL_TAGS = find_label_tags(label_config)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
    if LABEL_TAGS is None:
        raise HTTPException(
            status_code=422,
            detail="No labeling configuration received yet. Reconnect the model, then retry the test request.",
        )
    choice_name, image_name = LABEL_TAGS

    results: list[dict] = [
        {"result": [], "score": 0.0, "model_version": MODEL_VERSION} for _ in tasks
    ]
    started = time.perf_counter()
    readable_count = 0
    # The ML HTTP endpoint is synchronous.  Ensure only one request consumes
    # the classifier at once, then process/decode/infer bounded chunks.
    with PREDICTION_LOCK, ThreadPoolExecutor(max_workers=DECODE_WORKERS) as decoder:
        for start in range(0, len(tasks), INFERENCE_BATCH_SIZE):
            task_chunk = list(enumerate(tasks[start : start + INFERENCE_BATCH_SIZE], start))
            decoded = list(decoder.map(lambda item: load_task_tensor(*item), task_chunk))
            valid = [(position, tensor) for position, tensor in decoded if tensor is not None]
            readable_count += len(valid)
            if valid:
                positions, tensors = zip(*valid)
                batch = torch.stack(tensors).to(DEVICE, non_blocking=True)
                with torch.inference_mode(), torch.autocast(
                    device_type=DEVICE.type, dtype=torch.float16, enabled=DEVICE.type == "cuda"
                ):
                    probabilities = MODEL(batch).softmax(dim=1)
                scores, indices = probabilities.max(dim=1)
                for position, score, index in zip(positions, scores, indices):
                    label = CLASS_NAMES[int(index)]
                    confidence = float(score)
                    results[position] = {
                        "result": [
                            {
                                "from_name": choice_name,
                                "to_name": image_name,
                                "type": "choices",
                                "value": {"choices": [label]},
                            }
                        ],
                        "score": confidence,
                        "model_version": MODEL_VERSION,
                    }
            completed = min(start + INFERENCE_BATCH_SIZE, len(tasks))
            if completed == len(tasks) or completed % 1000 < INFERENCE_BATCH_SIZE:
                elapsed = max(time.perf_counter() - started, 0.001)
                rate = completed / elapsed
                print(f"Prediction progress: {completed:,}/{len(tasks):,} tasks ({rate:.1f} images/s)", flush=True)

    log_prediction_request(len(tasks), readable_count)
    # Label Studio's HTTP ML protocol expects an object with a ``results``
    # member. Returning the list directly makes Label Studio call ``.get`` on
    # a JSON array and fail with "'list' object has no attribute 'get'".
    response = {"results": results}
    log_prediction_trace(
        "predict_completed",
        task_count=len(tasks),
        readable_image_count=readable_count,
        response_result_count=len(results),
        classified_count=sum(bool(result["result"]) for result in results),
        sample_labels=[result["result"][0]["value"]["choices"][0] for result in results if result["result"]][:5],
    )
    return response


if __name__ == "__main__":
    print(f"Loaded {MODEL_VERSION} on {DEVICE.type.upper()} from {MODEL_PATH}")
    print(f"Resolving Label Studio images from {IMAGE_ROOT}")
    print(f"Listening on {SERVER_HOST}:{SERVER_PORT}")
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
