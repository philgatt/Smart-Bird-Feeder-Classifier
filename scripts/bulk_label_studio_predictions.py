"""Create Label Studio predictions reliably for a large local project.

The Label Studio Data Manager's ``Retrieve predictions`` action sends every
selected task in one synchronous request. That request normally times out for
large projects. This command invokes that same supported action in modest
chunks, allowing Label Studio to save every chunk internally in bulk. It never
changes annotations.

Set ``LABEL_STUDIO_API_TOKEN`` to the API token from Label Studio's Account &
Settings page, then run this script from the repository root.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def request_json(
    url: str,
    *,
    method: str = "GET",
    token: str | None = None,
    auth_scheme: str = "Token",
    payload: Any | None = None,
    timeout: int = 120,
) -> Any:
    """Make one JSON request and raise a useful error if Label Studio rejects it."""
    headers = {"Accept": "application/json"}
    data = None
    if token:
        headers["Authorization"] = f"{auth_scheme} {token}"
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")
    request = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - user chose local URL.
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed ({error.code}): {body[:500]}") from error
    except URLError as error:
        raise RuntimeError(f"Could not reach {url}: {error.reason}") from error


def list_project_tasks(
    base_url: str, project_id: int, token: str, auth_scheme: str, page_size: int
) -> list[dict[str, Any]]:
    """Read all tasks using Label Studio's paginated project-task endpoint."""
    tasks: list[dict[str, Any]] = []
    page = 1
    while True:
        query = urlencode({"page": page, "page_size": page_size})
        payload = request_json(
            f"{base_url}/api/projects/{project_id}/tasks?{query}",
            token=token,
            auth_scheme=auth_scheme,
            timeout=180,
        )
        if isinstance(payload, dict):
            page_tasks = payload.get("tasks", payload.get("results", []))
            total = payload.get("total", payload.get("count"))
        elif isinstance(payload, list):
            page_tasks = payload
            total = None
        else:
            raise RuntimeError("Label Studio returned an unexpected task-list response.")
        if not isinstance(page_tasks, list):
            raise RuntimeError("Label Studio task-list response did not contain a task list.")
        tasks.extend(task for task in page_tasks if isinstance(task, dict))
        print(f"Loaded {len(tasks):,}" + (f" / {total:,}" if isinstance(total, int) else "") + " tasks", flush=True)
        if not page_tasks or len(page_tasks) < page_size or (isinstance(total, int) and len(tasks) >= total):
            return tasks
        page += 1


def retrieve_prediction_batch(
    base_url: str,
    token: str,
    auth_scheme: str,
    project_id: int,
    task_ids: list[int],
) -> dict[str, Any]:
    """Run Label Studio's native bulk prediction action for one safe chunk."""
    response = request_json(
        f"{base_url}/api/dm/actions/?id=retrieve_tasks_predictions&project={project_id}",
        method="POST",
        token=token,
        auth_scheme=auth_scheme,
        payload={"selectedItems": {"all": False, "included": task_ids}},
        # The server performs both model inference and its bulk persistence
        # inside this request.  500 images remains comfortably below the
        # Label Studio ML prediction timeout on this machine.
        timeout=600,
    )
    if not isinstance(response, dict):
        raise RuntimeError("Label Studio returned an unexpected response for the bulk prediction action.")
    return response


def refresh_personal_access_token(base_url: str, personal_token: str) -> str:
    """Exchange the saved personal refresh token for a fresh bearer token."""
    refreshed = request_json(
        f"{base_url}/api/token/refresh/",
        method="POST",
        payload={"refresh": personal_token},
        timeout=30,
    )
    access_token = refreshed.get("access") if isinstance(refreshed, dict) else None
    if not isinstance(access_token, str) or not access_token:
        raise RuntimeError("Label Studio did not return an access token for the supplied personal token.")
    return access_token


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-id", type=int, default=5, help="Label Studio project ID (default: 5).")
    parser.add_argument("--label-studio-url", default="http://localhost:8080")
    parser.add_argument("--backend-url", default="http://127.0.0.1:9090", help=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, default=500, help="Tasks per native Label Studio batch.")
    parser.add_argument(
        "--activate-model-version",
        help="Set Label Studio's active project model version before processing (enables score filtering).",
    )
    parser.add_argument(
        "--activate-only",
        action="store_true",
        help="Set --activate-model-version and exit without requesting predictions.",
    )
    parser.add_argument(
        "--include-existing-predictions",
        action="store_true",
        help="Also create another prediction for tasks that already have one (normally skipped).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # Copy/paste from the Label Studio UI can accidentally include a trailing
    # newline or space; authentication tokens themselves never contain either.
    token = os.environ.get("LABEL_STUDIO_API_TOKEN", "").strip()
    if not token:
        print(
            "LABEL_STUDIO_API_TOKEN is not set. Copy your token from Label Studio's Account & Settings page, then run:\n"
            "$env:LABEL_STUDIO_API_TOKEN = 'paste-token-here'",
            file=sys.stderr,
        )
        return 2
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")

    base_url = args.label_studio_url.rstrip("/")
    # Legacy tokens authenticate directly with ``Token``. Personal access
    # tokens are refresh JWTs, which must be exchanged for a short-lived
    # bearer token before raw HTTP API calls.
    auth_scheme = "Token"
    personal_refresh_token: str | None = None
    try:
        project = request_json(
            f"{base_url}/api/projects/{args.project_id}", token=token, auth_scheme=auth_scheme
        )
    except RuntimeError as error:
        if "failed (401)" not in str(error):
            raise
        personal_refresh_token = token
        token = refresh_personal_access_token(base_url, personal_refresh_token)
        auth_scheme = "Bearer"
        print("Authenticated with a personal access token.", flush=True)
        project = request_json(
            f"{base_url}/api/projects/{args.project_id}", token=token, auth_scheme=auth_scheme
        )
    if not isinstance(project, dict):
        raise RuntimeError("Could not read the Label Studio project.")
    if args.activate_only and not args.activate_model_version:
        raise ValueError("--activate-only requires --activate-model-version.")
    if args.activate_model_version:
        project = request_json(
            f"{base_url}/api/projects/{args.project_id}",
            method="PATCH",
            token=token,
            auth_scheme=auth_scheme,
            payload={"model_version": args.activate_model_version},
            timeout=60,
        )
        print(f"Active project model version set to: {args.activate_model_version}", flush=True)
        if args.activate_only:
            return 0
    print("Using Label Studio's native batch prediction action.", flush=True)

    all_tasks = list_project_tasks(base_url, args.project_id, token, auth_scheme, page_size=1000)
    tasks = all_tasks if args.include_existing_predictions else [
        task for task in all_tasks if not task.get("predictions")
    ]
    print(
        f"Will create predictions for {len(tasks):,} tasks"
        + (f"; skipped {len(all_tasks) - len(tasks):,} with existing predictions." if not args.include_existing_predictions else "."),
        flush=True,
    )
    if not tasks:
        return 0

    stored = 0
    started = time.perf_counter()
    for start in range(0, len(tasks), args.batch_size):
        batch_tasks = tasks[start : start + args.batch_size]
        # Personal access tokens produce deliberately short-lived bearer
        # tokens. Refresh immediately before every native batch action.
        if personal_refresh_token is not None:
            token = refresh_personal_access_token(base_url, personal_refresh_token)
        response = retrieve_prediction_batch(
            base_url,
            token,
            auth_scheme,
            args.project_id,
            [int(task["id"]) for task in batch_tasks],
        )
        batch_stored = int(response.get("processed_items", len(batch_tasks)))
        stored += batch_stored
        completed = min(start + len(batch_tasks), len(tasks))
        elapsed = max(time.perf_counter() - started, 0.001)
        print(
            f"Progress: {completed:,}/{len(tasks):,} tasks; {stored:,} predictions requested "
            f"({completed / elapsed:.1f} tasks/s)",
            flush=True,
        )
    print(f"Done. Requested {stored:,} predictions without changing any annotations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
