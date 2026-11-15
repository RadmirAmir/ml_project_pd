"""Smoke test for a running scoring API.

Checks the three public endpoints and prints a compact report::

    python scripts/smoke_test_api.py --url http://127.0.0.1:8000

Exit code 0 means every request succeeded and returned a plausible payload.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import requests

SAMPLE_CLIENTS = [
    {
        "LIMIT_BAL": 20_000,
        "SEX": 2,
        "EDUCATION": 2,
        "MARRIAGE": 1,
        "AGE": 24,
        "PAY_0": 2,
        "BILL_AMT1": 3913,
        "PAY_AMT1": 0,
    },
    {
        "LIMIT_BAL": 120_000,
        "SEX": 2,
        "EDUCATION": 2,
        "MARRIAGE": 2,
        "AGE": 26,
        "PAY_0": -1,
        "BILL_AMT1": 2682,
        "PAY_AMT1": 0,
    },
    {
        "LIMIT_BAL": 50_000,
        "SEX": 1,
        "EDUCATION": 3,
        "MARRIAGE": 1,
        "AGE": 45,
        "PAY_0": 4,
        "BILL_AMT1": 45_000,
        "PAY_AMT1": 500,
    },
]


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def wait_for_service(base_url: str, timeout: int = 60) -> dict:
    """Poll ``/health`` until the service answers or the timeout expires."""
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                return response.json()
        except requests.RequestException as error:  # service not up yet
            last_error = error
        time.sleep(1.5)
    raise RuntimeError(f"service at {base_url} did not become healthy: {last_error}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="base URL of the API")
    parser.add_argument("--timeout", type=int, default=60, help="seconds to wait for /health")
    args = parser.parse_args(argv)
    base_url = args.url.rstrip("/")

    health = wait_for_service(base_url, args.timeout)
    print(f"[1/4] /health         -> {health}")
    _check(health.get("model_loaded") is True, "model is not loaded in the service")

    info = requests.get(f"{base_url}/model/info", timeout=10).json()
    print(
        f"[2/4] /model/info     -> {info['model_name']} (threshold={info['threshold']}, "
        f"{len(info['features']['numeric']) + len(info['features']['categorical'])} features)"
    )
    _check(bool(info["features"]["numeric"]), "the service reports no numeric features")

    prediction = requests.post(f"{base_url}/predict", json=SAMPLE_CLIENTS[0], timeout=10).json()
    print(f"[3/4] /predict        -> {json.dumps(prediction, ensure_ascii=False)}")
    _check(0.0 <= prediction["default_probability"] <= 1.0, "probability outside [0, 1]")
    _check(prediction["default_prediction"] in (0, 1), "prediction is not binary")

    batch = requests.post(f"{base_url}/predict/batch", json={"records": SAMPLE_CLIENTS}, timeout=10).json()
    print(
        f"[4/4] /predict/batch  -> n={batch['n_records']} flagged={batch['n_flagged']} "
        f"mean_pd={batch['mean_probability']:.4f}"
    )
    _check(batch["n_records"] == len(SAMPLE_CLIENTS), "batch size mismatch")

    print("\nAPI smoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
