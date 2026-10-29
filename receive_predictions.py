"""Receive ESP32 sound-classification predictions over HTTP."""

from __future__ import annotations

import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any

HOST = "0.0.0.0"
PORT = 8000
MAX_REQUEST_BYTES = 16_384
LATEST_PREDICTION: dict[str, Any] | None = None
LATEST_PREDICTION_LOCK = Lock()


def send_json(handler: BaseHTTPRequestHandler, status: int, payload: Any) -> None:
    response = json.dumps(payload, allow_nan=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(response)))
    handler.end_headers()
    handler.wfile.write(response)


class PredictionHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/api/predictions":
            with LATEST_PREDICTION_LOCK:
                prediction = LATEST_PREDICTION
            send_json(self, 200, {"prediction": prediction})
            return

        if self.path == "/dashboard-background.png":
            try:
                background = (
                    Path(__file__)
                    .with_name("dashboard-background.png")
                    .read_bytes()
                )
            except OSError as error:
                self.send_error(500, f"Could not read dashboard background: {error}")
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Cache-Control", "public, max-age=86400")
            self.send_header("Content-Length", str(len(background)))
            self.end_headers()
            self.wfile.write(background)
            return

        if self.path == "/":
            try:
                dashboard = (
                    Path(__file__)
                    .with_name("dashboard.html")
                    .read_bytes()
                )
            except OSError as error:
                self.send_error(500, f"Could not read dashboard: {error}")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(dashboard)))
            self.end_headers()
            self.wfile.write(dashboard)
            return

        if self.path not in ("/health", "/predict"):
            self.send_error(404, "Use GET / for status or POST /predict for predictions")
            return

        send_json(
            self,
            200,
            {
                "ok": True,
                "message": "Receiver is running. Send predictions with POST /predict.",
            },
        )

    def do_POST(self) -> None:
        if self.path != "/predict":
            self.send_error(404, "Use POST /predict")
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_error(400, "Invalid Content-Length")
            return
        if content_length <= 0 or content_length > MAX_REQUEST_BYTES:
            self.send_error(400, "Invalid request size")
            return

        try:
            payload: Any = json.loads(self.rfile.read(content_length))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self.send_error(400, "Request body must be valid JSON")
            return

        if not isinstance(payload, dict):
            self.send_error(400, "JSON body must be an object")
            return

        device_id = payload.get("device_id")
        label = payload.get("label")
        class_id = payload.get("class_id")
        confidence = payload.get("confidence")
        if (
            not isinstance(device_id, str)
            or not isinstance(label, str)
            or not isinstance(class_id, int)
            or isinstance(class_id, bool)
            or not isinstance(confidence, (int, float))
            or isinstance(confidence, bool)
            or not math.isfinite(confidence)
            or not 0.0 <= confidence <= 1.0
        ):
            self.send_error(400, "Invalid prediction fields")
            return

        scores = payload.get("scores")
        if not isinstance(scores, dict) or not scores:
            self.send_error(400, "scores must be a non-empty object")
            return
        if any(
            not isinstance(name, str)
            or not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not math.isfinite(score)
            or not 0.0 <= score <= 1.0
            for name, score in scores.items()
        ):
            self.send_error(400, "Each score must be between 0 and 1")
            return

        global LATEST_PREDICTION
        with LATEST_PREDICTION_LOCK:
            LATEST_PREDICTION = payload

        print(
            f"{device_id}: {label} (class_id={class_id}, "
            f"confidence={confidence:.3f})",
            flush=True,
        )
        send_json(self, 200, {"ok": True})

    def log_message(self, format: str, *args: object) -> None:
        print(f"HTTP {self.address_string()} - {format % args}", flush=True)


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), PredictionHandler)
    print(f"Listening for predictions on http://{HOST}:{PORT}/predict", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping prediction server.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
