#!/usr/bin/env python3
"""Exercise auth and truncated Responses lifecycles on the production server."""

import argparse
import http.client
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import time

API_KEY = "q27-production-gate-key"
CODEX_HEADERS = (
    ("none", {}),
    ("legacy", {"x-codex-installation-id": "q27-gate"}),
    ("current", {"x-codex-turn-metadata": "{}"}),
)


def fail(message):
    raise RuntimeError(message)


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def request(port, method, path, body=None, headers=None, timeout=300):
    payload = None if body is None else json.dumps(body).encode()
    request_headers = dict(headers or {})
    if payload is not None:
        request_headers["Content-Type"] = "application/json"
        request_headers["Content-Length"] = str(len(payload))
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        connection.request(method, path, body=payload, headers=request_headers)
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def require_status(label, actual, expected):
    if actual != expected:
        fail(f"{label}: expected HTTP {expected}, got {actual}")


def parse_json(label, payload):
    try:
        return json.loads(payload)
    except Exception as error:
        fail(f"{label}: invalid JSON: {error}: {payload[:500]!r}")


def parse_sse(label, payload):
    events = []
    for line in payload.decode("utf-8").splitlines():
        if line.startswith("data: "):
            events.append(parse_json(label, line[6:]))
    if not events:
        fail(f"{label}: no SSE data events: {payload[:500]!r}")
    return events


def assert_incomplete_response(label, response, reasoning_limited):
    if response.get("status") != "incomplete":
        fail(f"{label}: terminal status is not incomplete: {response!r}")
    if response.get("incomplete_details") != {"reason": "max_output_tokens"}:
        fail(f"{label}: missing max_output_tokens details: {response!r}")
    usage = response.get("usage", {})
    details = usage.get("output_tokens_details", {})
    if details.get("reasoning_budget_exceeded") is not reasoning_limited:
        fail(f"{label}: wrong reasoning budget state: {details!r}")
    output = response.get("output")
    if not isinstance(output, list) or not output:
        fail(f"{label}: expected at least one output item: {response!r}")
    statuses = [item.get("status") for item in output if isinstance(item, dict)]
    if "incomplete" not in statuses:
        fail(f"{label}: no incomplete output item: {statuses!r}")


def run_response_case(port, header_label, codex_headers, stream, reasoning_limited):
    label = f"{header_label}/{'stream' if stream else 'nonstream'}/{'reasoning' if reasoning_limited else 'token'}"
    headers = {"Authorization": f"Bearer {API_KEY}", **codex_headers}
    body = {
        "input": "Reply with the words hello world and nothing else.",
        "stream": stream,
        "max_output_tokens": 8 if reasoning_limited else 1,
        "enable_thinking": reasoning_limited,
    }
    if reasoning_limited:
        body["thinking_token_budget"] = 0
    status, response_headers, payload = request(
        port, "POST", "/v1/responses", body, headers
    )
    require_status(label, status, 200)
    if stream:
        content_type = response_headers.get("Content-Type", "")
        if not content_type.startswith("text/event-stream"):
            fail(f"{label}: wrong content type {content_type!r}")
        events = parse_sse(label, payload)
        terminal = events[-1]
        if terminal.get("type") != "response.incomplete":
            fail(f"{label}: wrong terminal event: {terminal!r}")
        assert_incomplete_response(label, terminal.get("response", {}), reasoning_limited)
        done_items = [
            event.get("item", {})
            for event in events
            if event.get("type") == "response.output_item.done"
        ]
        if not done_items or "incomplete" not in [item.get("status") for item in done_items]:
            fail(f"{label}: stream has no incomplete output_item.done: {done_items!r}")
    else:
        assert_incomplete_response(label, parse_json(label, payload), reasoning_limited)
    print(f"{label}: PASS")


def wait_ready(process, port, log_path):
    deadline = time.monotonic() + 300
    last_error = "not started"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            break
        try:
            status, _, _ = request(port, "GET", "/health", timeout=2)
            if status == 200:
                return
            last_error = f"HTTP {status}"
        except Exception as error:
            last_error = str(error)
        time.sleep(0.5)
    log_tail = Path(log_path).read_text(errors="replace")[-4000:]
    fail(f"server failed readiness ({last_error}); log tail:\n{log_tail}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    args = parser.parse_args()

    server = Path(args.server).resolve()
    model = Path(args.model).resolve()
    tokenizer = Path(args.tokenizer).resolve()
    for label, path in (("server", server), ("model", model), ("tokenizer", tokenizer)):
        if not path.is_file():
            fail(f"{label} does not exist: {path}")

    port = free_port()
    with tempfile.NamedTemporaryFile(prefix="q27-responses-gate-", suffix=".log", delete=False) as log:
        log_path = log.name
        process = subprocess.Popen(
            [
                str(server), str(model), str(tokenizer),
                "--host", "127.0.0.1", "--port", str(port),
                "--ctx", "4096", "--slots", "1", "--request-think",
                "--api-key", API_KEY,
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    try:
        wait_ready(process, port, log_path)
        status, _, payload = request(port, "GET", "/health")
        require_status("health exemption", status, 200)
        if parse_json("health exemption", payload).get("status") != "ok":
            fail(f"health exemption: unexpected body {payload!r}")

        for label, headers, expected in (
            ("models missing key", {}, 401),
            ("models wrong key", {"Authorization": "Bearer wrong"}, 401),
            ("models bearer key", {"Authorization": f"Bearer {API_KEY}"}, 200),
            ("models x-api-key", {"x-api-key": API_KEY}, 200),
        ):
            status, _, _ = request(port, "GET", "/v1/models", headers=headers)
            require_status(label, status, expected)
            print(f"{label}: PASS")

        rejected_body = {"input": "This must not reach generation.", "max_output_tokens": 1}
        for label, headers in (
            ("responses missing key", {}),
            ("responses wrong key", {"x-api-key": "wrong"}),
        ):
            status, _, _ = request(port, "POST", "/v1/responses", rejected_body, headers)
            require_status(label, status, 401)
            print(f"{label}: PASS")

        for header_label, codex_headers in CODEX_HEADERS:
            for stream in (False, True):
                for reasoning_limited in (False, True):
                    run_response_case(
                        port, header_label, codex_headers, stream, reasoning_limited
                    )
        print("all production Responses integration tests passed")
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        Path(log_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
