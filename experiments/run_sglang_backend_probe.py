#!/usr/bin/env python3
"""Launch a temporary sglang server for one backend pair and probe a prompt."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request


MODEL_PATH = os.environ.get("B12X_MODEL_PATH", "/data/models/Qwen3.5-397B-A17B-NVFP4")


def _http_get(url: str, *, timeout: float) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode()


def _http_post(url: str, payload: dict, *, timeout: float) -> str:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode()


def _wait_ready(base_url: str, *, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            _http_get(f"{base_url}/model_info", timeout=2.0)
            return
        except Exception:
            time.sleep(1.0)
    raise TimeoutError(f"Timed out waiting for {base_url} to become ready")


def _terminate(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=30)
        return
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--moe-backend", required=True)
    parser.add_argument("--fp4-backend", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--cuda-visible-devices", default="0,1,2,3")
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument("--prompt", default="What is 2+2?")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--startup-timeout", type=float, default=240.0)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--warmup-requests", type=int, default=0)
    parser.add_argument("--measure-requests", type=int, default=1)
    parser.add_argument("--log-path", type=pathlib.Path, default=None)
    args = parser.parse_args()

    log_path = args.log_path
    if log_path is None:
        log_path = pathlib.Path(
            f"/tmp/sglang-probe-{args.moe_backend}-{args.fp4_backend}-{args.port}.log"
        )

    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "16",
            "SGLANG_ENABLE_SPEC_V2": "True",
            "SGLANG_IMAGE_MAX_PIXELS": str(1280 * 28 * 28),
            "TORCHINDUCTOR_CACHE_DIR": "/cache/torchinductor",
            "CUTE_DSL_ARCH": "sm_120a",
            "CUDA_VISIBLE_DEVICES": args.cuda_visible_devices,
        }
    )

    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model",
        MODEL_PATH,
        "--served-model-name",
        "Qwen3.5",
        "--reasoning-parser",
        "qwen3",
        "--tool-call-parser",
        "qwen3_coder",
        "--tensor-parallel-size",
        "4",
        "--quantization",
        "modelopt_fp4",
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--trust-remote-code",
        "--attention-backend",
        "triton",
        "--moe-runner-backend",
        args.moe_backend,
        "--fp4-gemm-backend",
        args.fp4_backend,
        "--model-loader-extra-config",
        '{"enable_multithread_load": "true","num_threads": 32}',
        "--max-running-requests",
        "16",
        "--mem-fraction-static",
        "0.7",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
    ]
    if args.disable_cuda_graph:
        cmd.append("--disable-cuda-graph")
    else:
        cmd.extend(["--cuda-graph-max-bs", "16"])

    base_url = f"http://127.0.0.1:{args.port}"
    with log_path.open("w") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=os.environ.get("SGLANG_ROOT"),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            start_ready = time.perf_counter()
            _wait_ready(base_url, timeout_s=args.startup_timeout)
            ready_ms = (time.perf_counter() - start_ready) * 1000.0
            payload = {
                "model": "Qwen3.5",
                "messages": [{"role": "user", "content": args.prompt}],
                "max_tokens": args.max_tokens,
                "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False},
                "separate_reasoning": False,
            }
            total_requests = args.warmup_requests + args.measure_requests
            request_ms = []
            response = None
            for req_idx in range(total_requests):
                start_request = time.perf_counter()
                response = _http_post(
                    f"{base_url}/v1/chat/completions",
                    payload,
                    timeout=args.request_timeout,
                )
                elapsed_ms = (time.perf_counter() - start_request) * 1000.0
                if req_idx >= args.warmup_requests:
                    request_ms.append(elapsed_ms)
            print(
                json.dumps(
                    {
                        "moe_backend": args.moe_backend,
                        "fp4_backend": args.fp4_backend,
                        "port": args.port,
                        "disable_cuda_graph": args.disable_cuda_graph,
                        "log_path": str(log_path),
                        "ready_ms": ready_ms,
                        "request_ms": request_ms,
                        "warmup_requests": args.warmup_requests,
                        "measure_requests": args.measure_requests,
                        "response": json.loads(response),
                    },
                    indent=2,
                )
            )
        finally:
            _terminate(proc)


if __name__ == "__main__":
    main()
