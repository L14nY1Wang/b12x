"""Multi-process TP launch.

Uses mp.Process (not mp.spawn) for direct process lifecycle control.
Parent process manages worker lifetimes and handles cleanup.
"""

from __future__ import annotations

import os
import signal
import sys
from typing import Optional

import datetime

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from serve.tp.group import TPGroup


def _find_free_port() -> int:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def launch_tp(
    fn,
    world_size: int,
    args: tuple = (),
    gpu_ids: list[int] | None = None,
):
    """Launch *world_size* processes and run *fn(tp_group, *args)* in each.

    If world_size == 1, runs in the current process (no distributed).
    Rank 0 runs in the current process to preserve stdin (interactive mode).
    Followers are spawned as daemon processes.
    """
    if gpu_ids is None:
        gpu_ids = list(range(world_size))
    assert len(gpu_ids) == world_size

    if world_size == 1:
        torch.cuda.set_device(gpu_ids[0])
        fn(None, *args)
        return

    port = _find_free_port()
    mp.set_start_method("spawn", force=True)

    # Spawn followers (ranks 1..N-1).
    followers = []
    for rank in range(1, world_size):
        p = mp.Process(
            target=_worker,
            args=(rank, world_size, gpu_ids, port, fn, args),
            daemon=True,
        )
        p.start()
        followers.append(p)

    # Run rank 0 in the current process (preserves stdin for interactive mode).
    try:
        _worker(0, world_size, gpu_ids, port, fn, args)
    finally:
        for p in followers:
            if p.is_alive():
                p.kill()
                p.join(timeout=5)


def _worker(rank, world_size, gpu_ids, port, fn, args):
    """Per-process worker."""
    device_id = gpu_ids[rank]

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)

    # Silence non-rank-0 completely.
    if rank != 0:
        import warnings, logging
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, 1)  # Replace fd 1 (stdout).
        os.dup2(devnull_fd, 2)  # Replace fd 2 (stderr) — catches C++ output too.
        os.close(devnull_fd)
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")
        warnings.filterwarnings("ignore")
        logging.disable(logging.CRITICAL)

    torch.cuda.set_device(device_id)
    torch.set_grad_enabled(False)
    # Long timeout for first-pass kernel compilation (Triton JIT + CuTe DSL).
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(minutes=30),
    )

    tp_group = TPGroup(
        rank=rank,
        world_size=world_size,
        device=torch.device("cuda", device_id),
        process_group=dist.group.WORLD,
    )

    try:
        fn(tp_group, *args)
    finally:
        if dist.is_initialized():
            # Flush any pending NCCL work and let all ranks reach teardown
            # before destroying the shared process group.
            try:
                torch.cuda.synchronize(device_id)
            except Exception:
                pass
            try:
                dist.barrier(device_ids=[device_id])
            except Exception:
                pass
        TPGroup.destroy()
