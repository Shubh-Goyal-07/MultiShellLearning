"""Deterministic seeding, RNG checkpoint support, and run-environment capture."""

from __future__ import annotations

import hashlib
import os
import platform
import random
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch

_PACKAGE_ROOT = Path(__file__).resolve().parent


def derive_stage_seed(root_seed: int, stage: str, *components: object) -> int:
    """Derive a stable seed so adding unrelated stages does not perturb a run."""

    if not stage:
        raise ValueError("stage must be a non-empty string")
    message = "\x1f".join([str(int(root_seed)), stage, *(str(value) for value in components)])
    digest = hashlib.sha256(message.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


stage_seed = derive_stage_seed


def seed_everything(seed: int, *, deterministic: bool = True) -> dict[str, Any]:
    """Seed Python, NumPy, PyTorch CPU, and every visible CUDA device."""

    seed = int(seed)
    if deterministic:
        # cuBLAS reads this before its first operation; setdefault respects a
        # user-selected valid deterministic workspace configuration.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = False if deterministic else torch.backends.cudnn.benchmark
    return {
        "seed": seed,
        "deterministic_algorithms": bool(deterministic),
        "cudnn_deterministic": bool(getattr(torch.backends.cudnn, "deterministic", False)),
        "cudnn_benchmark": bool(getattr(torch.backends.cudnn, "benchmark", False)),
        "cuda_available": bool(torch.cuda.is_available()),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def make_torch_generator(seed: int, *, device: str = "cpu") -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def seed_worker(worker_id: int) -> None:
    """DataLoader worker initializer compatible with a seeded generator."""

    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    required = {"python", "numpy", "torch_cpu"}
    missing = required - set(state)
    if missing:
        raise ValueError(f"RNG state is missing keys: {sorted(missing)}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint has CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def source_tree_hash(root: Path = _PACKAGE_ROOT) -> str:
    """Hash every Python source file of the package, independent of git."""

    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_commit(root: Path = _PACKAGE_ROOT) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def environment_metadata() -> dict[str, Any]:
    """Software versions, hardware, and code revision recorded with every run."""

    try:
        import torchvision

        torchvision_version: str | None = torchvision.__version__
    except Exception:  # pragma: no cover - optional binary dependency
        torchvision_version = None
    cuda = torch.cuda.is_available()
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torchvision": torchvision_version,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if cuda else None,
        "device": torch.cuda.get_device_name(0) if cuda else "cpu",
        "code_revision": {"git_commit": _git_commit(), "source_tree_hash": source_tree_hash()},
    }


@contextmanager
def seeded(seed: int, *, deterministic: bool = True) -> Iterator[None]:
    """Temporarily seed all RNGs, restoring their previous state afterward."""

    state = capture_rng_state()
    deterministic_before = torch.are_deterministic_algorithms_enabled()
    cudnn_deterministic_before = bool(getattr(torch.backends.cudnn, "deterministic", False))
    cudnn_benchmark_before = bool(getattr(torch.backends.cudnn, "benchmark", False))
    seed_everything(seed, deterministic=deterministic)
    try:
        yield
    finally:
        restore_rng_state(state)
        torch.use_deterministic_algorithms(deterministic_before, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = cudnn_deterministic_before
            torch.backends.cudnn.benchmark = cudnn_benchmark_before


__all__ = [
    "capture_rng_state",
    "derive_stage_seed",
    "environment_metadata",
    "make_torch_generator",
    "restore_rng_state",
    "seed_everything",
    "seed_worker",
    "seeded",
    "source_tree_hash",
    "stage_seed",
]
