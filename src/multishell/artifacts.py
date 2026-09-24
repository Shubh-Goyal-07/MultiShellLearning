"""Small, dependency-light helpers for immutable research artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


class ArtifactError(RuntimeError):
    """Base error for invalid or inconsistent artifacts."""


class ArtifactMismatchError(ArtifactError):
    """Raised when a downstream artifact does not match its declared upstream."""


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    if is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(to_jsonable(item) for item in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a value deterministically for hashing and provenance."""

    return json.dumps(
        to_jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def hash_array(array: Any) -> str:
    """Hash array dtype, shape, and contiguous bytes."""

    if hasattr(array, "detach") and hasattr(array, "cpu"):
        array = array.detach().cpu().numpy()
    value = np.asarray(array)
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(canonical_json_bytes(list(contiguous.shape)))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def hash_class_order(class_ids: Sequence[Any], class_names: Sequence[str] | None = None) -> str:
    payload: dict[str, Any] = {"class_ids": list(class_ids)}
    if class_names is not None:
        if len(class_names) != len(class_ids):
            raise ValueError("class_names and class_ids must have equal length")
        payload["class_names"] = list(class_names)
    return stable_hash(payload)


def hash_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


sha256_file = hash_file


def utc_timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _prepare_destination(path: Path, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")


def save_json(
    path: str | Path,
    payload: Any,
    *,
    overwrite: bool = False,
    indent: int = 2,
) -> Path:
    """Atomically save JSON, refusing overwrite by default."""

    destination = Path(path)
    _prepare_destination(destination, overwrite)
    serializable = to_jsonable(payload)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(
                serializable,
                handle,
                indent=indent,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return destination


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_npz(
    path: str | Path,
    arrays: Mapping[str, Any],
    *,
    compressed: bool = True,
    overwrite: bool = False,
) -> Path:
    """Atomically save named arrays without enabling pickle."""

    destination = Path(path)
    _prepare_destination(destination, overwrite)
    clean = {
        name: np.asarray(value.detach().cpu().numpy())
        if hasattr(value, "detach") and hasattr(value, "cpu")
        else np.asarray(value)
        for name, value in arrays.items()
    }
    for name, value in clean.items():
        if value.dtype.hasobject:
            raise TypeError(f"array {name!r} has object dtype and cannot be saved safely")
    temporary: Path | None = None
    compression = zipfile.ZIP_DEFLATED if compressed else zipfile.ZIP_STORED
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            # A fixed member timestamp makes identical arrays byte-identical files.
            with zipfile.ZipFile(handle, mode="w", compression=compression) as archive:
                for name, value in clean.items():
                    member = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                    member.compress_type = compression
                    with archive.open(member, "w", force_zip64=True) as stream:
                        np.lib.format.write_array(stream, value, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return destination


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


@dataclass(frozen=True)
class ArtifactPaths:
    arrays: Path
    metadata: Path


def save_array_artifact(
    directory: str | Path,
    arrays: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    basename: str = "artifact",
    overwrite: bool = False,
) -> ArtifactPaths:
    """Save an NPZ/JSON artifact pair with content hashes in the metadata."""

    root = Path(directory)
    array_path = root / f"{basename}.npz"
    metadata_path = root / f"{basename}_metadata.json"
    normalized_arrays = {
        name: np.asarray(value.detach().cpu().numpy())
        if hasattr(value, "detach") and hasattr(value, "cpu")
        else np.asarray(value)
        for name, value in arrays.items()
    }
    enriched = dict(metadata)
    enriched.setdefault("created_at", utc_timestamp())
    enriched["array_hashes"] = {
        name: hash_array(value) for name, value in sorted(normalized_arrays.items())
    }
    enriched["payload_hash"] = stable_hash(
        {"metadata": metadata, "array_hashes": enriched["array_hashes"]}
    )
    save_npz(array_path, normalized_arrays, overwrite=overwrite)
    try:
        save_json(metadata_path, enriched, overwrite=overwrite)
    except Exception:
        # Avoid presenting an incomplete pair as a valid artifact on first write.
        if not overwrite and array_path.exists():
            array_path.unlink()
        raise
    return ArtifactPaths(array_path, metadata_path)


def load_array_artifact(
    directory: str | Path,
    *,
    basename: str = "artifact",
    validate_hashes: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    root = Path(directory)
    arrays = load_npz(root / f"{basename}.npz")
    metadata = load_json(root / f"{basename}_metadata.json")
    if validate_hashes:
        expected = metadata.get("array_hashes", {})
        actual = {name: hash_array(value) for name, value in arrays.items()}
        if expected != actual:
            raise ArtifactMismatchError(
                f"array hashes do not match metadata: expected {expected}, got {actual}"
            )
    return arrays, metadata


def validate_hash(name: str, expected: str | None, actual: str | None) -> None:
    if expected is not None and actual != expected:
        raise ArtifactMismatchError(f"{name} hash mismatch: expected {expected!r}, got {actual!r}")


def validate_upstream_hashes(expected: Mapping[str, str], actual: Mapping[str, str]) -> None:
    missing = sorted(set(expected) - set(actual))
    if missing:
        raise ArtifactMismatchError(f"missing upstream hashes: {missing}")
    for name, expected_hash in expected.items():
        validate_hash(name, expected_hash, actual.get(name))


__all__ = [
    "ArtifactError",
    "ArtifactMismatchError",
    "ArtifactPaths",
    "canonical_json_bytes",
    "hash_array",
    "hash_class_order",
    "hash_file",
    "load_array_artifact",
    "load_json",
    "load_npz",
    "save_array_artifact",
    "save_json",
    "save_npz",
    "sha256_file",
    "stable_hash",
    "to_jsonable",
    "utc_timestamp",
    "validate_hash",
    "validate_upstream_hashes",
]
