# SPDX-License-Identifier: Apache-2.0
"""Inspect and extract the BF16 o_proj tensors used by the local overlay.

The command has two deliberately separate modes.  ``headers-only`` reads the
two checkpoint indexes and the safetensors headers for the thirty target
tensors.  ``extract`` performs the same checks and then downloads only the
target tensor byte ranges.  No model framework is imported and no complete
checkpoint shard is downloaded by this module.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as _datetime
import hashlib
import io
import json
import os
import re
import struct
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import BinaryIO, Callable, Iterable

try:  # Linux uses the required advisory flock; Windows keeps import/tests usable.
    import fcntl
except ImportError:  # pragma: no cover - exercised by Windows CI
    fcntl = None

ROOT = Path(__file__).resolve().parents[1]
BASE_REPO = "nvidia/GLM-5.3-Flash-NVFP4"
DONOR_REPO = "dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4"
DONOR_REVISION = "745aac2ff0f10acf961f396df3f9418598aa7327"
TARGET_LAYERS = (*range(15, 44), 45)
TARGET_DTYPE = "BF16"
INDEX_NAME = "model.safetensors.index.json"
TOOL_VERSION = "1"
UPSTREAM_LICENSE = "https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/LICENSE"
JST = _datetime.timezone(_datetime.timedelta(hours=9))

_SHARD_NAME = re.compile(r"model-\d{5}-of-\d{5}\.safetensors\Z")
_CONTENT_RANGE = re.compile(r"bytes ([0-9]+)-([0-9]+)/([0-9]+)\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MAX_INDEX_BYTES = 64 * 1024 * 1024
_MAX_HEADER_BYTES = 64 * 1024 * 1024
_COPY_CHUNK = 1024 * 1024


def _absolute(path: Path) -> Path:
    """Make a user path absolute without resolving a final symlink."""

    return path.expanduser().absolute()


class PrepareError(RuntimeError):
    """An expected contract, transport, or output validation failure."""


class RangeError(PrepareError):
    """The origin did not return the exact byte range requested."""


@dataclasses.dataclass
class TransferStats:
    """Private accounting for bytes received from the public origin."""

    transfer_bytes: int = 0
    range_requests: int = 0
    index_requests: int = 0

    def add(self, count: int) -> None:
        if count < 0:
            raise ValueError("negative transfer count")
        self.transfer_bytes += count


class RangeClient:
    """Small stdlib-only HTTP client with strict range response checks."""

    def __init__(
        self,
        endpoint: str = "https://huggingface.co",
        *,
        opener: Callable[..., object] | None = None,
        timeout: float = 60.0,
        stats: TransferStats | None = None,
    ) -> None:
        endpoint = endpoint.rstrip("/")
        parsed = urllib.parse.urlparse(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("endpoint must be an absolute HTTP(S) URL")
        self.endpoint = endpoint
        self.opener = opener or urllib.request.urlopen
        self.timeout = timeout
        self.stats = stats or TransferStats()

    def url(self, repo: str, revision: str, name: str) -> str:
        if not repo or not revision or not name:
            raise ValueError("repo, revision, and name are required")
        # Repository names contain one slash; keep it as a path separator while
        # quoting every other component so a filename cannot escape ``resolve``.
        repo_part = urllib.parse.quote(repo, safe="/")
        revision_part = urllib.parse.quote(revision, safe="")
        name_part = urllib.parse.quote(name, safe="/")
        return f"{self.endpoint}/{repo_part}/resolve/{revision_part}/{name_part}"

    @staticmethod
    def _status(response: object) -> int:
        value = getattr(response, "status", None)
        if value is None:
            value = response.getcode()  # type: ignore[attr-defined]
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise PrepareError("HTTP response has no numeric status") from exc

    @staticmethod
    def _header(response: object, name: str) -> str | None:
        headers = getattr(response, "headers", None)
        if headers is not None:
            value = headers.get(name)
            if value is not None:
                return str(value)
            items = getattr(headers, "items", None)
            if items is not None:
                for key, candidate in items():
                    if str(key).lower() == name.lower():
                        return str(candidate)
        getter = getattr(response, "getheader", None)
        if getter is not None:
            value = getter(name)
            if value is not None:
                return str(value)
        return None

    @staticmethod
    def _read_limited(response: object, limit: int, stats: TransferStats) -> bytes:
        result = bytearray()
        while True:
            try:
                chunk = response.read(min(_COPY_CHUNK, limit - len(result) + 1))  # type: ignore[attr-defined]
            except OSError as exc:
                raise PrepareError("HTTP response read failed") from exc
            if not chunk:
                break
            stats.add(len(chunk))
            result.extend(chunk)
            if len(result) > limit:
                raise PrepareError("HTTP response exceeds the allowed size")
        return bytes(result)

    def get_json(self, url: str) -> dict:
        request = urllib.request.Request(
            url,
            headers={"Accept": "application/json"},
            method="GET",
        )
        self.stats.index_requests += 1
        try:
            with self.opener(request, timeout=self.timeout) as response:  # type: ignore[misc]
                if self._status(response) != 200:
                    raise PrepareError("checkpoint index did not return HTTP 200")
                raw = self._read_limited(response, _MAX_INDEX_BYTES, self.stats)
        except urllib.error.HTTPError as exc:
            raise PrepareError(
                f"checkpoint index request failed: HTTP {exc.code}"
            ) from exc
        except urllib.error.URLError as exc:
            raise PrepareError("checkpoint index request failed") from exc
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PrepareError("checkpoint index is not valid UTF-8 JSON") from exc
        if not isinstance(data, dict) or not isinstance(data.get("weight_map"), dict):
            raise PrepareError("checkpoint index has no weight_map object")
        return data

    @contextlib.contextmanager
    def _range_response(
        self,
        url: str,
        start: int,
        end: int,
        *,
        total_size: int | None = None,
    ) -> Iterable[tuple[object, int]]:
        if start < 0 or end < start:
            raise ValueError("invalid byte range")
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/octet-stream",
                "Range": f"bytes={start}-{end}",
            },
            method="GET",
        )
        expected_length = end - start + 1
        self.stats.range_requests += 1
        try:
            response = self.opener(request, timeout=self.timeout)  # type: ignore[misc]
            with response:
                if self._status(response) != 206:
                    raise RangeError("origin did not honor Range (expected HTTP 206)")
                content_range = self._header(response, "Content-Range")
                if content_range is None:
                    raise RangeError("HTTP 206 response has no Content-Range")
                match = _CONTENT_RANGE.fullmatch(content_range.strip())
                if match is None:
                    raise RangeError("invalid Content-Range header")
                actual_start, actual_end, actual_total = map(int, match.groups())
                if (actual_start, actual_end) != (start, end):
                    raise RangeError("Content-Range does not match the requested range")
                if actual_total <= actual_end:
                    raise RangeError("Content-Range total is too small")
                if total_size is not None and actual_total != total_size:
                    raise RangeError("shard length changed between range requests")
                content_length = self._header(response, "Content-Length")
                if content_length is None:
                    raise RangeError("HTTP 206 response has no Content-Length")
                try:
                    actual_length = int(content_length.strip())
                except ValueError as exc:
                    raise RangeError("invalid Content-Length header") from exc
                if actual_length != expected_length:
                    raise RangeError(
                        "Content-Length does not match the requested range"
                    )
                yield response, actual_total
        except urllib.error.HTTPError as exc:
            raise RangeError(f"range request failed: HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise RangeError("range request failed") from exc

    def read_range(
        self,
        url: str,
        start: int,
        end: int,
        *,
        total_size: int | None = None,
    ) -> tuple[bytes, int]:
        expected_length = end - start + 1
        buffer = io.BytesIO()
        with self._range_response(url, start, end, total_size=total_size) as (
            response,
            actual_total,
        ):
            self._copy_response(response, buffer, expected_length)
        return buffer.getvalue(), actual_total

    def stream_range(
        self,
        url: str,
        start: int,
        end: int,
        destination: BinaryIO,
        *,
        total_size: int | None = None,
    ) -> int:
        expected_length = end - start + 1
        with self._range_response(url, start, end, total_size=total_size) as (
            response,
            actual_total,
        ):
            self._copy_response(response, destination, expected_length)
        return actual_total

    def _copy_response(
        self, response: object, destination: BinaryIO, expected_length: int
    ) -> None:
        received = 0
        while received < expected_length:
            try:
                chunk = response.read(  # type: ignore[attr-defined]
                    min(_COPY_CHUNK, expected_length - received)
                )
            except OSError as exc:
                raise RangeError("range response read failed") from exc
            if not chunk:
                raise RangeError("range response was truncated")
            if len(chunk) > expected_length - received:
                raise RangeError("range response exceeded Content-Length")
            destination.write(chunk)
            received += len(chunk)
            self.stats.add(len(chunk))
        try:
            extra = response.read(1)  # type: ignore[attr-defined]
        except OSError as exc:
            raise RangeError("range response read failed") from exc
        if extra:
            self.stats.add(len(extra))
            raise RangeError("range response contained bytes beyond Content-Length")


@dataclasses.dataclass(frozen=True)
class TensorSpec:
    layer: int
    key: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int
    donor_shard: str
    donor_offsets: tuple[int, int]
    donor_data_start: int
    donor_size: int
    base_shard: str
    base_offsets: tuple[int, int]
    base_data_start: int
    base_size: int

    @property
    def filename(self) -> str:
        return f"layer-{self.layer:02d}.safetensors"


@dataclasses.dataclass
class Inspection:
    base_repo: str
    base_revision: str
    donor_repo: str
    donor_revision: str
    specs: list[TensorSpec]

    def report(self) -> dict:
        tensors = []
        for spec in self.specs:
            tensors.append(
                {
                    "layer": spec.layer,
                    "key": spec.key,
                    "dtype": spec.dtype,
                    "shape": list(spec.shape),
                    "nbytes": spec.nbytes,
                    "donor": {
                        "shard": spec.donor_shard,
                        "data_offsets": list(spec.donor_offsets),
                        "data_start": spec.donor_data_start,
                        "shard_nbytes": spec.donor_size,
                    },
                    "base": {
                        "shard": spec.base_shard,
                        "data_offsets": list(spec.base_offsets),
                        "data_start": spec.base_data_start,
                        "shard_nbytes": spec.base_size,
                        "compatible": True,
                    },
                }
            )
        return {
            "schema": 1,
            "mode": "headers-only",
            "base": {"repo": self.base_repo, "revision": self.base_revision},
            "donor": {"repo": self.donor_repo, "revision": self.donor_revision},
            "tensor_count": len(tensors),
            "tensors": tensors,
        }


class RemoteCheckpoint:
    """Index and lazily fetched safetensors headers for one revision."""

    def __init__(self, client: RangeClient, repo: str, revision: str) -> None:
        self.client = client
        self.repo = repo
        self.revision = revision
        self.index = client.get_json(client.url(repo, revision, INDEX_NAME))
        weight_map = self.index["weight_map"]
        self.weight_map = dict(weight_map)
        self.headers: dict[str, tuple[int, int, dict]] = {}

    def shard_for(self, key: str) -> str:
        value = self.weight_map.get(key)
        if not isinstance(value, str) or not _SHARD_NAME.fullmatch(value):
            raise PrepareError(f"{self.repo} index has an invalid shard for {key}")
        return value

    def header_for(self, shard: str) -> tuple[int, int, dict]:
        cached = self.headers.get(shard)
        if cached is not None:
            return cached
        url = self.client.url(self.repo, self.revision, shard)
        prefix, total_size = self.client.read_range(url, 0, 7)
        if len(prefix) != 8:
            raise PrepareError(f"safetensors header prefix is truncated: {shard}")
        header_size = struct.unpack("<Q", prefix)[0]
        if not 0 < header_size <= _MAX_HEADER_BYTES:
            raise PrepareError(f"invalid safetensors header length: {shard}")
        header_end = 8 + header_size - 1
        if header_end >= total_size:
            raise PrepareError(f"safetensors header exceeds shard: {shard}")
        header_raw, repeated_total = self.client.read_range(
            url, 8, header_end, total_size=total_size
        )
        if repeated_total != total_size or len(header_raw) != header_size:
            raise PrepareError(f"safetensors header length mismatch: {shard}")
        try:
            header = json.loads(header_raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PrepareError(
                f"safetensors header is not valid JSON: {shard}"
            ) from exc
        if not isinstance(header, dict):
            raise PrepareError(f"safetensors header is not an object: {shard}")
        result = (total_size, header_size, header)
        self.headers[shard] = result
        return result


def pinned_base_revision(lock_path: Path | None = None) -> str:
    """Read the base revision from the repository's runtime lock."""

    path = lock_path or ROOT / "config" / "runtime.lock.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PrepareError("cannot read config/runtime.lock.json") from exc
    if data.get("model") != BASE_REPO or not isinstance(data.get("revision"), str):
        raise PrepareError("runtime lock does not pin the expected base checkpoint")
    if not re.fullmatch(r"[0-9a-f]{40}", data["revision"]):
        raise PrepareError("runtime lock has an invalid base revision")
    return data["revision"]


def _validate_target_meta(
    checkpoint: RemoteCheckpoint, key: str
) -> tuple[str, int, tuple[int, ...], tuple[int, int], int, int]:
    shard = checkpoint.shard_for(key)
    total_size, header_size, header = checkpoint.header_for(shard)
    value = header.get(key)
    if not isinstance(value, dict):
        raise PrepareError(f"{checkpoint.repo} shard header lacks {key}")
    if value.get("dtype") != TARGET_DTYPE:
        raise PrepareError(f"{checkpoint.repo} target is not BF16: {key}")
    shape = value.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(type(item) is not int or item <= 0 for item in shape)
    ):
        raise PrepareError(f"{checkpoint.repo} target shape is invalid: {key}")
    offsets = value.get("data_offsets")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or any(type(item) is not int for item in offsets)
        or offsets[0] < 0
        or offsets[1] < offsets[0]
    ):
        raise PrepareError(f"invalid data_offsets for {checkpoint.repo}: {key}")
    start, end = offsets
    nbytes = end - start
    expected = shape[0] * shape[1] * 2
    if nbytes != expected:
        raise PrepareError(f"shape/byte size mismatch for {checkpoint.repo}: {key}")
    # The offsets are relative to the data section.  A conservative check using
    # the minimum possible header prefix catches impossible ranges.  The range
    # request is still bounded by the complete remote shard length below.
    data_start = 8 + header_size
    if data_start + end > total_size:
        raise PrepareError(f"tensor data exceeds shard for {checkpoint.repo}: {key}")
    return shard, total_size, tuple(shape), (start, end), nbytes, data_start


def inspect_sources(
    client: RangeClient,
    *,
    base_repo: str = BASE_REPO,
    base_revision: str | None = None,
    donor_repo: str = DONOR_REPO,
    donor_revision: str = DONOR_REVISION,
    target_layers: tuple[int, ...] = TARGET_LAYERS,
) -> Inspection:
    """Read both indexes and all target shard headers, without tensor bytes."""

    if base_revision is None:
        base_revision = pinned_base_revision()
    if len(set(target_layers)) != len(target_layers):
        raise PrepareError("target layer list contains duplicates")
    base = RemoteCheckpoint(client, base_repo, base_revision)
    donor = RemoteCheckpoint(client, donor_repo, donor_revision)
    specs = []
    for layer in target_layers:
        key = f"model.language_model.layers.{layer}.self_attn.o_proj.weight"
        (
            base_shard,
            base_size,
            base_shape,
            base_offsets,
            base_nbytes,
            base_data_start,
        ) = _validate_target_meta(base, key)
        (
            donor_shard,
            donor_size,
            donor_shape,
            donor_offsets,
            donor_nbytes,
            donor_data_start,
        ) = _validate_target_meta(donor, key)
        if base_shape != donor_shape or base_nbytes != donor_nbytes:
            raise PrepareError(f"base/donor tensor shape or size mismatch: {key}")
        specs.append(
            TensorSpec(
                layer=layer,
                key=key,
                dtype=TARGET_DTYPE,
                shape=donor_shape,
                nbytes=donor_nbytes,
                donor_shard=donor_shard,
                donor_offsets=donor_offsets,
                donor_data_start=donor_data_start,
                donor_size=donor_size,
                base_shard=base_shard,
                base_offsets=base_offsets,
                base_data_start=base_data_start,
                base_size=base_size,
            )
        )
    if len(specs) != len(target_layers) or len({spec.key for spec in specs}) != len(
        specs
    ):
        raise PrepareError("target tensor set is duplicated")
    return Inspection(base_repo, base_revision, donor_repo, donor_revision, specs)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_COPY_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, data: dict) -> None:
    if path.is_symlink():
        raise PrepareError(f"refusing to replace symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _manifest_row(spec: TensorSpec, hashes: dict[str, str]) -> dict:
    return {
        "layer": spec.layer,
        "key": spec.key,
        "file": spec.filename,
        "dtype": spec.dtype,
        "shape": list(spec.shape),
        "nbytes": spec.nbytes,
        "sha256": hashes["sha256"],
        "tensor_sha256": hashes["tensor_sha256"],
        "shard": spec.donor_shard,
        "base": {
            "key": spec.key,
            "dtype": spec.dtype,
            "shape": list(spec.shape),
            "nbytes": spec.nbytes,
            "shard": spec.base_shard,
            "data_offsets": list(spec.base_offsets),
            "compatible": True,
        },
    }


def _manifest_header(inspection: Inspection) -> dict:
    return {
        "schema": 1,
        "base": {"repo": inspection.base_repo, "revision": inspection.base_revision},
        "donor": {
            "repo": inspection.donor_repo,
            "revision": inspection.donor_revision,
        },
        "tool": {"name": "prepare_abliteration.py", "version": TOOL_VERSION},
    }


def _manifest_data(inspection: Inspection, rows: list[dict]) -> dict:
    data = _manifest_header(inspection)
    data["acquired_date"] = _datetime.datetime.now(JST).date().isoformat()
    data["license_references"] = {
        "base_model_card": f"https://huggingface.co/{inspection.base_repo}",
        "base_revision_snapshot": (
            f"https://huggingface.co/{inspection.base_repo}/tree/{inspection.base_revision}"
        ),
        "donor_model_card": f"https://huggingface.co/{inspection.donor_repo}",
        "donor_revision_snapshot": (
            f"https://huggingface.co/{inspection.donor_repo}/tree/{inspection.donor_revision}"
        ),
        "donor_license": (
            f"https://huggingface.co/{inspection.donor_repo}/blob/"
            f"{inspection.donor_revision}/LICENSE"
        ),
        "upstream_license": UPSTREAM_LICENSE,
    }
    data["tensors"] = rows
    return data


def _manifest_identity_matches(data: object, inspection: Inspection) -> bool:
    if not isinstance(data, dict):
        return False
    expected = _manifest_header(inspection)
    for key, value in expected.items():
        if key == "tool":
            actual_tool = data.get("tool")
            if not isinstance(actual_tool, dict):
                return False
            if any(
                actual_tool.get(name) != tool_value
                for name, tool_value in value.items()
            ):
                return False
        elif data.get(key) != value:
            return False
    return True


def _validate_row(row: object, spec: TensorSpec) -> bool:
    if not isinstance(row, dict):
        return False
    base = row.get("base")
    return (
        row.get("layer") == spec.layer
        and row.get("key") == spec.key
        and row.get("file") == spec.filename
        and row.get("dtype") == spec.dtype
        and row.get("shape") == list(spec.shape)
        and row.get("nbytes") == spec.nbytes
        and isinstance(row.get("sha256"), str)
        and _SHA256.fullmatch(row["sha256"]) is not None
        and isinstance(row.get("tensor_sha256"), str)
        and _SHA256.fullmatch(row["tensor_sha256"]) is not None
        and row.get("shard") == spec.donor_shard
        and isinstance(base, dict)
        and base.get("key") == spec.key
        and base.get("dtype") == spec.dtype
        and base.get("shape") == list(spec.shape)
        and base.get("nbytes") == spec.nbytes
        and base.get("shard") == spec.base_shard
        and base.get("compatible") is True
    )


def _load_state(path: Path, inspection: Inspection) -> dict[int, dict]:
    if not path.is_file() or path.is_symlink():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PrepareError("partial manifest is invalid") from exc
    if not _manifest_identity_matches(data, inspection):
        raise PrepareError("partial manifest checkpoint identity mismatch")
    rows = data.get("tensors")
    if not isinstance(rows, list):
        raise PrepareError("partial manifest tensors is not a list")
    by_layer = {}
    specs = {spec.layer: spec for spec in inspection.specs}
    for row in rows:
        if not isinstance(row, dict) or type(row.get("layer")) is not int:
            raise PrepareError("partial manifest has an invalid tensor layer")
        if row.get("layer") in by_layer:
            raise PrepareError("partial manifest has duplicate tensor rows")
        layer = row.get("layer")
        if layer not in specs or not _validate_row(row, specs[layer]):
            raise PrepareError("partial manifest tensor row is incompatible")
        by_layer[layer] = row
    return by_layer


def _verify_safetensor_file(path: Path, spec: TensorSpec, row: dict) -> bool:
    if not path.is_file() or path.is_symlink():
        return False
    if _sha256_file(path) != row["sha256"]:
        return False
    try:
        with path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                return False
            header_size = struct.unpack("<Q", prefix)[0]
            if not 0 < header_size <= _MAX_HEADER_BYTES:
                return False
            header = json.loads(stream.read(header_size).decode("utf-8"))
            if not isinstance(header, dict) or len(header) != 1:
                return False
            value = header.get(spec.key)
            if not isinstance(value, dict):
                return False
            if value.get("dtype") != spec.dtype or value.get("shape") != list(
                spec.shape
            ):
                return False
            offsets = value.get("data_offsets")
            if offsets != [0, spec.nbytes]:
                return False
            if path.stat().st_size != 8 + header_size + spec.nbytes:
                return False
            tensor_digest = hashlib.sha256()
            remaining = spec.nbytes
            while remaining:
                chunk = stream.read(min(_COPY_CHUNK, remaining))
                if not chunk:
                    return False
                tensor_digest.update(chunk)
                remaining -= len(chunk)
            return tensor_digest.hexdigest() == row["tensor_sha256"]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, struct.error):
        return False


def _verify_part(path: Path, spec: TensorSpec, row: dict | None) -> bool:
    if not path.is_file() or path.is_symlink():
        return False
    try:
        if row is None:
            # A part without a manifest row has no trusted digest.  Its bytes
            # may have come from a crashed or replaced origin, so restart this
            # tensor from its declared donor range.
            return False
        if path.stat().st_size != spec.nbytes:
            return False
        return _sha256_file(path) == row["tensor_sha256"]
    except OSError:
        return False


def _materialize_safetensor(
    part: Path, final: Path, spec: TensorSpec
) -> dict[str, str]:
    if not part.is_file() or part.is_symlink() or part.stat().st_size != spec.nbytes:
        raise PrepareError(f"tensor part is incomplete: {part.name}")
    if final.exists() and final.is_symlink():
        raise PrepareError(f"refusing to replace symlink: {final}")
    header_json = json.dumps(
        {
            spec.key: {
                "dtype": spec.dtype,
                "shape": list(spec.shape),
                "data_offsets": [0, spec.nbytes],
            }
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    # The safetensors format aligns the JSON header to an eight-byte boundary.
    header_raw = header_json + b" " * (-len(header_json) % 8)
    temporary = final.with_name(f".{final.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    tensor_digest = hashlib.sha256()
    with part.open("rb") as source, temporary.open("wb") as target:
        target.write(struct.pack("<Q", len(header_raw)))
        target.write(header_raw)
        remaining = spec.nbytes
        while remaining:
            chunk = source.read(min(_COPY_CHUNK, remaining))
            if not chunk:
                raise PrepareError(f"tensor part ended early: {part.name}")
            target.write(chunk)
            tensor_digest.update(chunk)
            remaining -= len(chunk)
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary, final)
    return {
        "sha256": _sha256_file(final),
        "tensor_sha256": tensor_digest.hexdigest(),
    }


def _download_tensor(
    client: RangeClient,
    spec: TensorSpec,
    output: Path,
    state_row: dict | None,
    *,
    donor_repo: str,
    donor_revision: str,
) -> dict[str, str]:
    part = output / f"{spec.filename}.part"
    final = output / spec.filename
    if part.exists() and not _verify_part(part, spec, state_row):
        if part.is_symlink():
            raise PrepareError(f"refusing to replace symlink: {part}")
        part.unlink()
    if final.exists():
        if state_row is not None and _verify_safetensor_file(final, spec, state_row):
            return {
                "sha256": state_row["sha256"],
                "tensor_sha256": state_row["tensor_sha256"],
            }
        if state_row is not None:
            raise PrepareError(f"existing tensor is corrupt: {final.name}")
        raise PrepareError(f"orphan tensor file requires review: {final.name}")
    if not part.exists():
        if part.is_symlink():
            raise PrepareError(f"refusing to replace symlink: {part}")
        part.touch(mode=0o600, exist_ok=False)
    current = part.stat().st_size
    if current < spec.nbytes:
        start = spec.donor_data_start + spec.donor_offsets[0] + current
        end = spec.donor_data_start + spec.donor_offsets[1] - 1
        url = client.url(donor_repo, donor_revision, spec.donor_shard)
        with part.open("ab") as destination:
            client.stream_range(
                url,
                start,
                end,
                destination,
                total_size=spec.donor_size,
            )
    if part.stat().st_size != spec.nbytes:
        raise PrepareError(f"tensor part has an unexpected length: {part.name}")
    return _materialize_safetensor(part, final, spec)


@contextlib.contextmanager
def _output_lock(output: Path) -> Iterable[None]:
    lock = output.with_name(f".{output.name}.lock")
    if lock.is_symlink():
        raise PrepareError(f"refusing to use symlink lock: {lock}")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+b") as stream:
        if fcntl is not None:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            return
        if os.name != "nt":
            raise PrepareError("extract requires a platform lock implementation")
        import msvcrt  # pragma: no cover - exercised by Windows CI

        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def extract(
    client: RangeClient,
    inspection: Inspection,
    output: Path,
) -> dict:
    """Download each target range independently and atomically build output."""

    output = _absolute(output)
    if output.exists() and output.is_symlink():
        raise PrepareError("output directory must not be a symlink")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    partial_path = output / "manifest.json.part"
    with _output_lock(output):
        if manifest_path.is_file() and not manifest_path.is_symlink():
            try:
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise PrepareError("manifest.json is invalid") from exc
            if not _manifest_identity_matches(existing, inspection):
                raise PrepareError("manifest checkpoint identity mismatch")
            rows = existing.get("tensors")
            if not isinstance(rows, list) or len(rows) != len(inspection.specs):
                raise PrepareError("manifest tensor set is incomplete")
            by_layer = {}
            for row in rows:
                layer = row.get("layer") if isinstance(row, dict) else None
                if type(layer) is not int or layer in by_layer:
                    raise PrepareError("manifest tensor set contains duplicates")
                by_layer[layer] = row
            for spec in inspection.specs:
                row = by_layer.get(spec.layer)
                if not _validate_row(row, spec) or not _verify_safetensor_file(
                    output / spec.filename, spec, row
                ):
                    raise PrepareError(f"manifest tensor is corrupt: {spec.filename}")
            return existing
        if manifest_path.exists():
            raise PrepareError("manifest.json must be a regular file")
        by_layer = _load_state(partial_path, inspection)
        rows = []
        for spec in inspection.specs:
            row = by_layer.get(spec.layer)
            hashes = _download_tensor(
                client,
                spec,
                output,
                row,
                donor_repo=inspection.donor_repo,
                donor_revision=inspection.donor_revision,
            )
            if row is not None and (
                row["sha256"] != hashes["sha256"]
                or row["tensor_sha256"] != hashes["tensor_sha256"]
            ):
                raise PrepareError(f"tensor changed during resume: {spec.filename}")
            row = _manifest_row(spec, hashes)
            by_layer[spec.layer] = row
            rows = [
                by_layer[specification.layer]
                for specification in inspection.specs
                if specification.layer in by_layer
            ]
            _atomic_json(partial_path, _manifest_data(inspection, rows))
        result = _manifest_data(inspection, rows)
        _atomic_json(manifest_path, result)
        if partial_path.exists():
            partial_path.unlink()
        for spec in inspection.specs:
            part = output / f"{spec.filename}.part"
            if part.exists():
                if part.is_symlink():
                    raise PrepareError(f"refusing to remove symlink: {part}")
                part.unlink()
        return result


def _receipt_path(output: Path, supplied: Path | None) -> Path:
    if supplied is not None:
        return _absolute(supplied)
    output = _absolute(output)
    return output.parent / (f".{output.name}.prepare-abliteration.receipt.json")


def _write_receipt(
    path: Path, stats: TransferStats, *, status: str, error: str | None = None
) -> None:
    payload = {
        "schema": 1,
        "tool": "prepare_abliteration.py",
        "version": TOOL_VERSION,
        "mode": "extract",
        "status": status,
        "transfer_bytes": stats.transfer_bytes,
        "range_requests": stats.range_requests,
        "index_requests": stats.index_requests,
        "recorded_at": _datetime.datetime.now(JST).isoformat(),
    }
    if error:
        payload["error"] = error
    _atomic_json(path, payload)


def _write_report(path: Path, report: dict) -> None:
    _atomic_json(path, report)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        nargs="?",
        choices=("inspect", "headers-only", "extract"),
        help="operation to perform; inspect is the headers-only mode",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--headers-only", action="store_true")
    mode.add_argument("--extract", action="store_true")
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--output", type=Path, help="extract directory or report path")
    parser.add_argument(
        "--report", type=Path, help="write the inspection/extraction report"
    )
    parser.add_argument("--receipt", type=Path, help="private transfer receipt path")
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    selected = args.command
    if args.headers_only:
        if selected is not None:
            parser.error("choose either a command or --headers-only")
        selected = "headers-only"
    if args.extract:
        if selected is not None:
            parser.error("choose either a command or --extract")
        selected = "extract"
    if selected is None:
        parser.error("select headers-only or extract")
    if selected == "inspect":
        selected = "headers-only"
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if selected == "extract" and args.output is None:
        parser.error("extract requires --output")
    if selected == "headers-only" and args.receipt is not None:
        parser.error("--receipt is only valid for extract")
    stats = TransferStats()
    try:
        client = RangeClient(args.endpoint, timeout=args.timeout, stats=stats)
        inspection = inspect_sources(
            client,
            base_repo=BASE_REPO,
            base_revision=pinned_base_revision(),
            donor_repo=DONOR_REPO,
            donor_revision=DONOR_REVISION,
        )
        if selected == "headers-only":
            report = inspection.report()
            if args.report is not None:
                _write_report(_absolute(args.report), report)
            elif args.output is not None:
                output = _absolute(args.output)
                if output.is_symlink():
                    raise PrepareError("inspection output must not be a symlink")
                if output.exists() and output.is_dir():
                    _write_report(output / "headers.json", report)
                elif output.suffix.lower() == ".json":
                    _write_report(output, report)
                else:
                    output.mkdir(parents=True, exist_ok=True)
                    _write_report(output / "headers.json", report)
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        output = _absolute(args.output)
        receipt = _receipt_path(output, args.receipt)
        try:
            manifest = extract(client, inspection, output)
            if args.report is not None:
                _write_report(_absolute(args.report), manifest)
            _write_receipt(receipt, stats, status="complete")
            print(json.dumps(manifest, indent=2, sort_keys=True))
            return 0
        except Exception as exc:
            try:
                _write_receipt(receipt, stats, status="failed", error=str(exc))
            except OSError:
                pass
            raise
    except (PrepareError, OSError, ValueError) as exc:
        print(f"prepare_abliteration: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
