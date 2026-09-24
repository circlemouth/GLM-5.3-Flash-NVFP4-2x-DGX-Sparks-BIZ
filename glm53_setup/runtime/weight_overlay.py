# SPDX-License-Identifier: Apache-2.0
"""Optional, fail-closed BF16 o_proj replacement at GLM weight-load time.

The checkpoint iterator remains the owner of all non-target tensors.  Each
replacement is read separately so the donor is never held in memory at once.
"""

import hashlib
import json
import os
import re
from pathlib import Path

BASE_REPO = "nvidia/GLM-5.3-Flash-NVFP4"
DONOR_REPO = "dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4"
TARGET_LAYERS = (*range(15, 44), 45)
KEY = "model.language_model.layers.{layer}.self_attn.o_proj.weight"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_manifest(path, expected_digest, base_revision, donor_revision=None):
    """Check the complete immutable contract before a donor file is opened."""
    path = Path(path)
    if not SHA256.fullmatch(expected_digest) or _digest(path) != expected_digest:
        raise ValueError("Overlay manifest SHA-256 mismatch")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != 1 or data.get("base") != {
        "repo": BASE_REPO,
        "revision": base_revision,
    }:
        raise ValueError(
            "Overlay base checkpoint does not match the pinned NVIDIA revision"
        )
    donor = data.get("donor")
    if not isinstance(donor, dict) or donor.get("repo") != DONOR_REPO:
        raise ValueError("Overlay donor identity is missing")
    if not re.fullmatch(r"[0-9a-f]{40}", str(donor.get("revision", ""))):
        raise ValueError("Overlay donor revision must be immutable")
    if donor_revision is not None and donor["revision"] != donor_revision:
        raise ValueError("Overlay donor revision does not match launch profile")
    rows = data.get("tensors")
    if not isinstance(rows, list) or len(rows) != len(TARGET_LAYERS):
        raise ValueError("Overlay manifest must contain exactly 30 tensors")
    found = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Invalid overlay tensor row")
        layer = row.get("layer")
        if type(layer) is not int or layer not in TARGET_LAYERS or layer in found:
            raise ValueError("Unexpected or duplicated overlay layer")
        key = KEY.format(layer=layer)
        name = row.get("file")
        if row.get("key") != key or name != f"layer-{layer:02d}.safetensors":
            raise ValueError("Overlay tensor key or file does not match its layer")
        shape = row.get("shape")
        if (
            row.get("dtype") != "BF16"
            or not isinstance(shape, list)
            or len(shape) != 2
            or type(shape[0]) is not int
            or type(shape[1]) is not int
            or shape[0] != 4096
            or shape[1] not in (8192, 16384)
        ):
            raise ValueError("Overlay tensor dtype or shape is unsupported")
        if row.get("nbytes") != shape[0] * shape[1] * 2:
            raise ValueError("Overlay tensor byte size is invalid")
        if not all(
            SHA256.fullmatch(str(row.get(field, "")))
            for field in ("sha256", "tensor_sha256")
        ):
            raise ValueError("Overlay tensor SHA-256 is missing")
        if not isinstance(row.get("shard"), str) or not re.fullmatch(
            r"model-\d{5}-of-\d{5}\.safetensors", row["shard"]
        ):
            raise ValueError("Overlay donor shard is invalid")
        base = row.get("base")
        if (
            not isinstance(base, dict)
            or base.get("key") != key
            or base.get("dtype") != "BF16"
            or base.get("shape") != row["shape"]
            or base.get("nbytes") != row["nbytes"]
            or base.get("compatible") is not True
            or not isinstance(base.get("shard"), str)
            or not re.fullmatch(r"model-\d{5}-of-\d{5}\.safetensors", base["shard"])
        ):
            raise ValueError("Overlay target is incompatible with the NVIDIA tensor")
        found[layer] = row
    if set(found) != set(TARGET_LAYERS):
        raise ValueError("Overlay layer set is incomplete")
    return found


def verify_assets(directory, expected_digest, base_revision, donor_revision=None):
    """Verify all local extracted files without opening a model or GPU."""
    directory = Path(directory)
    rows = read_manifest(
        directory / "manifest.json", expected_digest, base_revision, donor_revision
    )
    total = 0
    for row in rows.values():
        path = directory / row["file"]
        if not path.is_file() or path.is_symlink() or _digest(path) != row["sha256"]:
            raise ValueError(f"Overlay file missing or changed: {row['file']}")
        total += row["nbytes"]
    return {
        "tensor_count": len(rows),
        "tensor_bytes": total,
        "manifest_sha256": expected_digest,
    }


def overlay_weights(
    weights,
    scope,
    *,
    manifest_path=None,
    manifest_sha256=None,
    base_revision=None,
    donor_revision=None,
    tensor_loader=None,
):
    """Replace only the scope's target tuples; preserve all extra tuple items."""
    manifest_path = manifest_path or os.environ.get("GLM53_WEIGHT_OVERLAY_MANIFEST")
    if manifest_path is None:
        yield from weights
        return
    manifest_sha256 = manifest_sha256 or os.environ.get("GLM53_WEIGHT_OVERLAY_SHA256")
    base_revision = base_revision or os.environ.get(
        "GLM53_WEIGHT_OVERLAY_BASE_REVISION"
    )
    donor_revision = donor_revision or os.environ.get(
        "GLM53_WEIGHT_OVERLAY_DONOR_REVISION"
    )
    if (
        scope not in {"main", "mtp"}
        or not manifest_sha256
        or not base_revision
        or not donor_revision
    ):
        raise ValueError("Overlay load configuration is incomplete")
    rows = read_manifest(manifest_path, manifest_sha256, base_revision, donor_revision)
    expected = set(range(15, 44)) if scope == "main" else {45}
    by_key = {rows[layer]["key"]: rows[layer] for layer in expected}
    seen = set()
    if tensor_loader is None:
        from safetensors.torch import load_file

        def tensor_loader(path, key):
            return load_file(str(path), device="cpu")[key]

    for item in weights:
        if not isinstance(item, tuple) or len(item) < 2:
            raise ValueError("Unexpected checkpoint weight tuple")
        name, original = item[:2]
        row = by_key.get(name)
        if row is None:
            yield item
            continue
        if name in seen:
            raise ValueError(f"Duplicate overlay target: {name}")
        seen.add(name)
        if (
            str(original.dtype) != "torch.bfloat16"
            or list(original.shape) != row["shape"]
        ):
            raise ValueError(f"Unsupported target dtype or shape: {name}")
        path = Path(manifest_path).parent / row["file"]
        if not path.is_file() or path.is_symlink() or _digest(path) != row["sha256"]:
            raise ValueError(f"Overlay tensor file missing or changed: {name}")
        donor = tensor_loader(path, name)
        if str(donor.dtype) != "torch.bfloat16" or list(donor.shape) != row["shape"]:
            raise ValueError(f"Donor tensor dtype or shape mismatch: {name}")
        yield (name, donor, *item[2:])
    if seen != set(by_key):
        missing = sorted(set(by_key) - seen)
        raise ValueError(
            f"Overlay target missing from checkpoint iterator: {missing[:2]}"
        )
