# SPDX-License-Identifier: Apache-2.0
"""Compare rank-local BF16 o_proj bytes with the pinned source tensor shards.

The worker RPC returns digests, names and shapes only. This host tool reads one
row at a time from Safetensors and never materializes a complete tensor.
"""

import argparse
import hashlib
import json
import re
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glm53_setup.runtime.weight_overlay import (  # noqa: E402
    KEY,
    read_manifest,
    verify_assets,
)


def tensor_location(path, key):
    with Path(path).open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError("Incomplete Safetensors header length")
        length = struct.unpack("<Q", raw)[0]
        if length > 32 * 1024 * 1024:
            raise ValueError("Safetensors header is too large")
        header = json.loads(stream.read(length))
        if key not in header:
            raise ValueError(f"Missing source tensor: {key}")
        item = header[key]
        if item.get("dtype") != "BF16" or len(item.get("shape", [])) != 2:
            raise ValueError(f"Source tensor is not BF16 matrix: {key}")
        shape = item["shape"]
        if any(type(dim) is not int or dim <= 0 for dim in shape):
            raise ValueError("Invalid source tensor shape")
        offsets = item.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or offsets[1] - offsets[0] != shape[0] * shape[1] * 2
            or offsets[0] < 0
        ):
            raise ValueError("Invalid source tensor offsets")
        start = 8 + length + offsets[0]
        if Path(path).stat().st_size < 8 + length + offsets[1]:
            raise ValueError("Incomplete Safetensors tensor")
    return start, shape


def expected_shard_sha256(path, key, loaded_shape, rank):
    """Hash the exact rank half in row-major order, with no implicit transform."""
    start, shape = tensor_location(path, key)
    if loaded_shape == shape:
        first_row, rows, first_col, cols = 0, shape[0], 0, shape[1]
        placement = "replicated"
    elif loaded_shape == [shape[0] // 2, shape[1]] and shape[0] % 2 == 0:
        first_row, rows, first_col, cols = (
            rank * loaded_shape[0],
            loaded_shape[0],
            0,
            shape[1],
        )
        placement = "row_split"
    elif loaded_shape == [shape[0], shape[1] // 2] and shape[1] % 2 == 0:
        first_row, rows, first_col, cols = (
            0,
            shape[0],
            rank * loaded_shape[1],
            loaded_shape[1],
        )
        placement = "column_split"
    else:
        raise ValueError(
            f"Loaded shape {loaded_shape} cannot be a TP=2 BF16 shard of {shape}"
        )
    digest = hashlib.sha256()
    row_bytes = shape[1] * 2
    chunk_bytes = cols * 2
    with Path(path).open("rb") as stream:
        for row in range(first_row, first_row + rows):
            stream.seek(start + row * row_bytes + first_col * 2)
            chunk = stream.read(chunk_bytes)
            if len(chunk) != chunk_bytes:
                raise ValueError("Source tensor ended during shard read")
            digest.update(chunk)
    return digest.hexdigest(), placement


def source_for_layer(layer, source, assets, snapshot, index, rows):
    key = KEY.format(layer=layer)
    if source == "donor":
        return assets / rows[layer]["file"], key
    shard = index.get(key)
    if not isinstance(shard, str) or Path(shard).name != shard:
        raise ValueError(f"Base index missing or invalid for layer {layer}")
    return snapshot / shard, key


def loaded_parameter(rank_rows, layer):
    suffix = re.compile(rf"(?:^|\.)layers\.{layer}\.self_attn\.o_proj\.weight$")
    matches = [row for row in rank_rows if suffix.search(row["name"])]
    if layer == 45 and not matches:
        matches = [
            row
            for row in rank_rows
            if re.fullmatch(
                r"[^:]+:(?:model\.)?layers\.(?:0|45\.mtp_block)\.self_attn\.o_proj\.weight",
                row["name"],
            )
        ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one loaded o_proj for layer {layer}, found {len(matches)}"
        )
    row = matches[0]
    if row.get("dtype") != "BF16" or not re.fullmatch(
        r"[0-9a-f]{64}", row.get("sha256", "")
    ):
        raise ValueError(f"Invalid loaded o_proj digest for layer {layer}")
    return row


def selected_layers(source, mtp, only_layers=None):
    expected = (*range(15, 44), 44) if source == "base" else tuple(range(15, 44))
    if mtp:
        expected += (45,)
    if only_layers is None:
        return expected
    if (
        not only_layers
        or len(set(only_layers)) != len(only_layers)
        or not set(only_layers) <= set(expected)
    ):
        raise ValueError("Requested readback layers are outside the selected source")
    return tuple(layer for layer in expected if layer in only_layers)


def verify(
    loaded,
    *,
    source,
    assets,
    snapshot,
    manifest_sha256,
    base_revision,
    donor_revision,
    mtp,
    only_layers=None,
):
    rows = read_manifest(
        assets / "manifest.json", manifest_sha256, base_revision, donor_revision
    )
    if source == "donor":
        verify_assets(assets, manifest_sha256, base_revision, donor_revision)
    index = {}
    if source == "base":
        index = json.loads((snapshot / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
    expected_layers = selected_layers(source, mtp, only_layers)
    if {entry.get("rank") for entry in loaded} != {0, 1} or len(loaded) != 2:
        raise ValueError("Readback must contain exactly ranks 0 and 1")
    results = []
    for rank_record in loaded:
        rank = rank_record["rank"]
        rank_rows = rank_record.get("rows")
        if not isinstance(rank_rows, list):
            raise ValueError("Missing rank readback rows")
        for layer in expected_layers:
            actual = loaded_parameter(rank_rows, layer)
            path, key = source_for_layer(layer, source, assets, snapshot, index, rows)
            expected, placement = expected_shard_sha256(
                path, key, actual["shape"], rank
            )
            results.append(
                {
                    "rank": rank,
                    "layer": layer,
                    "name": actual["name"],
                    "placement": placement,
                    "expected_sha256": expected,
                    "loaded_sha256": actual["sha256"],
                    "match": expected == actual["sha256"],
                }
            )
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Live profile for worker readback")
    parser.add_argument(
        "--loaded-json", type=Path, help="Saved worker readback instead of live RPC"
    )
    parser.add_argument("--source", choices=("base", "donor"), required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--base-snapshot", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--donor-revision", required=True)
    parser.add_argument("--mtp", action="store_true")
    parser.add_argument(
        "--only-layer",
        type=int,
        action="append",
        dest="only_layers",
        help="Limit comparison to one source layer; repeat for more than one",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if bool(args.config) == bool(args.loaded_json):
        parser.error("Specify exactly one of --config and --loaded-json")
    if args.loaded_json:
        loaded = json.loads(args.loaded_json.read_text())
    else:
        from glm53_setup import server, server_config

        profile = server_config.load(args.config)
        loaded = server.collective_rpc(profile, "o_proj_sha256")
    results = verify(
        loaded,
        source=args.source,
        assets=args.assets,
        snapshot=args.base_snapshot,
        manifest_sha256=args.manifest_sha256,
        base_revision=args.base_revision,
        donor_revision=args.donor_revision,
        mtp=args.mtp,
        only_layers=args.only_layers,
    )
    result = {
        "source": args.source,
        "mtp": args.mtp,
        "readback": loaded,
        "rows": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    matches = sum(row["match"] for row in results)
    print(
        f"o_proj readback: {matches}/{len(results)} rank-local tensors match {args.source}"
    )
    return 0 if matches == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
