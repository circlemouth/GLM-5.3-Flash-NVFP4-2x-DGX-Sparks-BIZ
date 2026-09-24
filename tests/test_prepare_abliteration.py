# SPDX-License-Identifier: Apache-2.0

import hashlib
import io
import json
import struct
import tempfile
import unittest
from pathlib import Path
from urllib.parse import urlparse

from tools import prepare_abliteration as prepare


class _Response:
    def __init__(self, status, headers, content):
        self.status = status
        self.headers = headers
        self._content = io.BytesIO(content)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def getcode(self):
        return self.status

    def read(self, size=-1):
        return self._content.read(size)


class _Origin:
    def __init__(
        self,
        *,
        base_repo="base/model",
        donor_repo="donor/model",
        shapes=None,
        donor_shapes=None,
    ):
        self.base_repo = base_repo
        self.donor_repo = donor_repo
        self.base_revision = "base-revision"
        self.donor_revision = "donor-revision"
        self.shapes = shapes or {15: (1, 4)}
        self.donor_shapes = donor_shapes or self.shapes
        self.keys = {
            layer: f"model.language_model.layers.{layer}.self_attn.o_proj.weight"
            for layer in self.shapes
        }
        self.key = self.keys[15]
        self.raw = b"abcdefgh"
        self.base_shard = self._shard(self.shapes)
        self.donor_shard = self._shard(self.donor_shapes)
        self.shard = self.donor_shard
        index = json.dumps(
            {
                "weight_map": {
                    self.keys[layer]: "model-00001-of-00001.safetensors"
                    for layer in self.shapes
                }
            }
        ).encode()
        self.files = {
            f"/{self.base_repo}/resolve/{self.base_revision}/{prepare.INDEX_NAME}": index,
            f"/{self.donor_repo}/resolve/{self.donor_revision}/{prepare.INDEX_NAME}": index,
            f"/{self.base_repo}/resolve/{self.base_revision}/model-00001-of-00001.safetensors": self.base_shard,
            f"/{self.donor_repo}/resolve/{self.donor_revision}/model-00001-of-00001.safetensors": self.donor_shard,
        }
        self.ranges = []
        self.no_range = False
        self.truncate_once = False
        self._truncated = False

    def _shard(self, shapes):
        offset = 0
        payload = bytearray()
        entries = {}
        for layer, shape in shapes.items():
            size = shape[0] * shape[1] * 2
            raw = (
                self.raw
                if layer == 15 and size == len(self.raw)
                else bytes([layer]) * size
            )
            key = self.keys[layer]
            entries[key] = {
                "dtype": "BF16",
                "shape": list(shape),
                "data_offsets": [offset, offset + size],
            }
            payload.extend(raw)
            offset += size
        header = json.dumps(entries, separators=(",", ":")).encode()
        return struct.pack("<Q", len(header)) + header + payload

    @staticmethod
    def _range(request):
        for name, value in request.header_items():
            if name.lower() == "range":
                start, end = value.removeprefix("bytes=").split("-")
                return int(start), int(end)
        return None

    def __call__(self, request, timeout=0):
        del timeout
        path = urlparse(request.full_url).path
        content = self.files[path]
        requested = self._range(request)
        if requested is None:
            return _Response(200, {}, content)
        start, end = requested
        self.ranges.append((path, start, end))
        if self.no_range:
            return _Response(200, {}, content)
        body = content[start : end + 1]
        headers = {
            "Content-Range": f"bytes {start}-{end}/{len(content)}",
            "Content-Length": str(end - start + 1),
        }
        if self.truncate_once and not self._truncated and start > 8:
            self._truncated = True
            body = body[: max(1, len(body) // 2)]
        return _Response(206, headers, body)


class PrepareAbliterationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.origin = _Origin()
        self.client = prepare.RangeClient(
            "http://synthetic.test", opener=self.origin, stats=prepare.TransferStats()
        )

    def _inspection(self, target_layers=(15,)):
        return prepare.inspect_sources(
            self.client,
            base_repo=self.origin.base_repo,
            base_revision=self.origin.base_revision,
            donor_repo=self.origin.donor_repo,
            donor_revision=self.origin.donor_revision,
            target_layers=target_layers,
        )

    def _spec(self):
        return prepare.TensorSpec(
            layer=15,
            key=self.origin.key,
            dtype="BF16",
            shape=(1, 4),
            nbytes=8,
            donor_shard="model-00001-of-00001.safetensors",
            donor_offsets=(0, 8),
            donor_data_start=len(self.origin.shard) - len(self.origin.raw),
            donor_size=len(self.origin.shard),
            base_shard="model-00001-of-00001.safetensors",
            base_offsets=(0, 8),
            base_data_start=len(self.origin.shard) - len(self.origin.raw),
            base_size=len(self.origin.shard),
        )

    def test_headers_only_reads_indexes_and_shard_headers(self):
        inspection = self._inspection()
        report = inspection.report()
        self.assertEqual(report["tensor_count"], 1)
        self.assertTrue(report["tensors"][0]["base"]["compatible"])
        self.assertEqual(len(self.origin.ranges), 4)
        self.assertTrue(
            all(end < len(self.origin.shard) for _, _, end in self.origin.ranges)
        )
        self.assertNotIn(
            self.origin.raw,
            [
                self.origin.shard[start : end + 1]
                for _, start, end in self.origin.ranges
            ],
        )

    def test_each_layer_uses_its_header_shape_and_byte_size(self):
        origin = _Origin(shapes={15: (1, 4), 16: (1, 2)})
        client = prepare.RangeClient(
            "http://synthetic.test", opener=origin, stats=prepare.TransferStats()
        )
        inspection = prepare.inspect_sources(
            client,
            base_repo=origin.base_repo,
            base_revision=origin.base_revision,
            donor_repo=origin.donor_repo,
            donor_revision=origin.donor_revision,
            target_layers=(15, 16),
        )
        self.assertEqual([spec.shape for spec in inspection.specs], [(1, 4), (1, 2)])
        self.assertEqual([spec.nbytes for spec in inspection.specs], [8, 4])

    def test_base_and_donor_shape_mismatch_is_rejected(self):
        origin = _Origin(shapes={15: (1, 4)}, donor_shapes={15: (1, 2)})
        client = prepare.RangeClient(
            "http://synthetic.test", opener=origin, stats=prepare.TransferStats()
        )
        with self.assertRaisesRegex(prepare.PrepareError, "shape or size mismatch"):
            prepare.inspect_sources(
                client,
                base_repo=origin.base_repo,
                base_revision=origin.base_revision,
                donor_repo=origin.donor_repo,
                donor_revision=origin.donor_revision,
                target_layers=(15,),
            )

    def test_range_non_support_is_rejected_before_bytes_are_written(self):
        self.origin.no_range = True
        with self.assertRaisesRegex(prepare.RangeError, "HTTP 206"):
            self.client.read_range(
                self.client.url(
                    self.origin.donor_repo,
                    self.origin.donor_revision,
                    "model-00001-of-00001.safetensors",
                ),
                0,
                7,
            )

    def test_truncated_range_discards_unverified_part_before_retry(self):
        spec = self._spec()
        output = Path(self.temp.name) / "overlay"
        output.mkdir()
        self.origin.truncate_once = True
        with self.assertRaises(prepare.RangeError):
            prepare._download_tensor(
                self.client,
                spec,
                output,
                None,
                donor_repo=self.origin.donor_repo,
                donor_revision=self.origin.donor_revision,
            )
        part = output / "layer-15.safetensors.part"
        self.assertGreater(part.stat().st_size, 0)
        hashes = prepare._download_tensor(
            self.client,
            spec,
            output,
            None,
            donor_repo=self.origin.donor_repo,
            donor_revision=self.origin.donor_revision,
        )
        self.assertEqual(
            hashes["tensor_sha256"], hashlib.sha256(self.origin.raw).hexdigest()
        )
        data_ranges = [
            item for item in self.origin.ranges if item[1] >= spec.donor_data_start
        ]
        self.assertGreaterEqual(len(data_ranges), 2)
        self.assertEqual(data_ranges[-1][1], spec.donor_data_start)

    def test_corrupt_resumable_part_is_discarded(self):
        spec = self._spec()
        output = Path(self.temp.name) / "overlay"
        output.mkdir()
        part = output / "layer-15.safetensors.part"
        part.write_bytes(b"XXXXXXXX")
        expected = prepare._manifest_row(
            spec,
            {
                "sha256": "a" * 64,
                "tensor_sha256": hashlib.sha256(self.origin.raw).hexdigest(),
            },
        )
        # The complete part has the right length but the wrong digest, so it is
        # not adopted.  The next request starts at the donor data start.
        hashes = prepare._download_tensor(
            self.client,
            spec,
            output,
            expected,
            donor_repo=self.origin.donor_repo,
            donor_revision=self.origin.donor_revision,
        )
        self.assertEqual(
            hashes["tensor_sha256"], hashlib.sha256(self.origin.raw).hexdigest()
        )
        self.assertEqual(
            self.origin.ranges[-1][1], spec.donor_data_start + spec.donor_offsets[0]
        )

    def test_duplicate_target_layer_is_rejected(self):
        with self.assertRaisesRegex(prepare.PrepareError, "duplicates"):
            prepare.inspect_sources(
                self.client,
                base_repo=self.origin.base_repo,
                base_revision=self.origin.base_revision,
                donor_repo=self.origin.donor_repo,
                donor_revision=self.origin.donor_revision,
                target_layers=(15, 15),
            )

    def test_extract_writes_manifest_and_one_tensor_file(self):
        inspection = self._inspection()
        output = Path(self.temp.name) / "overlay"
        manifest = prepare.extract(self.client, inspection, output)
        self.assertEqual(manifest["schema"], 1)
        self.assertEqual(manifest["tool"]["name"], "prepare_abliteration.py")
        self.assertEqual(manifest["tool"]["version"], prepare.TOOL_VERSION)
        self.assertIn("donor_license", manifest["license_references"])
        self.assertRegex(manifest["acquired_date"], r"20[0-9]{2}-[0-9]{2}-[0-9]{2}")
        self.assertEqual(
            manifest["tensors"][0]["base"]["shard"], "model-00001-of-00001.safetensors"
        )
        self.assertTrue(manifest["tensors"][0]["base"]["compatible"])
        self.assertTrue((output / "manifest.json").is_file())
        self.assertTrue((output / "layer-15.safetensors").is_file())
        self.assertFalse((output / "layer-15.safetensors.part").exists())
        with (output / "layer-15.safetensors").open("rb") as stream:
            header_size = struct.unpack("<Q", stream.read(8))[0]
        self.assertEqual((8 + header_size) % 8, 0)


if __name__ == "__main__":
    unittest.main()
