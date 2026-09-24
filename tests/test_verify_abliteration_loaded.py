# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for BF16 source-to-rank readback matching."""

import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

from tools.verify_abliteration_loaded import expected_shard_sha256, loaded_parameter


class RankReadbackTests(unittest.TestCase):
    def test_exact_full_row_and_column_shards(self):
        key = "model.language_model.layers.15.self_attn.o_proj.weight"
        # Four rows by four BF16 elements; the two-byte words are distinct.
        data = b"".join(i.to_bytes(2, "little") for i in range(16))
        header = json.dumps(
            {key: {"dtype": "BF16", "shape": [4, 4], "data_offsets": [0, len(data)]}}
        ).encode()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "part.safetensors"
            path.write_bytes(struct.pack("<Q", len(header)) + header + data)
            full, placement = expected_shard_sha256(path, key, [4, 4], 0)
            self.assertEqual((full, placement), (hashlib.sha256(data).hexdigest(), "replicated"))
            for rank in (0, 1):
                rows, placement = expected_shard_sha256(path, key, [2, 4], rank)
                self.assertEqual(placement, "row_split")
                self.assertEqual(
                    rows, hashlib.sha256(data[rank * 16 : (rank + 1) * 16]).hexdigest()
                )
                columns, placement = expected_shard_sha256(path, key, [4, 2], rank)
                self.assertEqual(placement, "column_split")
                expected = b"".join(data[row * 8 + rank * 4 : row * 8 + (rank + 1) * 4] for row in range(4))
                self.assertEqual(columns, hashlib.sha256(expected).hexdigest())
            with self.assertRaises(ValueError):
                expected_shard_sha256(path, key, [3, 3], 0)

    def test_mtp_alias_requires_one_unique_candidate(self):
        row = {
            "name": "draft:model.layers.0.self_attn.o_proj.weight",
            "shape": [2, 2],
            "dtype": "BF16",
            "sha256": "a" * 64,
        }
        self.assertEqual(loaded_parameter([row], 45), row)
        with self.assertRaisesRegex(ValueError, "found 2"):
            loaded_parameter([row, row], 45)
        with self.assertRaisesRegex(ValueError, "found 0"):
            loaded_parameter([dict(row, name="draft:model.layers.1.self_attn.o_proj.weight")], 45)


if __name__ == "__main__":
    unittest.main()
