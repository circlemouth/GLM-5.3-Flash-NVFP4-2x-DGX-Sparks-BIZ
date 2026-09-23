# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from glm53_setup import server, server_config
from glm53_setup.runtime.weight_overlay import (
    BASE_REPO,
    KEY,
    TARGET_LAYERS,
    overlay_weights,
    read_manifest,
    verify_assets,
)

BASE_REV = "a" * 40
DONOR_REV = "b" * 40


class FakeTensor:
    dtype = "torch.bfloat16"
    shape = (4096, 16384)


class WeightOverlayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.original = FakeTensor()
        self.replacement = FakeTensor()
        self.rows = []
        for layer in TARGET_LAYERS:
            name = f"layer-{layer:02d}.safetensors"
            content = f"synthetic-{layer}".encode()
            (self.root / name).write_bytes(content)
            key = KEY.format(layer=layer)
            self.rows.append(
                {
                    "layer": layer,
                    "key": key,
                    "file": name,
                    "dtype": "BF16",
                    "shape": [4096, 16384],
                    "nbytes": 4096 * 16384 * 2,
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "tensor_sha256": hashlib.sha256(content).hexdigest(),
                    "shard": "model-00001-of-00120.safetensors",
                    "base": {
                        "key": key,
                        "dtype": "BF16",
                        "shape": [4096, 16384],
                        "nbytes": 4096 * 16384 * 2,
                        "shard": "model-00001-of-00033.safetensors",
                        "compatible": True,
                    },
                }
            )
        self.manifest = {
            "schema": 1,
            "base": {"repo": BASE_REPO, "revision": BASE_REV},
            "donor": {
                "repo": "dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4",
                "revision": DONOR_REV,
            },
            "tensors": self.rows,
        }
        self.save_manifest()

    def save_manifest(self):
        raw = json.dumps(self.manifest, sort_keys=True).encode()
        self.path = self.root / "manifest.json"
        self.path.write_bytes(raw)
        self.digest = hashlib.sha256(raw).hexdigest()

    def apply(self, weights, scope="main", loader=None):
        return list(
            overlay_weights(
                weights,
                scope,
                manifest_path=self.path,
                manifest_sha256=self.digest,
                base_revision=BASE_REV,
                donor_revision=DONOR_REV,
                tensor_loader=loader or (lambda _path, _key: self.replacement),
            )
        )

    def test_main_replaces_only_29_and_keeps_metadata_and_non_targets(self):
        weights = [
            (KEY.format(layer=layer), self.original, {"source": layer})
            for layer in range(46)
        ]
        output = self.apply(weights)
        self.assertEqual(len(output), len(weights))
        for layer, item in enumerate(output):
            self.assertEqual(item[2], weights[layer][2])
            self.assertIs(
                item[1], self.replacement if 15 <= layer <= 43 else self.original
            )
        self.assertIs(weights[15][1], self.original)

    def test_mtp_replaces_only_layer_45(self):
        weights = [(KEY.format(layer=layer), self.original) for layer in range(46)]
        output = self.apply(weights, "mtp")
        self.assertEqual(
            [layer for layer, item in enumerate(output) if item[1] is self.replacement],
            [45],
        )

    def test_mixed_attention_widths_follow_the_verified_manifest(self):
        row = next(item for item in self.rows if item["layer"] == 16)
        row["shape"] = [4096, 8192]
        row["nbytes"] = 4096 * 8192 * 2
        row["base"]["shape"] = row["shape"]
        row["base"]["nbytes"] = row["nbytes"]
        self.save_manifest()
        narrow = FakeTensor()
        narrow.shape = (4096, 8192)
        donor_narrow = FakeTensor()
        donor_narrow.shape = (4096, 8192)
        weights = [
            (KEY.format(layer=layer), narrow if layer == 16 else self.original)
            for layer in range(15, 44)
        ]
        result = self.apply(
            weights,
            loader=lambda _path, key: (
                donor_narrow if ".16." in key else self.replacement
            ),
        )
        self.assertIs(result[1][1], donor_narrow)
        self.assertIs(result[0][1], self.replacement)

    def test_disabled_does_not_open_donor(self):
        weights = [("ordinary", self.original, {"k": 1})]
        self.assertEqual(list(overlay_weights(iter(weights), "main")), weights)

    def test_missing_and_duplicate_fail(self):
        complete = [(KEY.format(layer=layer), self.original) for layer in range(15, 44)]
        with self.assertRaisesRegex(ValueError, "missing"):
            self.apply(complete[:-1])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.apply(complete + complete[:1])

    def test_hash_dtype_shape_and_partial_file_fail(self):
        complete = [(KEY.format(layer=layer), self.original) for layer in range(15, 44)]
        (self.root / "layer-15.safetensors").write_bytes(b"partial")
        with self.assertRaisesRegex(ValueError, "changed"):
            self.apply(complete)
        (self.root / "layer-15.safetensors").write_bytes(b"synthetic-15")
        bad_dtype = FakeTensor()
        bad_dtype.dtype = "torch.float16"
        with self.assertRaisesRegex(ValueError, "dtype or shape"):
            self.apply([(complete[0][0], bad_dtype), *complete[1:]])
        bad_shape = FakeTensor()
        bad_shape.shape = (1, 2)
        with self.assertRaisesRegex(ValueError, "dtype or shape"):
            self.apply([(complete[0][0], bad_shape), *complete[1:]])

    def test_manifest_identity_and_duplicate_rows_fail(self):
        self.assertEqual(len(read_manifest(self.path, self.digest, BASE_REV)), 30)
        self.assertEqual(
            verify_assets(self.root, self.digest, BASE_REV)["tensor_count"], 30
        )
        with self.assertRaisesRegex(ValueError, "base checkpoint"):
            read_manifest(self.path, self.digest, "c" * 40)
        with self.assertRaisesRegex(ValueError, "donor revision"):
            read_manifest(self.path, self.digest, BASE_REV, "c" * 40)
        self.manifest["tensors"][-1] = self.manifest["tensors"][0]
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "duplicated"):
            read_manifest(self.path, self.digest, BASE_REV)

    def test_profile_opt_in_changes_identity_and_mounts_read_only(self):
        profile = server_config.load(
            Path(__file__).resolve().parents[1] / "examples/server.example.toml"
        )
        baseline = server_config.fingerprint(profile)
        self.assertNotIn(
            "GLM53_WEIGHT_OVERLAY_MANIFEST", server_config.environment(profile, 0)
        )
        profile["runtime"]["weight_overlay"] = {
            "enabled": True,
            "path": str(self.root),
            "manifest_sha256": self.digest,
            "donor_revision": DONOR_REV,
        }
        with self.assertRaisesRegex(ValueError, "distinct served_model_name"):
            server_config.validate(profile)
        profile["api"]["served_model_name"] = "glm-5.3-flash-ablit"
        server_config.validate(profile)
        self.assertNotEqual(server_config.fingerprint(profile), baseline)
        for rank in (0, 1):
            args = server.command(
                profile, self.root / "server.toml", rank, "test", cache=self.root
            )
            self.assertIn(f"{self.root}:/weight-overlay:ro", args)
            env = server_config.environment(profile, rank)
            self.assertEqual(env["GLM53_WEIGHT_OVERLAY_SHA256"], self.digest)
        derived = copy.deepcopy(profile)
        derived["runtime"]["derived_checkpoint"] = {
            "path": "/derived",
            "requant_target": "test",
            "overlays": [
                {
                    "target": "model.py",
                    "source": "/source.py",
                    "sha256": "a" * 64,
                    "base_sha256": "b" * 64,
                    "marker": "test",
                }
            ],
        }
        with self.assertRaisesRegex(ValueError, "does not support"):
            server_config.validate(derived)


if __name__ == "__main__":
    unittest.main()
