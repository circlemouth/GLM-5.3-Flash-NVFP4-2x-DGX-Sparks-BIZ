# SPDX-License-Identifier: Apache-2.0
"""Patch the pinned GLM main and native MTP loader at their weight iterators."""

import argparse
import ast
import hashlib
import json
from pathlib import Path

from .pinned_patch import default_package, replace_once

SOURCES = {
    "models/glm5next/nvidia/model.py": "48f60a65afab1afd737b282d109c6c11f97d881d3ec774442e174fab09c2f525",
    "models/glm5next/nvidia/mtp.py": "5e17988481502878146f848c5216c7865da0e29de4f147b25728717c00ea8b50",
}
MARKER = "GLM53_BF16_OPROJ_OVERLAY_API=1"


def patch_model(text):
    tree = ast.parse(text)
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "Glm5NextForConditionalGeneration"
    )
    if any(
        isinstance(n, ast.FunctionDef) and n.name == "load_weights" for n in node.body
    ):
        raise ValueError("Main loader already overrides load_weights")
    lines = text.splitlines(keepends=True)
    lines.insert(
        node.end_lineno,
        "\n    def load_weights(self, weights):\n"
        "        import os\n"
        "        from glm53_setup.runtime.weight_overlay import overlay_weights\n"
        "        enabled = bool(os.environ.get('GLM53_WEIGHT_OVERLAY_MANIFEST'))\n"
        "        if enabled and getattr(self, '_glm53_overlay_loaded', False):\n"
        "            raise ValueError('Overlay main loader called twice on one model')\n"
        "        loaded = super().load_weights(overlay_weights(weights, 'main'))\n"
        "        if enabled:\n"
        "            self._glm53_overlay_loaded = True\n"
        "        return loaded\n",
    )
    result = (
        "# Modified by BIZ fork contributors: optional BF16 o_proj weight overlay.\n"
        + "".join(lines)
    )
    compile(result, "model.py", "exec")
    return result


def patch_mtp(text):
    result = replace_once(
        text,
        "    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:\n"
        "        stacked_params_mapping = [\n",
        "    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:\n"
        "        import os\n"
        "        from glm53_setup.runtime.weight_overlay import overlay_weights\n"
        "        enabled = bool(os.environ.get('GLM53_WEIGHT_OVERLAY_MANIFEST'))\n"
        "        if enabled and getattr(self, '_glm53_overlay_loaded', False):\n"
        "            raise ValueError('Overlay MTP loader called twice on one model')\n"
        "        weights = overlay_weights(weights, 'mtp')\n"
        "        stacked_params_mapping = [\n",
    )
    result = replace_once(
        result,
        "        for name, loaded_weight in weights:\n",
        "        for args in weights:\n"
        "            name, loaded_weight = args[:2]\n"
        "            kwargs = args[2] if len(args) > 2 else {}\n",
    )
    result = replace_once(
        result,
        "                weight_loader(param, loaded_weight, shard_id)\n",
        "                weight_loader(param, loaded_weight, shard_id, **kwargs)\n",
    )
    result = replace_once(
        result,
        "                        return_success=True,\n",
        "                        return_success=True,\n"
        "                        **kwargs,\n",
    )
    result = replace_once(
        result,
        "                    weight_loader(param, loaded_weight)\n",
        "                    weight_loader(param, loaded_weight, **kwargs)\n",
    )
    result = replace_once(
        result,
        "        return loaded_params\n",
        "        if enabled:\n"
        "            self._glm53_overlay_loaded = True\n"
        "        return loaded_params\n",
    )
    result = (
        "# Modified by BIZ fork contributors: optional BF16 MTP o_proj overlay.\n"
        + result
    )
    compile(result, "mtp.py", "exec")
    return result


def prepare(package):
    originals = {name: (package / name).read_bytes() for name in SOURCES}
    for name, content in originals.items():
        if hashlib.sha256(content).hexdigest() != SOURCES[name]:
            raise ValueError(f"Overlay source hash mismatch: {name}")
    model, mtp = SOURCES
    return {
        model: patch_model(originals[model].decode("utf-8")).encode(),
        mtp: patch_mtp(originals[mtp].decode("utf-8")).encode(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=default_package())
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    outputs = prepare(args.package)
    result = {
        name: hashlib.sha256(content).hexdigest() for name, content in outputs.items()
    }
    if not args.check:
        for name, content in outputs.items():
            (args.package / name).write_bytes(content)
        (args.package.parent / "glm53-weight-overlay-patch.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
    print(
        json.dumps(
            {
                "source_hashes": SOURCES,
                "patched_hashes": result,
                "check_only": args.check,
            }
        )
    )


if __name__ == "__main__":
    main()
