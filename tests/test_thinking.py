# SPDX-License-Identifier: Apache-2.0
import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from glm53_setup import chat_template, thinking


class ThinkingRequestTests(unittest.TestCase):
    def test_four_managed_modes_are_request_scoped(self):
        original = {"messages": [{"role": "user", "content": "hello"}]}
        for mode in ("off", "low", "high", "max"):
            with self.subTest(mode=mode):
                request = {**original, "reasoning_effort": mode}
                result = thinking.normalize_request(request, "low")
                enabled = mode != "off"
                self.assertEqual(
                    result["chat_template_kwargs"]["enable_thinking"], enabled
                )
                self.assertEqual(result["chat_template_kwargs"]["thinking"], enabled)
                if enabled:
                    self.assertEqual(result["reasoning_effort"], mode)
                    self.assertEqual(
                        result["chat_template_kwargs"]["reasoning_effort"], mode
                    )
                else:
                    self.assertNotIn("reasoning_effort", result)
                    self.assertNotIn("reasoning_effort", result["chat_template_kwargs"])
                self.assertEqual(original, {"messages": original["messages"]})

    def test_unspecified_keeps_each_managed_entry_default(self):
        for default in ("off", "low", "high", "max"):
            result = thinking.normalize_request({}, default)
            self.assertEqual(
                result["chat_template_kwargs"]["enable_thinking"],
                default != "off",
            )
            if default == "off":
                self.assertNotIn("reasoning_effort", result)
            else:
                self.assertEqual(result["reasoning_effort"], default)

    def test_explicit_false_beats_an_on_default(self):
        result = thinking.normalize_request(
            {"chat_template_kwargs": {"enable_thinking": False}}, "max"
        )
        self.assertNotIn("reasoning_effort", result)
        self.assertEqual(
            result["chat_template_kwargs"],
            {"enable_thinking": False, "thinking": False},
        )

    def test_managed_conflicts_and_bad_types_fail_closed(self):
        invalid = [
            {"reasoning_effort": "medium"},
            {"reasoning_effort": False},
            {"chat_template_kwargs": []},
            {"chat_template_kwargs": {"thinking": "false"}},
            {
                "chat_template_kwargs": {
                    "thinking": False,
                    "enable_thinking": True,
                }
            },
            {
                "reasoning_effort": "off",
                "chat_template_kwargs": {"enable_thinking": True},
            },
            {
                "reasoning_effort": "high",
                "chat_template_kwargs": {"thinking": False},
            },
            {
                "reasoning_effort": "low",
                "chat_template_kwargs": {"reasoning_effort": "max"},
            },
        ]
        for request in invalid:
            with self.subTest(request=request), self.assertRaises(ValueError):
                thinking.normalize_request(request, "low")

    def test_no_state_leaks_from_off_to_on(self):
        base = {"chat_template_kwargs": {"clear_thinking": True}}
        before = copy.deepcopy(base)
        off = thinking.normalize_request({**base, "reasoning_effort": "off"}, "low")
        high = thinking.normalize_request({**base, "reasoning_effort": "high"}, "low")
        self.assertFalse(off["chat_template_kwargs"]["thinking"])
        self.assertTrue(high["chat_template_kwargs"]["thinking"])
        self.assertEqual(base, before)


class ThinkingTemplateDerivationTests(unittest.TestCase):
    def setUp(self):
        self.source = (
            "[gMASK]<sop>\n"
            + chat_template.OLD_HEAD
            + "\nTOOLS IMAGES HISTORY TOOL_CALL_IDS\n"
            + chat_template.OLD_TAIL
            + "\n"
        ).encode()
        self.derived = chat_template.derive(self.source)

    def test_only_the_two_pinned_blocks_change(self):
        self.assertEqual(
            self.derived,
            self.source.replace(
                chat_template.OLD_HEAD.encode(), chat_template.NEW_HEAD.encode()
            ).replace(chat_template.OLD_TAIL.encode(), chat_template.NEW_TAIL.encode()),
        )
        self.assertIn(b"TOOLS IMAGES HISTORY TOOL_CALL_IDS", self.derived)
        self.assertIn(b"thinking_enabled and effective_reasoning_effort", self.derived)
        self.assertIn(b"'<think></think>'", self.derived)

    def test_default_off_variant_changes_only_the_undefined_default(self):
        default_off = chat_template.derive(self.source, default_thinking=False)
        self.assertEqual(
            default_off,
            self.derived.replace(
                b"{%- set thinking_enabled = true -%}",
                b"{%- set thinking_enabled = false -%}",
            ),
        )
        self.assertIn(b"thinking if thinking is defined else false", default_off)
        self.assertIn(b"'<think></think>'", default_off)

    def test_default_selector_rejects_non_boolean_values(self):
        for value in (None, 0, 1, "off"):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "default_thinking must be true or false"
            ):
                chat_template.derive(self.source, default_thinking=value)

    def test_missing_or_duplicate_blocks_fail(self):
        for raw in (
            b"not the pinned template",
            self.source + chat_template.OLD_TAIL.encode(),
        ):
            with self.subTest(raw=raw[:20]), self.assertRaises(ValueError):
                chat_template.derive(raw)

    def test_hash_gate_idempotence_and_unknown_output_refusal(self):
        source_hash = hashlib.sha256(self.source).hexdigest()
        derived_hash = hashlib.sha256(self.derived).hexdigest()
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.multiple(
                chat_template,
                SOURCE_SHA256=source_hash,
                DERIVED_SHA256=derived_hash,
            ),
        ):
            root = Path(tmp)
            source = root / "source.jinja"
            output = root / "derived.jinja"
            source.write_bytes(self.source)
            result = chat_template.write_derived(source, output)
            self.assertEqual(result["status"], "created")
            self.assertEqual(output.read_bytes(), self.derived)
            self.assertEqual(os.stat(output).st_mode & 0o777, 0o444)
            self.assertEqual(
                chat_template.write_derived(source, output)["status"], "unchanged"
            )
            output.chmod(0o644)
            output.write_text("unknown", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown existing output"):
                chat_template.write_derived(source, output)

    def test_default_off_hash_gate_and_idempotence(self):
        source_hash = hashlib.sha256(self.source).hexdigest()
        derived = chat_template.derive(self.source, default_thinking=False)
        derived_hash = hashlib.sha256(derived).hexdigest()
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.multiple(
                chat_template,
                SOURCE_SHA256=source_hash,
                DEFAULT_OFF_DERIVED_SHA256=derived_hash,
            ),
        ):
            root = Path(tmp)
            source = root / "source.jinja"
            output = root / "derived-default-off.jinja"
            source.write_bytes(self.source)
            result = chat_template.write_derived(
                source, output, default_thinking=False
            )
            self.assertEqual(result["derived_sha256"], derived_hash)
            self.assertEqual(output.read_bytes(), derived)
            self.assertEqual(
                chat_template.write_derived(
                    source, output, default_thinking=False
                )["status"],
                "unchanged",
            )

    def test_missing_and_wrong_source_hash_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "missing"):
                chat_template.write_derived(root / "missing", root / "out")
            source = root / "source"
            source.write_bytes(self.source)
            with self.assertRaisesRegex(ValueError, "unknown source"):
                chat_template.write_derived(source, root / "out")

    def test_receipt_is_json_serializable(self):
        receipt = {
            "source_sha256": chat_template.SOURCE_SHA256,
            "derived_sha256": chat_template.DERIVED_SHA256,
        }
        self.assertEqual(json.loads(json.dumps(receipt)), receipt)


if __name__ == "__main__":
    unittest.main()
