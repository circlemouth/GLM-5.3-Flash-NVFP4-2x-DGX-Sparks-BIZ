# SPDX-License-Identifier: Apache-2.0
"""Derive and verify the request-scoped GLM thinking template."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

SOURCE_SHA256 = "34d5ee66b12fa6446cdae131c352b8f68cd85369e0e6fda115583805fada3891"
DERIVED_SHA256 = "96ed83160b243de213e95eb2fa19bde4ac13b676661cfec477d18e45e9fcca3a"
DEFAULT_OFF_DERIVED_SHA256 = (
    "186a9894b17aa797eb2bee669d055fef55e303f738aa1e103fdf3d484c844059"
)

OLD_HEAD = """{%- set effective_reasoning_effort = reasoning_effort if reasoning_effort is defined and reasoning_effort in ['low', 'high'] else 'max' -%}
{%- if effective_reasoning_effort is not none -%}<|system|>Reasoning Effort: {{ effective_reasoning_effort | capitalize }}{%- endif -%}"""
NEW_HEAD = """{%- if thinking is defined or enable_thinking is defined -%}
{%- set thinking_enabled = (thinking if thinking is defined else false) or (enable_thinking if enable_thinking is defined else false) -%}
{%- else -%}
{%- set thinking_enabled = true -%}
{%- endif -%}
{%- set effective_reasoning_effort = reasoning_effort if reasoning_effort is defined and reasoning_effort in ['low', 'high'] else 'max' -%}
{%- if thinking_enabled and effective_reasoning_effort is not none -%}<|system|>Reasoning Effort: {{ effective_reasoning_effort | capitalize }}{%- endif -%}"""
NEW_HEAD_DEFAULT_OFF = NEW_HEAD.replace(
    "{%- set thinking_enabled = true -%}",
    "{%- set thinking_enabled = false -%}",
)
OLD_TAIL = """{%- if add_generation_prompt -%}
    <|assistant|>{{- '<think>' -}}
{%- endif -%}"""
NEW_TAIL = """{%- if add_generation_prompt -%}
    <|assistant|>{{- '<think>' if thinking_enabled else '<think></think>' -}}
{%- endif -%}"""


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def derive(raw, default_thinking=True):
    """Apply the only two permitted edits to the pinned NVIDIA template."""

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Source template is not UTF-8") from exc
    if text.count(OLD_HEAD) != 1 or text.count(OLD_TAIL) != 1:
        raise ValueError("Expected exactly one reasoning header and generation block")
    if type(default_thinking) is not bool:
        raise ValueError("default_thinking must be true or false")
    head = NEW_HEAD if default_thinking else NEW_HEAD_DEFAULT_OFF
    return text.replace(OLD_HEAD, head).replace(OLD_TAIL, NEW_TAIL).encode()


def verify_source(raw, default_thinking=True):
    actual = sha256(raw)
    if actual != SOURCE_SHA256:
        raise ValueError(f"Refusing unknown source template: {actual}")
    derived = derive(raw, default_thinking=default_thinking)
    actual = sha256(derived)
    expected = DERIVED_SHA256 if default_thinking else DEFAULT_OFF_DERIVED_SHA256
    if actual != expected:
        raise ValueError(f"Derived template hash mismatch: {actual}")
    return derived


def write_derived(source, output, default_thinking=True):
    """Write atomically, or accept an already-identical regular output."""

    source = Path(source)
    output = Path(output)
    if not source.is_file():
        raise ValueError(f"Pinned source template is missing: {source}")
    derived = verify_source(
        source.read_bytes(), default_thinking=default_thinking
    )
    expected = DERIVED_SHA256 if default_thinking else DEFAULT_OFF_DERIVED_SHA256

    if output.exists() or output.is_symlink():
        if output.is_symlink() or not output.is_file():
            raise ValueError(f"Refusing unsafe output: {output}")
        actual = sha256(output.read_bytes())
        if actual == expected:
            return {
                "status": "unchanged",
                "source_sha256": SOURCE_SHA256,
                "derived_sha256": actual,
            }
        raise ValueError(f"Refusing unknown existing output: {actual}")

    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(derived)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o444)
        os.replace(temporary, output)
        directory_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "status": "created",
        "source_sha256": SOURCE_SHA256,
        "derived_sha256": expected,
    }


def verify_derived(path, expected_sha256=DERIVED_SHA256):
    """Verify the regular host file that will be mounted read-only."""

    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Chat template is missing or unsafe: {path}")
    actual = sha256(path.read_bytes())
    if actual != expected_sha256:
        raise ValueError(f"Chat template hash mismatch: {actual}")
    return path.resolve()
