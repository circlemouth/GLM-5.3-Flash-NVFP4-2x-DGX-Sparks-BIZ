# SPDX-License-Identifier: Apache-2.0
"""Request-scoped GLM thinking-mode normalization.

The managed BIZ surface exposes ``off``, ``low``, ``high`` and ``max``.
``off`` is a local control value: it must never be forwarded as vLLM's
top-level ``reasoning_effort``.
"""

from __future__ import annotations

import copy

MODES = frozenset({"off", "low", "high", "max"})
_FLAGS = ("thinking", "enable_thinking")


def _mode(value, where):
    if type(value) is not str or value not in MODES:
        raise ValueError(f"{where} must be one of off, low, high or max")
    return value


def normalize_request(request, default_mode):
    """Return an isolated request with one unambiguous thinking mode.

    The top-level and template reasoning values are managed aliases. Explicit
    boolean flags may select ``off`` when no effort was supplied, but may not
    contradict an explicit/default mode. The returned object never shares its
    template-kwargs mapping with the caller.
    """

    _mode(default_mode, "default reasoning_effort")
    if not isinstance(request, dict):
        raise ValueError("request must be an object")
    body = copy.deepcopy(request)
    template = body.get("chat_template_kwargs")
    if template is None:
        template = {}
        body["chat_template_kwargs"] = template
    elif not isinstance(template, dict):
        raise ValueError("chat_template_kwargs must be an object")

    explicit_modes = []
    if "reasoning_effort" in body:
        explicit_modes.append(
            ("reasoning_effort", _mode(body["reasoning_effort"], "reasoning_effort"))
        )
    if "reasoning_effort" in template:
        explicit_modes.append(
            (
                "chat_template_kwargs.reasoning_effort",
                _mode(
                    template["reasoning_effort"],
                    "chat_template_kwargs.reasoning_effort",
                ),
            )
        )
    if explicit_modes and any(
        mode != explicit_modes[0][1] for _, mode in explicit_modes
    ):
        names = " and ".join(name for name, _ in explicit_modes)
        raise ValueError(f"Conflicting thinking modes in {names}")

    flags = []
    for name in _FLAGS:
        if name in template:
            value = template[name]
            if type(value) is not bool:
                raise ValueError(f"chat_template_kwargs.{name} must be true or false")
            flags.append((name, value))
    if flags and any(value != flags[0][1] for _, value in flags):
        raise ValueError("thinking and enable_thinking must not contradict")

    if explicit_modes:
        selected = explicit_modes[0][1]
    elif flags and flags[0][1] is False:
        selected = "off"
    else:
        selected = default_mode

    enabled = selected != "off"
    if flags and flags[0][1] is not enabled:
        raise ValueError(f"Thinking flags contradict reasoning mode {selected}")

    if enabled:
        body["reasoning_effort"] = selected
        template["reasoning_effort"] = selected
    else:
        body.pop("reasoning_effort", None)
        template.pop("reasoning_effort", None)
    template["enable_thinking"] = enabled
    template["thinking"] = enabled
    return body
