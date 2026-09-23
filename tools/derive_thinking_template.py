#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Derive the request-scoped thinking template from a pinned model template."""

import argparse
import json

from glm53_setup.chat_template import write_derived


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--default-thinking",
        choices=("on", "off"),
        default="on",
        help="Default only when a request omits both managed thinking flags",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            write_derived(
                args.source,
                args.output,
                default_thinking=args.default_thinking == "on",
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
