# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack.core.component import component

_component_instance = component()


@_component_instance
class Hello:
    @_component_instance.output_types(output=str)
    def run(self, word: str):
        """Takes a string in input and returns "Hello, <string>!"in output."""
        return {"output": f"Hello, {word}!"}
