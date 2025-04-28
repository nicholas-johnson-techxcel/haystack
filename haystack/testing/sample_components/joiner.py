# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

from haystack.core.component import component
from haystack.core.component.types import Variadic

_component_instance = component()


@_component_instance
class StringJoiner:
    @_component_instance.output_types(output=str)
    def run(self, input_str: Variadic[str]):
        """
        Take strings from multiple input nodes and join them into a single one returned in output.

        Since `input_str` is Variadic, we know we'll receive a List[str].
        """
        return {"output": " ".join(input_str)}


_component_instance = component()


@_component_instance
class StringListJoiner:
    @_component_instance.output_types(output=str)
    def run(self, inputs: Variadic[List[str]]):
        """
        Take list of strings from multiple input nodes and join them into a single one returned in output.

        Since `input_str` is Variadic, we know we'll receive a List[List[str]].
        """
        retval: List[str] = []
        for list_of_strings in inputs:
            retval += list_of_strings

        return {"output": retval}
