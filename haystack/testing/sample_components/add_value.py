# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

from haystack.core.component import component

_component_instance = component()


@_component_instance
class AddFixedValue:
    """
    Adds two values together.
    """

    def __init__(self, add: int = 1):
        self.add = add

    @_component_instance.output_types(result=int)
    def run(self, value: int, add: Optional[int] = None):
        """
        Adds two values together.
        """
        if add is None:
            add = self.add
        return {"result": value + add}
