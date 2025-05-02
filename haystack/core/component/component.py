# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import inspect
from collections.abc import Callable, Coroutine
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
from types import new_class
from typing import Any, Protocol, overload, runtime_checkable

from haystack import logging
from haystack.core.errors import ComponentError

from .sockets import Sockets
from .types import InputSocket, OutputSocket, _empty

logger = logging.getLogger(__name__)


@dataclass
class PreInitHookPayload:
    callback: Callable[[type[Any], dict[str, Any]], None]
    in_progress: bool = False


_COMPONENT_PRE_INIT_HOOK: ContextVar[PreInitHookPayload | None] = ContextVar("component_pre_init_hook", default=None)


@contextmanager
def _hook_component_init(callback: Callable[[type[Any], dict[str, Any]], None]) -> Any:
    """
    Context manager to set a callback that will be invoked before a component's constructor is called.

    The callback receives the component class and the init parameters (as keyword arguments) and can modify the init
    parameters in place.

    :param callback:
        Callback function to invoke.
    """
    token = _COMPONENT_PRE_INIT_HOOK.set(PreInitHookPayload(callback))
    try:
        yield
    finally:
        _COMPONENT_PRE_INIT_HOOK.reset(token)


@runtime_checkable
class Component(Protocol):
    """
    Note this is only used by type checking tools.

    In order to implement the `Component` protocol, custom components need to
    have a `run` method. The signature of the method and its return value
    won't be checked, i.e. classes with the following methods:

        def run(self, param: str) -> Dict[str, Any]:
            ...

    and

        def run(self, **kwargs):
            ...

    will be both considered as respecting the protocol. This makes the type
    checking much weaker, but we have other places where we ensure code is
    dealing with actual Components.

    The protocol is runtime checkable so it'll be possible to assert:

        isinstance(MyComponent, Component)
    """

    def run(self, *args: Any, **kwargs: Any) -> dict[str, Any]: ...


class ComponentMeta(type):
    @staticmethod
    def _positional_to_kwargs(cls_type: type[Component], args: tuple[Any, ...]) -> dict[str, Any]:
        """
        Convert positional arguments to keyword arguments based on the signature of the `__init__` method.
        """
        init_signature = inspect.signature(cls_type.__init__)
        init_params = {name: info for name, info in init_signature.parameters.items() if name != "self"}

        out = {}
        for arg, (name, info) in zip(args, init_params.items()):
            if info.kind == inspect.Parameter.VAR_POSITIONAL:
                raise ComponentError(
                    "Pre-init hooks do not support components with variadic positional args in their init method"
                )

            assert info.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.POSITIONAL_ONLY)
            out[name] = arg
        return out

    @staticmethod
    def _parse_and_set_output_sockets(instance: Component) -> None:
        has_async_run = hasattr(instance, "run_async")

        # If `component.set_output_types()` was called in the component constructor,
        # `__haystack_output__` is already populated, no need to do anything.
        if "__haystack_output__" not in instance.__dict__:
            # If that's not the case, we need to populate `__haystack_output__`
            #
            # If either of the run methods were decorated, they'll have a field assigned that
            # stores the output specification. If both run methods were decorated, we ensure that
            # outputs are the same. We deepcopy the content of the cache to transfer ownership from
            # the class method to the actual instance, so that different instances of the same class
            # won't share this data.

            run_output_types = getattr(instance.run, "_output_types_cache", {})
            async_run_output_types = getattr(instance.run_async, "_output_types_cache", {}) if has_async_run else {}

            if has_async_run and run_output_types != async_run_output_types:
                raise ComponentError("Output type specifications of 'run' and 'run_async' methods must be the same")
            output_types_cache = run_output_types

            instance.__haystack_output__ = Sockets(instance, deepcopy(output_types_cache), OutputSocket)

    @staticmethod
    def _parse_and_set_input_sockets(component_cls: type[Component], instance: Any) -> None:
        def inner(method: Callable[..., Any], sockets: Sockets):
            from inspect import Parameter

            run_signature = inspect.signature(method)

            for param_name, param_info in run_signature.parameters.items():
                if param_name == "self" or param_info.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
                    continue

                socket_kwargs = {"name": param_name, "type": param_info.annotation}
                if param_info.default != Parameter.empty:
                    socket_kwargs["default_value"] = param_info.default

                new_socket = InputSocket(**socket_kwargs)

                # Also ensure that new sockets don't override existing ones.
                existing_socket = sockets.get(param_name)
                if existing_socket is not None and existing_socket != new_socket:
                    raise ComponentError(
                        "set_input_types()/set_input_type() cannot override the parameters of the 'run' method"
                    )

                sockets[param_name] = new_socket

            return run_signature

        # Create the sockets if set_input_types() wasn't called in the constructor.
        if not hasattr(instance, "__haystack_input__"):
            instance.__haystack_input__ = Sockets(instance, {}, InputSocket)

        inner(getattr(component_cls, "run"), instance.__haystack_input__)

        # Ensure that the sockets are the same for the async method, if it exists.
        async_run = getattr(component_cls, "run_async", None)
        if async_run is not None:
            run_sockets = Sockets(instance, {}, InputSocket)
            async_run_sockets = Sockets(instance, {}, InputSocket)

            # Can't use the sockets from above as they might contain
            # values set with set_input_types().
            run_sig = inner(getattr(component_cls, "run"), run_sockets)
            async_run_sig = inner(async_run, async_run_sockets)

            if async_run_sockets != run_sockets or run_sig.parameters != async_run_sig.parameters:
                sig_diff = _compare_run_methods_signatures(run_sig, async_run_sig)
                raise ComponentError(
                    f"Parameters of 'run' and 'run_async' methods must be the same.\nDifferences found:\n{sig_diff}"
                )

    def __call__[T](cls, *args: Any, **kwargs: Any) -> T:
        """
        This method is called when clients instantiate a Component and runs before __new__ and __init__.
        """
        pre_init_hook = _COMPONENT_PRE_INIT_HOOK.get()
        if pre_init_hook is None or pre_init_hook.in_progress:
            instance = super().__call__(*args, **kwargs)
        else:
            try:
                pre_init_hook.in_progress = True
                named_positional_args = ComponentMeta._positional_to_kwargs(cls, args)
                assert set(named_positional_args.keys()).intersection(kwargs.keys()) == set(), (
                    "positional and keyword arguments overlap"
                )
                kwargs.update(named_positional_args)
                pre_init_hook.callback(cls, kwargs)
                instance = super().__call__(**kwargs)
            finally:
                pre_init_hook.in_progress = False

        # Before returning, we have the chance to modify the newly created
        # Component instance, so we take the chance and set up the I/O sockets
        has_async_run = hasattr(instance, "run_async")
        if has_async_run and not inspect.iscoroutinefunction(instance.run_async):
            raise ComponentError(f"Method 'run_async' of component '{cls.__name__}' must be a coroutine")
        instance.__haystack_supports_async__ = has_async_run

        ComponentMeta._parse_and_set_input_sockets(cls, instance)
        ComponentMeta._parse_and_set_output_sockets(instance)

        # Since a Component can't be used in multiple Pipelines at the same time
        # we need to know if it's already owned by a Pipeline when adding it to one.
        # We use this flag to check that.
        instance.__haystack_added_to_pipeline__ = None

        return instance


def _component_repr(component: Component) -> str:
    """
    All Components override their __repr__ method with this one.

    It prints the component name and the input/output sockets.
    """
    result = object.__repr__(component)
    if pipeline := getattr(component, "__haystack_added_to_pipeline__", None):
        result += f"\n{pipeline.get_component_name(component)}"

    return (
        f"{result}\n{getattr(component, '__haystack_input__', '<invalid_input_sockets>')}"
        f"\n{getattr(component, '__haystack_output__', '<invalid_output_sockets>')}"
    )


def _component_run_has_kwargs(component_cls: type[Component]) -> bool:
    run_method = getattr(component_cls, "run", None)
    if run_method is None:
        return False
    return any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in inspect.signature(run_method).parameters.values()
    )


def _compare_run_methods_signatures(run_sig: inspect.Signature, async_run_sig: inspect.Signature) -> str:
    """
    Builds a detailed error message with the differences between the signatures of the run and run_async methods.

    :param run_sig: The signature of the run method
    :param async_run_sig: The signature of the run_async method

    :returns:
        A detailed error message if signatures don't match, empty string if they do
    """
    differences: list[str] = []
    run_params = list(run_sig.parameters.items())
    async_params = list(async_run_sig.parameters.items())

    if len(run_params) != len(async_params):
        differences.append(
            f"Different number of parameters: run has {len(run_params)}, run_async has {len(async_params)}"
        )

    for (run_name, run_param), (async_name, async_param) in zip(run_params, async_params):
        if run_name != async_name:
            differences.append(f"Parameter name mismatch: {run_name} vs {async_name}")

        if run_param.annotation != async_param.annotation:
            differences.append(
                f"Parameter '{run_name}' type mismatch: {run_param.annotation} vs {async_param.annotation}"
            )

        if run_param.default != async_param.default:
            differences.append(
                f"Parameter '{run_name}' default value mismatch: {run_param.default} vs {async_param.default}"
            )

        if run_param.kind != async_param.kind:
            differences.append(
                f"Parameter '{run_name}' kind (POSITIONAL, KEYWORD, etc.) mismatch: "
                f"{run_param.kind} vs {async_param.kind}"
            )

    return "\n".join(differences)


class _Component:
    """
    See module's docstring.

    Args:
        cls: the class that should be used as a component.

    Returns:
        A class that can be recognized as a component.

    Raises:
        ComponentError: if the class provided has no `run()` method or otherwise doesn't respect the component contract.
    """

    def __init__(self) -> None:
        self.registry: dict[str, type[Component]] = {}

    @staticmethod
    def set_input_type(instance: Any, name: str, type: Any, default: Any = _empty) -> None:
        """
        Add a single input socket to the component instance.

        Replaces any existing input socket with the same name.

        :param instance: Component instance where the input type will be added.
        :param name: name of the input socket.
        :param type: type of the input socket.
        :param default: default value of the input socket, defaults to _empty
        """
        if not _component_run_has_kwargs(instance.__class__):
            raise ComponentError(
                "Cannot set input types on a component that doesn't have a kwargs parameter in the 'run' method"
            )
        if not hasattr(instance, "__haystack_input__"):
            instance.__haystack_input__ = Sockets(instance, {}, InputSocket)
        instance.__haystack_input__[name] = InputSocket(name=name, type=type, default_value=default)

    def set_input_types(self, instance: Component, **types: Any) -> None:
        """
        Method that specifies the input types when 'kwargs' is passed to the run method.

        Use as:

        ```python
        _component_instance = component()


        @_component_instance
        class MyComponent:

            def __init__(self, value: int):
                component.set_input_types(self, value_1=str, value_2=str)
                ...

            @_component_instance.output_types(output_1=int, output_2=str)
            def run(self, **kwargs):
                return {"output_1": kwargs["value_1"], "output_2": ""}
        ```

        Note that if the `run()` method also specifies some parameters, those will take precedence.

        For example:

        ```python
        _component_instance = component()


        @_component_instance
        class MyComponent:

            def __init__(self, value: int):
                component.set_input_types(self, value_1=str, value_2=str)
                ...

            @_component_instance.output_types(output_1=int, output_2=str)
            def run(self, value_0: str, value_1: Optional[str] = None, **kwargs):
                return {"output_1": kwargs["value_1"], "output_2": ""}
        ```

        would add a mandatory `value_0` parameters, make the `value_1`
        parameter optional with a default None, and keep the `value_2`
        parameter mandatory as specified in `set_input_types`.

        """
        if not _component_run_has_kwargs(instance.__class__):
            raise ComponentError(
                "Cannot set input types on a component that doesn't have a kwargs parameter in the 'run' method"
            )

        instance.__haystack_input__ = Sockets(
            instance, {name: InputSocket(name=name, type=type_) for name, type_ in types.items()}, InputSocket
        )

    def set_output_types(self, instance: Component, **types: Any) -> None:
        """
        Method that specifies the output types when the 'run' method is not decorated with 'component.output_types'.

        Use as:

        ```python
        _component_instance = component()

        @_component_instance
        class MyComponent:

            def __init__(self, value: int):
                component.set_output_types(self, output_1=int, output_2=str)
                ...

            # no decorators here
            def run(self, value: int):
                return {"output_1": 1, "output_2": "2"}
        ```
        """
        has_decorator = hasattr(instance.run, "_output_types_cache")
        if has_decorator:
            raise ComponentError(
                "Cannot call `set_output_types` on a component that already has "
                "the 'output_types' decorator on its `run` method"
            )

        instance.__haystack_output__ = Sockets(
            instance, {name: OutputSocket(name=name, type=type_) for name, type_ in types.items()}, OutputSocket
        )

    def output_types[T: Callable[..., Any]](self, **types: Any) -> Callable[[T], T]:
        """
        Decorator factory that specifies the output types of a component.

        Use as:
        ```python
        _component_instance = component()

        @_component_instance
        class MyComponent:
            @_component_instance.output_types(output_1=int, output_2=str)
            def run(self, value: int):
                return {"output_1": 1, "output_2": "2"}
        ```
        """

        def output_types_decorator(run_method: T) -> T:
            """
            Decorator that sets the output types of the decorated method.

            This happens at class creation time, and since we don't have the decorated
            class available here, we temporarily store the output types as an attribute of
            the decorated method. The ComponentMeta metaclass will use this data to create
            sockets at instance creation time.
            """
            # Check if the method is asynchronous
            is_async = inspect.iscoroutinefunction(run_method)

            # Create the return type
            if is_async:
                return_type = Coroutine[Any, Any, dict[str, Any]]
            else:
                return_type = dict[str, Any]

            # Attach the output types as metadata to the method
            method_name = run_method.__name__
            if method_name not in ("run", "run_async"):
                raise ComponentError("'output_types' decorator can only be used on 'run' and 'run_async' methods")

            setattr(
                run_method,
                "_output_types_cache",
                {name: OutputSocket(name=name, type=type_) for name, type_ in types.items()},
            )

            # We now use the inferred return_type (Coroutine or Dict) in the annotation
            run_method.__annotations__["return"] = return_type

            return run_method

        return output_types_decorator

    def _component[T: type[Component]](self, cls: T) -> T:
        """
        Decorator validating the structure of the component and registering it in the components registry.
        """
        logger.debug("Registering {component} as a component", component=cls)

        if not hasattr(cls, "run"):
            raise ComponentError(f"{cls.__name__} must have a 'run()' method. See the docs for more information.")

        def copy_class_namespace(namespace: dict[str, Any]) -> None:
            """
            This is the callback that `typing.new_class` will use to populate the newly created class.

            Simply copy the whole namespace from the decorated class.
            """
            for key, val in dict(cls.__dict__).items():
                if key in ("__dict__", "__weakref__"):
                    continue
                namespace[key] = val

        new_cls: T = new_class(cls.__name__, cls.__bases__, {"metaclass": ComponentMeta}, copy_class_namespace)

        class_path = f"{new_cls.__module__}.{new_cls.__name__}"
        if class_path in self.registry:
            logger.debug(
                "Component {component} is already registered. Previous imported from '{module_name}', \
                new imported from '{new_module_name}'",
                component=class_path,
                module_name=self.registry[class_path],
                new_module_name=new_cls,
            )
        self.registry[class_path] = new_cls
        logger.debug("Registered Component {component}", component=new_cls)

        new_cls.__repr__ = lambda: _component_repr(cls)

        return new_cls

    @overload
    def __call__[T: type[Component]](self, cls: T) -> T: ...

    @overload
    def __call__[T: type[Component]](self) -> Callable[[type[T]], T]: ...

    def __call__[T: type[Component]](self, cls: T | None = None) -> T | Callable[[T], T]:
        def wrap(clss: T) -> T:
            return self._component(cls=clss)

        if cls:
            return wrap(cls)

        return wrap


component = _Component
