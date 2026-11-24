"""Observe the saved-tensor effects of a traced autograd.Function."""

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypedDict

import torch


if TYPE_CHECKING:
    from torch._dynamo.output_graph import SubgraphTracer
    from torch._dynamo.variables.base import VariableTracker
    from torch._dynamo.variables.misc import AutogradFunctionContextVariable
    from torch.fx import Graph
    from torch.fx.proxy import Proxy


AUTOGRAD_SAVED_TENSOR_EFFECTS = "autograd_saved_tensor_effects"


class SavedTensorEffects(TypedDict):
    saved_names: tuple[str | None, ...]
    used_slots: tuple[int, ...]
    read_prefix: tuple[str, ...]


@dataclass
class SavedTensorTrace:
    ctx: "AutogradFunctionContextVariable"
    arguments: Sequence["VariableTracker"]
    reads: list[tuple["Graph", tuple[str, ...]]] = field(default_factory=list)
    ambiguous: bool = False
    backward: bool = False
    saved_before_backward: tuple["VariableTracker", ...] | None = None

    @contextmanager
    def observing(
        self,
        *,
        backward: bool = False,
    ) -> Iterator[None]:
        previous = self.backward
        self.backward = backward
        token = _active_trace.set(self)
        try:
            yield
        finally:
            self.backward = previous
            _active_trace.reset(token)

    def prepare_backward(
        self,
        output: "VariableTracker",
        attributes: Mapping[str, "VariableTracker"],
    ) -> None:
        current = self.ctx.saved_tensors
        if current is not None and _valid_topology(self.ctx, output, attributes):
            self.saved_before_backward = tuple(current.tensors)

    def record(
        self,
        forward_tracer: "SubgraphTracer",
        backward_graph: "Graph",
        backward_freevars: Mapping["Proxy", "Proxy"],
    ) -> SavedTensorEffects | None:
        saved = self.saved_before_backward
        current = self.ctx.saved_tensors
        if (
            saved is None
            or current is None
            or self.ambiguous
            or len(self.reads) != 1
            or self.reads[0][0] is not backward_graph
            or any(
                node.op == "call_function"
                and isinstance(node.target, torch._ops.HigherOrderOperator)
                for node in backward_graph.nodes
            )
            or tuple(map(id, saved)) != tuple(map(id, current.tensors))
        ):
            return None

        return _save_record(
            saved, self.arguments, forward_tracer, backward_freevars, self.reads[0][1]
        )


_active_trace: ContextVar[SavedTensorTrace | None] = ContextVar(
    "_autograd_saved_tensor_trace", default=None
)


def observe_context_read(
    ctx: "AutogradFunctionContextVariable",
    graph: "Graph",
    name: str,
    value: "VariableTracker",
) -> None:
    trace = _active_trace.get()
    if trace is None or trace.ctx is not ctx:
        return
    if name == "saved_tensors":
        if trace.backward:
            trace.reads.append((graph, _read_prefix(graph)))
        else:
            # Dynamo's simulated context exposes the tuple before eager registration.
            trace.ambiguous = True
    elif trace.backward and not _is_literal(value):
        # A tensor attribute may share a proxy with a save without going through
        # autograd's unpack and hook substitution.
        trace.ambiguous = True


def observe_external_tensor_source() -> None:
    trace = _active_trace.get()
    if trace is not None and trace.backward:
        trace.ambiguous = True


def observe_external_cell(value: "VariableTracker") -> None:
    trace = _active_trace.get()
    if trace is None or not trace.backward:
        return
    from .variables.constant import ConstantVariable
    from .variables.functions import UserFunctionVariable

    if not isinstance(value, (ConstantVariable, UserFunctionVariable)):
        trace.ambiguous = True


def _read_prefix(graph: "Graph") -> tuple[str, ...]:
    return tuple(
        f"{node.op}:{getattr(node.target, '__name__', node.target)}"
        for node in graph.nodes
        if node.op != "placeholder"
        and not (
            node.op == "call_function"
            and node.target is torch._C._set_grad_enabled
            and node.args == (False,)
            and not node.kwargs
        )
    )


def _is_literal(value: "VariableTracker") -> bool:
    from .variables.constant import ConstantVariable

    return isinstance(value, ConstantVariable) and type(value.value) in (
        bool,
        int,
        float,
        complex,
        str,
        type(None),
    )


def _valid_topology(
    ctx: "AutogradFunctionContextVariable",
    output: "VariableTracker",
    attributes: Mapping[str, "VariableTracker"],
) -> bool:
    from .variables.constant import ConstantVariable
    from .variables.lists import TupleVariable

    if output.is_tensor():
        outputs = (output,)
    elif type(output) is TupleVariable and all(
        value.is_tensor() for value in output.items
    ):
        outputs = output.items
    else:
        return False

    # Pending instance attribute reads can bypass the context getter.
    for name, value in attributes.items():
        if name == "needs_input_grad":
            if isinstance(value, TupleVariable) and all(
                isinstance(flag, ConstantVariable) and type(flag.value) is bool
                for flag in value.items
            ):
                continue
            return False
        if not _is_literal(value):
            return False
        if name == "materialize_grads" and not (
            isinstance(value, ConstantVariable) and value.value is True
        ):
            return False

    if not ctx.dirty_tensors:
        return True
    if any(not value.is_tensor() for value in ctx.dirty_tensors):
        return False
    nodes = tuple(value.as_proxy().node for value in outputs)
    dirty = tuple(value.as_proxy().node for value in ctx.dirty_tensors)
    # The HOP lowers dirty roles to output indexes and loses multiplicity.
    return len(set(dirty)) == len(dirty) and all(nodes.count(node) == 1 for node in dirty)


def _save_record(
    saved: Sequence["VariableTracker"],
    arguments: Sequence["VariableTracker"],
    tracer: "SubgraphTracer",
    backward_freevars: Mapping["Proxy", "Proxy"],
    read_prefix: tuple[str, ...],
) -> SavedTensorEffects | None:
    from .variables.constant import ConstantVariable

    def inner_node(proxy: "Proxy") -> torch.fx.Node:
        return tracer.lifted_freevars.get(proxy, proxy).node

    argument_nodes = {
        inner_node(value.as_proxy()) for value in arguments if value.is_tensor()
    }
    names: list[str | None] = []
    used: dict[str, int] = {}
    for slot, value in enumerate(saved):
        if isinstance(value, ConstantVariable) and value.value is None:
            names.append(None)
            continue
        if not value.is_tensor():
            return None
        node = inner_node(value.as_proxy())
        if (
            node.graph is not tracer.graph
            or node.name in used
            or node.op in ("placeholder", "get_attr")
            and node not in argument_nodes
        ):
            return None
        names.append(node.name)
        used[node.name] = slot

    slots: list[int] = []
    for proxy in backward_freevars:
        node = inner_node(proxy)
        if node.graph is not tracer.graph or node.name not in used:
            return None
        slots.append(used[node.name])
    return {
        "saved_names": tuple(names),
        "used_slots": tuple(slots),
        "read_prefix": read_prefix,
    }
