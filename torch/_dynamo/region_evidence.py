# mypy: allow-untyped-defs
"""Experimental compiler-owned evidence for linking guarded Dynamo regions."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from os import getpid
from threading import current_thread, Thread
from types import CodeType
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Iterator

    from torch._guards import CompileId

    from .output_graph import OutputGraph
    from .source import Source
    from .variables.base import VariableTracker
    from .variables.dicts import ConstDictVariable
    from .variables.lists import ListVariable
    from .codegen import GraphOutputEntry


@dataclass(frozen=True)
class GraphInput:
    source_kind: str
    local_name: str | None
    tensorize: bool


@dataclass(frozen=True)
class TensorCall:
    backend_id: str
    inputs: tuple[GraphInput, ...]


@dataclass(frozen=True)
class ResumeSite:
    source_code: CodeType
    code: CodeType
    resume_offset: int
    stack_size: int
    local_names: tuple[str, ...]


@dataclass(frozen=True)
class ResumeCall:
    frame_code: CodeType
    break_offset: int
    site: ResumeSite


@dataclass(frozen=True)
class OperandSource:
    scope: str
    name: str
    attributes: tuple[str, ...] = ()
    entry_argument: bool = False


@dataclass(frozen=True)
class CallOperand:
    tracker: str
    source: OperandSource | None
    same_locals: tuple[str, ...]
    constant_int: int | None


@dataclass(frozen=True)
class HostCall:
    frame_code: CodeType
    offset: int
    target: CallOperand
    receiver: CallOperand | None
    arguments: tuple[CallOperand, ...]
    keywords: tuple[tuple[str, CallOperand], ...]


@dataclass(frozen=True)
class ListReplayItem:
    tracker: str
    source: OperandSource | None
    graph_output: int | None
    constant_int: int | None


@dataclass(frozen=True)
class ListReplayWrite:
    target: OperandSource | None
    items: tuple[ListReplayItem, ...]


@dataclass(frozen=True)
class DictReplayItem:
    key: ListReplayItem
    literal_key: str | None
    value: ListReplayItem


@dataclass(frozen=True)
class DictReplayWrite:
    target: OperandSource | None
    clear_first: bool
    items: tuple[DictReplayItem, ...]


@dataclass(frozen=True)
class GuardedFrameEvidence:
    frame_code: CodeType
    emitted_code: CodeType
    guarded_code: CodeType
    compile_id: "CompileId | None"
    isolation_id: int
    resume: ResumeSite | None
    calls: tuple[TensorCall, ...]
    resumes: tuple[ResumeCall, ...]
    host_calls: tuple[HostCall, ...]
    replayed_mutations: tuple[str, ...]
    list_replays: tuple[ListReplayWrite, ...]
    dict_replays: tuple[DictReplayWrite, ...]


@dataclass
class _Capture:
    source: CodeType
    frames: list[GuardedFrameEvidence]
    pid: int
    thread: Thread
    active: bool = True


_active: ContextVar[_Capture | None] = ContextVar(
    "dynamo_region_evidence", default=None
)


@contextmanager
def capture_region_evidence(source: CodeType) -> "Iterator[list[GuardedFrameEvidence]]":
    """Collect completed compiles for one source and its generated resumes.

    Evidence describes the code passed back to native dispatch. An observer must
    establish installation and execution separately. Code identities are local
    to the current process and only retained by the active scope or its caller.
    """
    capture = _Capture(source, [], getpid(), current_thread())
    token = _active.set(capture)
    try:
        yield capture.frames
    finally:
        capture.active = False
        capture.frames = []
        _active.reset(token)


def _current_capture() -> _Capture | None:
    capture = _active.get()
    if (
        capture is not None
        and capture.active
        and capture.pid == getpid()
        and capture.thread is current_thread()
    ):
        return capture
    return None


def capture_active() -> bool:
    return _current_capture() is not None


def resume_site(code: CodeType) -> ResumeSite | None:
    from .resume_execution import ContinueExecutionCache

    meta = ContinueExecutionCache.generated_code_metadata.get(code)
    if meta is None or meta.site is None:
        return None
    resume_offset, stack_size, local_names = meta.site
    return ResumeSite(meta.code, code, resume_offset, stack_size, local_names)


def _operand_source(source: "Source | None") -> OperandSource | None:
    from .source import AttrSource, GlobalSource, LocalSource

    attributes = []
    while type(source) is AttrSource:
        attributes.append(source.member)
        source = source.base
    path = tuple(reversed(attributes))
    if type(source) is LocalSource:
        return OperandSource(
            "local",
            source.local_name,
            path,
            source.is_input and not source.is_derefed_cell_contents,
        )
    if type(source) is GlobalSource:
        return OperandSource("global", source.global_name, path)
    return None


def _peek(variable: "VariableTracker") -> "VariableTracker":
    from .variables.lazy import (
        ComputedLazyConstantVariable,
        LazyConstantVariable,
        LazyVariableTracker,
    )

    if type(variable) in (
        LazyVariableTracker,
        LazyConstantVariable,
        ComputedLazyConstantVariable,
    ):
        return variable.unwrap()
    return variable


def _operand(
    variable: "VariableTracker",
    locals_: "dict[str, VariableTracker]",
) -> CallOperand:
    from .variables.misc import GetAttrVariable

    variable = _peek(variable)
    source = _operand_source(variable.source)
    if source is None and type(variable) is GetAttrVariable:
        parent = _operand_source(variable.obj.source)
        if parent is not None:
            source = OperandSource(
                parent.scope,
                parent.name,
                (*parent.attributes, variable.name),
                parent.entry_argument,
            )
    return CallOperand(
        type(variable).__name__,
        source,
        tuple(name for name, value in locals_.items() if _peek(value) is variable),
        _constant_int(variable),
    )


def _constant_int(variable: "VariableTracker") -> int | None:
    from .variables.constant import ConstantVariable

    if (
        type(variable) is ConstantVariable
        and variable.source is None
        and type(variable.value) is int
        and -(2**31) <= variable.value < 2**31
    ):
        return variable.value
    return None


def observe_list_replay(
    variable: "ListVariable", graph_outputs: "dict[int, GraphOutputEntry]"
) -> ListReplayWrite:
    """Project a native list reconstruction and its observed direct output uses."""
    return ListReplayWrite(
        _operand_source(variable.source),
        tuple(_replay_item(_peek(value), graph_outputs) for value in variable.items),
    )


def _replay_item(
    value: "VariableTracker", graph_outputs: "dict[int, GraphOutputEntry]"
) -> ListReplayItem:
    from .variables.tensor import TensorVariable

    slot = None
    if type(value) is TensorVariable and value.source is None:
        slot = next(
            (entry.index for entry in graph_outputs.values() if entry.variable is value),
            None,
        )
    return ListReplayItem(
        type(value).__name__, _operand_source(value.source), slot, _constant_int(value)
    )


def observe_dict_replay(
    variable: "ConstDictVariable",
    pairs: "list[tuple[VariableTracker, VariableTracker]]",
    graph_outputs: "dict[int, GraphOutputEntry]",
) -> DictReplayWrite:
    from .variables.constant import ConstantVariable

    items = []
    for raw_key, value in pairs:
        key = _peek(raw_key)
        literal_key = None
        if (
            type(key) is ConstantVariable
            and key.source is None
            and type(key.value) is str
            and len(key.value) <= 64
        ):
            literal_key = key.value
        items.append(
            DictReplayItem(
                _replay_item(key, graph_outputs),
                literal_key,
                _replay_item(_peek(value), graph_outputs),
            )
        )
    return DictReplayWrite(
        _operand_source(variable.source), variable.should_reconstruct_all, tuple(items)
    )


def observe_host_call(
    frame_code: CodeType,
    offset: int,
    operands: "list[VariableTracker]",
    locals_: "dict[str, VariableTracker]",
    has_keywords: bool,
) -> HostCall | None:
    """Read the CALL operands without realizing lazy values or adding guards."""
    from .variables.constant import ConstantVariable
    from .variables.lists import TupleVariable
    from .variables.misc import NullVariable

    target, receiver, *arguments = operands
    keywords = ()
    if has_keywords:
        names = _peek(arguments.pop())
        if type(names) is not TupleVariable:
            return None
        names = tuple(_peek(name) for name in names.items)
        if len(names) > len(arguments) or any(
            type(name) is not ConstantVariable or type(name.value) is not str
            for name in names
        ):
            return None
        if names:
            keywords = tuple(
                (name.value, _operand(argument, locals_))
                for name, argument in zip(names, arguments[-len(names) :], strict=True)
            )
            del arguments[-len(names) :]
    return HostCall(
        frame_code,
        offset,
        _operand(target, locals_),
        None if type(receiver) is NullVariable else _operand(receiver, locals_),
        tuple(_operand(argument, locals_) for argument in arguments),
        keywords,
    )


def record_guarded_frame(
    output: "OutputGraph",
    frame_code: CodeType,
    emitted_code: CodeType,
    guarded_code: CodeType,
    compile_id: "CompileId | None",
    replayed_mutations: tuple[str, ...],
) -> None:
    capture = _current_capture()
    if capture is None:
        return
    resume = resume_site(frame_code)
    if frame_code is not capture.source and (
        resume is None or resume.source_code is not capture.source
    ):
        return
    from .eval_frame import get_eval_frame_isolate_recompiles_id

    capture.frames.append(
        GuardedFrameEvidence(
            frame_code,
            emitted_code,
            guarded_code,
            compile_id,
            get_eval_frame_isolate_recompiles_id(),
            resume,
            tuple(output.region_tensor_calls or ()),
            tuple(output.region_resumes or ()),
            tuple(output.region_host_calls or ()),
            replayed_mutations,
            tuple(output.region_list_replays or ()),
            tuple(output.region_dict_replays or ()),
        )
    )
