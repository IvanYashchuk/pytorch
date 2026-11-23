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
class GuardedFrameEvidence:
    frame_code: CodeType
    emitted_code: CodeType
    guarded_code: CodeType
    compile_id: "CompileId | None"
    isolation_id: int
    resume: ResumeSite | None
    calls: tuple[TensorCall, ...]
    resumes: tuple[ResumeCall, ...]


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


def record_guarded_frame(
    output: "OutputGraph",
    frame_code: CodeType,
    emitted_code: CodeType,
    guarded_code: CodeType,
    compile_id: "CompileId | None",
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
        )
    )
