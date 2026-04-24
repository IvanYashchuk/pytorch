import argparse
import datetime
import re
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Callable, Pattern, Protocol

import torch
from torch.autograd import DeviceType
from torch.utils._ordered_set import OrderedSet

from .runtime.benchmarking import benchmarker
from .runtime.runtime_utils import create_bandwidth_info_str, get_num_bytes


class BenchmarkCallableType(Protocol):
    def __call__(self, times: int, repeat: int) -> float: ...


@dataclass(frozen=True)
class KernelBenchmarkInfo:
    kernel: Any
    device_type: str
    arg_names: list[str]
    category: str
    num_gb: float | None = None
    launcher: Any | None = None


KernelBenchmarkProvider = Callable[[ModuleType], KernelBenchmarkInfo | None]


_kernel_category_choices = [
    "foreach",
    "persistent_reduction",
    "pointwise",
    "reduction",
    "split_scan",
    "template",
]

_kernel_benchmark_providers: dict[str, KernelBenchmarkProvider] = {}


def _validate_backend_name(name: str) -> None:
    if not name:
        raise ValueError("Backend name must be non-empty")
    if not name.isidentifier():
        raise ValueError(f"Backend name must be a valid Python identifier: {name!r}")


def register_kernel_benchmark_provider(
    backend: str, provider: KernelBenchmarkProvider
) -> None:
    """
    Register backend-specific generated-kernel discovery for wrapper benchmarks.

    Providers should return ``None`` when a module does not contain their
    backend's kernel. This keeps Triton as the built-in default while allowing
    out-of-tree backends to make ``output_code.py --benchmark-kernels`` work
    without PyTorch importing backend-specific runtime classes.
    """
    _validate_backend_name(backend)
    existing = _kernel_benchmark_providers.get(backend)
    if existing is not None and existing is not provider:
        raise ValueError(f"Kernel benchmark provider {backend!r} is already registered")
    _kernel_benchmark_providers[backend] = provider
_kernel_category_source_patterns: list[tuple[Pattern[str], str]] = [
    (re.compile(re.escape(f"@triton_heuristics.{choice}")), choice)
    for choice in _kernel_category_choices
]


def register_kernel_category_pattern(
    pattern: str | Pattern[str],
    category: str,
) -> None:
    """
    Register a source-code pattern used to classify generated kernels.

    Out-of-tree backends can use this to make wrapper benchmarks and metadata
    recognize their generated source without patching this module.
    """
    compiled = re.compile(pattern) if isinstance(pattern, str) else pattern
    entry = (compiled, category)
    if any(
        existing.pattern == compiled.pattern and existing_category == category
        for existing, existing_category in _kernel_category_source_patterns
    ):
        return
    _kernel_category_source_patterns.append(entry)


def get_kernel_category_by_source_code(src_code: str) -> str:
    """
    Similar to get_kernel_category but use the source code. Call this API
    if we have not compile the src_code to module yet.
    """
    choices = [
        category
        for pattern, category in _kernel_category_source_patterns
        if pattern.search(src_code)
    ]
    if len(choices) == 1:
        return choices[0]
    else:
        return "unknown"


def get_kernel_category(kernel_mod: ModuleType) -> str:
    """
    Given the module defining a triton kernel, return the category of the kernel.
    Category can be one of:
    - pointwise
    - reduction
    - persistent_reduction

    Currently we simply decide the category depending on what decorator is imported
    by the kernel.
    """
    choices = [ch for ch in _kernel_category_choices if ch in kernel_mod.__dict__]
    if len(choices) == 1:
        return choices[0]
    else:
        return "unknown"


def get_triton_kernel(mod: ModuleType):  # type: ignore[no-untyped-def]
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner

    cand_list = [
        v
        for k, v in mod.__dict__.items()
        if k.startswith("triton_") and isinstance(v, CachingAutotuner)
    ]
    assert len(cand_list) == 1
    return cand_list[0]


def _get_triton_kernel_benchmark_info(
    mod: ModuleType,
) -> KernelBenchmarkInfo | None:
    try:
        triton_kernel = get_triton_kernel(mod)
    except AssertionError:
        return None
    launcher = (
        triton_kernel.launchers[0] if len(triton_kernel.launchers) == 1 else None
    )
    return KernelBenchmarkInfo(
        kernel=triton_kernel,
        device_type=triton_kernel.device_props.type,
        arg_names=list(triton_kernel.fn.arg_names),
        category=get_kernel_category(mod),
        num_gb=triton_kernel.inductor_meta.get("kernel_num_gb", None),
        launcher=launcher,
    )


def get_kernel_benchmark_info(mod: ModuleType) -> KernelBenchmarkInfo:
    for provider in _kernel_benchmark_providers.values():
        info = provider(mod)
        if info is not None:
            return info

    info = _get_triton_kernel_benchmark_info(mod)
    assert info is not None
    return info


def benchmark_all_kernels(
    benchmark_name: str, benchmark_all_configs: dict[Any, Any] | None
) -> None:
    """
    An experimental API used only when config.benchmark_kernel is true.

    Run the kernel benchmarks for all the kernels cached in PyCodeCache.
    Used in the compiled modules.

    Put this method here rather than codegen it for convenience since its implementation
    does not change based on different graph modules being compiled.
    """
    from torch._inductor.codecache import PyCodeCache

    nfound = 0
    for kernel_mod in PyCodeCache.modules:
        kernel_key = kernel_mod.key
        if not hasattr(kernel_mod, "get_args") or not hasattr(kernel_mod, "call"):
            continue

        kernel_info = get_kernel_benchmark_info(kernel_mod)
        args = kernel_mod.get_args()
        num_in_out_ptrs = len(
            [
                arg_name
                for arg_name in kernel_info.arg_names
                if arg_name.startswith("in_out_ptr")
            ]
        )
        num_gb = kernel_info.num_gb
        if num_gb is None:
            num_gb = get_num_bytes(*args, num_in_out_args=num_in_out_ptrs) / 1e9

        def get_info_str(
            ms: float,
            n_regs: Any | None,
            n_spills: Any | None,
            shared: Any | None,
            prefix: str = "",
        ) -> str:
            if not any(x is None for x in [n_regs, n_spills, shared]):
                kernel_detail_str = (
                    f"  {n_regs:3} regs  {n_spills:3} spills  {shared:8} shared mem"
                )
            else:
                kernel_detail_str = ""

            gb_per_s = num_gb / (ms / 1e3)
            return create_bandwidth_info_str(
                ms, num_gb, gb_per_s, prefix=prefix, suffix=kernel_detail_str
            )

        kernel_desc = (
            f"{benchmark_name:20} {kernel_info.category[:3].upper()} {kernel_key[:10]}"
        )
        if benchmark_all_configs:
            assert hasattr(kernel_mod, "benchmark_all_configs")
            bench_result = kernel_mod.benchmark_all_configs(args)
            print(kernel_desc)
            for launcher, ms in bench_result.items():
                print(
                    f"  {get_info_str(ms, launcher.n_regs, launcher.n_spills, launcher.shared)} @ {launcher.config}"
                )
        else:
            ms = benchmarker.benchmark(
                lambda: kernel_mod.call(args),
                device=kernel_info.device_type,
                rep=40,
            )
            launcher = kernel_info.launcher
            print(
                get_info_str(
                    ms,
                    getattr(launcher, "n_regs", None),
                    getattr(launcher, "n_spills", None),
                    getattr(launcher, "shared", None),
                    prefix=f"{kernel_desc} ",
                )
            )

        nfound += 1
    if nfound == 0:
        print(
            "No kernel with benchmark functionality found. Make sure you run inductor with config.benchmark_kernel being True"
        )


@dataclass
class ProfileEvent:
    category: str
    key: str
    self_device_time_ms: float
    # the benchmark is run multiple times and we average the count across all the
    # runs. It should be an integer but define a float just in case.
    count: float


def parse_profile_event_list(
    benchmark_name: str,
    event_list: torch.autograd.profiler_util.EventList,
    wall_time_ms: float,
    nruns: int,
    device_name: str,
) -> None:
    """
    Parse and generate a report for an event_list.
    """

    def get_self_device_time(
        ev: torch.autograd.profiler_util.EventList,
    ) -> float:
        """
        ev.self_device_time_total is in microsecond. Convert to millisecond.
        """
        return ev.self_device_time_total / 1000 / nruns  # type: ignore[attr-defined]

    all_events: dict[str, list[ProfileEvent]] = defaultdict(list)

    def add_event(
        ev: torch.autograd.profiler_util.EventList,
        category: str,
    ) -> None:
        profile_ev = ProfileEvent(
            category=category,
            key=ev.key,  # type: ignore[attr-defined]
            self_device_time_ms=get_self_device_time(ev),
            count=ev.count / nruns,  # type: ignore[operator] # average across all runs
        )
        all_events[category].append(profile_ev)

    for ev in event_list:
        assert not ev.is_legacy, "Don't support the legacy profiler"
        if ev.device_type == DeviceType.CPU:
            # ignore the event on CPU side
            continue

        category = "unknown"
        if ev.key.startswith("triton_"):
            if ev.key.startswith("triton_poi"):
                category = "triton_pointwise"
            elif ev.key.startswith("triton_red"):
                category = "triton_reduction"
            elif ev.key.startswith("triton_per"):
                category = "triton_persistent_reduction"
            else:
                category = "triton_unknown"

        add_event(ev, category)

    def report_category(category: str, profile_events: list[ProfileEvent]) -> float:
        if not device_name:
            return 0.0

        from tabulate import tabulate

        profile_events.sort(key=lambda ev: ev.self_device_time_ms, reverse=True)

        rows = []
        total_time = 0.0
        print(f"\n  == {category} category kernels == ")
        for ev in profile_events:
            total_time += ev.self_device_time_ms
            percent = f"{ev.self_device_time_ms / wall_time_ms * 100:.2f}%"
            rows.append([ev.key[:120], ev.self_device_time_ms, ev.count, percent])
        rows.append(
            ["Total", total_time, "", f"{total_time / wall_time_ms * 100:.2f}%"]
        )
        print(
            tabulate(
                rows,
                headers=[
                    "Kernel",
                    f"Self {device_name.upper()} TIME (ms)",
                    "Count",
                    "Percent",
                ],
            )
        )
        return total_time

    def report() -> None:
        category_list = [
            "triton_pointwise",
            "triton_reduction",
            "triton_persistent_reduction",
            "triton_unknown",
            "unknown",
        ]
        assert OrderedSet(all_events.keys()).issubset(OrderedSet(category_list)), (
            f"{list(all_events.keys())}"
        )

        per_category_wall_time = {}
        total_device_ms = 0.0
        for category in category_list:
            if category in all_events:
                _time = report_category(category, all_events[category])
                per_category_wall_time[category] = _time
                total_device_ms += _time

        device_busy_percent = f"{total_device_ms / wall_time_ms * 100:.2f}%"
        if device_name:
            print(
                f"\nPercent of time when {device_name.upper()} is busy: {device_busy_percent}"
            )
        else:
            print("No device detected")

        print(f"Total wall time {wall_time_ms:.3f} ms")

        # output such a line so we can gather such line from all compiled modules from all
        # benchmarks and tabulate it!
        # Columns: benchmark_name, pointwise_percent, reduction_percent, persistent_reduction_percent,
        #   unknown_category_percent, device_busy_percent, wall_time_ms
        tabulate_line = f"Output for tabulate: {benchmark_name}"
        for category in category_list:
            percent = (
                f"{per_category_wall_time.get(category, 0.0) / wall_time_ms * 100:.2f}%"
            )
            tabulate_line += f", {percent}"
        tabulate_line += f", {device_busy_percent}, {wall_time_ms:.3f}ms"

        print(tabulate_line)

    report()


PROFILE_DIR = tempfile.gettempdir()
PROFILE_PATH = f"{PROFILE_DIR}/compiled_module_profile.json"


def perf_profile(
    wall_time_ms: float,
    times: int,
    repeat: int,
    benchmark_name: str,
    benchmark_compiled_module_fn: BenchmarkCallableType,
) -> None:
    with torch.profiler.profile(record_shapes=True) as p:
        benchmark_compiled_module_fn(times=times, repeat=repeat)

    path = PROFILE_PATH
    p.export_chrome_trace(path)
    print(f"Profiling result for a compiled module of benchmark {benchmark_name}:")
    print(f"Chrome trace for the profile is written to {path}")
    event_list = p.key_averages(group_by_input_shape=True)
    print(event_list.table(sort_by="self_device_time_total", row_limit=10))
    parse_profile_event_list(
        benchmark_name, event_list, wall_time_ms, times * repeat, p.use_device or ""
    )


def ncu_analyzer(
    benchmark_name: str,
    benchmark_compiled_module_fn: BenchmarkCallableType,
    args: argparse.Namespace,
) -> None:
    import inspect
    import os
    import subprocess

    kernel_regex = args.ncu_kernel_regex
    metrics = args.ncu_metrics

    module_file = inspect.getfile(benchmark_compiled_module_fn)
    module_dir = os.path.dirname(module_file)
    module_name = os.path.splitext(os.path.basename(module_file))[0]

    ncu_dir = tempfile.gettempdir()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ncu_output = os.path.join(ncu_dir, f"ncu_output_{timestamp}.ncu-rep")
    python_cmd = (
        f"""import sys; sys.path.insert(0, '{module_dir}'); """
        f"""from {module_name} import benchmark_compiled_module; """
        """benchmark_compiled_module(times=1, repeat=1)"""
    )

    ncu_cmd = [
        "ncu",
        "--target-processes",
        "all",
        "--replay-mode",
        "kernel",
        "--kernel-name-base",
        "function",
        "--print-units",
        "base",
        "--import-source",
        "yes",
        "--force-overwrite",
        "--export",
        ncu_output,
    ]

    if kernel_regex:
        ncu_cmd.extend(["--kernel-name", f"regex:{kernel_regex}"])

    if metrics:
        ncu_cmd.extend(["--metrics", metrics])
    else:
        ncu_cmd.extend(["--set", "full"])

    ncu_cmd.extend(
        [
            "python",
            "-c",
            python_cmd,
        ]
    )

    try:
        subprocess.run(ncu_cmd, check=True)
        print(f"\nNCU profiling results for benchmark {benchmark_name}:")
        print(f"NCU report has been written to {ncu_output}")

    except subprocess.CalledProcessError as e:
        print(f"NCU profiling failed with error: {e}")
        return


def collect_memory_snapshot(
    benchmark_compiled_module_fn: BenchmarkCallableType,
) -> None:
    assert torch.cuda.is_available()

    torch.cuda.memory._record_memory_history(max_entries=100000)
    benchmark_compiled_module_fn(times=10, repeat=1)  # run 10 times
    snapshot_path = f"{tempfile.gettempdir()}/memory_snapshot.pickle"
    torch.cuda.memory._dump_snapshot(snapshot_path)
    torch.cuda.memory._record_memory_history(enabled=None)
    print(f"The collect memory snapshot has been written to {snapshot_path}")


# With AOTAutograd cache, we directly call the compiled module. So prevent
# Dynamo from reentering
@torch.compiler.disable  # type: ignore[misc]
def compiled_module_main(
    benchmark_name: str, benchmark_compiled_module_fn: BenchmarkCallableType
) -> None:
    """
    This is the function called in __main__ block of a compiled module.
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark-kernels",
        "-k",
        action="store_true",
        help="Whether to benchmark each individual kernels",
    )
    parser.add_argument(
        "--benchmark-all-configs",
        "-c",
        action="store_true",
        help="Whether to benchmark each individual config for a kernel",
    )
    parser.add_argument(
        "--profile",
        "-p",
        action="store_true",
        help="Whether to profile the compiled module",
    )
    parser.add_argument(
        "--cuda-memory-snapshot",
        action="store_true",
        help="""
            Whether to collect CUDA memory snapshot. Refer to
            "https://pytorch.org/blog/understanding-gpu-memory-1/
            for details about how to visualize the collected snapshot
        """,
    )
    parser.add_argument(
        "--ncu",
        action="store_true",
        help="Whether to run ncu analysis",
    )
    parser.add_argument(
        "--ncu-kernel-regex",
        type=str,
        default=None,
        help=(
            "Filter kernels profiled by NCU using a regex (e.g., '^triton_.*'). "
            "Maps to '--kernel-name regex:<regex>'. "
            "If None, NCU will profile all kernels."
        ),
    )
    parser.add_argument(
        "--ncu-metrics",
        type=str,
        default=None,
        help=(
            "Comma-separated list of NCU metrics to collect (e.g., 'dram__bytes.sum.per_second'). "
            "If None, NCU will use '--set full'."
        ),
    )
    parser.add_argument(
        "--times",
        type=int,
        default=10,
        help="Number of times to run each benchmark iteration",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=10,
        help="Number of repetitions of each benchmark run",
    )

    args = parser.parse_args()

    if args.benchmark_kernels:
        benchmark_all_kernels(benchmark_name, args.benchmark_all_configs)
    else:
        times = args.times
        repeat = args.repeat

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        wall_time_ms = benchmark_compiled_module_fn(times=times, repeat=repeat) * 1000

        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated()
            print(f"Peak GPU memory usage {peak_mem / 1e6:.3f} MB")

        if torch.cuda.is_available() and args.cuda_memory_snapshot:
            collect_memory_snapshot(benchmark_compiled_module_fn)

        if args.profile:
            perf_profile(
                wall_time_ms,
                times,
                repeat,
                benchmark_name,
                benchmark_compiled_module_fn,
            )
        if args.ncu:
            ncu_analyzer(
                benchmark_name,
                benchmark_compiled_module_fn,
                args=args,
            )
