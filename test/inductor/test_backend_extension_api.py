# Owner(s): ["module: inductor"]

import sys
import tempfile
from types import ModuleType
from unittest import mock
from pathlib import Path

import torch
from torch._inductor import config, metrics, wrapper_benchmark
from torch._inductor.async_compile import (
    _async_compile_backends,
    AsyncCompile,
    register_async_compile_backend,
)
from torch._inductor.codegen import common
from torch._inductor.codegen.cuda_combined_scheduling import CUDACombinedScheduling
from torch._inductor.codegen.triton import (
    TileKernel,
    TileKernelScheduling,
    TritonKernel,
    TritonScheduling,
)
from torch._inductor.runtime import triton_heuristics
from torch._inductor.runtime.hints import HeuristicType
from torch._inductor.runtime.triton_compat import Config
from torch._inductor.scheduler import (
    BaseScheduling,
    ForeachKernelSchedulerNode,
    Scheduler,
)
from torch.testing._internal.common_utils import TestCase


class BackendExtensionAPITests(TestCase):
    def tearDown(self):
        common._cuda_backends.pop("dummy_cuda_backend", None)
        common._constexpr_syntaxes.pop("dummy_constexpr_backend", None)
        common._dtype_propagation_backends.pop("dummy_dtype_backend", None)
        common._backend_wrapper_imports.pop("dummy_wrapper_backend", None)
        metrics._kernel_metadata_providers.pop("dummy_metrics_backend", None)
        wrapper_benchmark._kernel_benchmark_providers.pop(
            "dummy_benchmark_backend", None
        )
        _async_compile_backends.pop("dummy_async_backend", None)
        if hasattr(AsyncCompile, "dummy_async_backend"):
            delattr(AsyncCompile, "dummy_async_backend")
        common.init_backend_registration.cache_clear()
        super().tearDown()

    def test_register_cuda_backend_before_init(self):
        class DummyCudaScheduling(CUDACombinedScheduling):
            pass

        common.init_backend_registration.cache_clear()

        with (
            mock.patch.dict(common.device_codegens, {}, clear=True),
            mock.patch.dict(common._cuda_backends, {}, clear=True),
            config.patch(cuda_backend="dummy_cuda_backend"),
        ):
            common.register_cuda_backend("dummy_cuda_backend", DummyCudaScheduling)
            common.init_backend_registration()

            scheduling_ctor = common.get_scheduling_for_device("cuda")
            self.assertIsNotNone(scheduling_ctor)
            self.assertIsInstance(scheduling_ctor(None), DummyCudaScheduling)

    def test_register_cuda_backend_after_init(self):
        class DummyCudaScheduling(CUDACombinedScheduling):
            pass

        common.init_backend_registration()
        common.register_cuda_backend("dummy_cuda_backend", DummyCudaScheduling)

        with config.patch(cuda_backend="dummy_cuda_backend"):
            scheduling_ctor = common.get_scheduling_for_device("cuda")
            self.assertIsNotNone(scheduling_ctor)
            self.assertIsInstance(scheduling_ctor(None), DummyCudaScheduling)

    def test_register_cuda_backend_rejects_conflicting_duplicate(self):
        class DummyCudaScheduling(CUDACombinedScheduling):
            pass

        class OtherDummyCudaScheduling(CUDACombinedScheduling):
            pass

        common.register_cuda_backend("dummy_cuda_backend", DummyCudaScheduling)
        common.register_cuda_backend("dummy_cuda_backend", DummyCudaScheduling)

        with self.assertRaisesRegex(ValueError, "already registered"):
            common.register_cuda_backend(
                "dummy_cuda_backend", OtherDummyCudaScheduling
            )

    def test_register_cuda_backend_rejects_invalid_name(self):
        class DummyCudaScheduling(CUDACombinedScheduling):
            pass

        for name in ("", "not-valid", "class"):
            with self.assertRaisesRegex(ValueError, "valid Python identifier|non-empty"):
                common.register_cuda_backend(name, DummyCudaScheduling)

    def test_backend_registries_accept_reloaded_equivalent_symbols(self):
        class DummyCudaScheduling(CUDACombinedScheduling):
            pass

        class ReloadedDummyCudaScheduling(CUDACombinedScheduling):
            pass

        ReloadedDummyCudaScheduling.__module__ = DummyCudaScheduling.__module__
        ReloadedDummyCudaScheduling.__qualname__ = DummyCudaScheduling.__qualname__

        def dummy_async_backend(self, kernel_name: str, source_code: str):
            return "original", kernel_name, source_code

        def reloaded_dummy_async_backend(self, kernel_name: str, source_code: str):
            return "reloaded", kernel_name, source_code

        reloaded_dummy_async_backend.__module__ = dummy_async_backend.__module__
        reloaded_dummy_async_backend.__qualname__ = dummy_async_backend.__qualname__

        def dummy_metadata_provider(*args):
            return None

        def reloaded_dummy_metadata_provider(*args):
            return None

        reloaded_dummy_metadata_provider.__module__ = (
            dummy_metadata_provider.__module__
        )
        reloaded_dummy_metadata_provider.__qualname__ = (
            dummy_metadata_provider.__qualname__
        )

        def dummy_benchmark_provider(*args):
            return None

        def reloaded_dummy_benchmark_provider(*args):
            return None

        reloaded_dummy_benchmark_provider.__module__ = (
            dummy_benchmark_provider.__module__
        )
        reloaded_dummy_benchmark_provider.__qualname__ = (
            dummy_benchmark_provider.__qualname__
        )

        common.register_cuda_backend("dummy_cuda_backend", DummyCudaScheduling)
        common.register_cuda_backend(
            "dummy_cuda_backend", ReloadedDummyCudaScheduling
        )
        self.assertIs(
            common._cuda_backends["dummy_cuda_backend"],
            ReloadedDummyCudaScheduling,
        )

        register_async_compile_backend("dummy_async_backend", dummy_async_backend)
        register_async_compile_backend(
            "dummy_async_backend", reloaded_dummy_async_backend
        )
        self.assertEqual(
            AsyncCompile().dummy_async_backend("kernel0", "source"),
            ("reloaded", "kernel0", "source"),
        )

        metrics.register_kernel_metadata_provider(
            "dummy_metrics_backend", dummy_metadata_provider
        )
        metrics.register_kernel_metadata_provider(
            "dummy_metrics_backend", reloaded_dummy_metadata_provider
        )
        self.assertIs(
            metrics._kernel_metadata_providers["dummy_metrics_backend"],
            reloaded_dummy_metadata_provider,
        )

        wrapper_benchmark.register_kernel_benchmark_provider(
            "dummy_benchmark_backend", dummy_benchmark_provider
        )
        wrapper_benchmark.register_kernel_benchmark_provider(
            "dummy_benchmark_backend", reloaded_dummy_benchmark_provider
        )
        self.assertIs(
            wrapper_benchmark._kernel_benchmark_providers[
                "dummy_benchmark_backend"
            ],
            reloaded_dummy_benchmark_provider,
        )

    def test_register_backend_wrapper_import(self):
        common.register_backend_wrapper_import(
            "dummy_wrapper_backend", "import dummy_backend"
        )
        common.register_backend_wrapper_import(
            "dummy_wrapper_backend", "import dummy_backend"
        )
        common.register_backend_wrapper_import(
            "dummy_wrapper_backend", "dummy_backend.register()"
        )

        self.assertEqual(
            common.get_backend_wrapper_imports("dummy_wrapper_backend"),
            ("import dummy_backend", "dummy_backend.register()"),
        )

    def test_register_backend_wrapper_import_rejects_invalid_input(self):
        with self.assertRaisesRegex(ValueError, "valid Python identifier"):
            common.register_backend_wrapper_import("not-valid", "import dummy")
        with self.assertRaisesRegex(ValueError, "non-empty"):
            common.register_backend_wrapper_import("dummy_wrapper_backend", " ")

    def test_backend_wrapper_import_restores_async_compile_registration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            package_path = Path(tmpdir) / "dummy_backend_package.py"
            package_path.write_text(
                "\n".join(
                    [
                        "from torch._inductor.async_compile import register_async_compile_backend",
                        "",
                        "def compile_dummy(self, kernel_name, source_code):",
                        "    return kernel_name, source_code",
                        "",
                        "register_async_compile_backend('dummy_async_backend', compile_dummy)",
                    ]
                )
            )

            sys.path.insert(0, tmpdir)
            try:
                common.register_backend_wrapper_import(
                    "dummy_wrapper_backend", "import dummy_backend_package"
                )
                _async_compile_backends.pop("dummy_async_backend", None)
                if hasattr(AsyncCompile, "dummy_async_backend"):
                    delattr(AsyncCompile, "dummy_async_backend")
                sys.modules.pop("dummy_backend_package", None)

                source = "\n".join(
                    [
                        "from torch._inductor.async_compile import AsyncCompile",
                        *common.get_backend_wrapper_imports("dummy_wrapper_backend"),
                        "async_compile = AsyncCompile()",
                        "compiled = async_compile.dummy_async_backend('kernel', 'source')",
                    ]
                )
                namespace: dict[str, object] = {}
                exec(source, namespace)

                self.assertEqual(namespace["compiled"], ("kernel", "source"))
            finally:
                sys.path.remove(tmpdir)
                sys.modules.pop("dummy_backend_package", None)

    def test_unknown_cuda_backend_has_actionable_error(self):
        common.init_backend_registration()

        with config.patch(cuda_backend="missing_backend"):
            scheduling_ctor = common.get_scheduling_for_device("cuda")
            self.assertIsNotNone(scheduling_ctor)
            with self.assertRaisesRegex(KeyError, "Available CUDA backends"):
                scheduling_ctor(None)

    def test_register_async_compile_backend_installs_instance_method(self):
        def dummy_async_backend(self, kernel_name: str, source_code: str):
            return type(self).__name__, kernel_name, source_code

        register_async_compile_backend("dummy_async_backend", dummy_async_backend)
        register_async_compile_backend("dummy_async_backend", dummy_async_backend)

        self.assertEqual(
            AsyncCompile().dummy_async_backend("kernel0", "source"),
            ("AsyncCompile", "kernel0", "source"),
        )

    def test_register_async_compile_backend_rejects_conflicting_duplicate(self):
        def dummy_async_backend(self, kernel_name: str, source_code: str):
            return kernel_name, source_code

        def other_dummy_async_backend(self, kernel_name: str, source_code: str):
            return source_code, kernel_name

        register_async_compile_backend("dummy_async_backend", dummy_async_backend)

        with self.assertRaisesRegex(ValueError, "already registered"):
            register_async_compile_backend(
                "dummy_async_backend", other_dummy_async_backend
            )

    def test_register_async_compile_backend_rejects_invalid_name(self):
        def dummy_async_backend(self, kernel_name: str, source_code: str):
            return kernel_name, source_code

        for name in ("", "not-valid", "class"):
            with self.assertRaisesRegex(ValueError, "valid Python identifier|non-empty"):
                register_async_compile_backend(name, dummy_async_backend)

    def test_cuda_combined_scheduling_kernel_scheduler_is_overridable(self):
        class DummyKernelScheduling(TritonScheduling):
            pass

        class DummyCombinedScheduling(CUDACombinedScheduling):
            kernel_scheduling_class = DummyKernelScheduling

        scheduling = DummyCombinedScheduling(None)

        self.assertIsInstance(
            scheduling._kernel_scheduling, DummyKernelScheduling
        )

    def test_combo_kernel_support_capability(self):
        class DummyTileKernelScheduling(TileKernelScheduling):
            pass

        class DummyCombinedScheduling(CUDACombinedScheduling):
            kernel_scheduling_class = DummyTileKernelScheduling

        self.assertFalse(BaseScheduling(None).supports_combo_kernels())
        self.assertFalse(DummyTileKernelScheduling(None).supports_combo_kernels())
        self.assertTrue(TritonScheduling(None).supports_combo_kernels())
        self.assertTrue(CUDACombinedScheduling(None).supports_combo_kernels())
        self.assertFalse(DummyCombinedScheduling(None).supports_combo_kernels())

    def test_create_combo_kernel_nodes_skips_unsupported_backend(self):
        device = torch.device("cuda")

        class DummyNode:
            def __init__(self, name, min_order):
                self._name = name
                self.min_order = min_order

            def get_device(self):
                return device

            def get_name(self):
                return self._name

            def is_template(self):
                return False

            def is_reduction(self):
                return False

        nodes = [DummyNode("node0", 0), DummyNode("node1", 1)]
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.nodes = nodes[:]
        scheduler.node_to_stream = {}
        scheduler.name_to_fused_node = {}
        scheduler.get_backend = mock.Mock(return_value=BaseScheduling(None))
        scheduler.speedup_by_combo_kernel = mock.Mock(
            side_effect=AssertionError("unsupported backend should skip benchmarking")
        )
        scheduler.topological_sort_schedule = lambda nodes: list(nodes)
        scheduler.prune_redundant_deps = mock.Mock()

        with mock.patch.object(
            ForeachKernelSchedulerNode,
            "group_algorithm_for_combo_kernels",
            lambda scheduler: [nodes],
        ):
            Scheduler.create_combo_kernel_nodes(scheduler)

        self.assertEqual(scheduler.nodes, nodes)
        scheduler.get_backend.assert_called_once_with(device)
        scheduler.speedup_by_combo_kernel.assert_not_called()

    def test_triton_tile_kernel_base_classes_are_exposed(self):
        self.assertTrue(issubclass(TritonKernel, TileKernel))
        self.assertTrue(issubclass(TritonScheduling, TileKernelScheduling))
        self.assertIs(TritonScheduling.kernel_type, TritonKernel)
        self.assertEqual(TritonKernel.backend(), "triton")

    def test_triton_bool_load_cast_is_backend_hook(self):
        kernel = TritonKernel.__new__(TritonKernel)

        self.assertTrue(hasattr(TileKernel, "codegen_bool_load_cast"))
        self.assertEqual(kernel.codegen_bool_load_cast("tmp0"), "tmp0.to(tl.int1)")

    def test_triton_looped_reduction_syntax_is_backend_hook(self):
        kernel = TritonKernel.__new__(TritonKernel)
        kernel.index_to_str = lambda expr: "ADVANCE"

        self.assertTrue(hasattr(TileKernel, "codegen_looped_reduction_range"))
        self.assertTrue(hasattr(TileKernel, "codegen_block_ptr_advance"))
        self.assertEqual(
            kernel.codegen_looped_reduction_range("r0", "0", "r0numel"),
            "for r0offset in tl.range(0, r0numel, R0BLOCK):",
        )
        self.assertEqual(
            kernel.codegen_block_ptr_advance("ptr", [1]),
            "ptr = tl.advance(ptr, ADVANCE)",
        )

    def test_register_constexpr_syntax(self):
        self.assertEqual(
            common.ArgName("BLOCK", is_constexpr=True).full_name(),
            "BLOCK : tl.constexpr",
        )

        common.register_constexpr_syntax(
            "dummy_constexpr_backend", " : dummy.Constant"
        )
        common.register_constexpr_syntax(
            "dummy_constexpr_backend", " : dummy.Constant"
        )

        self.assertEqual(
            common.ArgName(
                "BLOCK", is_constexpr=True, backend="dummy_constexpr_backend"
            ).full_name(),
            "BLOCK : dummy.Constant",
        )
        self.assertEqual(
            common.ArgName("arg", backend="dummy_constexpr_backend").full_name(),
            "arg",
        )

    def test_register_constexpr_syntax_rejects_conflicting_duplicate(self):
        common.register_constexpr_syntax(
            "dummy_constexpr_backend", " : dummy.Constant"
        )

        with self.assertRaisesRegex(ValueError, "already registered"):
            common.register_constexpr_syntax(
                "dummy_constexpr_backend", " : other.Constant"
            )

    def test_register_constexpr_syntax_rejects_invalid_name(self):
        for name in ("", "not-valid", "class"):
            with self.assertRaisesRegex(ValueError, "valid Python identifier|non-empty"):
                common.register_constexpr_syntax(name, " : dummy.Constant")

    def test_unknown_constexpr_syntax_has_actionable_error(self):
        with self.assertRaisesRegex(KeyError, "Available constexpr syntax backends"):
            common.ArgName(
                "BLOCK", is_constexpr=True, backend="missing_backend"
            ).full_name()

    def test_register_dtype_propagation_backend(self):
        self.assertFalse(common._uses_dtype_propagation("dummy_dtype_backend"))

        common.register_dtype_propagation_backend("dummy_dtype_backend")
        common.register_dtype_propagation_backend("dummy_dtype_backend")

        self.assertTrue(common._uses_dtype_propagation("dummy_dtype_backend"))
        self.assertTrue(common._requires_output_dtype("dummy_dtype_backend"))

    def test_register_dtype_propagation_backend_rejects_conflicting_duplicate(self):
        common.register_dtype_propagation_backend(
            "dummy_dtype_backend", require_output_dtype=True
        )

        with self.assertRaisesRegex(ValueError, "already registered"):
            common.register_dtype_propagation_backend(
                "dummy_dtype_backend", require_output_dtype=False
            )

    def test_register_dtype_propagation_backend_rejects_invalid_name(self):
        for name in ("", "not-valid", "class"):
            with self.assertRaisesRegex(ValueError, "valid Python identifier|non-empty"):
                common.register_dtype_propagation_backend(name)

    def test_reduction_heuristic_uses_shared_prepare(self):
        configs = [Config({"XBLOCK": 1, "R0_BLOCK": 1})]
        inductor_meta = {"kernel_name": "dummy"}
        triton_meta = {"signature": {}}

        with (
            mock.patch.object(
                triton_heuristics,
                "_prepare_reduction",
                return_value=(configs, {"x": 1, "r0_": 1}, inductor_meta),
            ) as prepare,
            mock.patch.object(
                triton_heuristics, "cached_autotune", return_value="decorator"
            ) as cached_autotune,
        ):
            result = triton_heuristics.reduction(
                {"x": 1, "r0_": 1},
                reduction_hint=False,
                triton_meta=triton_meta,
                filename="dummy.py",
                inductor_meta=inductor_meta,
            )

        self.assertEqual(result, "decorator")
        prepare.assert_called_once_with(
            {"x": 1, "r0_": 1},
            reduction_hint=False,
            triton_meta=triton_meta,
            filename="dummy.py",
            inductor_meta=inductor_meta,
        )
        cached_autotune.assert_called_once_with(
            {"x": 1, "r0_": 1},
            configs=configs,
            triton_meta=triton_meta,
            inductor_meta=inductor_meta,
            heuristic_type=HeuristicType.REDUCTION,
            filename="dummy.py",
        )

    def test_register_kernel_metadata_provider(self):
        row = {
            "kernel_name": "dummy_kernel",
            "kernel_path": "dummy.py",
            "kernel_category": "pointwise",
            "size_hints": "{'x': 16}",
            "reduction_hint": None,
            "line_of_code": 1,
            "num_load": 1,
            "num_store": 1,
            "num_for_loop": 0,
            "num_atomic_add": 0,
            "num_args": 2,
            "xnumel": 16,
            "ynumel": None,
            "rnumel": None,
            "kernel_args_num_gb": None,
        }

        def provider(kernel_name, kernel_path, kernel_module_code, kernel_category):
            self.assertEqual(kernel_name, "dummy_kernel")
            self.assertEqual(kernel_path, "dummy.py")
            self.assertEqual(kernel_module_code, "dummy source")
            self.assertEqual(kernel_category, "unknown")
            return row

        metrics.register_kernel_metadata_provider("dummy_metrics_backend", provider)
        metrics.register_kernel_metadata_provider("dummy_metrics_backend", provider)

        with mock.patch.object(
            metrics.get_metric_table("kernel_metadata"), "add_row"
        ) as add_row:
            metrics.log_kernel_metadata("dummy_kernel", "dummy.py", "dummy source")

        add_row.assert_called_once()
        self.assertEqual(add_row.call_args.args[0](), row)

        with self.assertRaisesRegex(ValueError, "already registered"):
            metrics.register_kernel_metadata_provider(
                "dummy_metrics_backend", lambda *args: None
            )

    def test_register_kernel_benchmark_provider(self):
        mod = ModuleType("dummy_module")
        info = wrapper_benchmark.KernelBenchmarkInfo(
            kernel=object(),
            device_type="cuda",
            arg_names=["in_ptr0", "out_ptr0"],
            category="pointwise",
            num_gb=1.0,
        )

        def provider(candidate):
            return info if candidate is mod else None

        wrapper_benchmark.register_kernel_benchmark_provider(
            "dummy_benchmark_backend", provider
        )
        wrapper_benchmark.register_kernel_benchmark_provider(
            "dummy_benchmark_backend", provider
        )

        self.assertIs(wrapper_benchmark.get_kernel_benchmark_info(mod), info)

        with self.assertRaisesRegex(ValueError, "already registered"):
            wrapper_benchmark.register_kernel_benchmark_provider(
                "dummy_benchmark_backend", lambda candidate: None
            )


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    run_tests()
