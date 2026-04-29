# Owner(s): ["module: inductor"]

import contextlib
import os
import sys
import tempfile
from types import ModuleType, SimpleNamespace
from unittest import mock
from pathlib import Path

import sympy
import torch
from torch._inductor import (
    config,
    metrics,
    select_algorithm,
    wrapper_benchmark,
)
from torch._inductor.async_compile import (
    _async_compile_backends,
    AsyncCompile,
    register_async_compile_backend,
)
from torch._inductor.codegen import common
from torch._inductor.codegen.cuda_combined_scheduling import CUDACombinedScheduling
from torch._inductor.codegen.triton import (
    BlockDescriptorOptions,
    BlockParameters,
    TileKernel,
    TileIODescriptor,
    TileKernelScheduling,
    TritonSymbols,
    TritonKernel,
    TritonScheduling,
)
from torch._inductor.kernel import bmm as bmm_kernel
from torch._inductor.kernel import mm as mm_kernel
from torch._inductor.kernel.flex import flex_attention as flex_attention_kernel
from torch._inductor.runtime import triton_heuristics
from torch._inductor.runtime.hints import HeuristicType
from torch._inductor.runtime.triton_compat import Config
from torch._inductor.scheduler import (
    BaseScheduling,
    ForeachKernelSchedulerNode,
    Scheduler,
)
from torch._inductor.virtualized import V
from torch.testing._internal.common_utils import TestCase
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.symbol import SymT


class BackendExtensionAPITests(TestCase):
    class _FakeBuffer:
        def __init__(
            self,
            size,
            *,
            dtype=torch.float32,
            device=torch.device("cuda"),
            stride=None,
            offset=0,
        ):
            self._size = list(size)
            self._dtype = dtype
            self._device = device
            self._stride = None if stride is None else list(stride)
            self._layout = SimpleNamespace(offset=sympy.Integer(offset))

        def get_dtype(self):
            return self._dtype

        def get_device(self):
            return self._device

        def get_size(self):
            return self._size

        def get_stride(self):
            if self._stride is not None:
                return self._stride
            stride = []
            running = 1
            for size in reversed(self._size):
                stride.append(running)
                running *= size
            return list(reversed(stride))

        def get_layout(self):
            return self._layout

        def maybe_get_layout(self):
            return self._layout

    class _DummyChoice:
        def __init__(self, name, description=""):
            self.name = name
            self.description = description
            self.annotations = {}
            self.failed = False

        def output_node(self):
            return f"{self.name}_node"

    def tearDown(self):
        common._cuda_backends.pop("dummy_cuda_backend", None)
        common._constexpr_syntaxes.pop("dummy_constexpr_backend", None)
        common._dtype_propagation_backends.pop("dummy_dtype_backend", None)
        common._backend_wrapper_imports.pop("dummy_wrapper_backend", None)
        metrics._kernel_metadata_providers.pop("dummy_metrics_backend", None)
        wrapper_benchmark._kernel_benchmark_providers.pop(
            "dummy_benchmark_backend", None
        )
        mm_kernel._gemm_template_providers.pop("dummy_gemm_backend", None)
        select_algorithm._flex_attention_template_providers.pop(
            "dummy_flex_backend", None
        )
        _async_compile_backends.pop("dummy_async_backend", None)
        if hasattr(AsyncCompile, "dummy_async_backend"):
            delattr(AsyncCompile, "dummy_async_backend")
        common.init_backend_registration.cache_clear()
        super().tearDown()

    def _run_tuned_mm_provider_harness(
        self,
        *,
        is_nonzero=True,
        out_dtype=None,
        input_dtype=torch.float32,
        active_backend="dummy_gemm_backend",
        max_autotune=False,
        max_autotune_gemm=True,
        max_autotune_gemm_backends="DUMMY_GEMM_BACKEND",
        enable_autoheuristic=False,
        template_config_returns=(),
        selector=None,
    ):
        class FakeChoices:
            def __init__(self, returns):
                self.returns = list(returns)

            def get_template_configs(self, *args, **kwargs):
                if self.returns:
                    return self.returns.pop(0)
                return []

        mat1 = self._FakeBuffer([16, 32], dtype=input_dtype)
        mat2 = self._FakeBuffer([32, 8], dtype=input_dtype)
        layout = SimpleNamespace(
            device=torch.device("cuda"),
            dtype=out_dtype or input_dtype,
        )

        if selector is None:

            def selector(name, choices, input_nodes, layout, **kwargs):
                return "selected_node", None

        with contextlib.ExitStack() as stack:
            stack.enter_context(
                config.patch(
                    cuda_backend=active_backend,
                    max_autotune=max_autotune,
                    max_autotune_gemm=max_autotune_gemm,
                    max_autotune_gemm_backends=max_autotune_gemm_backends,
                    remote_gemm_autotune_cache=False,
                )
            )
            stack.enter_context(
                mock.patch.dict(
                    os.environ,
                    {
                        "TORCHINDUCTOR_AUTOHEURISTIC_USE": "mm"
                        if enable_autoheuristic
                        else "",
                        "TORCHINDUCTOR_AUTOHEURISTIC_COLLECT": "",
                    },
                )
            )
            stack.enter_context(
                mm_kernel.V.set_choices_handler(FakeChoices(template_config_returns))
            )
            stack.enter_context(
                mock.patch.object(mm_kernel, "use_native_matmul", return_value=False)
            )
            stack.enter_context(
                mock.patch.object(
                    mm_kernel,
                    "mm_args",
                    return_value=(16, 8, 32, layout, mat1, mat2),
                )
            )
            stack.enter_context(
                mock.patch.object(
                    mm_kernel,
                    "_is_static_problem",
                    return_value=(True, is_nonzero),
                )
            )
            stack.enter_context(
                mock.patch.object(mm_kernel, "use_aten_gemm_kernels", return_value=False)
            )
            stack.enter_context(
                mock.patch.object(
                    mm_kernel,
                    "use_triton_template",
                    return_value=enable_autoheuristic,
                )
            )
            stack.enter_context(
                mock.patch.object(mm_kernel, "use_decompose_k_choice", return_value=False)
            )
            stack.enter_context(
                mock.patch.object(
                    mm_kernel,
                    "use_triton_blackwell_tma_template",
                    return_value=False,
                )
            )
            stack.enter_context(
                mock.patch.object(mm_kernel, "use_triton_tma_template", return_value=False)
            )
            stack.enter_context(
                mock.patch.object(mm_kernel, "use_cutlass_template", return_value=False)
            )
            stack.enter_context(
                mock.patch.object(mm_kernel, "use_ck_gemm_template", return_value=False)
            )
            stack.enter_context(
                mock.patch.object(
                    mm_kernel, "use_ck_tile_gemm_template", return_value=False
                )
            )
            stack.enter_context(
                mock.patch.object(
                    mm_kernel, "use_nv_universal_gemm_template", return_value=False
                )
            )
            stack.enter_context(
                mock.patch.object(mm_kernel, "use_cpp_gemm_template", return_value=False)
            )
            stack.enter_context(
                mock.patch.object(
                    mm_kernel.distributed_autotune,
                    "maybe_autotune_remote",
                    return_value=None,
                )
            )
            stack.enter_context(mock.patch.object(mm_kernel, "is_triton", return_value=True))
            stack.enter_context(mock.patch.object(mm_kernel, "mm_autoheuristic", return_value=[]))
            stack.enter_context(
                mock.patch.object(mm_kernel, "autotune_select_algorithm", selector)
            )

            return mm_kernel.tuned_mm.__wrapped__(
                mat1, mat2, out_dtype=out_dtype
            )

    def _run_tuned_addmm_provider_harness(
        self,
        *,
        is_nonzero=True,
        alpha=1,
        beta=1,
        active_backend="dummy_gemm_backend",
        max_autotune=False,
        max_autotune_gemm=True,
        max_autotune_gemm_backends="DUMMY_GEMM_BACKEND",
        selector=None,
    ):
        class FakeChoices:
            def get_template_configs(self, *args, **kwargs):
                return []

        inp = self._FakeBuffer([16, 8])
        mat1 = self._FakeBuffer([16, 32])
        mat2 = self._FakeBuffer([32, 8])
        inp_expanded = self._FakeBuffer([16, 8])
        layout = SimpleNamespace(device=torch.device("cuda"), dtype=torch.float32)

        if selector is None:

            def selector(name, choices, input_nodes, layout, **kwargs):
                return "selected_node", None

        with contextlib.ExitStack() as stack:
            stack.enter_context(
                config.patch(
                    cuda_backend=active_backend,
                    max_autotune=max_autotune,
                    max_autotune_gemm=max_autotune_gemm,
                    max_autotune_gemm_backends=max_autotune_gemm_backends,
                )
            )
            stack.enter_context(
                mm_kernel.V.set_choices_handler(FakeChoices())
            )
            stack.enter_context(
                mock.patch.object(mm_kernel, "use_native_matmul", return_value=False)
            )
            stack.enter_context(
                mock.patch.object(
                    mm_kernel,
                    "mm_args",
                    return_value=(16, 8, 32, layout, mat1, mat2, inp_expanded),
                )
            )
            stack.enter_context(
                mock.patch.object(
                    mm_kernel,
                    "_is_static_problem",
                    return_value=(True, is_nonzero),
                )
            )
            stack.enter_context(
                mock.patch.object(mm_kernel, "use_aten_gemm_kernels", return_value=False)
            )
            stack.enter_context(
                mock.patch.object(mm_kernel, "use_triton_template", return_value=False)
            )
            stack.enter_context(
                mock.patch.object(mm_kernel, "use_cutlass_template", return_value=False)
            )
            stack.enter_context(
                mock.patch.object(mm_kernel, "use_ck_gemm_template", return_value=False)
            )
            stack.enter_context(
                mock.patch.object(mm_kernel, "use_cpp_gemm_template", return_value=False)
            )
            stack.enter_context(
                mock.patch.object(mm_kernel, "autotune_select_algorithm", selector)
            )

            return mm_kernel.tuned_addmm.__wrapped__(
                inp, mat1, mat2, alpha=alpha, beta=beta
            )

    def _run_tuned_bmm_provider_harness(
        self,
        *,
        is_nonzero=True,
        out_dtype=None,
        input_dtype=torch.float32,
        active_backend="dummy_gemm_backend",
        max_autotune=False,
        max_autotune_gemm=True,
        max_autotune_gemm_backends="DUMMY_GEMM_BACKEND",
        selector=None,
    ):
        class FakeChoices:
            def get_template_configs(self, *args, **kwargs):
                return []

        mat1 = self._FakeBuffer([4, 16, 32], dtype=input_dtype)
        mat2 = self._FakeBuffer([4, 32, 8], dtype=input_dtype)
        layout = SimpleNamespace(
            device=torch.device("cuda"),
            dtype=out_dtype or input_dtype,
        )

        if selector is None:

            def selector(name, choices, input_nodes, layout, **kwargs):
                return "selected_node", None

        with contextlib.ExitStack() as stack:
            stack.enter_context(
                config.patch(
                    cuda_backend=active_backend,
                    max_autotune=max_autotune,
                    max_autotune_gemm=max_autotune_gemm,
                    max_autotune_gemm_backends=max_autotune_gemm_backends,
                )
            )
            stack.enter_context(bmm_kernel.V.set_choices_handler(FakeChoices()))
            stack.enter_context(
                mock.patch.object(bmm_kernel, "use_native_matmul", return_value=False)
            )
            stack.enter_context(
                mock.patch.object(
                    bmm_kernel,
                    "mm_args",
                    return_value=(16, 8, 32, layout, mat1, mat2),
                )
            )
            stack.enter_context(
                mock.patch.object(
                    bmm_kernel,
                    "_is_static_problem",
                    return_value=(True, is_nonzero),
                )
            )
            stack.enter_context(
                mock.patch.object(bmm_kernel, "use_aten_gemm_kernels", return_value=False)
            )
            stack.enter_context(
                mock.patch.object(bmm_kernel, "use_triton_template", return_value=False)
            )
            stack.enter_context(
                mock.patch.object(
                    bmm_kernel,
                    "is_batch_stride_largest_or_zero",
                    return_value=False,
                )
            )
            stack.enter_context(
                mock.patch.object(bmm_kernel, "use_cutlass_template", return_value=False)
            )
            stack.enter_context(
                mock.patch.object(bmm_kernel, "use_cpp_bmm_template", return_value=False)
            )
            stack.enter_context(
                mock.patch.object(bmm_kernel, "use_ck_gemm_template", return_value=False)
            )
            stack.enter_context(
                mock.patch.object(
                    bmm_kernel, "use_nv_universal_gemm_template", return_value=False
                )
            )
            stack.enter_context(
                mock.patch.object(bmm_kernel, "autotune_select_algorithm", selector)
            )

            return bmm_kernel.tuned_bmm.__wrapped__(
                mat1, mat2, out_dtype=out_dtype
            )

    def _run_tuned_baddbmm_provider_harness(
        self,
        *,
        is_nonzero=True,
        alpha=1,
        beta=1,
        active_backend="dummy_gemm_backend",
        max_autotune=False,
        max_autotune_gemm=True,
        max_autotune_gemm_backends="DUMMY_GEMM_BACKEND",
        selector=None,
    ):
        class FakeChoices:
            def get_template_configs(self, *args, **kwargs):
                return []

        inp = self._FakeBuffer([4, 16, 8])
        mat1 = self._FakeBuffer([4, 16, 32])
        mat2 = self._FakeBuffer([4, 32, 8])
        inp_expanded = self._FakeBuffer([4, 16, 8])
        layout = SimpleNamespace(device=torch.device("cuda"), dtype=torch.float32)

        if selector is None:

            def selector(name, choices, input_nodes, layout, **kwargs):
                return "selected_node", None

        with contextlib.ExitStack() as stack:
            stack.enter_context(
                config.patch(
                    cuda_backend=active_backend,
                    max_autotune=max_autotune,
                    max_autotune_gemm=max_autotune_gemm,
                    max_autotune_gemm_backends=max_autotune_gemm_backends,
                )
            )
            stack.enter_context(bmm_kernel.V.set_choices_handler(FakeChoices()))
            stack.enter_context(
                mock.patch.object(bmm_kernel, "use_native_matmul", return_value=False)
            )
            stack.enter_context(
                mock.patch.object(
                    bmm_kernel,
                    "mm_args",
                    return_value=(16, 8, 32, layout, mat1, mat2, inp_expanded),
                )
            )
            stack.enter_context(
                mock.patch.object(
                    bmm_kernel,
                    "_is_static_problem",
                    return_value=(True, is_nonzero),
                )
            )
            stack.enter_context(
                mock.patch.object(bmm_kernel, "use_aten_gemm_kernels", return_value=False)
            )
            stack.enter_context(
                mock.patch.object(bmm_kernel, "use_triton_template", return_value=False)
            )
            stack.enter_context(
                mock.patch.object(bmm_kernel, "autotune_select_algorithm", selector)
            )

            return bmm_kernel.tuned_baddbmm.__wrapped__(
                inp, mat1, mat2, alpha=alpha, beta=beta
            )

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

        def dummy_gemm_provider(context):
            return ()

        def reloaded_dummy_gemm_provider(context):
            return ()

        reloaded_dummy_gemm_provider.__module__ = dummy_gemm_provider.__module__
        reloaded_dummy_gemm_provider.__qualname__ = dummy_gemm_provider.__qualname__

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

        mm_kernel.register_gemm_template_provider(
            "dummy_gemm_backend", dummy_gemm_provider
        )
        mm_kernel.register_gemm_template_provider(
            "dummy_gemm_backend", reloaded_dummy_gemm_provider
        )
        self.assertIs(
            mm_kernel._gemm_template_providers["dummy_gemm_backend"],
            reloaded_dummy_gemm_provider,
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

    def test_register_gemm_template_provider(self):
        seen_context = None

        def dummy_gemm_provider(context):
            nonlocal seen_context
            seen_context = context
            return ("choice0",)

        mm_kernel.register_gemm_template_provider(
            "dummy_gemm_backend", dummy_gemm_provider
        )
        mm_kernel.register_gemm_template_provider(
            "dummy_gemm_backend", dummy_gemm_provider
        )

        self.assertIs(
            mm_kernel.get_gemm_template_provider("dummy_gemm_backend"),
            dummy_gemm_provider,
        )
        context = mm_kernel.GemmTemplateProviderContext(
            op_name="mm",
            kernel_inputs="kernel_inputs",
            layout="layout",
            mat1="mat1",
            mat2="mat2",
            m=16,
            n=32,
            k=64,
            out_dtype=None,
            static_shape=False,
            is_nonzero=True,
        )
        with config.patch(
            max_autotune_gemm=True,
            max_autotune_gemm_backends="DUMMY_GEMM_BACKEND",
        ):
            self.assertEqual(
                mm_kernel.get_backend_gemm_template_choices(
                    "dummy_gemm_backend",
                    context=context,
                ),
                ["choice0"],
            )
        self.assertIs(seen_context, context)
        self.assertEqual(seen_context.op_name, "mm")
        self.assertEqual(seen_context.kernel_inputs, "kernel_inputs")
        self.assertEqual(seen_context.m, 16)
        self.assertEqual(seen_context.out_dtype, None)
        self.assertFalse(seen_context.static_shape)
        self.assertIsNone(seen_context.batch_size)
        with config.patch(
            max_autotune_gemm=True,
            max_autotune_gemm_backends="TRITON",
        ):
            self.assertEqual(
                mm_kernel.get_backend_gemm_template_choices(
                    "dummy_gemm_backend",
                    context=context,
                ),
                [],
            )
        with config.patch(max_autotune=False, max_autotune_gemm=False):
            self.assertEqual(
                mm_kernel.get_backend_gemm_template_choices(
                    "dummy_gemm_backend",
                    context=context,
                ),
                [],
            )
        self.assertEqual(
            mm_kernel.get_backend_gemm_template_choices(
                "missing_gemm_backend",
                context=context,
            ),
            [],
        )

    def test_register_flex_attention_template_provider(self):
        seen_context = None

        def dummy_flex_provider(context):
            nonlocal seen_context
            seen_context = context
            return (self._DummyChoice("dummy_flex_choice"),)

        select_algorithm.register_flex_attention_template_provider(
            "dummy_flex_backend", dummy_flex_provider
        )
        select_algorithm.register_flex_attention_template_provider(
            "dummy_flex_backend", dummy_flex_provider
        )

        self.assertIs(
            select_algorithm.get_flex_attention_template_provider(
                "dummy_flex_backend"
            ),
            dummy_flex_provider,
        )
        context = select_algorithm.FlexAttentionTemplateProviderContext(
            op_name="flex_attention",
            input_nodes=["query", "key", "value"],
            layout="layout",
            subgraphs=["score_mod", "mask_mod"],
            mutated_inputs=["logsumexp"],
            call_sizes=[2, 4, 16, 32],
            kernel_options={"BLOCK_M": 16},
        )
        choices = select_algorithm.get_backend_flex_attention_template_choices(
            "dummy_flex_backend",
            context=context,
        )

        self.assertIs(seen_context, context)
        self.assertEqual(len(choices), 1)
        self.assertEqual(
            choices[0].annotations[
                select_algorithm.FLEX_ATTENTION_TEMPLATE_PROVIDER_ANNOTATION
            ],
            "dummy_flex_backend",
        )
        self.assertEqual(
            select_algorithm.get_backend_flex_attention_template_choices(
                "missing_flex_backend",
                context=context,
            ),
            [],
        )

    def test_register_flex_attention_template_provider_rejects_duplicate(self):
        def dummy_flex_provider(context):
            return ()

        def other_dummy_flex_provider(context):
            return ()

        select_algorithm.register_flex_attention_template_provider(
            "dummy_flex_backend", dummy_flex_provider
        )

        with self.assertRaisesRegex(ValueError, "already registered"):
            select_algorithm.register_flex_attention_template_provider(
                "dummy_flex_backend", other_dummy_flex_provider
            )

    def test_builtin_triton_flex_template_gated_by_backend(self):
        self.assertTrue(
            flex_attention_kernel._use_builtin_triton_flex_template("triton", "AUTO")
        )
        self.assertTrue(
            flex_attention_kernel._use_builtin_triton_flex_template(
                "triton", "TRITON"
            )
        )
        self.assertTrue(
            flex_attention_kernel._use_builtin_triton_flex_template(
                "triton", "TRITON_DECODE"
            )
        )
        self.assertFalse(
            flex_attention_kernel._use_builtin_triton_flex_template(
                "dummy_flex_backend", "TRITON"
            )
        )
        self.assertFalse(
            flex_attention_kernel._use_builtin_triton_flex_template(
                "dummy_flex_backend", "TRITON_DECODE"
            )
        )
        self.assertFalse(
            flex_attention_kernel._use_builtin_triton_flex_template(
                "dummy_flex_backend", "AUTO"
            )
        )
        with self.assertRaisesRegex(
            NotImplementedError,
            "BACKEND='TRITON' requires config.cuda_backend='triton'",
        ):
            flex_attention_kernel._validate_explicit_triton_flex_backend(
                "dummy_flex_backend", "TRITON"
            )
        flex_attention_kernel._validate_explicit_triton_flex_backend(
            "triton", "TRITON"
        )

    def test_gemm_provider_backend_neutral_template_caller_reaches_multi_template_buffer(
        self,
    ):
        class DummyBenchmarkRequest:
            module_path = "/tmp/dummy_template.py"
            module_cache_key = "dummy_cache_key"
            num_stages = 1
            num_warps = 2
            n_regs = None

            def benchmark(self, *args, **kwargs):
                raise AssertionError("not used")

        class DummyTemplateCaller(select_algorithm.TemplateCaller):
            backend = "Dummy"

        class MinimalBenchmarkRequest:
            def benchmark(self, *args, **kwargs):
                return 3.14

        class FakeGraph:
            def __init__(self):
                self.next_buffer_index = 0
                self.operations = []

            def register_buffer(self, buffer):
                name = f"buf{self.next_buffer_index}"
                self.next_buffer_index += 1
                return name

            def register_operation(self, operation):
                self.operations.append(operation)

        layout = select_algorithm.ir.FixedLayout(
            torch.device("cuda"), torch.float32, [16, 8]
        )

        def make_kernel_render():
            return "dummy-render"

        provider_choice = DummyTemplateCaller(
            "dummy_template_0",
            (),
            layout,
            make_kernel_render,
            "provider",
            DummyBenchmarkRequest(),
            log_info={"tile_shape": "(16, 32, 8)"},
            allowed_prologue_inps=OrderedSet(["arg0"]),
        )

        def dummy_gemm_provider(context):
            return [provider_choice]

        mm_kernel.register_gemm_template_provider(
            "dummy_gemm_backend", dummy_gemm_provider
        )
        context = mm_kernel.GemmTemplateProviderContext(
            op_name="mm",
            kernel_inputs="kernel_inputs",
            layout=layout,
            mat1=self._FakeBuffer([16, 32]),
            mat2=self._FakeBuffer([32, 8]),
            m=16,
            n=8,
            k=32,
            out_dtype=None,
            static_shape=True,
            is_nonzero=True,
        )

        with config.patch(
            max_autotune_gemm=True,
            max_autotune_gemm_backends="DUMMY_GEMM_BACKEND",
        ):
            choices = mm_kernel.get_backend_gemm_template_choices(
                "dummy_gemm_backend",
                context=context,
            )

        self.assertEqual(choices, [provider_choice])
        self.assertEqual(
            provider_choice.annotations[
                select_algorithm.GEMM_TEMPLATE_PROVIDER_ANNOTATION
            ],
            "dummy_gemm_backend",
        )
        self.assertIsInstance(
            provider_choice, select_algorithm.ir.TritonTemplateCallerBase
        )
        self.assertNotIsInstance(
            provider_choice, select_algorithm.TritonTemplateCaller
        )

        with select_algorithm.ir.V.set_graph_handler(FakeGraph()):
            multi_template_buffer = select_algorithm.ir.MultiTemplateBuffer(
                layout=layout,
                inputs=(),
                choice_timings_fn=lambda hint_override: {provider_choice: 1.0},
                unfiltered_choices=choices,
                allowed_prologue_inps=OrderedSet(["arg0"]),
            )

            self.assertTrue(multi_template_buffer.output_plannable)
            multi_template_buffer.finalize_as_template_caller(provider_choice)
            self.assertIs(
                multi_template_buffer.make_kernel_render, make_kernel_render
            )
            multi_template_buffer.finalize_as_triton_caller(provider_choice)
            self.assertIs(
                multi_template_buffer.make_kernel_render, make_kernel_render
            )

            multi_template_buffer.make_kernel_render = None
            with multi_template_buffer.swap_as_template_caller(provider_choice):
                self.assertIs(
                    multi_template_buffer.make_kernel_render, make_kernel_render
                )
            self.assertIsNone(multi_template_buffer.make_kernel_render)

            with multi_template_buffer.swap_as_triton_caller(provider_choice):
                self.assertIs(
                    multi_template_buffer.make_kernel_render, make_kernel_render
                )
            self.assertIsNone(multi_template_buffer.make_kernel_render)

        minimal_choice = DummyTemplateCaller(
            "dummy_minimal_0",
            (),
            layout,
            make_kernel_render,
            "minimal",
            MinimalBenchmarkRequest(),
        )
        self.assertIn("MinimalBenchmarkRequest", str(minimal_choice))
        self.assertIn("dummy_minimal", minimal_choice.hash_key())
        with (
            config.patch(profile_bandwidth_with_do_bench_using_profiling=True),
            mock.patch.object(
                select_algorithm,
                "do_bench_using_profiling",
                side_effect=lambda fn: fn(),
            ),
        ):
            self.assertEqual(minimal_choice.benchmark(out=object()), 3.14)

    def test_tuned_mm_gemm_provider_choice_reaches_selection(self):
        provider_choice = self._DummyChoice("dummy_provider", "provider")
        ah_choice = self._DummyChoice("autoheuristic", "autoheuristic")
        provider_contexts = []
        selector_choices = []

        def dummy_gemm_provider(context):
            provider_contexts.append(context)
            return [provider_choice]

        def selector(name, choices, input_nodes, layout, **kwargs):
            selector_choices.extend(choices)
            return choices[-1].output_node(), choices[-1]

        mm_kernel.register_gemm_template_provider(
            "dummy_gemm_backend", dummy_gemm_provider
        )

        result = self._run_tuned_mm_provider_harness(
            enable_autoheuristic=True,
            template_config_returns=([], [ah_choice]),
            selector=selector,
        )

        self.assertEqual(result, "dummy_provider_node")
        self.assertEqual(selector_choices, [provider_choice])
        self.assertEqual(len(provider_contexts), 1)
        context = provider_contexts[0]
        self.assertEqual(context.op_name, "mm")
        self.assertEqual(context.m, 16)
        self.assertEqual(context.n, 8)
        self.assertEqual(context.k, 32)
        self.assertIsNone(context.out_dtype)
        self.assertTrue(context.static_shape)
        self.assertTrue(context.is_nonzero)
        self.assertEqual(
            provider_choice.annotations[
                select_algorithm.GEMM_TEMPLATE_PROVIDER_ANNOTATION
            ],
            "dummy_gemm_backend",
        )

    def test_tuned_addmm_gemm_provider_choice_reaches_selection(self):
        provider_choice = self._DummyChoice("dummy_provider", "provider")
        provider_contexts = []
        selector_choices = []
        selector_input_nodes = []

        def dummy_gemm_provider(context):
            provider_contexts.append(context)
            return [provider_choice]

        def selector(name, choices, input_nodes, layout, **kwargs):
            selector_choices.extend(choices)
            selector_input_nodes.extend(input_nodes)
            return choices[-1].output_node(), choices[-1]

        mm_kernel.register_gemm_template_provider(
            "dummy_gemm_backend", dummy_gemm_provider
        )

        result = self._run_tuned_addmm_provider_harness(
            alpha=1,
            beta=1,
            selector=selector,
        )

        self.assertEqual(result, "dummy_provider_node")
        self.assertEqual(selector_choices, [provider_choice])
        self.assertEqual(len(provider_contexts), 1)
        context = provider_contexts[0]
        self.assertEqual(context.op_name, "addmm")
        self.assertIs(context.inp, selector_input_nodes[0])
        self.assertIs(context.mat1, selector_input_nodes[1])
        self.assertIs(context.mat2, selector_input_nodes[2])
        self.assertEqual(context.alpha, 1)
        self.assertEqual(context.beta, 1)
        self.assertEqual(context.m, 16)
        self.assertEqual(context.n, 8)
        self.assertEqual(context.k, 32)
        self.assertTrue(context.static_shape)
        self.assertTrue(context.is_nonzero)
        self.assertEqual(
            provider_choice.annotations[
                select_algorithm.GEMM_TEMPLATE_PROVIDER_ANNOTATION
            ],
            "dummy_gemm_backend",
        )

    def test_tuned_bmm_gemm_provider_choice_reaches_selection(self):
        provider_choice = self._DummyChoice("dummy_provider", "provider")
        provider_contexts = []
        selector_choices = []
        selector_input_nodes = []

        def dummy_gemm_provider(context):
            provider_contexts.append(context)
            return [provider_choice]

        def selector(name, choices, input_nodes, layout, **kwargs):
            selector_choices.extend(choices)
            selector_input_nodes.extend(input_nodes)
            return choices[-1].output_node(), choices[-1]

        mm_kernel.register_gemm_template_provider(
            "dummy_gemm_backend", dummy_gemm_provider
        )

        result = self._run_tuned_bmm_provider_harness(selector=selector)

        self.assertEqual(result, "dummy_provider_node")
        self.assertEqual(selector_choices, [provider_choice])
        self.assertEqual(len(provider_contexts), 1)
        context = provider_contexts[0]
        self.assertEqual(context.op_name, "bmm")
        self.assertIsNone(context.inp)
        self.assertIs(context.mat1, selector_input_nodes[0])
        self.assertIs(context.mat2, selector_input_nodes[1])
        self.assertEqual(context.batch_size, 4)
        self.assertEqual(context.m, 16)
        self.assertEqual(context.n, 8)
        self.assertEqual(context.k, 32)
        self.assertIsNone(context.out_dtype)
        self.assertTrue(context.static_shape)
        self.assertTrue(context.is_nonzero)
        self.assertEqual(
            provider_choice.annotations[
                select_algorithm.GEMM_TEMPLATE_PROVIDER_ANNOTATION
            ],
            "dummy_gemm_backend",
        )

    def test_tuned_baddbmm_gemm_provider_choice_reaches_selection(self):
        provider_choice = self._DummyChoice("dummy_provider", "provider")
        provider_contexts = []
        selector_choices = []
        selector_input_nodes = []

        def dummy_gemm_provider(context):
            provider_contexts.append(context)
            return [provider_choice]

        def selector(name, choices, input_nodes, layout, **kwargs):
            selector_choices.extend(choices)
            selector_input_nodes.extend(input_nodes)
            return choices[-1].output_node(), choices[-1]

        mm_kernel.register_gemm_template_provider(
            "dummy_gemm_backend", dummy_gemm_provider
        )

        result = self._run_tuned_baddbmm_provider_harness(
            alpha=2,
            beta=3,
            selector=selector,
        )

        self.assertEqual(result, "dummy_provider_node")
        self.assertEqual(selector_choices, [provider_choice])
        self.assertEqual(len(provider_contexts), 1)
        context = provider_contexts[0]
        self.assertEqual(context.op_name, "baddbmm")
        self.assertIs(context.inp, selector_input_nodes[0])
        self.assertIs(context.mat1, selector_input_nodes[1])
        self.assertIs(context.mat2, selector_input_nodes[2])
        self.assertEqual(context.alpha, 2)
        self.assertEqual(context.beta, 3)
        self.assertEqual(context.batch_size, 4)
        self.assertEqual(context.m, 16)
        self.assertEqual(context.n, 8)
        self.assertEqual(context.k, 32)
        self.assertIsNone(context.out_dtype)
        self.assertTrue(context.static_shape)
        self.assertTrue(context.is_nonzero)
        self.assertEqual(
            provider_choice.annotations[
                select_algorithm.GEMM_TEMPLATE_PROVIDER_ANNOTATION
            ],
            "dummy_gemm_backend",
        )

    def test_tuned_addmm_gemm_provider_negative_gates(self):
        def run_case(
            *,
            register_provider=True,
            provider_result=(),
            provider_expected_calls=0,
            expected_choices=(),
            **harness_kwargs,
        ):
            provider_calls = []
            selector_choices = []

            def dummy_gemm_provider(context):
                provider_calls.append(context)
                return provider_result

            def selector(name, choices, input_nodes, layout, **kwargs):
                selector_choices.extend(choices)
                return "selected_node", None

            mm_kernel._gemm_template_providers.pop("dummy_gemm_backend", None)
            if register_provider:
                mm_kernel.register_gemm_template_provider(
                    "dummy_gemm_backend", dummy_gemm_provider
                )
            try:
                self._run_tuned_addmm_provider_harness(
                    selector=selector,
                    **harness_kwargs,
                )
            finally:
                mm_kernel._gemm_template_providers.pop("dummy_gemm_backend", None)

            self.assertEqual(len(provider_calls), provider_expected_calls)
            self.assertEqual(selector_choices, list(expected_choices))

        provider_choice = self._DummyChoice("dummy_provider")
        cases = {
            "default_config": {
                "max_autotune": False,
                "max_autotune_gemm": False,
            },
            "wrong_active_backend": {
                "active_backend": "other_gemm_backend",
            },
            "wrong_max_autotune_backend": {
                "max_autotune_gemm_backends": "TRITON",
            },
            "zero_size": {
                "is_nonzero": False,
            },
            "no_registered_provider": {
                "register_provider": False,
            },
            "provider_returns_no_choices": {
                "provider_result": (),
                "provider_expected_calls": 1,
            },
            "provider_returns_choice": {
                "provider_result": [provider_choice],
                "provider_expected_calls": 1,
                "expected_choices": [provider_choice],
            },
        }
        for name, kwargs in cases.items():
            with self.subTest(name=name):
                run_case(**kwargs)

    def test_tuned_bmm_gemm_provider_negative_gates(self):
        def run_case(
            *,
            register_provider=True,
            provider_result=(),
            provider_expected_calls=0,
            expected_choices=(),
            **harness_kwargs,
        ):
            provider_calls = []
            selector_choices = []

            def dummy_gemm_provider(context):
                provider_calls.append(context)
                return provider_result

            def selector(name, choices, input_nodes, layout, **kwargs):
                selector_choices.extend(choices)
                return "selected_node", None

            mm_kernel._gemm_template_providers.pop("dummy_gemm_backend", None)
            if register_provider:
                mm_kernel.register_gemm_template_provider(
                    "dummy_gemm_backend", dummy_gemm_provider
                )
            try:
                self._run_tuned_bmm_provider_harness(
                    selector=selector,
                    **harness_kwargs,
                )
            finally:
                mm_kernel._gemm_template_providers.pop("dummy_gemm_backend", None)

            self.assertEqual(len(provider_calls), provider_expected_calls)
            self.assertEqual(selector_choices, list(expected_choices))

        provider_choice = self._DummyChoice("dummy_provider")
        cases = {
            "default_config": {
                "max_autotune": False,
                "max_autotune_gemm": False,
            },
            "wrong_active_backend": {
                "active_backend": "other_gemm_backend",
            },
            "wrong_max_autotune_backend": {
                "max_autotune_gemm_backends": "TRITON",
            },
            "zero_size": {
                "is_nonzero": False,
            },
            "out_dtype": {
                "input_dtype": torch.float16,
                "out_dtype": torch.float32,
            },
            "no_registered_provider": {
                "register_provider": False,
            },
            "provider_returns_no_choices": {
                "provider_result": (),
                "provider_expected_calls": 1,
            },
            "provider_returns_choice": {
                "provider_result": [provider_choice],
                "provider_expected_calls": 1,
                "expected_choices": [provider_choice],
            },
        }
        for name, kwargs in cases.items():
            with self.subTest(name=name):
                run_case(**kwargs)

    def test_tuned_baddbmm_gemm_provider_negative_gates(self):
        def run_case(
            *,
            register_provider=True,
            provider_result=(),
            provider_expected_calls=0,
            expected_choices=(),
            **harness_kwargs,
        ):
            provider_calls = []
            selector_choices = []

            def dummy_gemm_provider(context):
                provider_calls.append(context)
                return provider_result

            def selector(name, choices, input_nodes, layout, **kwargs):
                selector_choices.extend(choices)
                return "selected_node", None

            mm_kernel._gemm_template_providers.pop("dummy_gemm_backend", None)
            if register_provider:
                mm_kernel.register_gemm_template_provider(
                    "dummy_gemm_backend", dummy_gemm_provider
                )
            try:
                self._run_tuned_baddbmm_provider_harness(
                    selector=selector,
                    **harness_kwargs,
                )
            finally:
                mm_kernel._gemm_template_providers.pop("dummy_gemm_backend", None)

            self.assertEqual(len(provider_calls), provider_expected_calls)
            self.assertEqual(selector_choices, list(expected_choices))

        provider_choice = self._DummyChoice("dummy_provider")
        cases = {
            "default_config": {
                "max_autotune": False,
                "max_autotune_gemm": False,
            },
            "wrong_active_backend": {
                "active_backend": "other_gemm_backend",
            },
            "wrong_max_autotune_backend": {
                "max_autotune_gemm_backends": "TRITON",
            },
            "zero_size": {
                "is_nonzero": False,
            },
            "no_registered_provider": {
                "register_provider": False,
            },
            "provider_returns_no_choices": {
                "provider_result": (),
                "provider_expected_calls": 1,
            },
            "provider_returns_choice": {
                "provider_result": [provider_choice],
                "provider_expected_calls": 1,
                "expected_choices": [provider_choice],
            },
        }
        for name, kwargs in cases.items():
            with self.subTest(name=name):
                run_case(**kwargs)

    def test_tuned_mm_gemm_provider_negative_gates(self):
        def run_case(
            *,
            register_provider=True,
            provider_result=(),
            provider_expected_calls=0,
            expected_choices=(),
            **harness_kwargs,
        ):
            provider_calls = []
            selector_choices = []

            def dummy_gemm_provider(context):
                provider_calls.append(context)
                return provider_result

            def selector(name, choices, input_nodes, layout, **kwargs):
                selector_choices.extend(choices)
                return "selected_node", None

            mm_kernel._gemm_template_providers.pop("dummy_gemm_backend", None)
            if register_provider:
                mm_kernel.register_gemm_template_provider(
                    "dummy_gemm_backend", dummy_gemm_provider
                )
            try:
                self._run_tuned_mm_provider_harness(
                    selector=selector,
                    **harness_kwargs,
                )
            finally:
                mm_kernel._gemm_template_providers.pop("dummy_gemm_backend", None)

            self.assertEqual(len(provider_calls), provider_expected_calls)
            self.assertEqual(selector_choices, list(expected_choices))

        provider_choice = self._DummyChoice("dummy_provider")
        cases = {
            "default_config": {
                "max_autotune": False,
                "max_autotune_gemm": False,
            },
            "wrong_active_backend": {
                "active_backend": "other_gemm_backend",
            },
            "wrong_max_autotune_backend": {
                "max_autotune_gemm_backends": "TRITON",
            },
            "zero_size": {
                "is_nonzero": False,
            },
            "out_dtype": {
                "input_dtype": torch.float16,
                "out_dtype": torch.float32,
            },
            "no_registered_provider": {
                "register_provider": False,
            },
            "provider_returns_no_choices": {
                "provider_result": (),
                "provider_expected_calls": 1,
            },
            "provider_returns_choice": {
                "provider_result": [provider_choice],
                "provider_expected_calls": 1,
                "expected_choices": [provider_choice],
            },
        }
        for name, kwargs in cases.items():
            with self.subTest(name=name):
                run_case(**kwargs)

    def test_remote_gemm_best_config_keeps_non_triton_provider_choice(self):
        class DummyTritonChoice(select_algorithm.ir.TritonTemplateCallerBase):
            def __init__(self, description, *, provider_backend=None):
                super().__init__("dummy_triton", [], None, description)
                if provider_backend is not None:
                    self.annotations[
                        select_algorithm.GEMM_TEMPLATE_PROVIDER_ANNOTATION
                    ] = provider_backend

            def call_name(self):
                return "dummy_triton"

            def to_callable(self):
                raise AssertionError("not used")

            def hash_key(self):
                return "dummy_triton"

            def output_node(self):
                return "dummy_triton_node"

            def get_make_kernel_render(self):
                return None

        best_config = {
            "ACC_TYPE": "tl.float32",
            "ALLOW_TF32": True,
            "BLOCK_K": 32,
            "BLOCK_M": 16,
            "BLOCK_N": 8,
            "EVEN_K": True,
            "GROUP_M": 8,
            "USE_FAST_ACCUM": False,
            "num_stages": 3,
            "num_warps": 4,
            "num_consumer_groups": 0,
            "num_buffers_warp_spec": 0,
        }
        matching_desc = " ".join(
            f"{key}={best_config[key]}"
            for key in select_algorithm._REMOTE_GEMM_AUTOTUNE_CACHE_IMPORTANT_KEYS
        )
        triton_match = DummyTritonChoice(matching_desc)
        triton_miss = DummyTritonChoice("BLOCK_M=999")
        cutile_provider_miss = DummyTritonChoice(
            "provider triton", provider_backend="cutile"
        )
        dummy_provider_miss = DummyTritonChoice(
            "provider dummy", provider_backend="dummy_gemm_backend"
        )
        triton_provider_miss = DummyTritonChoice(
            "provider triton", provider_backend="triton"
        )
        provider_choice = self._DummyChoice("dummy_provider", "provider")
        provider_choice.annotations[
            select_algorithm.GEMM_TEMPLATE_PROVIDER_ANNOTATION
        ] = "dummy_gemm_backend"

        filtered = select_algorithm._filter_choices_by_remote_gemm_best_config(
            [
                triton_match,
                triton_miss,
                provider_choice,
                cutile_provider_miss,
                dummy_provider_miss,
                triton_provider_miss,
            ],
            best_config,
        )

        self.assertIn(triton_match, filtered)
        self.assertIn(provider_choice, filtered)
        self.assertIn(cutile_provider_miss, filtered)
        self.assertIn(dummy_provider_miss, filtered)
        self.assertNotIn(triton_miss, filtered)
        self.assertNotIn(triton_provider_miss, filtered)

    def test_register_gemm_template_provider_rejects_conflicting_duplicate(self):
        def dummy_gemm_provider(context):
            return ()

        def other_dummy_gemm_provider(context):
            return ()

        mm_kernel.register_gemm_template_provider(
            "dummy_gemm_backend", dummy_gemm_provider
        )

        with self.assertRaisesRegex(ValueError, "already registered"):
            mm_kernel.register_gemm_template_provider(
                "dummy_gemm_backend", other_dummy_gemm_provider
            )

    def test_register_gemm_template_provider_rejects_invalid_name(self):
        def dummy_gemm_provider(context):
            return ()

        for name in ("", "not-valid", "class"):
            with self.assertRaisesRegex(ValueError, "valid Python identifier|non-empty"):
                mm_kernel.register_gemm_template_provider(
                    name, dummy_gemm_provider
                )

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

    def test_online_softmax_gate_uses_cuda_backend_feature(self):
        from torch._inductor.fx_passes.post_grad import prepare_softmax_extra_check

        match = SimpleNamespace(
            kwargs={
                "x": SimpleNamespace(
                    meta={"val": SimpleNamespace(device=torch.device("cuda"))}
                )
            }
        )

        class NoOnlineSoftmaxScheduling(BaseScheduling):
            pass

        class OnlineSoftmaxScheduling(BaseScheduling):
            def get_backend_features(self, device):
                return OrderedSet([common.BackendFeature.ONLINE_SOFTMAX])

        self.assertNotIn(
            common.BackendFeature.ONLINE_SOFTMAX,
            TileKernelScheduling(None).get_backend_features(torch.device("cuda")),
        )
        self.assertIn(
            common.BackendFeature.ONLINE_SOFTMAX,
            TritonScheduling(None).get_backend_features(torch.device("cuda")),
        )

        common.register_cuda_backend(
            "dummy_cuda_backend", NoOnlineSoftmaxScheduling
        )
        with config.patch(cuda_backend="dummy_cuda_backend"):
            self.assertFalse(prepare_softmax_extra_check(match))

        common._cuda_backends.pop("dummy_cuda_backend")
        common.register_cuda_backend(
            "dummy_cuda_backend", OnlineSoftmaxScheduling
        )
        with config.patch(cuda_backend="dummy_cuda_backend"):
            self.assertTrue(prepare_softmax_extra_check(match))

        with config.patch(
            cuda_backend="dummy_cuda_backend",
            online_softmax=False,
        ):
            self.assertFalse(prepare_softmax_extra_check(match))

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

    def _make_tile_io_descriptor(
        self,
        *,
        size=(32, 64),
        stride=None,
        offset=0,
        broadcasting_dims=(False, False),
        stride_sort_idx=(0, 1),
        prepared_index=None,
        prepare_indexing=None,
    ):
        xblock = TritonSymbols.block_sizes[SymT.XBLOCK]
        yblock = TritonSymbols.block_sizes[SymT.YBLOCK]
        xoffset = TritonSymbols.block_offsets[SymT.XBLOCK]
        yoffset = TritonSymbols.block_offsets[SymT.YBLOCK]

        buffer = self._FakeBuffer(size, stride=stride, offset=offset)

        class FakeSizeVars:
            @staticmethod
            def statically_known_equals(lhs, rhs):
                return sympy.simplify(lhs - rhs) == 0

        class FakeGraph:
            sizevars = FakeSizeVars()

            @staticmethod
            def get_buffer(name):
                self.assertEqual(name, "buf")
                return buffer

        kernel = TileKernel.__new__(TileKernel)
        kernel.prepare_indexing = (
            (lambda index: index) if prepare_indexing is None else prepare_indexing
        )

        y_tree = SimpleNamespace(
            prefix="y",
            tensor_dim=0,
            grid_dim=1,
            numel=sympy.Integer(size[0]),
            symt=SymT.YBLOCK,
        )
        x_tree = SimpleNamespace(
            prefix="x",
            tensor_dim=1,
            grid_dim=0,
            numel=sympy.Integer(size[1]),
            symt=SymT.XBLOCK,
        )
        kernel.active_range_trees = lambda: [y_tree, x_tree]

        indexing = BlockDescriptorOptions(
            params=BlockParameters(
                shape=[sympy.Integer(size[0]), sympy.Integer(size[1])],
                block_shape=[yblock, xblock],
                strides=[
                    sympy.Integer(buffer.get_stride()[0]),
                    sympy.Integer(buffer.get_stride()[1]),
                ],
                offsets=[yoffset, xoffset],
            ),
            constant_offset=sympy.S.Zero,
            order=[1, 0],
            mask_vars=OrderedSet(["ymask", "xmask"]),
            broadcast_shape=[yblock, xblock],
            broadcasting_dims=list(broadcasting_dims),
            final_shape=[yblock, xblock],
            stride_sorter=BlockParameters.StrideSorter(
                original_strides=buffer.get_stride(),
                sort_idx=list(stride_sort_idx),
            ),
            _boundary_check=[0, 1],
            prepared_index=prepared_index,
        )

        original_index = 64 * sympy.Symbol("yindex") + sympy.Symbol("xindex")
        with V.set_graph_handler(FakeGraph()):
            return kernel.tile_io_descriptor(
                "load",
                "buf",
                "in_ptr0",
                original_index,
                indexing,
            )

    def test_tile_io_descriptor_rank2_contiguous_access(self):
        descriptor = self._make_tile_io_descriptor()

        self.assertIsInstance(descriptor, TileIODescriptor)
        self.assertEqual(descriptor.access_kind, "load")
        self.assertEqual(descriptor.buffer_name, "buf")
        self.assertEqual(descriptor.arg_var, "in_ptr0")
        self.assertEqual(descriptor.logical_rank, 2)
        self.assertEqual(descriptor.logical_size, (32, 64))
        self.assertEqual(descriptor.logical_stride, (64, 1))
        self.assertEqual(descriptor.storage_offset, 0)
        self.assertEqual(descriptor.classification, "direct_contiguous")
        self.assertTrue(descriptor.statically_proven)
        self.assertEqual(
            tuple(str(dim) for dim in descriptor.block_shape),
            ("YBLOCK", "XBLOCK"),
        )
        self.assertEqual(descriptor.mask_vars, ("ymask", "xmask"))
        self.assertEqual(descriptor.boundary_check, (0, 1))
        self.assertEqual(descriptor.axes[0].prefix, "y")
        self.assertEqual(descriptor.axes[0].tensor_dim, 0)
        self.assertEqual(descriptor.axes[0].grid_dim, 1)
        self.assertEqual(str(descriptor.axes[0].block_size), "YBLOCK")
        self.assertEqual(descriptor.axes[1].prefix, "x")
        self.assertEqual(descriptor.axes[1].tensor_dim, 1)
        self.assertEqual(descriptor.axes[1].grid_dim, 0)
        self.assertEqual(str(descriptor.axes[1].block_size), "XBLOCK")

    def test_tile_io_descriptor_uses_existing_prepared_index(self):
        prepared_index = sympy.Symbol("prepared_index")

        def fail_if_recomputed(index):
            raise AssertionError("prepare_indexing should not run twice")

        descriptor = self._make_tile_io_descriptor(
            prepared_index=prepared_index,
            prepare_indexing=fail_if_recomputed,
        )

        self.assertEqual(descriptor.prepared_index, prepared_index)

    def test_tile_io_descriptor_classifies_unreviewed_layouts(self):
        self.assertEqual(
            self._make_tile_io_descriptor(stride=(0, 1)).classification,
            "broadcast",
        )
        self.assertEqual(
            self._make_tile_io_descriptor(
                broadcasting_dims=(True, False)
            ).classification,
            "broadcast",
        )
        self.assertEqual(
            self._make_tile_io_descriptor(stride_sort_idx=(1, 0)).classification,
            "transposed",
        )
        self.assertEqual(
            self._make_tile_io_descriptor(offset=1).classification,
            "nonzero_offset",
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

    def test_tile_heuristic_prepare_symbols_are_public(self):
        for name in (
            "prepare_pointwise_configs",
            "prepare_reduction_configs",
            "prepare_persistent_reduction_configs",
        ):
            self.assertTrue(callable(getattr(triton_heuristics, name)))

        with mock.patch.object(
            triton_heuristics,
            "prepare_pointwise_configs",
            return_value="pointwise",
        ):
            self.assertEqual(
                triton_heuristics._prepare_pointwise({"x": 1}, {"signature": {}}),
                "pointwise",
            )

        with mock.patch.object(
            triton_heuristics,
            "prepare_reduction_configs",
            return_value="reduction",
        ):
            self.assertEqual(
                triton_heuristics._prepare_reduction(
                    {"x": 1, "r0_": 1}, triton_meta={"signature": {}}
                ),
                "reduction",
            )

    def test_pointwise_heuristic_uses_public_prepare(self):
        configs = [Config({"XBLOCK": 1})]
        inductor_meta = {"kernel_name": "dummy"}
        triton_meta = {"signature": {}}

        with (
            mock.patch.object(
                triton_heuristics,
                "prepare_pointwise_configs",
                return_value=(configs, {"x": 1}, inductor_meta),
            ) as prepare,
            mock.patch.object(
                triton_heuristics, "cached_autotune", return_value="decorator"
            ) as cached_autotune,
        ):
            result = triton_heuristics.pointwise(
                {"x": 1},
                triton_meta=triton_meta,
                filename="dummy.py",
                inductor_meta=inductor_meta,
            )

        self.assertEqual(result, "decorator")
        prepare.assert_called_once_with(
            {"x": 1},
            triton_meta,
            tile_hint=None,
            filename="dummy.py",
            min_elem_per_thread=0,
            inductor_meta=inductor_meta,
        )
        cached_autotune.assert_called_once_with(
            {"x": 1},
            configs,
            triton_meta=triton_meta,
            inductor_meta=inductor_meta,
            heuristic_type=HeuristicType.POINTWISE,
            filename="dummy.py",
        )

    def test_reduction_heuristic_uses_public_prepare(self):
        configs = [Config({"XBLOCK": 1, "R0_BLOCK": 1})]
        inductor_meta = {"kernel_name": "dummy"}
        triton_meta = {"signature": {}}

        with (
            mock.patch.object(
                triton_heuristics,
                "prepare_reduction_configs",
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

    def test_persistent_reduction_heuristic_uses_public_prepare(self):
        configs = [Config({"XBLOCK": 1})]
        inductor_meta = {"kernel_name": "dummy"}
        triton_meta = {"signature": {}}

        with (
            mock.patch.object(
                triton_heuristics,
                "prepare_persistent_reduction_configs",
                return_value=(configs, None, inductor_meta),
            ) as prepare,
            mock.patch.object(
                triton_heuristics, "cached_autotune", return_value="decorator"
            ) as cached_autotune,
        ):
            result = triton_heuristics.persistent_reduction(
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
            None,
            configs,
            triton_meta=triton_meta,
            inductor_meta=inductor_meta,
            filename="dummy.py",
            heuristic_type=HeuristicType.PERSISTENT_REDUCTION,
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
