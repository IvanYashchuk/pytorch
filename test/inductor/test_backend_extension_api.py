# Owner(s): ["module: inductor"]

import unittest
from unittest import mock

import torch
from torch._inductor import config
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


class BackendExtensionAPITests(unittest.TestCase):
    def tearDown(self):
        common._cuda_backends.pop("dummy_cuda_backend", None)
        common._constexpr_syntaxes.pop("dummy_constexpr_backend", None)
        common._dtype_propagation_backends.pop("dummy_dtype_backend", None)
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

    def test_cuda_combined_scheduling_kernel_scheduler_is_overridable(self):
        class DummyKernelScheduling(TritonScheduling):
            pass

        class DummyCombinedScheduling(CUDACombinedScheduling):
            kernel_scheduling_class = DummyKernelScheduling

        scheduling = DummyCombinedScheduling(None)

        self.assertIsInstance(
            scheduling._kernel_scheduling, DummyKernelScheduling
        )

    def test_triton_tile_kernel_base_classes_are_exposed(self):
        self.assertTrue(issubclass(TritonKernel, TileKernel))
        self.assertTrue(issubclass(TritonScheduling, TileKernelScheduling))
        self.assertIs(TritonScheduling.kernel_type, TritonKernel)
        self.assertEqual(TritonKernel.backend(), "triton")

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


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    run_tests()
