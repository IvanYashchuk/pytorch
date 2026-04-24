# Owner(s): ["module: inductor"]

import unittest
from unittest import mock

import torch
from torch._inductor import config
from torch._inductor.codegen import common
from torch._inductor.codegen.cuda_combined_scheduling import CUDACombinedScheduling


class BackendExtensionAPITests(unittest.TestCase):
    def tearDown(self):
        common._cuda_backends.pop("dummy_cuda_backend", None)
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


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    run_tests()
