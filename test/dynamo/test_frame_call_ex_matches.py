# Owner(s): ["module: dynamo"]

import sys
import sysconfig
import unittest
from concurrent.futures import ThreadPoolExecutor

from torch._C._dynamo.eval_frame import frame_call_ex_matches, set_eval_frame

SUPPORTED = (
    sys.version_info[:3] == (3, 15, 2)
    and sys.version_info.releaselevel == "final"
    and not sysconfig.get_config_var("Py_GIL_DISABLED")
    and sys.platform != "win32"
)


class TestFrameCallExMatches(unittest.TestCase):
    def test_preserves_operand_identity_after_local_rebind(self):
        value = object()
        seen = []

        def target(arg, check):
            check(sys._getframe(1))
            return arg

        def impostor(arg, check):
            raise AssertionError("generator invoked the wrong function")

        def check(frame):
            self.assertEqual(frame.f_code.co_name, "<genexpr>")
            self.assertIs(frame_call_ex_matches(frame, target, operands), SUPPORTED)
            frame.f_locals["fn"] = impostor
            frame.f_locals["args"] = copied
            self.assertIs(frame.f_locals["fn"], impostor)
            self.assertIs(frame.f_locals["args"], copied)
            self.assertIs(frame_call_ex_matches(frame, target, operands), SUPPORTED)
            self.assertIs(frame_call_ex_matches(frame, impostor, operands), False)
            self.assertIs(frame_call_ex_matches(frame, target, copied), False)
            self.assertIs(frame_call_ex_matches(frame, impostor, copied), False)
            self.assertIs(frame_call_ex_matches(frame, target, list(operands)), False)
            with ThreadPoolExecutor(max_workers=1) as executor:
                foreign = executor.submit(
                    frame_call_ex_matches, frame, target, operands
                )
                self.assertIs(foreign.result(timeout=5), False)
            seen.append(frame)

        operands = (value, check)
        copied = (value, check)
        self.assertIsNot(operands, copied)
        prior = set_eval_frame(False)
        try:
            loop = (fn(*args) for fn, args in ((target, operands),))
            self.assertIs(next(loop), value)
            (frame,) = seen
            self.assertIs(loop.gi_frame, frame)
            self.assertIs(frame_call_ex_matches(frame, target, operands), False)
            loop.close()
            self.assertIs(frame_call_ex_matches(frame, target, operands), False)
        finally:
            set_eval_frame(prior)

    def test_declines_calls_with_kwargs_or_without_pep523(self):
        def target(arg):
            frame = sys._getframe(1)
            self.assertIs(frame_call_ex_matches(frame, target, operands), False)
            return arg

        value = object()
        operands = (value,)
        prior = set_eval_frame(False)
        try:
            result = next(
                fn(*args, **kwargs) for fn, args, kwargs in ((target, operands, {}),)
            )
            self.assertIs(result, value)
            set_eval_frame(None)
            self.assertIs(next(fn(*args) for fn, args in ((target, operands),)), value)
        finally:
            set_eval_frame(prior)


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
