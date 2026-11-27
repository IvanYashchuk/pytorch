# Owner(s): ["module: dynamo"]

import sys
import sysconfig
import unittest
from concurrent.futures import ThreadPoolExecutor
from dis import get_instructions
from types import MethodType

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

    def test_declines_calls_with_kwargs_or_inlined_callees(self):
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

    @unittest.skipUnless(SUPPORTED, "requires the audited CPython 3.15.2 ABI")
    def test_matches_adaptive_call_after_hook_is_disabled_in_callee(self):
        value = object()
        armed = False
        hooked = False
        opcodes = []

        def target(arg):
            if not armed:
                return arg
            frame = sys._getframe(1)
            (call,) = [
                instruction
                for instruction in get_instructions(frame.f_code, adaptive=True)
                if instruction.offset == frame.f_lasti
            ]
            opcodes.append(call.opname)
            if hooked:
                prior = set_eval_frame(None)
                try:
                    self.assertIs(prior, False)
                    self.assertIs(frame_call_ex_matches(frame, target, operands), True)
                finally:
                    set_eval_frame(prior)
            else:
                self.assertIs(frame_call_ex_matches(frame, target, operands), False)
            return arg

        operands = (value,)

        def invoke():
            return next(fn(*args) for fn, args in ((target, operands),))

        original = set_eval_frame(None)
        try:
            for _ in range(32):
                self.assertIs(invoke(), value)
            armed = True
            self.assertIs(invoke(), value)
            hooked = True
            set_eval_frame(False)
            self.assertIs(invoke(), value)
            self.assertEqual(opcodes, ["CALL_EX_PY", "CALL_EX_PY"])
        finally:
            set_eval_frame(original)

    @unittest.skipUnless(SUPPORTED, "requires the audited CPython 3.15.2 ABI")
    def test_matches_plain_function_after_bound_method_history(self):
        value = object()
        operands = (value,)
        hooked = False
        opcodes = []

        def warm(receiver, arg):
            return arg

        bound = MethodType(warm, object())

        def target(arg):
            frame = sys._getframe(1)
            (call,) = [
                instruction
                for instruction in get_instructions(frame.f_code, adaptive=True)
                if instruction.offset == frame.f_lasti
            ]
            opcodes.append(call.opname)
            prior = set_eval_frame(None)
            try:
                self.assertIs(prior, False if hooked else None)
                self.assertIs(frame_call_ex_matches(frame, target, operands), hooked)
            finally:
                set_eval_frame(prior)
            return arg

        def invoke(callee):
            return next(fn(*args) for fn, args in ((callee, operands),))

        original = set_eval_frame(None)
        try:
            for _ in range(32):
                self.assertIs(invoke(bound), value)
            self.assertIs(invoke(target), value)
            hooked = True
            set_eval_frame(False)
            self.assertIs(invoke(target), value)
            self.assertEqual(opcodes, ["CALL_EX_NON_PY_GENERAL"] * 2)
        finally:
            set_eval_frame(original)


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
