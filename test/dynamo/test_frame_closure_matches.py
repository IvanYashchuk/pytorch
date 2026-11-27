# Owner(s): ["module: dynamo"]

import ctypes
import sys
import sysconfig
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import CellType

from torch._C._dynamo.eval_frame import frame_closure_matches

SUPPORTED = sys.version_info[:2] == (3, 15) and not sysconfig.get_config_var(
    "Py_GIL_DISABLED"
)


class TestFrameClosureMatches(unittest.TestCase):
    def test_matches_ordered_cells_on_the_owning_thread(self):
        value = object()

        def make_target(left, right):
            def target(check):
                frame = sys._getframe()
                check(frame)
                return frame, left, right

            return target

        target = make_target(value, value)
        cells = target.__closure__
        clone = make_target(value, value)
        self.assertIs(target.__code__, clone.__code__)

        def check(frame):
            self.assertIs(frame_closure_matches(frame, (cells[0], cells[1])), SUPPORTED)
            self.assertIs(frame_closure_matches(frame, clone.__closure__), False)
            self.assertIs(frame_closure_matches(frame, cells[::-1]), False)
            self.assertIs(
                frame_closure_matches(frame, (cells[0], CellType(value))), False
            )
            self.assertIs(frame_closure_matches(frame, (value, value)), False)
            self.assertIs(frame_closure_matches(frame, [*cells]), False)
            self.assertIs(frame_closure_matches(None, cells), False)
            with ThreadPoolExecutor(max_workers=1) as executor:
                other_thread = executor.submit(frame_closure_matches, frame, cells)
                self.assertIs(other_thread.result(timeout=5), False)

        returned, left, right = target(check)
        self.assertIs(left, right)
        self.assertIs(frame_closure_matches(returned, cells), False)

    @unittest.skipUnless(SUPPORTED, "requires CPython 3.15 with the GIL")
    def test_declines_retargeted_function_while_its_frame_uses_old_cells(self):
        set_closure = ctypes.PYFUNCTYPE(
            ctypes.c_int, ctypes.py_object, ctypes.py_object
        )(("PyFunction_SetClosure", ctypes.pythonapi))

        def make_target(value):
            def target(check):
                check(sys._getframe())
                return value

            return target

        original_value = object()
        target = make_target(original_value)
        original = target.__closure__
        replacement = make_target(object()).__closure__

        def check(frame):
            self.assertIs(frame_closure_matches(frame, original), True)
            self.assertEqual(set_closure(target, replacement), 0)
            try:
                self.assertIs(target.__closure__, replacement)
                self.assertIs(frame_closure_matches(frame, original), False)
                self.assertIs(frame_closure_matches(frame, replacement), False)
            finally:
                self.assertEqual(set_closure(target, original), 0)

        self.assertIs(target(check), original_value)

    def test_declines_generator_frames(self):
        def suspended():
            frame = sys._getframe()
            yield frame, frame_closure_matches(frame, ())

        pending = suspended()
        frame, result = next(pending)
        self.assertIs(result, False)
        self.assertIs(frame_closure_matches(frame, ()), False)
        pending.close()


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
