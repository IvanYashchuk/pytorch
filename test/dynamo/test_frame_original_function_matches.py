# Owner(s): ["module: dynamo"]
import sys
import sysconfig
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import FunctionType

from torch._C._dynamo.eval_frame import frame_original_function_matches

import torch

SUPPORTED = (
    sys.platform == "linux"
    and sys.version_info[:3] == (3, 15, 2)
    and sys.version_info.releaselevel == "final"
    and not sysconfig.get_config_var("Py_GIL_DISABLED")
    and not sysconfig.get_config_var("Py_DEBUG")
)


@unittest.skipUnless(SUPPORTED, "requires the pinned Linux CPython evaluator")
class TestFrameOriginalFunctionMatches(unittest.TestCase):
    def test_ordinary_call_loop_preserves_original_identity(self):
        def make_leaf(offset):
            def leaf(value):
                return value.sin() + offset

            return leaf

        left = make_leaf(1.0)
        right = FunctionType(left.__code__, left.__globals__, closure=left.__closure__)
        self.assertIsNot(left, right)
        self.assertIs(left.__closure__, right.__closure__)
        value = torch.linspace(-1, 1, 6)
        expected = left(value)
        frames = []
        entries = []
        compiled_graphs = []
        spoofed = []
        spoof = False
        recurse = False
        fail = False
        outer = None

        def call_loop(children):
            outputs = []
            for child in children:
                outputs.append(child(value))
            return outputs

        def backend(graph, _inputs):
            compiled_graphs.append(graph)

            def execute(*args):
                nonlocal outer, recurse
                frame = sys._getframe(1)
                while frame is not None and frame.f_code.co_name != "leaf":
                    frame = frame.f_back
                self.assertIsNotNone(frame)
                frames.append(frame)
                if outer is not None:
                    self.assertIs(frame_original_function_matches(outer, left), False)
                matches = (
                    frame_original_function_matches(frame, left),
                    frame_original_function_matches(frame, right),
                )
                entries.append(matches)
                if len(frames) == 1:
                    self.assertIs(
                        executor.submit(
                            frame_original_function_matches, frame, left
                        ).result(timeout=5),
                        False,
                    )
                if spoof:
                    parent = frame.f_back
                    while (
                        parent is not None and parent.f_code is not call_loop.__code__
                    ):
                        parent = parent.f_back
                    self.assertIsNotNone(parent)
                    if parent.f_locals["child"] is compiled_right:
                        parent.f_locals["child"] = compiled_left
                        spoofed.append(parent.f_locals["child"] is compiled_left)
                        self.assertIs(
                            frame_original_function_matches(frame, left), False
                        )
                        self.assertIs(
                            frame_original_function_matches(frame, right), True
                        )
                if recurse:
                    recurse = False
                    outer = frame
                    try:
                        torch.testing.assert_close(compiled_right(value), expected)
                    finally:
                        outer = None
                    self.assertIs(frame_original_function_matches(frame, left), True)
                    self.assertIs(frame_original_function_matches(frame, right), False)
                if fail:
                    raise ValueError("native witness unwind")
                return graph.forward(*args)

            return execute

        optimize = torch._dynamo.optimize(backend, dynamic=False)
        compiled_left, compiled_right = optimize(left), optimize(right)
        with ThreadPoolExecutor(max_workers=1) as executor:
            torch.testing.assert_close(call_loop((compiled_left,))[0], expected)
            with torch._dynamo.config.patch(error_on_recompile=True):
                spoof = True
                for result in call_loop((compiled_left, compiled_right)):
                    torch.testing.assert_close(result, expected)
                spoof = False
                recurse = True
                torch.testing.assert_close(call_loop((compiled_left,))[0], expected)
                fail = True
                with self.assertRaisesRegex(ValueError, "native witness unwind"):
                    call_loop((compiled_right,))
                fail = False
                torch.testing.assert_close(call_loop((compiled_left,))[0], expected)

        self.assertEqual(spoofed, [True])
        self.assertEqual(len(compiled_graphs), 1)
        self.assertEqual(
            entries,
            [
                (True, False),
                (True, False),
                (False, True),
                (True, False),
                (False, True),
                (False, True),
                (True, False),
            ],
        )
        for frame in frames:
            self.assertIs(frame_original_function_matches(frame, left), False)
            self.assertIs(frame_original_function_matches(frame, right), False)


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
