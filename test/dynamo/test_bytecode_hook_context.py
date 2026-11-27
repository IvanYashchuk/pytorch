# Owner(s): ["module: dynamo"]

import builtins
import sys
import unittest
from contextlib import ExitStack
from io import StringIO
from types import FunctionType, MethodType, ModuleType

import torch
import torch._dynamo.test_case
from torch._dynamo.backends.debugging import eager
from torch._dynamo.cache_size import CacheSizeRelevantForFrame
from torch._dynamo.convert_frame import _compile, register_bytecode_hook
from torch._dynamo.eval_frame import _get_cache_entries_for_region
from torch._dynamo.hooks import Hooks
from torch._guards import CompileId


class TestBytecodeHookContext(torch._dynamo.test_case.TestCase):
    @unittest.skipIf(sys.version_info < (3, 11), "requires a native frame function")
    def test_context_owns_exact_function_and_namespace(self):
        def body(x, increment):
            return x.sin() + increment

        first_module = ModuleType("bytecode_context_one")
        second_module = ModuleType("bytecode_context_two")
        first_builtins = dict(vars(builtins))
        second_builtins = dict(vars(builtins))
        first_module.__dict__["__builtins__"] = first_builtins
        second_module.__dict__["__builtins__"] = second_builtins
        original = FunctionType(body.__code__, first_module.__dict__, "original", (1,))
        same_namespace = FunctionType(
            body.__code__, first_module.__dict__, "same_namespace", (2,)
        )
        other_namespace = FunctionType(
            body.__code__, second_module.__dict__, "other_namespace", (3,)
        )
        # Both original functions keep the builtin dictionary captured at creation.
        first_module.__dict__["__builtins__"] = second_builtins
        captures = []

        def inspect_context(old, new, context):
            if old is body.__code__:
                captures.append((new, context))

        x = torch.arange(4.0)
        function_namespaces = (
            (original, first_module.__dict__, first_builtins),
            (same_namespace, first_module.__dict__, first_builtins),
            (other_namespace, second_module.__dict__, second_builtins),
        )
        for module in (first_module, second_module):
            self.assertNotIn(module.__name__, sys.modules)
            sys.modules[module.__name__] = module
            self.addCleanup(sys.modules.pop, module.__name__, None)
        with register_bytecode_hook(inspect_context, with_context=True):
            for index, (fn, _, _) in enumerate(function_namespaces, 1):
                compiled = torch.compile(
                    fn,
                    backend="eager",
                    dynamic=False,
                    fullgraph=True,
                    isolate_recompiles=True,
                )
                self.assertEqual(compiled(x), fn(x))
                self.assertEqual(len(captures), index)
                self.assertEqual(compiled(x), fn(x))
                self.assertEqual(len(captures), index)

        buckets = set()
        for (new, context), (fn, namespace, builtin_namespace) in zip(
            captures, function_namespaces, strict=True
        ):
            self.assertIs(context.f_function, fn)
            self.assertIs(context.f_globals, namespace)
            self.assertIs(context.f_builtins, builtin_namespace)
            self.assertGreaterEqual(context.isolation_id, 0)
            buckets.add(context.isolation_id)
            self.assertTrue(
                any(
                    entry.code is new and entry.compile_id == context.compile_id
                    for entry in _get_cache_entries_for_region(
                        body.__code__, context.isolation_id
                    )
                )
            )
        self.assertEqual(len(buckets), len(function_namespaces))

    @unittest.skipIf(sys.version_info < (3, 11), "requires a native frame function")
    def test_context_observes_retained_bound_method_without_lookup(self):
        class Worker:
            def run(self, x, stream):
                y = x.sin()
                stream.read(1)
                return y + self.increment

        worker = Worker()
        worker.increment = 2
        bound = worker.run
        self.assertIs(type(bound), MethodType)
        lookups = []

        class Trap:
            def __get__(self, obj, owner):
                lookups.append(owner)
                raise AssertionError("re-resolved the retained bound method")

        Worker.run = Trap()
        captures = []

        def inspect_context(old, new, context):
            if old is bound.__func__.__code__:
                captures.append(context)

        compiled = torch.compile(
            bound, backend="eager", dynamic=False, isolate_recompiles=True
        )
        x = torch.arange(4.0)
        expected_stream = StringIO("LR")
        expected = bound(x, expected_stream)
        with (
            torch._dynamo.config.patch(nested_graph_breaks=False),
            register_bytecode_hook(inspect_context, with_context=True),
        ):
            for _ in range(2):
                stream = StringIO("LR")
                self.assertEqual(compiled(x, stream), expected)
                self.assertEqual(stream.tell(), expected_stream.tell())
        (context,) = captures
        self.assertIs(context.f_function, bound.__func__)
        self.assertIs(context.f_globals, bound.__func__.__globals__)
        self.assertIs(context.f_builtins, bound.__func__.__builtins__)
        self.assertEqual(lookups, [])

    def test_context_hooks_preserve_legacy_arity_order_and_emitted_code(self):
        def body(x):
            return x.sin() + 1

        events = []
        received = {}

        def before(old, new, context):
            if old is body.__code__:
                events.append("before")
                received["emitted"] = new
                received["context"] = context

        def legacy(*args):
            self.assertEqual(len(args), 2)
            old, new = args
            if old is body.__code__:
                events.append("legacy")
                self.assertIs(new, received["emitted"])
                modified = new.replace()
                received["modified"] = modified
                return modified
            return None

        def after(old, new, context):
            if old is body.__code__:
                events.append("after")
                self.assertIs(new, received["modified"])
                self.assertIs(context, received["context"])

        with ExitStack() as hooks:
            hooks.enter_context(register_bytecode_hook(before, with_context=True))
            hooks.enter_context(register_bytecode_hook(legacy))
            hooks.enter_context(register_bytecode_hook(after, with_context=True))
            compiled = torch.compile(
                body,
                backend="eager",
                dynamic=False,
                fullgraph=True,
            )
            x = torch.arange(4.0)
            self.assertEqual(compiled(x), body(x))
        self.assertEqual(events, ["before", "legacy", "after"])
        context = received["context"]
        self.assertEqual(context.isolation_id, -1)
        self.assertTrue(
            any(
                entry.code is received["modified"]
                and entry.compile_id == context.compile_id
                for entry in _get_cache_entries_for_region(
                    body.__code__, context.isolation_id
                )
            )
        )

    def test_synthetic_compile_has_no_frame_provenance(self):
        def body(x):
            return x.sin() + 1

        contexts = []
        requested_id = CompileId(frame_id=42, frame_compile_id=0)

        def observe(old, new, context):
            if old is body.__code__:
                contexts.append(context)

        with register_bytecode_hook(observe, with_context=True):
            _compile(
                body.__code__,
                body.__globals__,
                {"x": torch.arange(4.0)},
                body.__builtins__,
                (),
                compiler_fn=eager,
                one_graph=False,
                export=False,
                export_constraints=None,
                hooks=Hooks(),
                cache_entry=None,
                cache_entries=[],
                cache_entries_for_reasons=[],
                cache_size=CacheSizeRelevantForFrame(0, 0),
                frame=None,
                frame_state={},
                compile_id=requested_id,
            )
        (context,) = contexts
        self.assertEqual(context.compile_id, requested_id)
        self.assertIsNone(context.isolation_id)
        self.assertIsNone(context.f_globals)
        self.assertIsNone(context.f_builtins)
        self.assertIsNone(context.f_function)


if __name__ == "__main__":
    torch._dynamo.test_case.run_tests()
