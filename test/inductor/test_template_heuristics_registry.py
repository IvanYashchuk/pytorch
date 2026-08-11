# Owner(s): ["module: inductor"]
from unittest import mock

import torch
from torch._inductor.heuristics.registry import (
    _TEMPLATE_HEURISTIC_REGISTRY,
    clear_registry,
    get_template_heuristic,
    register_template_heuristic,
)
from torch._inductor.heuristics.template.base import TemplateConfigHeuristics
from torch._inductor.heuristics.template.triton import (
    BlackwellGPUGemmConfig,
    CUDAConfigHeuristic,
    FlexBwDConfig,
    FlexConfig,
)
from torch._inductor.kernel.flex.common import (
    create_causal_indices_fake_generator,
    create_causal_num_blocks_fake_generator,
)
from torch._inductor.kernel.flex.flex_attention import _mask_graph_is_causal
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.virtualized import V


class TestBlackwellGPUGemmConfig(TestCase):
    """Tests for BlackwellGPUGemmConfig class."""

    def test_default_values(self):
        """Test that BlackwellGPUGemmConfig has correct default values."""
        config = BlackwellGPUGemmConfig(
            block_m=128,
            block_n=256,
            block_k=64,
            num_stages=3,
            num_warps=8,
        )
        # Verify inherited GemmConfig fields
        self.assertEqual(config.block_m, 128)
        self.assertEqual(config.block_n, 256)
        self.assertEqual(config.block_k, 64)
        self.assertEqual(config.num_stages, 3)
        self.assertEqual(config.num_warps, 8)

        # Verify new BlackwellGPUGemmConfig-specific fields with default values
        self.assertEqual(config.epilogue_subtile, 1)  # default=1
        self.assertTrue(config.warp_specialize)  # default=True
        self.assertTrue(config.flatten)  # default=True

    def test_custom_values(self):
        """Test that BlackwellGPUGemmConfig accepts custom values for new fields."""
        config = BlackwellGPUGemmConfig(
            block_m=64,
            block_n=128,
            block_k=32,
            num_stages=2,
            num_warps=4,
            epilogue_subtile=2,
            warp_specialize=False,
            flatten=False,
        )
        # Verify custom values are set correctly
        self.assertEqual(config.epilogue_subtile, 2)
        self.assertFalse(config.warp_specialize)
        self.assertFalse(config.flatten)


class TestTemplateHeuristicsRegistry(TestCase):
    def setUp(self):
        super().setUp()
        # Save original registry state
        self.original_registry = _TEMPLATE_HEURISTIC_REGISTRY.copy()
        clear_registry()  # Test heuristic classes using the decorator registration

    def tearDown(self):
        # Restore original registry
        clear_registry()
        _TEMPLATE_HEURISTIC_REGISTRY.update(self.original_registry)
        super().tearDown()

    def test_register_class(self):
        """Test basic registration of a heuristic class."""
        # Clear registry for this isolated test
        clear_registry()

        @register_template_heuristic("test_mm", "cuda")
        class TestHeuristic(TemplateConfigHeuristics):
            pass

        # Verify registration
        key = ("test_mm", "cuda", None)
        self.assertIn(key, _TEMPLATE_HEURISTIC_REGISTRY)
        self.assertEqual(_TEMPLATE_HEURISTIC_REGISTRY[key], TestHeuristic)

    def test_assertion_existing_class(self):
        @register_template_heuristic("triton::mm", "cuda")
        class _CrossOpHeuristic(TemplateConfigHeuristics):
            """(template, device, None) - Cross-op for specific device"""

        """Test that registered class can be retrieved."""
        # The _CrossOpHeuristic is registered at module level for ("mm", "cuda", None)
        # Test retrieval - it should match for any op on cuda device
        heuristic = get_template_heuristic("triton::mm", "cuda", "bmm")
        self.assertIsInstance(heuristic, _CrossOpHeuristic)

    def test_hierarchy_lookup(self):
        """Test complete hierarchy: (template, device, op) -> (template, None, None)"""

        @register_template_heuristic("triton::mm", "cuda", op_name="scaled_mm")
        class _MostSpecificHeuristic(TemplateConfigHeuristics):
            """(template, device, op) - Most specific"""

        @register_template_heuristic("triton::mm", None, op_name="scaled_mm")
        class _CrossDeviceHeuristic(TemplateConfigHeuristics):
            """(template, None, op) - Cross-device for specific op"""

        @register_template_heuristic("triton::mm", "cuda")
        class _CrossOpHeuristic(TemplateConfigHeuristics):
            """(template, device, None) - Cross-op for specific device"""

        @register_template_heuristic("triton::mm", None)
        class _MostGeneralHeuristic(TemplateConfigHeuristics):
            """(template, None, None) - Most general"""

        # All classes are already registered via decorators:
        # _MostSpecificHeuristic: ("mm", "cuda", "scaled_mm") - Most specific
        # _CrossDeviceHeuristic: ("mm", None, "scaled_mm") - Cross-device for specific op
        # _CrossOpHeuristic: ("mm", "cuda", None) - Cross-op for specific device
        # _MostGeneralHeuristic: ("mm", None, None) - Most general

        # Test 1: Exact match - should get most specific
        heuristic = get_template_heuristic("triton::mm", "cuda", "scaled_mm")
        self.assertIsInstance(heuristic, _MostSpecificHeuristic)

        # Test 2: Different device, same op - should get cross-device
        heuristic = get_template_heuristic("triton::mm", "xpu", "scaled_mm")
        self.assertIsInstance(heuristic, _CrossDeviceHeuristic)

        # Test 3: Same device, different op - should get cross-op
        heuristic = get_template_heuristic("triton::mm", "cuda", "bmm")
        self.assertIsInstance(heuristic, _CrossOpHeuristic)

        # Test 4: Different device and op - should get most general
        heuristic = get_template_heuristic("triton::mm", "xpu", "bmm")
        self.assertIsInstance(heuristic, _MostGeneralHeuristic)

    def test_partial_hierarchy_scenarios(self):
        """Test hierarchy behavior with partial registrations"""

        # Scenario 1: Register partial hierarchy using decorators
        @register_template_heuristic("triton::tma", None, op_name="scaled_tma")
        class _TestCrossDeviceHeuristic(TemplateConfigHeuristics):
            pass

        @register_template_heuristic("triton::tma", None)
        class _TestGeneralHeuristic(TemplateConfigHeuristics):
            pass

        # Should get cross-device for matching op, regardless of device
        heuristic = get_template_heuristic("triton::tma", "cuda", "scaled_tma")
        self.assertIsInstance(heuristic, _TestCrossDeviceHeuristic)

        # Should fallback to general for different op
        heuristic = get_template_heuristic("triton::tma", "cuda", "scaled_mm")
        self.assertIsInstance(heuristic, _TestGeneralHeuristic)

        # Scenario 2: Only specific device exists
        @register_template_heuristic("triton::bmm", "cuda")
        class _TestDeviceSpecificHeuristic(TemplateConfigHeuristics):
            pass

        # Should get device-specific for cuda
        heuristic = get_template_heuristic("triton::bmm", "cuda", "any_op")
        self.assertIsInstance(heuristic, _TestDeviceSpecificHeuristic)

        # Should return fallback instance for other devices (no specific heuristic registered)
        heuristic = get_template_heuristic("triton::bmm", "xpu", "any_op")
        self.assertIsInstance(heuristic, TemplateConfigHeuristics)
        # Make sure it's not the registered specific heuristic
        self.assertNotIsInstance(heuristic, _TestDeviceSpecificHeuristic)

        # Scenario 3: Only most general exists
        @register_template_heuristic("triton::mm", None)
        class _TestMostGeneralHeuristic(TemplateConfigHeuristics):
            pass

        # Should always get general regardless of device/op
        heuristic = get_template_heuristic("triton::mm", "cuda", "scaled_addmm")
        self.assertIsInstance(heuristic, _TestMostGeneralHeuristic)

        heuristic = get_template_heuristic("triton::mm", "xpu", "regular_addmm")
        self.assertIsInstance(heuristic, _TestMostGeneralHeuristic)

    def test_fallback_behavior(self):
        """Test that fallback TemplateConfigHeuristics is returned when no heuristic is found"""

        # Test 1: Get fallback for unregistered template
        heuristic = get_template_heuristic("unknown_template", "cuda", "unknown_op")
        self.assertIsInstance(heuristic, TemplateConfigHeuristics)
        # Make sure it's the base class and not a subclass
        self.assertEqual(type(heuristic), TemplateConfigHeuristics)

        # Test 2: Verify fallback instances are NOT cached (new instance each time)
        heuristic2 = get_template_heuristic("unknown_template", "cuda", "unknown_op")
        self.assertIsInstance(heuristic2, TemplateConfigHeuristics)
        self.assertEqual(type(heuristic2), TemplateConfigHeuristics)
        # Should be different instances (not cached)
        self.assertIsNot(heuristic, heuristic2)

        # Test 3: After registering a heuristic, should get the registered one instead
        @register_template_heuristic("unknown_template", "cuda", op_name="unknown_op")
        class _NewlyRegisteredHeuristic(TemplateConfigHeuristics):
            pass

        # Now should get the registered heuristic, not the fallback
        heuristic3 = get_template_heuristic("unknown_template", "cuda", "unknown_op")
        self.assertIsInstance(heuristic3, _NewlyRegisteredHeuristic)
        self.assertNotEqual(type(heuristic3), TemplateConfigHeuristics)

        # Test 4: Verify registered instances ARE cached (same instance each time)
        heuristic4 = get_template_heuristic("unknown_template", "cuda", "unknown_op")
        self.assertIsInstance(heuristic4, _NewlyRegisteredHeuristic)
        self.assertIs(heuristic3, heuristic4)  # Should be same cached instance


class TestA100DefaultFlexConfig(TestCase):
    def test_head_dim_192_entries(self):
        """``(bf16, 192)`` and ``(fp16, 192)`` entries are required for
        DeepSeek V3 MLA (qk_nope_head_dim=128 + qk_rope_head_dim=64 = 192,
        v_head_dim=128) on every ``capability >= (8, 0)`` board.

        Without these entries, dispatch falls through to
        ``FlexConfig(64, 64, 3, 4)``, which exceeds the 99 KiB per-block
        shared-memory opt-in budget on sm_8.6 / sm_8.9 (A10G, L40, A2)
        and fails compilation with "No valid triton configs."

        The pinned tile fits the sm_8.6 budget with margin and was the
        empirical fastest among the candidates on A2, L40, and A100. If
        you want to retune, please re-validate on a real sm_8.6 board
        before changing this value (the formula-based SMEM estimate alone
        is not a reliable proxy for what triton actually allocates).
        """
        expected = FlexConfig(128, 32, 2, 8)
        h = CUDAConfigHeuristic()
        self.assertEqual(h.a100_default_flex_config[(torch.bfloat16, 192)], expected)
        self.assertEqual(h.a100_default_flex_config[(torch.float16, 192)], expected)


class TestRubinDefaultFlexConfig(TestCase):
    @mock.patch("torch.cuda.get_device_capability", return_value=(10, 7))
    def test_long_tanh_score_mod(self, _mock_capability):
        sizevars = mock.Mock()
        sizevars.statically_known_geq.side_effect = lambda value, limit: value >= limit
        with V.set_graph_handler(mock.Mock(sizevars=sizevars)):
            heuristic = CUDAConfigHeuristic()
            self.assertEqual(
                heuristic.get_flex_attn_fwd_configs(
                    128, 4096, torch.bfloat16, has_tanh_score_mod=True
                ),
                [FlexConfig(128, 128, 1, 8)],
            )
            self.assertEqual(
                heuristic.get_flex_attn_fwd_configs(
                    128, 2048, torch.bfloat16, has_tanh_score_mod=True
                ),
                [FlexConfig(128, 128, 2, 8)],
            )
            self.assertEqual(
                heuristic.get_flex_attn_fwd_configs(
                    128, 4096, torch.bfloat16, has_tanh_score_mod=False
                ),
                [FlexConfig(128, 128, 2, 8)],
            )
            self.assertEqual(
                heuristic.get_flex_attn_fwd_configs(
                    32, 4096, torch.bfloat16, has_tanh_score_mod=True
                ),
                [FlexConfig(64, 64, 3, 4)],
            )

    @mock.patch("torch.cuda.get_device_capability", return_value=(10, 7))
    def test_transcendental_score_mod_backward(self, _mock_capability):
        heuristic = CUDAConfigHeuristic()
        transcendental_config = FlexBwDConfig(32, 64, 64, 32, 3, 4)
        for dtype in (torch.bfloat16, torch.float16):
            for head_dim in (64, 128, 256):
                self.assertEqual(
                    heuristic.get_flex_attn_bwd_configs(
                        head_dim, dtype, has_transcendental_score_mod=True
                    ),
                    [transcendental_config],
                )

        self.assertEqual(
            heuristic.get_flex_attn_bwd_configs(
                128, torch.bfloat16, has_transcendental_score_mod=False
            ),
            [FlexBwDConfig(64, 128, 128, 64, 3, 4)],
        )
        self.assertEqual(
            heuristic.get_flex_attn_bwd_configs(
                32, torch.bfloat16, has_transcendental_score_mod=True
            ),
            [FlexBwDConfig(32, 64, 64, 32, 3, 4)],
        )


class TestFlexAttentionAutotuneInputs(TestCase):
    def test_causal_block_mask_generators(self):
        sizevars = mock.Mock()
        sizevars.optimization_hints.side_effect = lambda value: value
        num_blocks = mock.Mock()
        num_blocks.get_size.return_value = [1, 1, 4]
        num_blocks.get_dtype.return_value = torch.int32
        num_blocks.get_device.return_value = torch.device("cpu")
        indices = mock.Mock()
        indices.get_size.return_value = [1, 1, 4, 4]
        indices.get_dtype.return_value = torch.int32
        indices.get_device.return_value = torch.device("cpu")

        with V.set_graph_handler(mock.Mock(sizevars=sizevars)):
            partial_counts = create_causal_num_blocks_fake_generator(full=False)(
                num_blocks
            )
            full_kv_counts = create_causal_num_blocks_fake_generator(full=True)(
                num_blocks
            )
            full_q_counts = create_causal_num_blocks_fake_generator(
                full=True, transposed=True
            )(num_blocks)
            partial_indices = create_causal_indices_fake_generator(partial_block=True)(
                indices
            )
            full_q_indices = create_causal_indices_fake_generator(
                partial_block=False, transposed=True
            )(indices)

        self.assertEqual(partial_counts.tolist(), [[[1, 1, 1, 1]]])
        self.assertEqual(full_kv_counts.tolist(), [[[0, 1, 2, 3]]])
        self.assertEqual(full_q_counts.tolist(), [[[3, 2, 1, 0]]])
        self.assertEqual(
            partial_indices.tolist(),
            [[[[0, 1, 2, 3], [1, 0, 2, 3], [2, 0, 1, 3], [3, 0, 1, 2]]]],
        )
        self.assertEqual(
            full_q_indices.tolist(),
            [[[[1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2], [0, 1, 2, 3]]]],
        )

    def test_causal_mask_graph_detection(self):
        graph = torch.fx.Graph()
        placeholders = [graph.placeholder(f"arg{index}") for index in range(4)]
        graph.output(
            graph.call_function(
                torch.ops.aten.ge.Tensor,
                (placeholders[2], placeholders[3]),
            )
        )

        self.assertTrue(_mask_graph_is_causal(torch.fx.GraphModule({}, graph)))


if __name__ == "__main__":
    run_tests()
