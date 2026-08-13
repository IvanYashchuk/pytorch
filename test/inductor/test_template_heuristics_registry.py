# Owner(s): ["module: inductor"]
from unittest import mock

import sympy
import torch
from torch._inductor import config as inductor_config
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
from torch._inductor.kernel.flex.flex_attention import (
    _can_pack_all_gqa_heads,
    _can_pack_gqa_query_tail,
    _can_use_causal_fwd_autotune_inputs,
    _fast_tanh_min_rows_per_sm,
    _mask_graph_is_causal,
    _score_graph_is_softcap,
    _use_causal_load_balance,
    _use_fast_tanh_for_softcap,
    flex_attention_grid,
)
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
    def test_rubin_max_autotune_includes_stage2_rectangular_tile(
        self, _mock_capability
    ):
        heuristic = CUDAConfigHeuristic()
        candidate = FlexConfig(128, 64, 2, 8)
        sizevars = mock.Mock()
        sizevars.statically_known_geq.side_effect = (
            lambda value, limit: value >= limit
        )
        sizevars.statically_known_lt.side_effect = lambda value, limit: value < limit
        with (
            V.set_graph_handler(mock.Mock(sizevars=sizevars)),
            inductor_config.patch(max_autotune=True),
        ):
            self.assertIn(
                candidate,
                heuristic.get_flex_attn_fwd_configs(
                    256,
                    1024,
                    torch.bfloat16,
                    has_tanh_score_mod=True,
                    batch_heads=32,
                    is_causal=True,
                    is_gqa=True,
                ),
            )

        with inductor_config.patch(
            max_autotune=True, max_autotune_flex_search_space="EXHAUSTIVE"
        ):
            self.assertIn(
                candidate,
                heuristic.get_flex_attn_fwd_configs(256, 1024, torch.bfloat16),
            )

    @mock.patch("torch.cuda.get_device_capability", return_value=(10, 7))
    def test_long_tanh_score_mod(self, _mock_capability):
        sizevars = mock.Mock()
        sizevars.statically_known_geq.side_effect = (
            lambda value, limit: value >= limit
        )
        sizevars.statically_known_gt.side_effect = lambda value, limit: value > limit
        sizevars.statically_known_leq.side_effect = (
            lambda value, limit: value <= limit
        )
        sizevars.statically_known_lt.side_effect = lambda value, limit: value < limit
        with V.set_graph_handler(mock.Mock(sizevars=sizevars)):
            heuristic = CUDAConfigHeuristic()
            self.assertEqual(
                heuristic.get_flex_attn_fwd_configs(
                    128,
                    4096,
                    torch.bfloat16,
                    has_tanh_score_mod=True,
                    batch_heads=64,
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

    @mock.patch("torch.cuda.get_device_properties")
    @mock.patch("torch.cuda.get_device_capability", return_value=(10, 7))
    def test_long_forward_configs(self, _mock_capability, mock_properties):
        mock_properties.return_value.multi_processor_count = 212
        sizevars = mock.Mock()
        sizevars.statically_known_geq.side_effect = lambda value, limit: value >= limit
        with V.set_graph_handler(mock.Mock(sizevars=sizevars)):
            heuristic = CUDAConfigHeuristic()
            expected = {
                64: FlexConfig(64, 128, 3, 4),
                128: FlexConfig(128, 128, 2, 8),
                256: FlexConfig(64, 64, 3, 4),
            }
            for dtype in (torch.bfloat16, torch.float16):
                for head_dim, config in expected.items():
                    self.assertEqual(
                        heuristic.get_flex_attn_fwd_configs(
                            head_dim, 4096, dtype, batch_heads=1
                        ),
                        [config],
                    )

            tanh_cases = (
                (64, 32, False, FlexConfig(64, 64, 3, 4)),
                (64, 64, False, FlexConfig(128, 128, 1, 8)),
                (128, 32, False, FlexConfig(128, 128, 2, 8)),
                (128, 64, False, FlexConfig(128, 128, 1, 8)),
                (128, 64, True, FlexConfig(64, 64, 3, 4)),
                (256, 8, False, FlexConfig(64, 64, 3, 4)),
                (256, 32, True, FlexConfig(128, 128, 1, 8)),
            )
            for head_dim, batch_heads, is_causal, config in tanh_cases:
                self.assertEqual(
                    heuristic.get_flex_attn_fwd_configs(
                        head_dim,
                        4096,
                        torch.bfloat16,
                        has_tanh_score_mod=True,
                        batch_heads=batch_heads,
                        is_causal=is_causal,
                    ),
                    [config],
                )

            self.assertEqual(
                heuristic.get_flex_attn_fwd_configs(
                    128, 2048, torch.bfloat16, batch_heads=64
                ),
                [FlexConfig(128, 128, 2, 8)],
            )

    @mock.patch("torch.cuda.get_device_properties")
    @mock.patch("torch.cuda.get_device_capability", return_value=(10, 7))
    def test_gqa_forward_configs(self, _mock_capability, mock_properties):
        mock_properties.return_value.multi_processor_count = 212
        sizevars = mock.Mock()
        sizevars.statically_known_geq.side_effect = lambda value, limit: value >= limit
        sizevars.statically_known_lt.side_effect = lambda value, limit: value < limit
        sizevars.statically_known_leq.side_effect = (
            lambda value, limit: value <= limit
        )
        sizevars.statically_known_gt.side_effect = lambda value, limit: value > limit
        with V.set_graph_handler(mock.Mock(sizevars=sizevars)):
            heuristic = CUDAConfigHeuristic()
            for dtype in (torch.bfloat16, torch.float16):
                for seq_len, expected in (
                    (128, FlexConfig(64, 32, 3, 4)),
                    (384, FlexConfig(64, 32, 3, 4)),
                    (385, FlexConfig(128, 64, 2, 8)),
                    (512, FlexConfig(128, 64, 2, 8)),
                    (2048, FlexConfig(128, 64, 2, 8)),
                    (4096, FlexConfig(128, 64, 2, 8)),
                ):
                    self.assertEqual(
                        heuristic.get_flex_attn_fwd_configs(
                            256,
                            seq_len,
                            dtype,
                            has_tanh_score_mod=True,
                            batch_heads=8,
                            is_causal=True,
                            is_gqa=True,
                        ),
                        [expected],
                    )
                for head_dim in (64, 128):
                    self.assertEqual(
                        heuristic.get_flex_attn_fwd_configs(
                            head_dim,
                            2048,
                            dtype,
                            has_tanh_score_mod=True,
                            batch_heads=8,
                            is_causal=True,
                            is_gqa=True,
                        ),
                        [FlexConfig(64, 64, 3, 4)],
                    )

            short_gqa_configs = {
                (False, 64): FlexConfig(64, 128, 3, 4),
                (False, 128): FlexConfig(64, 128, 3, 4),
                (False, 256): FlexConfig(64, 64, 3, 4),
                (True, 64): FlexConfig(64, 64, 3, 4),
                (True, 128): FlexConfig(64, 64, 3, 4),
                (True, 256): FlexConfig(64, 64, 3, 4),
            }
            for dtype in (torch.bfloat16, torch.float16):
                for (has_tanh, head_dim), config in short_gqa_configs.items():
                    self.assertEqual(
                        heuristic.get_flex_attn_fwd_configs(
                            head_dim,
                            64,
                            dtype,
                            has_tanh_score_mod=has_tanh,
                            batch_heads=32,
                            is_gqa=True,
                        ),
                        [config],
                    )

            self.assertEqual(
                heuristic.get_flex_attn_fwd_configs(
                    64,
                    127,
                    torch.bfloat16,
                    batch_heads=32,
                    is_gqa=True,
                ),
                [FlexConfig(64, 128, 3, 4)],
            )
            self.assertEqual(
                heuristic.get_flex_attn_fwd_configs(
                    64,
                    128,
                    torch.bfloat16,
                    batch_heads=32,
                    is_gqa=True,
                ),
                [FlexConfig(64, 128, 3, 4)],
            )

            packed_gqa_configs = (
                (64, 128, 128, FlexConfig(64, 128, 3, 4)),
                (64, 266, 64, FlexConfig(64, 64, 3, 4)),
                (64, 1060, 64, FlexConfig(64, 64, 3, 4)),
                (64, 1061, 64, FlexConfig(128, 64, 3, 4)),
                (64, 1590, 64, FlexConfig(128, 64, 3, 4)),
                (64, 1591, 64, FlexConfig(64, 128, 3, 4)),
                (64, 2120, 64, FlexConfig(64, 128, 3, 4)),
                (64, 2121, 64, FlexConfig(128, 64, 3, 4)),
                (256, 128, 32, FlexConfig(64, 64, 3, 4)),
                (256, 2048, 128, FlexConfig(64, 64, 3, 4)),
            )
            for dtype in (torch.bfloat16, torch.float16):
                for head_dim, seq_len, batch_heads, config in packed_gqa_configs:
                    self.assertEqual(
                        heuristic.get_flex_attn_fwd_configs(
                            head_dim,
                            seq_len,
                            dtype,
                            batch_heads=batch_heads,
                            is_gqa=True,
                        ),
                        [config],
                    )

            # D128 retains its existing independent non-tanh policy.
            self.assertEqual(
                heuristic.get_flex_attn_fwd_configs(
                    128,
                    512,
                    torch.bfloat16,
                    batch_heads=32,
                    is_gqa=True,
                ),
                [FlexConfig(128, 128, 2, 8)],
            )

            softcap_gqa_configs = (
                (64, 256, 32, FlexConfig(64, 128, 3, 4)),
                (64, 512, 32, FlexConfig(64, 128, 3, 4)),
                (64, 1024, 32, FlexConfig(64, 64, 3, 4)),
                (64, 1536, 32, FlexConfig(128, 64, 3, 8)),
                (64, 2048, 32, FlexConfig(64, 64, 3, 4)),
                (64, 3072, 32, FlexConfig(128, 64, 2, 8)),
                (64, 4096, 32, FlexConfig(128, 64, 2, 8)),
                (64, 8192, 32, FlexConfig(128, 64, 2, 8)),
                (128, 256, 32, FlexConfig(64, 128, 3, 4)),
                (128, 768, 32, FlexConfig(128, 128, 2, 8)),
                (128, 1536, 32, FlexConfig(128, 128, 2, 8)),
                (128, 2048, 32, FlexConfig(128, 128, 2, 8)),
                (128, 4096, 32, FlexConfig(128, 128, 2, 8)),
                (256, 256, 32, FlexConfig(64, 64, 3, 4)),
                (256, 768, 32, FlexConfig(128, 64, 2, 8)),
                (256, 1024, 32, FlexConfig(128, 64, 2, 8)),
                (256, 1536, 32, FlexConfig(128, 64, 2, 8)),
                (256, 2048, 32, FlexConfig(128, 64, 2, 8)),
                (256, 4096, 32, FlexConfig(128, 64, 2, 8)),
            )
            for dtype in (torch.bfloat16, torch.float16):
                for head_dim, seq_len, batch_heads, config in softcap_gqa_configs:
                    self.assertEqual(
                        heuristic.get_flex_attn_fwd_configs(
                            head_dim,
                            seq_len,
                            dtype,
                            has_tanh_score_mod=True,
                            batch_heads=batch_heads,
                            is_gqa=True,
                            is_dense=True,
                        ),
                        [config],
                    )

            # Packed rows, not batch size alone, select the wave band.
            self.assertEqual(
                heuristic.get_flex_attn_fwd_configs(
                    64,
                    768,
                    torch.bfloat16,
                    has_tanh_score_mod=True,
                    batch_heads=64,
                    is_gqa=True,
                    is_dense=True,
                ),
                [FlexConfig(128, 64, 3, 8)],
            )

            # Causal softcap retains its separately tuned policy.
            self.assertEqual(
                heuristic.get_flex_attn_fwd_configs(
                    128,
                    4096,
                    torch.bfloat16,
                    has_tanh_score_mod=True,
                    batch_heads=32,
                    is_causal=True,
                    is_gqa=True,
                ),
                [FlexConfig(64, 64, 3, 4)],
            )

            self.assertEqual(
                heuristic.get_flex_attn_fwd_configs(
                    64,
                    64,
                    torch.bfloat16,
                    has_tanh_score_mod=True,
                    batch_heads=32,
                ),
                [FlexConfig(128, 64, 3, 4)],
            )

    @mock.patch("torch.cuda.get_device_capability", return_value=(10, 7))
    def test_rubin_backward_configs(self, _mock_capability):
        sizevars = mock.Mock()
        sizevars.statically_known_geq.side_effect = lambda value, limit: value >= limit
        sizevars.statically_known_gt.side_effect = lambda value, limit: value > limit
        sizevars.statically_known_leq.side_effect = lambda value, limit: value <= limit
        sizevars.statically_known_lt.side_effect = lambda value, limit: value < limit
        self.enterContext(V.set_graph_handler(mock.Mock(sizevars=sizevars)))
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

        small_config = FlexBwDConfig(32, 64, 64, 32, 3, 4)
        symmetric_config = FlexBwDConfig(64, 64, 64, 64, 3, 4)
        for dtype in (torch.bfloat16, torch.float16):
            for has_transcendental in (False, True):
                for head_dim, config in (
                    (64, symmetric_config),
                    (128, symmetric_config),
                    (256, small_config),
                ):
                    self.assertEqual(
                        heuristic.get_flex_attn_bwd_configs(
                            head_dim,
                            dtype,
                            has_transcendental_score_mod=has_transcendental,
                            seq_len_q=128,
                            batch_heads=32,
                            is_gqa=True,
                        ),
                        [config],
                    )

        self.assertEqual(
            heuristic.get_flex_attn_bwd_configs(
                128,
                torch.bfloat16,
                seq_len_q=129,
                batch_heads=32,
                is_gqa=True,
            ),
            [small_config],
        )
        self.assertEqual(
            heuristic.get_flex_attn_bwd_configs(
                128,
                torch.bfloat16,
                seq_len_q=256,
                batch_heads=32,
                is_gqa=True,
            ),
            [FlexBwDConfig(64, 128, 128, 64, 3, 4)],
        )
        self.assertEqual(
            heuristic.get_flex_attn_bwd_configs(
                128,
                torch.bfloat16,
                seq_len_q=64,
                batch_heads=64,
                is_gqa=True,
            ),
            [small_config],
        )
        self.assertEqual(
            heuristic.get_flex_attn_bwd_configs(
                64,
                torch.bfloat16,
                seq_len_q=64,
                batch_heads=256,
                is_gqa=True,
            ),
            [small_config],
        )

        # Long GQA D64 uses symmetric tiles below 128 packed query heads, but
        # tanh/softcap and higher-parallelism workloads retain the small tile.
        for dtype in (torch.bfloat16, torch.float16):
            for has_transcendental, has_tanh in (
                (False, False),
                (True, False),
            ):
                self.assertEqual(
                    heuristic.get_flex_attn_bwd_configs(
                        64,
                        dtype,
                        has_transcendental_score_mod=has_transcendental,
                        seq_len_q=4096,
                        batch_heads=32,
                        is_gqa=True,
                        has_tanh_score_mod=has_tanh,
                    ),
                    [symmetric_config],
                )
            for seq_len_q, batch_heads, has_tanh in (
                (129, 32, True),
                (4096, 32, True),
                (4096, 128, False),
            ):
                self.assertEqual(
                    heuristic.get_flex_attn_bwd_configs(
                        64,
                        dtype,
                        has_transcendental_score_mod=has_tanh,
                        seq_len_q=seq_len_q,
                        batch_heads=batch_heads,
                        is_gqa=True,
                        has_tanh_score_mod=has_tanh,
                    ),
                    [small_config],
                )

    @mock.patch("torch.cuda.get_device_capability", return_value=(10, 3))
    def test_dense_softcap_config_does_not_change_blackwell(self, _mock_capability):
        heuristic = CUDAConfigHeuristic()
        expected = {
            64: FlexConfig(128, 64, 3, 4),
            128: FlexConfig(128, 128, 2, 8),
            256: FlexConfig(64, 32, 3, 4),
        }
        for dtype in (torch.bfloat16, torch.float16):
            for head_dim, config in expected.items():
                self.assertEqual(
                    heuristic.get_flex_attn_fwd_configs(
                        head_dim,
                        512,
                        dtype,
                        has_tanh_score_mod=True,
                        batch_heads=32,
                        is_gqa=True,
                        is_dense=True,
                    ),
                    [config],
                )


class TestFlexAttentionAutotuneInputs(TestCase):
    def test_causal_gqa_full_head_packing_gate(self):
        sizevars = mock.Mock()
        sizevars.statically_known_true.side_effect = lambda expr: expr is sympy.true
        with V.set_graph_handler(mock.Mock(sizevars=sizevars)):
            for q_len in (1025, 1028, 1032):
                self.assertTrue(_can_pack_all_gqa_heads(q_len, 32, 128, 32, 212))
            for q_len in (1024, 1033, 1040, 1056, 1057, 1088, 897, 1665):
                self.assertFalse(_can_pack_all_gqa_heads(q_len, 32, 128, 32, 212))
            self.assertFalse(_can_pack_all_gqa_heads(1025, 64, 128, 32, 212))
            self.assertFalse(_can_pack_all_gqa_heads(1025, 32, 128, 32, 216))

            dynamic_q = sympy.Symbol("dynamic_q", integer=True, positive=True)
            self.assertFalse(_can_pack_all_gqa_heads(dynamic_q, 32, 128, 32, 212))

    def test_causal_gqa_head_packing_static_tail_gate(self):
        sizevars = mock.Mock()
        sizevars.statically_known_true.side_effect = lambda expr: expr is sympy.true
        with V.set_graph_handler(mock.Mock(sizevars=sizevars)):
            for q_len in (769, 800, 1665):
                self.assertTrue(_can_pack_gqa_query_tail(q_len, 32, 128, 32, 212))
            for q_len in (
                1,
                31,
                32,
                127,
                128,
                129,
                160,
                161,
                672,
                768,
                801,
                832,
                896,
                897,
                1025,
            ):
                self.assertFalse(_can_pack_gqa_query_tail(q_len, 32, 128, 32, 212))

            # The packed grid must remove a complete logical SM wave. Moving
            # the mocked SM count moves the gate, while exact equality stays
            # on the ordinary path.
            self.assertFalse(_can_pack_gqa_query_tail(6657, 4, 128, 32, 212))
            self.assertTrue(_can_pack_gqa_query_tail(769, 32, 128, 32, 223))
            self.assertFalse(_can_pack_gqa_query_tail(769, 32, 128, 32, 224))

            dynamic_q = sympy.Symbol("dynamic_q", integer=True, positive=True)
            dynamic_heads = sympy.Symbol("dynamic_heads", integer=True, positive=True)
            self.assertFalse(_can_pack_gqa_query_tail(dynamic_q, 32, 128, 32, 212))
            self.assertFalse(
                _can_pack_gqa_query_tail(769, dynamic_heads, 128, 32, 212)
            )

    def test_causal_mask_classifier_rejects_constant_output(self):
        graph = torch.fx.Graph()
        for name in ("b", "h", "q", "kv"):
            graph.placeholder(name)
        graph.output(True)
        self.assertFalse(_mask_graph_is_causal(torch.fx.GraphModule({}, graph)))

    def test_causal_gqa_head_packing_grid(self):
        packed_meta = {
            "BLOCK_M": 128,
            "QUERY_TILE_M": 32,
            "PACK_GQA_HEADS": True,
            "GQA_SHARED_HEADS": 4,
        }
        expected_programs = {
            129: 40,
            160: 40,
            769: 200,
            800: 200,
            897: 232,
            1025: 264,
        }
        for q_len, programs in expected_programs.items():
            self.assertEqual(
                flex_attention_grid(1, 32, q_len, 256, packed_meta),
                (programs, 1, 1),
            )
            self.assertEqual(
                flex_attention_grid(
                    1,
                    32,
                    q_len,
                    256,
                    {**packed_meta, "CAUSAL_LOAD_BALANCE": True},
                ),
                (programs, 1, 1),
            )

        full_meta = {
            "BLOCK_M": 128,
            "QUERY_TILE_M": 32,
            "PACK_ALL_GQA_HEADS": True,
            "GQA_SHARED_HEADS": 4,
        }
        self.assertEqual(
            flex_attention_grid(1, 32, 1025, 256, full_meta), (33, 1, 8)
        )
        self.assertEqual(
            flex_attention_grid(
                1,
                32,
                1025,
                256,
                {**full_meta, "CAUSAL_LOAD_BALANCE": True},
            ),
            (264, 1, 1),
        )

        # Shapes outside the static 1..32 tail gate retain the exact ordinary
        # grid. Causal load balancing still flattens that legacy work.
        ordinary_meta = {"BLOCK_M": 128, "PACK_GQA_HEADS": False}
        for q_len, query_blocks in (
            (31, 1),
            (128, 1),
            (161, 2),
            (768, 6),
            (831, 7),
            (832, 7),
            (833, 7),
            (895, 7),
            (896, 7),
        ):
            self.assertEqual(
                flex_attention_grid(1, 32, q_len, 256, ordinary_meta),
                (query_blocks, 1, 32),
            )
            self.assertEqual(
                flex_attention_grid(
                    1,
                    32,
                    q_len,
                    256,
                    {**ordinary_meta, "CAUSAL_LOAD_BALANCE": True},
                ),
                (query_blocks * 32, 1, 1),
            )

    def test_causal_load_balanced_grid(self):
        self.assertEqual(
            flex_attention_grid(
                2,
                32,
                769,
                256,
                {"BLOCK_M": 128, "CAUSAL_LOAD_BALANCE": True},
            ),
            (448, 1, 1),
        )
        self.assertEqual(
            flex_attention_grid(
                2,
                32,
                769,
                256,
                {"BLOCK_M": 128, "CAUSAL_LOAD_BALANCE": False},
            ),
            (7, 2, 32),
        )

    def test_rubin_fast_tanh_thresholds(self):
        for head_dim in (64, 128, 256):
            self.assertIsNone(
                _fast_tanh_min_rows_per_sm(
                    head_dim, dense_attention=True, causal_prefix_attention=False
                )
            )

        self.assertEqual(
            _fast_tanh_min_rows_per_sm(
                64, dense_attention=False, causal_prefix_attention=True
            ),
            128,
        )
        self.assertEqual(
            _fast_tanh_min_rows_per_sm(
                128, dense_attention=False, causal_prefix_attention=True
            ),
            64,
        )
        self.assertEqual(
            _fast_tanh_min_rows_per_sm(
                256, dense_attention=False, causal_prefix_attention=True
            ),
            58,
        )

        self.assertEqual(
            _fast_tanh_min_rows_per_sm(
                128, dense_attention=False, causal_prefix_attention=False
            ),
            128,
        )
        self.assertEqual(
            _fast_tanh_min_rows_per_sm(
                256, dense_attention=False, causal_prefix_attention=False
            ),
            96,
        )

    def test_rubin_causal_load_balance_scope(self):
        graph = torch.fx.Graph()
        placeholders = [graph.placeholder(f"arg{index}") for index in range(5)]
        divided = graph.call_function(
            torch.ops.aten.div.Tensor, (placeholders[0], 20.0)
        )
        tanh = graph.call_function(torch.ops.aten.tanh.default, (divided,))
        graph.output(graph.call_function(torch.ops.aten.mul.Tensor, (tanh, 20.0)))
        softcap_graph = torch.fx.GraphModule({}, graph)

        with mock.patch.object(
            torch.cuda, "get_device_capability", return_value=(10, 7)
        ):
            self.assertTrue(
                _use_causal_load_balance(
                    softcap_graph,
                    torch.bfloat16,
                    torch.device("cuda"),
                    256,
                    True,
                )
            )
            for dtype, head_dim, is_causal in (
                (torch.float32, 256, True),
                (torch.bfloat16, 128, True),
                (torch.bfloat16, 256, False),
            ):
                self.assertFalse(
                    _use_causal_load_balance(
                        softcap_graph,
                        dtype,
                        torch.device("cuda"),
                        head_dim,
                        is_causal,
                    )
                )

            arbitrary_graph = torch.fx.Graph()
            arbitrary = [
                arbitrary_graph.placeholder(f"arg{index}") for index in range(5)
            ]
            arbitrary_graph.output(
                arbitrary_graph.call_function(
                    torch.ops.aten.tanh.default, (arbitrary[0],)
                )
            )
            self.assertFalse(
                _use_causal_load_balance(
                    torch.fx.GraphModule({}, arbitrary_graph),
                    torch.bfloat16,
                    torch.device("cuda"),
                    256,
                    True,
                )
            )

        with mock.patch.object(
            torch.cuda, "get_device_capability", return_value=(10, 3)
        ):
            self.assertFalse(
                _use_causal_load_balance(
                    softcap_graph,
                    torch.bfloat16,
                    torch.device("cuda"),
                    256,
                    True,
                )
            )

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

    def test_rectangular_causal_block_mask_generators(self):
        sizevars = mock.Mock()
        sizevars.optimization_hints.side_effect = lambda value: value
        num_blocks = mock.Mock()
        num_blocks.get_size.return_value = [1, 1, 4]
        num_blocks.get_dtype.return_value = torch.int32
        num_blocks.get_device.return_value = torch.device("cpu")
        indices = mock.Mock()
        indices.get_size.return_value = [1, 1, 4, 32]
        indices.get_dtype.return_value = torch.int32
        indices.get_device.return_value = torch.device("cpu")

        with V.set_graph_handler(mock.Mock(sizevars=sizevars)):
            full_counts = create_causal_num_blocks_fake_generator(full=True)(num_blocks)
            partial_indices = create_causal_indices_fake_generator(partial_block=True)(
                indices
            )

        self.assertEqual(full_counts.tolist(), [[[0, 1, 2, 3]]])
        self.assertEqual(
            partial_indices[0, 0, :, 0].tolist(),
            [0, 1, 2, 3],
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

        causal_graph = torch.fx.GraphModule({}, graph)
        self.assertTrue(_mask_graph_is_causal(causal_graph))

        sizevars = mock.Mock()
        sizevars.statically_known_leq.side_effect = lambda lhs, rhs: lhs <= rhs
        with V.set_graph_handler(mock.Mock(sizevars=sizevars)):
            self.assertTrue(
                _can_use_causal_fwd_autotune_inputs(causal_graph, 512, 4096, 128, 128)
            )
            self.assertFalse(
                _can_use_causal_fwd_autotune_inputs(causal_graph, 8192, 4096, 128, 128)
            )
            self.assertFalse(
                _can_use_causal_fwd_autotune_inputs(causal_graph, 512, 4096, 64, 128)
            )

    def test_softcap_score_graph_detection(self):
        graph = torch.fx.Graph()
        placeholders = [graph.placeholder(f"arg{index}") for index in range(5)]
        divided = graph.call_function(
            torch.ops.aten.div.Tensor, (placeholders[0], 50.0)
        )
        tanh = graph.call_function(torch.ops.aten.tanh.default, (divided,))
        graph.output(graph.call_function(torch.ops.aten.mul.Tensor, (tanh, 50.0)))
        softcap_graph = torch.fx.GraphModule({}, graph)

        self.assertTrue(_score_graph_is_softcap(softcap_graph))
        sizevars = mock.Mock()
        sizevars.statically_known_geq.side_effect = lambda value, limit: value >= limit
        properties = mock.Mock(multi_processor_count=216)
        with (
            V.set_graph_handler(mock.Mock(sizevars=sizevars)),
            mock.patch.object(
                torch.cuda, "get_device_capability", return_value=(10, 7)
            ),
            mock.patch.object(
                torch.cuda, "get_device_properties", return_value=properties
            ),
        ):
            self.assertTrue(
                _use_fast_tanh_for_softcap(
                    softcap_graph,
                    torch.bfloat16,
                    torch.device("cuda"),
                    128,
                    1024,
                    32,
                    128,
                )
            )
            self.assertFalse(
                _use_fast_tanh_for_softcap(
                    softcap_graph,
                    torch.bfloat16,
                    torch.device("cuda"),
                    256,
                    512,
                    32,
                    96,
                )
            )
            self.assertTrue(
                _use_fast_tanh_for_softcap(
                    softcap_graph,
                    torch.bfloat16,
                    torch.device("cuda"),
                    128,
                    512,
                    32,
                    64,
                )
            )
            self.assertTrue(
                _use_fast_tanh_for_softcap(
                    softcap_graph,
                    torch.bfloat16,
                    torch.device("cuda"),
                    128,
                    128,
                    32,
                    None,
                )
            )
            self.assertFalse(
                _use_fast_tanh_for_softcap(
                    softcap_graph,
                    torch.bfloat16,
                    torch.device("cuda"),
                    128,
                    256,
                    32,
                    64,
                )
            )
            self.assertFalse(
                _use_fast_tanh_for_softcap(
                    softcap_graph,
                    torch.bfloat16,
                    torch.device("cuda"),
                    128,
                    512,
                    32,
                    128,
                )
            )
            self.assertTrue(
                _use_fast_tanh_for_softcap(
                    softcap_graph,
                    torch.bfloat16,
                    torch.device("cuda"),
                    256,
                    768,
                    32,
                    96,
                )
            )
            self.assertFalse(
                _use_fast_tanh_for_softcap(
                    softcap_graph,
                    torch.float32,
                    torch.device("cuda"),
                    128,
                    1024,
                    32,
                    128,
                )
            )
        with mock.patch.object(
            torch.cuda, "get_device_capability", return_value=(10, 3)
        ):
            self.assertFalse(
                _use_fast_tanh_for_softcap(
                    softcap_graph,
                    torch.bfloat16,
                    torch.device("cuda"),
                    128,
                    1024,
                    32,
                    128,
                )
            )

        graph = torch.fx.Graph()
        score = graph.placeholder("score")
        for index in range(4):
            graph.placeholder(f"unused{index}")
        graph.output(
            graph.call_function(
                torch.ops.aten.tanh.default,
                (graph.call_function(torch.ops.aten.add.Tensor, (score, 1.0)),),
            )
        )
        arbitrary_tanh_graph = torch.fx.GraphModule({}, graph)
        self.assertFalse(_score_graph_is_softcap(arbitrary_tanh_graph))

        graph = torch.fx.Graph()
        score = graph.placeholder("score")
        for index in range(4):
            graph.placeholder(f"unused{index}")
        divided = graph.call_function(torch.ops.aten.div.Tensor, (score, 50.0))
        tanh = graph.call_function(torch.ops.aten.tanh.default, (divided,))
        graph.output(graph.call_function(torch.ops.aten.mul.Tensor, (tanh, 20.0)))
        mismatched_cap_graph = torch.fx.GraphModule({}, graph)
        self.assertFalse(_score_graph_is_softcap(mismatched_cap_graph))


if __name__ == "__main__":
    run_tests()
