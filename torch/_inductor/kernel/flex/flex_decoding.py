# mypy: allow-untyped-defs
"""Triton Implementation of the flex_attention Kernel for short query length (FlexDecoding)"""

import logging
from typing import Any

import sympy

import torch
from torch._dynamo.device_interface import get_interface_for_device
from torch._inductor.virtualized import ops, V
from torch.utils._sympy.functions import FloorDiv, Max, Min, Mod

from ... import ir
from ...ir import FixedLayout, FlexibleLayout
from ...lowering import empty, empty_strided, lowerings
from ...runtime.runtime_utils import ceildiv, is_power_of_2, next_power_of_2
from ...select_algorithm import (
    autotune_select_algorithm,
    SymbolicGridFn,
    TritonTemplate,
)
from ...utils import can_use_tma
from .common import (
    _flex_kernel_options_example,
    _flex_kernel_tuning_options,
    can_skip_boundary_checks,
    create_indices_fake,
    create_num_blocks_fake_generator,
    freeze_irnodes,
    get_fwd_subgraph_outputs,
    is_tensor_ir_node,
    load_flex_template,
    maybe_realize,
    set_head_dim_values,
)
from .flex_decoding_split_autotune import _autotune_rubin_flex_decode_split_kv


aten = torch.ops.aten
prims = torch.ops.prims

log = logging.getLogger(__name__)


def raise_flex_decoding_kernel_options_error(
    kernel_options: dict[str, Any],
    sparse_q_block_size: int,
    sparse_kv_block_size: int,
) -> None:
    formated_kernel_options = ", ".join(
        f"{name}={kernel_options[name]}" for name in ("BLOCK_M", "BLOCK_N")
    )
    raise ValueError(
        "Invalid FlexAttention decode kernel options: Q and KV block sizes must "
        "be divisible by the selected tile sizes. Got "
        f"SPARSE_Q_BLOCK_SIZE={sparse_q_block_size}, "
        f"SPARSE_KV_BLOCK_SIZE={sparse_kv_block_size}, and "
        f"{formated_kernel_options}. "
        "Pass compatible values with kernel_options. Available decode tuning "
        f"options are {_flex_kernel_tuning_options('decode')}. For example: "
        f"{_flex_kernel_options_example('decode')}. If you did not pin "
        "these options, compiling with mode='max-autotune-no-cudagraphs' "
        "can also fix this by trying more FlexAttention configs."
    )


def _use_flex_decoding(query, kv_indices, value, kernel_options, enable_gqa) -> bool:
    """Decide which kernel to use, return true if use flex decoding kernel.
    Note:
       Since the number of splits is calculated based on the number of batch and head dims
       we need to ensure that the batch and head dims are statically known. Otherwise we just
       use the main flex_attention kernel.
    """
    force_flex = kernel_options.get("FORCE_USE_FLEX_ATTENTION", False)

    # Decode eligibility is an optimization choice, not a user-visible contract,
    # so every predicate below uses guard_or_false (case 2 of Note [guard_or_]:
    # the program behaves equivalently whether we pick decode or the general
    # kernel). guard_or_false returns the real result when ShapeEnv can prove or
    # guard it, and conservatively returns False for an unprovable unbacked size.
    # We deliberately do NOT use torch._check here: failing to prove eligibility
    # must fall back to the general FlexAttention kernel, never impose a runtime
    # constraint on the user's shapes. We also do NOT use guard_or_true: an
    # unknown predicate should disable decode, not silently enable it.
    short_query_length = V.graph.sizevars.guard_or_false(
        sympy.Lt(query.get_size()[-2], 128)
    )
    non_zero_length = V.graph.sizevars.guard_or_false(sympy.Gt(query.get_size()[-2], 0))
    static_batch = isinstance(query.get_size()[0], (int, sympy.Integer))
    static_num_heads = isinstance(query.get_size()[1], (int, sympy.Integer))
    if enable_gqa:
        # in the current flex decoding triton kernel, grouped query heads for the
        # same kv head are handled by the same block. So it's hard to support different
        # kv num blocks for grouped query heads. We just fall back to main flex_attention
        # kernel where each query head is handled by a separate block.
        valid_block_mask_num_heads = V.graph.sizevars.guard_or_false(
            sympy.Eq(kv_indices.get_size()[1], 1)
        )
    else:
        valid_block_mask_num_heads = V.graph.sizevars.guard_or_false(
            sympy.Or(
                sympy.Eq(kv_indices.get_size()[1], 1),
                sympy.Eq(kv_indices.get_size()[1], query.get_size()[1]),
            )
        )

    Hq = query.get_size()[1]
    Hkv = value.get_size()[1]
    ratio = FloorDiv(Hq, Hkv)

    pw_of_two = V.graph.sizevars.guard_or_false(
        sympy.And(sympy.Gt(ratio, 0), sympy.Eq(ratio & (ratio - 1), 0))
    )

    out = (
        not force_flex
        and not kernel_options.get("OUTPUT_MAX", False)
        and short_query_length
        and static_batch
        and static_num_heads
        and non_zero_length
        and valid_block_mask_num_heads
        and pw_of_two
    )
    log.debug(
        "Use flex decoding %s, force_flex_attention=%s, short_query_length=%s, static_batch=%s, static_num_heads=%s",
        out,
        force_flex,
        short_query_length,
        static_batch,
        static_num_heads,
    )
    return out


@SymbolicGridFn
def flex_decoding_grid(batch_size, kv_heads, gqa_group_size, seq_len_q, d_model, meta):
    """How is this kernel parallelized?
    We create a grid of (batch_size * kv_heads, SPLIT_KV, 1)
    Each block is responsible for iterating over blocks of keys and values calculating
    the local output for their tile of keys and values over all full length of query.
    groups of SPLIT_KV blocks then combine their output to produce the final result.
    """

    BLOCK_M = meta["BLOCK_M"]
    num_block_m = ceildiv(seq_len_q * gqa_group_size, BLOCK_M)
    return (num_block_m, batch_size * kv_heads, meta["SPLIT_KV"])


flex_decoding_template = TritonTemplate(
    name="flex_decoding",
    grid=flex_decoding_grid,
    source=load_flex_template("flex_decode")
    + load_flex_template("utilities")
    + load_flex_template("common"),
    always_freeze_layout=True,
)


def get_split_k(
    B: int,
    H: int,
    max_kv_work: int,
    num_block_m: int,
    device: torch.device,
) -> int:
    device_interface = get_interface_for_device(device.type)
    properties = device_interface.get_device_properties(device)
    if device.type == "xpu":
        num_SM = properties.gpu_subslice_count
        is_rubin = False
    else:
        num_SM = properties.multi_processor_count
        is_rubin = (
            device.type == "cuda"
            and torch.version.hip is None
            and (properties.major, properties.minor) == (10, 7)
        )
    bh = max(B * H, 1)  # NOTE: Handle B*h=0 case
    if not isinstance(bh, (int, sympy.Integer)):
        raise AssertionError("B and H must be concrete integers")
    split_k = max(num_SM // bh * 2, 1)
    # On Rubin Q1 decode, measured split counts up to six can undershoot two waves.
    # Use block-mask capacity as a static work proxy; short work can regress.
    if (
        is_rubin
        and isinstance(num_block_m, (int, sympy.Integer))
        and num_block_m == 1
        and split_k <= 6
        and bh < num_SM * 2
        and isinstance(max_kv_work, (int, sympy.Integer))
        and max_kv_work >= split_k * 4096
    ):
        split_k = ceildiv(num_SM, bh) * 2
    # TODO: workload evening at runtime for splits fully masked out.

    return split_k


def _get_rubin_flex_decode_runtime_split_policy(
    B: int,
    Hq: int,
    Hkv: int,
    seq_len_q: int,
    qk_head_dim: int,
    v_head_dim: int,
    dtype: torch.dtype,
    sparse_kv_block_size: int,
    max_kv_work: int,
    num_block_m: int,
    device: torch.device,
) -> tuple[int, tuple[tuple[int, int, int], ...]] | None:
    """Select the measured long-context Rubin GQA decode split surface."""
    values = (
        B,
        Hq,
        Hkv,
        seq_len_q,
        qk_head_dim,
        v_head_dim,
        sparse_kv_block_size,
        max_kv_work,
        num_block_m,
    )
    if not all(isinstance(value, (int, sympy.Integer)) for value in values):
        return None
    if (
        device.type != "cuda"
        or torch.version.hip is not None
        or int(B) not in (64, 128)
        or (int(Hq), int(Hkv)) != (32, 2)
        or int(seq_len_q) != 1
        or (int(qk_head_dim), int(v_head_dim)) != (256, 256)
        or dtype != torch.bfloat16
        or int(sparse_kv_block_size) != 64
        or int(max_kv_work) < 131072
        or int(num_block_m) != 1
    ):
        return None
    properties = get_interface_for_device(device.type).get_device_properties(device)
    if (properties.major, properties.minor) != (
        10,
        7,
    ) or properties.multi_processor_count not in (212, 216):
        return None
    sm_count = properties.multi_processor_count
    if sm_count == 212 and int(B) == 64:
        return (
            16,
            (
                (18, 64, 14),
                (20, 64, 13),
                (22, 64, 12),
                (24, 64, 11),
                (26, 64, 10),
                (28, 64, 9),
                (33, 64, 16),
                (39, 64, 14),
                (44, 64, 13),
                (55, 64, 5),
                (63, 64, 13),
            ),
        )
    if sm_count == 212:
        return (
            16,
            (
                (24, 64, 9),
                (32, 64, 16),
                (38, 64, 7),
                (44, 64, 13),
                (47, 64, 6),
                (52, 64, 16),
                (58, 64, 14),
                (80, 64, 16),
                (111, 64, 5),
                (127, 64, 14),
            ),
        )
    if int(B) == 64:
        return (
            16,
            (
                (18, 64, 15),
                (19, 64, 14),
                (21, 64, 13),
                (23, 64, 12),
                (24, 64, 11),
                (27, 64, 10),
                (30, 64, 9),
                (34, 64, 16),
                (39, 64, 15),
                (42, 64, 7),
                (44, 64, 13),
                (55, 64, 5),
                (63, 64, 13),
            ),
        )
    return (
        16,
        (
            (24, 64, 9),
            (34, 64, 16),
            (38, 64, 7),
            (43, 64, 13),
            (47, 64, 6),
            (53, 64, 11),
            (57, 64, 10),
            (63, 64, 14),
            (80, 64, 16),
            (111, 64, 5),
            (127, 64, 14),
        ),
    )


def _create_runtime_split_band_buffer(
    kv_num_blocks,
    full_kv_num_blocks,
    has_full_blocks: bool,
    runtime_default: int,
    runtime_split_bands: tuple[tuple[int, int, int], ...],
    sparse_tiles_per_block: int,
):
    """Evaluate a runtime split policy and cap it at each row's tile work."""
    kv_num_blocks_loader = kv_num_blocks.make_loader()
    full_kv_num_blocks_loader = (
        full_kv_num_blocks.make_loader() if has_full_blocks else None
    )
    dtype = kv_num_blocks.get_dtype()
    sparse_batch = V.graph.sizevars.guard_int(kv_num_blocks.get_size()[0])

    def inner_fn(index):
        sparse_z, sparse_h, sparse_m = index
        effective_split = ops.constant(runtime_default, dtype)
        for batch_threshold, min_blocks, split in runtime_split_bands:
            load_index = (batch_threshold % sparse_batch, sparse_h, sparse_m)
            num_blocks = kv_num_blocks_loader(load_index)
            if full_kv_num_blocks_loader is not None:
                num_blocks = ops.maximum(
                    num_blocks, full_kv_num_blocks_loader(load_index)
                )
            effective_split = ops.where(
                ops.ge(num_blocks, ops.constant(min_blocks, dtype)),
                ops.constant(split, dtype),
                effective_split,
            )
        row_index = (sparse_z, sparse_h, sparse_m)
        row_blocks = kv_num_blocks_loader(row_index)
        if full_kv_num_blocks_loader is not None:
            row_blocks = ops.maximum(row_blocks, full_kv_num_blocks_loader(row_index))
        row_split_limit = ops.maximum(
            ops.mul(row_blocks, ops.constant(sparse_tiles_per_block, dtype)),
            ops.constant(1, dtype),
        )
        return ops.minimum(effective_split, row_split_limit)

    result = ir.TensorBox.create(
        ir.Pointwise(
            device=kv_num_blocks.get_device(),
            dtype=dtype,
            inner_fn=inner_fn,
            ranges=kv_num_blocks.get_size(),
        )
    )
    result.realize()
    return result


def _mask_inactive_runtime_splits(
    buffer,
    effective_split_kv,
    gqa_shared_heads,
    sparse_q_block_size,
    fill_value,
):
    """Mask scratch loads from runtime-inactive decode splits."""
    buffer_loader = buffer.make_loader()
    effective_split_loader = effective_split_kv.make_loader()
    effective_z, effective_h, effective_m = effective_split_kv.get_size()

    def inner_fn(index):
        effective_index = (
            Mod(index[0], effective_z),
            Mod(FloorDiv(index[2], gqa_shared_heads), effective_h),
            Mod(FloorDiv(index[3], sparse_q_block_size), effective_m),
        )
        active = ops.lt(
            ops.index_expr(index[1], torch.int32),
            effective_split_loader(effective_index),
        )
        return ops.masked(active, lambda: buffer_loader(index), fill_value)

    return ir.TensorBox.create(
        ir.Pointwise(
            device=buffer.get_device(),
            dtype=buffer.get_dtype(),
            inner_fn=inner_fn,
            ranges=buffer.get_size(),
        )
    )


def _get_split_policy_max_kv_work(
    partial_capacity: int,
    full_capacity: int | None,
    sparse_kv_block_size: int,
    seq_len_kv: int,
):
    max_kv_blocks = partial_capacity
    if full_capacity is not None:
        max_kv_blocks = Max(max_kv_blocks, full_capacity)
    return Min(max_kv_blocks * sparse_kv_block_size, seq_len_kv)


def create_flex_decoding_kernel(*args, **kwargs):
    """Flex decode lowering that is optimized for small Q_LEN and GQA packing"""
    (
        query,
        key,
        value,
        block_mask,
        scale,
        kernel_options,
        score_mod_subgraph,
        mask_mod_subgraph,
        score_mod_other_buffers,
        mask_mod_other_buffers,
        score_mod_graph,
    ) = args
    (
        _,  # q_length
        _,  # kv_length
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,  # full_kv_num_blocks,
        full_kv_indices,  # full_kv_indices,
        _,  # q_num_blocks
        _,  # q_indices
        _,  # full_q_num_blocks,
        _,  # full_q_indices,
        _,  # dq_write_order
        _,  # dq_write_order_full
        _,  # dq_kv_order
        _,  # dq_kv_order_spt
        SPARSE_Q_BLOCK_SIZE,
        SPARSE_KV_BLOCK_SIZE,
        _,
    ) = block_mask

    Bq, Hq, seq_len_q, qk_head_dim = query.get_size()
    Bkv, Hkv, seq_len_kv, v_head_dim = value.get_size()

    if not V.graph.sizevars.evaluate_expr(sympy.Eq(Bq, Bkv) | sympy.Eq(Bkv, 1)):
        raise AssertionError(
            f"Bq and Bkv must broadcastable. Got Bq={Bq} and Bkv={Bkv}"
        )

    B = Bq
    kernel_options = dict(kernel_options)
    # Mark symbols in custom kernel options as static shapes and add guards.
    kernel_options = {
        k: V.graph.sizevars.guard_int(v) if isinstance(v, sympy.Symbol) else v
        for k, v in kernel_options.items()
    }
    # Forward-prefixed options must be normalized before computing any layout,
    # grid, or safety predicate that they can affect.
    for name in list(kernel_options):
        if name.startswith("fwd_"):
            kernel_options[name[4:]] = kernel_options.pop(name)
    seq_q_divisible = can_skip_boundary_checks(seq_len_q, SPARSE_Q_BLOCK_SIZE)
    seq_kv_divisible = can_skip_boundary_checks(seq_len_kv, SPARSE_KV_BLOCK_SIZE)
    if seq_q_divisible and seq_kv_divisible:
        kernel_options.setdefault("IS_DIVISIBLE", True)
    else:
        kernel_options.setdefault("IS_DIVISIBLE", False)
    SPARSE_Q_BLOCK_SIZE = V.graph.sizevars.guard_int(SPARSE_Q_BLOCK_SIZE)
    SPARSE_KV_BLOCK_SIZE = V.graph.sizevars.guard_int(SPARSE_KV_BLOCK_SIZE)

    # Calculate GQA head sharing
    gqa_shared_heads = FloorDiv(Hq, Hkv)
    if not is_power_of_2(gqa_shared_heads):
        raise ValueError(
            "Number of shared query heads sharing the same KV head must be power of 2. "
        )
    kernel_options.setdefault("GQA_SHARED_HEADS", gqa_shared_heads)
    kernel_options.setdefault(
        "BLOCK_M",
        max(
            next_power_of_2(
                V.graph.sizevars.optimization_hint(seq_len_q) * gqa_shared_heads
            ),
            1 if query.get_device().type == "xpu" else 16,
        ),
    )
    block_m = kernel_options["BLOCK_M"]
    num_block_m = ceildiv(seq_len_q * gqa_shared_heads, block_m)

    # Determine if there are "full" blocks where we only need to apply score_mod, and can skip mask_mod
    has_full_blocks = full_kv_num_blocks is not None
    kernel_options.setdefault("HAS_FULL_BLOCKS", has_full_blocks)
    max_kv_work = _get_split_policy_max_kv_work(
        kv_indices.get_size()[-1],
        full_kv_indices.get_size()[-1] if has_full_blocks else None,
        SPARSE_KV_BLOCK_SIZE,
        seq_len_kv,
    )
    if not has_full_blocks:
        # Create a placeholder full block list in case it is empty
        full_kv_num_blocks, full_kv_indices = (
            empty(0, device=query.get_device()) for _ in range(2)
        )

    (
        query,
        key,
        value,
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
    ) = maybe_realize(
        [
            query,
            key,
            value,
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
        ]
    )
    score_mod_other_buffers = maybe_realize(score_mod_other_buffers)
    mask_mod_other_buffers = maybe_realize(mask_mod_other_buffers)

    freeze_irnodes(score_mod_other_buffers)
    freeze_irnodes(mask_mod_other_buffers)

    choices: list[Any] = []
    dtype = key.get_dtype()
    head_dim = V.graph.sizevars.guard_int(key.get_size()[-1])
    configs = V.choices.get_flex_decode_configs(
        head_dim,
        dtype,
        query.get_device().type,
        sparse_kv_block_size=SPARSE_KV_BLOCK_SIZE,
    )

    kernel_options.setdefault("SM_SCALE", scale)
    runtime_option_names = (
        "SPLIT_KV",
        "RUNTIME_SPLIT_KV",
        "RUNTIME_SPLIT_KV_DEFAULT",
        "RUNTIME_SPLIT_KV_BANDS",
        "RUNTIME_SPLIT_KV_LOW",
        "RUNTIME_SPLIT_KV_HIGH",
        "RUNTIME_SPLIT_KV_BATCH_THRESHOLD",
        "RUNTIME_SPLIT_KV_MIN_BLOCKS",
    )
    if not any(name in kernel_options for name in runtime_option_names):
        runtime_policy = _get_rubin_flex_decode_runtime_split_policy(
            B,
            Hq,
            Hkv,
            seq_len_q,
            qk_head_dim,
            v_head_dim,
            dtype,
            SPARSE_KV_BLOCK_SIZE,
            max_kv_work,
            num_block_m,
            query.get_device(),
        )
        if runtime_policy is not None:
            runtime_default, runtime_bands = runtime_policy
            kernel_options["RUNTIME_SPLIT_KV_DEFAULT"] = runtime_default
            kernel_options["RUNTIME_SPLIT_KV_BANDS"] = runtime_bands
    runtime_split_bands = kernel_options.get("RUNTIME_SPLIT_KV_BANDS")
    runtime_split_kv = bool(kernel_options.get("RUNTIME_SPLIT_KV", False)) or (
        runtime_split_bands is not None
    )
    split_kv_was_explicit = "SPLIT_KV" in kernel_options or runtime_split_kv
    if runtime_split_kv:
        runtime_split_capacity = 0
        if runtime_split_bands is not None:
            two_level_keys = (
                "RUNTIME_SPLIT_KV_LOW",
                "RUNTIME_SPLIT_KV_HIGH",
                "RUNTIME_SPLIT_KV_BATCH_THRESHOLD",
                "RUNTIME_SPLIT_KV_MIN_BLOCKS",
            )
            if any(key in kernel_options for key in two_level_keys):
                raise ValueError(
                    "runtime SPLIT_KV bands and two-level options are mutually exclusive"
                )
            runtime_default = kernel_options.get("RUNTIME_SPLIT_KV_DEFAULT")
            if not isinstance(runtime_default, (int, sympy.Integer)):
                raise ValueError("runtime SPLIT_KV default must be a concrete integer")
            runtime_default = int(runtime_default)
            if runtime_default < 1:
                raise ValueError("runtime SPLIT_KV default must be positive")
            if not isinstance(runtime_split_bands, (tuple, list)):
                raise ValueError("runtime SPLIT_KV bands must be a tuple or list")
            normalized_bands = []
            previous_batch_threshold = -1
            for band in runtime_split_bands:
                if not isinstance(band, (tuple, list)) or len(band) != 3:
                    raise ValueError(
                        "each runtime SPLIT_KV band must be "
                        "(batch_threshold, min_blocks, split)"
                    )
                if not all(isinstance(value, (int, sympy.Integer)) for value in band):
                    raise ValueError("runtime SPLIT_KV bands must contain integers")
                batch_threshold, min_blocks, split = map(int, band)
                if not 0 <= batch_threshold < int(B):
                    raise ValueError(
                        "runtime SPLIT_KV band threshold must index the static batch"
                    )
                if batch_threshold <= previous_batch_threshold:
                    raise ValueError(
                        "runtime SPLIT_KV batch thresholds must be strictly increasing"
                    )
                if min(min_blocks, split) < 1:
                    raise ValueError(
                        "runtime SPLIT_KV band split and work threshold must be positive"
                    )
                normalized_bands.append((batch_threshold, min_blocks, split))
                previous_batch_threshold = batch_threshold
            if not normalized_bands:
                raise ValueError("runtime SPLIT_KV bands must not be empty")
            runtime_split_bands = tuple(normalized_bands)
            runtime_split_capacity = max(
                runtime_default, *(band[2] for band in runtime_split_bands)
            )
            kernel_options["RUNTIME_SPLIT_KV"] = True
            kernel_options["RUNTIME_SPLIT_KV_DEFAULT"] = runtime_default
            kernel_options["RUNTIME_SPLIT_KV_BANDS"] = runtime_split_bands
        else:
            runtime_split_low = kernel_options.get("RUNTIME_SPLIT_KV_LOW")
            runtime_split_high = kernel_options.get("RUNTIME_SPLIT_KV_HIGH")
            runtime_batch_threshold = kernel_options.get(
                "RUNTIME_SPLIT_KV_BATCH_THRESHOLD"
            )
            runtime_min_blocks = kernel_options.get("RUNTIME_SPLIT_KV_MIN_BLOCKS")
            runtime_values = (
                runtime_split_low,
                runtime_split_high,
                runtime_batch_threshold,
                runtime_min_blocks,
            )
            if not all(
                isinstance(value, (int, sympy.Integer)) for value in runtime_values
            ):
                raise ValueError("runtime SPLIT_KV options must be concrete integers")
            runtime_split_low = int(runtime_split_low)
            runtime_split_high = int(runtime_split_high)
            runtime_batch_threshold = int(runtime_batch_threshold)
            runtime_min_blocks = int(runtime_min_blocks)
            if min(runtime_split_low, runtime_split_high, runtime_min_blocks) < 1:
                raise ValueError(
                    "runtime SPLIT_KV counts and work threshold must be positive"
                )
            if not 0 <= runtime_batch_threshold < int(B):
                raise ValueError(
                    "runtime SPLIT_KV batch threshold must index the static batch"
                )
            runtime_split_capacity = max(runtime_split_low, runtime_split_high)
            kernel_options["RUNTIME_SPLIT_KV_LOW"] = runtime_split_low
            kernel_options["RUNTIME_SPLIT_KV_HIGH"] = runtime_split_high
            kernel_options["RUNTIME_SPLIT_KV_BATCH_THRESHOLD"] = runtime_batch_threshold
            kernel_options["RUNTIME_SPLIT_KV_MIN_BLOCKS"] = runtime_min_blocks
        if (
            "SPLIT_KV" in kernel_options
            and int(kernel_options["SPLIT_KV"]) != runtime_split_capacity
        ):
            raise ValueError(
                "SPLIT_KV must equal the largest runtime split for scratch ownership"
            )
        kernel_options["SPLIT_KV"] = runtime_split_capacity
    else:
        kernel_options.setdefault(
            "SPLIT_KV",
            get_split_k(B, Hkv, max_kv_work, num_block_m, query.get_device()),
        )
    split_kv = kernel_options["SPLIT_KV"]
    if not isinstance(split_kv, (int, sympy.Integer)) or int(split_kv) < 1:
        raise ValueError(f"SPLIT_KV must be a positive integer, got {split_kv!r}")
    kernel_options["SPLIT_KV"] = int(split_kv)

    effective_split_kv = None
    if runtime_split_bands is not None:
        if has_full_blocks and (
            kv_num_blocks.get_size() != full_kv_num_blocks.get_size()
        ):
            raise ValueError(
                "runtime SPLIT_KV bands require matching partial/full count shapes"
            )
        configured_block_n = kernel_options.get("BLOCK_N")
        if configured_block_n is None:
            min_block_n = min(
                min(conf.block_n, SPARSE_KV_BLOCK_SIZE) for conf in configs
            )
        elif isinstance(configured_block_n, (int, sympy.Integer)):
            min_block_n = int(configured_block_n)
        else:
            raise ValueError("runtime SPLIT_KV BLOCK_N must be a concrete integer")
        if min_block_n < 1 or SPARSE_KV_BLOCK_SIZE % min_block_n != 0:
            raise ValueError(
                "runtime SPLIT_KV requires BLOCK_N to divide the sparse KV block"
            )
        effective_split_kv = _create_runtime_split_band_buffer(
            kv_num_blocks,
            full_kv_num_blocks,
            has_full_blocks,
            kernel_options["RUNTIME_SPLIT_KV_DEFAULT"],
            runtime_split_bands,
            SPARSE_KV_BLOCK_SIZE // min_block_n,
        )

    set_head_dim_values(kernel_options, qk_head_dim, v_head_dim, V.graph.sizevars)

    if not split_kv_was_explicit:
        selected_split_kv = _autotune_rubin_flex_decode_split_kv(
            query,
            key,
            value,
            block_mask,
            scale,
            kernel_options,
            score_mod_graph,
            score_mod_other_buffers,
            mask_mod_other_buffers,
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
            has_full_blocks,
            SPARSE_Q_BLOCK_SIZE,
            SPARSE_KV_BLOCK_SIZE,
            configs,
        )
        if selected_split_kv is not None:
            kernel_options["SPLIT_KV"] = selected_split_kv

    # The selected split owns its complete scratch layout. Split autotuning
    # benchmarks candidate-private layouts before reaching this point.
    split_kv = kernel_options["SPLIT_KV"]
    buf_ACC_shape = [B, split_kv, Hq, seq_len_q, v_head_dim]
    buf_ML_shape = buf_ACC_shape[:-1]
    buf_M = empty_strided(
        buf_ML_shape,
        None,
        dtype=torch.float32,
        device=query.get_device(),
    )
    buf_L = empty_strided(
        buf_ML_shape,
        None,
        dtype=torch.float32,
        device=query.get_device(),
    )

    layout_acc = FixedLayout(
        query.get_device(),
        torch.float32,
        buf_ACC_shape,
        FlexibleLayout.contiguous_strides(buf_ACC_shape),
    )

    query = ir.ExternKernel.realize_input(query)
    stride_b, stride_hq, stride_seq_len_q, stride_qk_head_dim = query.get_stride()

    # Reshape query for GQA: [B, Hq, Mq, D] -> [B, Hkv, G, Mq, D]
    gqa_query_shape = (B, Hkv, gqa_shared_heads, seq_len_q, qk_head_dim)
    gqa_query_stride = (
        stride_b,
        stride_hq * gqa_shared_heads,
        stride_hq,
        stride_seq_len_q,
        stride_qk_head_dim,
    )
    query = lowerings[aten.as_strided](query, gqa_query_shape, gqa_query_stride)

    kernel_options.setdefault(
        "SAFE_M_BOUNDARY",
        Mod(seq_len_q * gqa_shared_heads, block_m) == 0,
    )
    # TODO: This feels sketchy
    kernel_options.setdefault("SAFE_N_BOUNDARY", True)
    original_kernel_options = kernel_options.copy()
    # Note, we don't need to pass in the captured buffers explicitly
    # because they're implicitly added by the score_mod function
    # We do need to explicitly pass it in for autotuning though.

    # Default config for warp specialization
    num_consumer_groups, num_buffers_warp_spec = 0, 0
    invalid_block_options: dict[str, Any] | None = None

    for conf in configs:
        cur_kernel_options = original_kernel_options.copy()
        # Remove prefix for forward kernels options and delete backward kernel options.
        for k in list(cur_kernel_options.keys()):
            if k.startswith("fwd_"):
                v = cur_kernel_options.pop(k)
                cur_kernel_options[k[4:]] = v
            if k.startswith("bwd_"):
                cur_kernel_options.pop(k)
        # Performance tuning
        cur_kernel_options.setdefault(
            "BLOCK_N", min(conf.block_n, SPARSE_KV_BLOCK_SIZE)
        )
        cur_kernel_options.setdefault("SPARSE_Q_BLOCK_SIZE", SPARSE_Q_BLOCK_SIZE)
        cur_kernel_options.setdefault("SPARSE_KV_BLOCK_SIZE", SPARSE_KV_BLOCK_SIZE)
        cur_kernel_options.setdefault("num_warps", conf.num_warps)
        cur_kernel_options.setdefault("num_stages", conf.num_stages)

        if (
            cur_kernel_options["SPARSE_Q_BLOCK_SIZE"] % cur_kernel_options["BLOCK_M"]
            != 0
            or cur_kernel_options["SPARSE_KV_BLOCK_SIZE"]
            % cur_kernel_options["BLOCK_N"]
            != 0
        ):
            invalid_block_options = cur_kernel_options
            if len(configs) == 1:
                raise_flex_decoding_kernel_options_error(
                    cur_kernel_options,
                    SPARSE_Q_BLOCK_SIZE,
                    SPARSE_KV_BLOCK_SIZE,
                )
            continue

        if cur_kernel_options.get("num_consumer_groups", False):
            cur_kernel_options.setdefault("num_consumer_groups", num_consumer_groups)
            cur_kernel_options.setdefault(
                "num_buffers_warp_spec", num_buffers_warp_spec
            )

        # Intel GPU enables TMA by default
        cur_kernel_options.setdefault("USE_TMA", bool(torch.xpu.is_available()))

        if cur_kernel_options["USE_TMA"] and not can_use_tma(query, key, value):
            cur_kernel_options["USE_TMA"] = False

        # Add ROCm-specific parameters if they exist in the config
        for attrib in ["kpack", "matrix_instr_nonkdim", "waves_per_eu"]:
            if hasattr(conf, attrib):
                cur_kernel_options[attrib] = getattr(conf, attrib)

        template_input_nodes = [
            query,
            key,
            value,
            buf_M,
            buf_L,
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
        ]
        if effective_split_kv is not None:
            template_input_nodes.append(effective_split_kv)

        flex_decoding_template.maybe_append_choice(
            choices=choices,
            input_nodes=template_input_nodes,
            layout=layout_acc,
            subgraphs=[
                score_mod_subgraph,
                mask_mod_subgraph,
            ],
            mutated_inputs=[buf_M, buf_L],
            call_sizes=query.get_size(),
            **cur_kernel_options,
        )

    if not choices and invalid_block_options is not None:
        raise_flex_decoding_kernel_options_error(
            invalid_block_options,
            SPARSE_Q_BLOCK_SIZE,
            SPARSE_KV_BLOCK_SIZE,
        )

    inputs_for_flex_decoding = (
        [
            query,
            key,
            value,
            buf_M,
            buf_L,
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
        ]
        + ([effective_split_kv] if effective_split_kv is not None else [])
        + list(score_mod_other_buffers)
        + list(mask_mod_other_buffers)
    )

    # Producer configs are tuned against their established full-capacity fake
    # inputs. The outer complete-graph selector independently benchmarks
    # half-full capture-bucket occupancy while using these same producer configs.
    input_gen_fns = {
        5: create_num_blocks_fake_generator(kv_indices),
        6: create_indices_fake,
        7: create_num_blocks_fake_generator(full_kv_indices),
        8: create_indices_fake,
    }
    if effective_split_kv is not None:

        def create_effective_split_fake(x):
            size = V.graph.sizevars.optimization_hints(x.get_size())
            return torch.full(
                size,
                kernel_options["RUNTIME_SPLIT_KV_DEFAULT"],
                dtype=x.get_dtype(),
                device=x.get_device(),
            )

        input_gen_fns[9] = create_effective_split_fake

    buf_ACC, _ = autotune_select_algorithm(
        "flex_decoding",
        choices,
        # Autotuning materializes benchmark tensors. Scalar shape captures stay
        # in subgraph_inps below for dependency tracking and codegen.
        [x for x in inputs_for_flex_decoding if is_tensor_ir_node(x)],
        layout_acc,
        input_gen_fns=input_gen_fns,
    )

    # need subgraph inputs and outputs to analyze all symints used in flex attention
    buf_ACC.data.data.subgraph_inps = list(score_mod_other_buffers) + list(
        mask_mod_other_buffers
    )
    buf_ACC.data.data.subgraph_outs = get_fwd_subgraph_outputs(
        score_mod_subgraph, mask_mod_subgraph
    )

    # Reduction

    if effective_split_kv is not None:
        buf_M = _mask_inactive_runtime_splits(
            buf_M,
            effective_split_kv,
            gqa_shared_heads,
            SPARSE_Q_BLOCK_SIZE,
            -float("inf"),
        )
        buf_L = _mask_inactive_runtime_splits(
            buf_L,
            effective_split_kv,
            gqa_shared_heads,
            SPARSE_Q_BLOCK_SIZE,
            0.0,
        )
        buf_ACC = _mask_inactive_runtime_splits(
            buf_ACC,
            effective_split_kv,
            gqa_shared_heads,
            SPARSE_Q_BLOCK_SIZE,
            0.0,
        )

    g_M = lowerings[aten.max](buf_M, dim=1, keepdim=True)[0]
    # See [Note] Handle fully masked out rows:
    # g_M Is the global max among split kv blocks.
    masked_rows = lowerings[aten.eq](g_M, -float("inf"))
    adj_M = lowerings[aten.sub](buf_M, g_M)
    adj_M = lowerings[aten.where](masked_rows, 0, adj_M)
    alpha = lowerings[aten.exp2](adj_M)

    buf_L = lowerings[aten.mul](buf_L, alpha)
    g_L = lowerings[aten.sum](buf_L, axis=1)
    masked_rows_squeezed = lowerings[aten.squeeze](masked_rows, dim=1)
    g_L = lowerings[aten.where](masked_rows_squeezed, 1.0, g_L)
    logsumexp = lowerings[aten.log2](g_L)
    logsumexp = lowerings[aten.add](logsumexp, lowerings[aten.squeeze](g_M, dim=1))

    alpha_unseq = lowerings[aten.unsqueeze](alpha, 4)
    buf_ACC = lowerings[aten.mul](buf_ACC, alpha_unseq)
    output = lowerings[aten.sum](buf_ACC, axis=1)
    L_unseq = lowerings[aten.unsqueeze](g_L, 3)
    output = lowerings[aten.div](output, L_unseq)
    output = lowerings[prims.convert_element_type](output, query.get_dtype())

    return (
        output,
        logsumexp,
    )
