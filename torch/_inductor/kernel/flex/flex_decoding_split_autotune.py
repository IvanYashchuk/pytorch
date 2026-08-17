# mypy: allow-untyped-defs

import operator

import sympy

import torch
from torch._inductor.virtualized import V

from ... import config
from ...codegen.subgraph import SubgraphTemplate
from ...ir import FixedLayout, FlexibleLayout
from ...runtime.runtime_utils import ceildiv
from ...select_algorithm import autotune_select_algorithm
from .common import is_tensor_ir_node


flex_decode_split_template = SubgraphTemplate(name="flex_decode_split")

_RUBIN_FLEX_DECODE_SPLIT_KV_CAP = 16
_RUBIN_FLEX_DECODE_PROXY_VERSION = "half_batch_full_capacity_v1"


def _get_split_kv_autotune_candidates(
    current_split: int,
    sm_count: int,
    base_ctas: int,
    max_generated_split: int,
) -> tuple[int, ...]:
    """Bounded split candidates around the default and dense upper-half waves."""
    if min(current_split, sm_count, base_ctas, max_generated_split) < 1:
        raise AssertionError("Flex decode split candidate inputs must be positive")

    if current_split > max_generated_split:
        generated = []
        split = 1
        while split < max_generated_split:
            generated.append(split)
            split *= 2
        generated.append(max_generated_split)
    else:
        generated = [
            max(1, current_split // 2),
            current_split * 2,
            current_split * 4,
            ceildiv(3 * sm_count, base_ctas),
        ]
    # Rubin decode latency can oscillate between adjacent high split counts;
    # testing only powers/multiples misses the measured 13/15 winners. Preserve
    # the cheap legacy neighborhood, then make the bounded upper half dense.
    generated.extend(
        range(
            max(1, ceildiv(max_generated_split, 2)),
            max_generated_split + 1,
        )
    )
    candidates = [current_split]
    for split in generated:
        split = min(split, max_generated_split)
        if split not in candidates:
            candidates.append(split)
    return tuple(candidates)


def _create_bucket_num_blocks_fake(count: int):
    def create_num_blocks_fake(node) -> torch.Tensor:
        size = V.graph.sizevars.optimization_hints(node.get_size())
        values = torch.zeros(
            size,
            dtype=node.get_dtype(),
            device=node.get_device(),
        )
        # Batch shapes are commonly capture buckets, while the live prefix can
        # begin just above half full. This proxy avoids tuning only for the
        # near-full endpoint, whose best split can cliff at the bucket boundary.
        values[: ceildiv(size[0], 2)].fill_(count)
        return values

    return create_num_blocks_fake


def _create_bounded_indices_fake(start: int, physical_blocks: int):
    def create_indices(node) -> torch.Tensor:
        size = V.graph.sizevars.optimization_hints(node.get_size())
        indices = torch.arange(
            start,
            start + size[-1],
            dtype=node.get_dtype(),
            device=node.get_device(),
        )
        indices = indices.remainder(max(physical_blocks, 1))
        return indices.expand(size).contiguous()

    return create_indices


def _graph_module_cache_key(graph_module: torch.fx.GraphModule) -> str | None:
    from ...codecache import BypassFxGraphCache, FxGraphCachePickler

    if any(
        isinstance(value, torch.Tensor) and value.numel() > 1024
        for _, value in graph_module.named_parameters(remove_duplicate=False)
    ) or any(
        isinstance(value, torch.Tensor) and value.numel() > 1024
        for _, value in graph_module.named_buffers(remove_duplicate=False)
    ):
        return None
    try:
        return FxGraphCachePickler(graph_module).get_hash(graph_module)
    except BypassFxGraphCache:
        return None


def _current_flex_decode_live_outputs() -> tuple[int, ...]:
    current_node = V.graph.current_node
    if current_node is None:
        return (0, 1)

    used_indices = set()
    for user in current_node.users:
        if len(user.users) == 0:
            continue
        if (
            user.op == "call_function"
            and user.target is operator.getitem
            and len(user.args) >= 2
            and user.args[1] in (0, 1)
        ):
            used_indices.add(user.args[1])
        else:
            return (0, 1)
    return tuple(sorted(used_indices)) or (0, 1)


def _mask_batch_matches_query(query_batch: int, mask_nodes) -> bool:
    return all(
        int(node.get_size()[0]) == query_batch
        for node in mask_nodes
        if node is not None
    )


def _autotune_rubin_flex_decode_split_kv(
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
    sparse_q_block_size,
    sparse_kv_block_size,
    configs,
) -> int | None:
    """Tune SPLIT_KV over complete decode-and-combine subgraphs on Rubin."""
    if (
        not config.max_autotune
        or query.get_device().type != "cuda"
        or torch.version.hip is not None
        or torch.cuda.get_device_capability(query.get_device()) != (10, 7)
    ):
        return None

    shapes = (
        query.get_size()
        + query.get_stride()
        + key.get_size()
        + key.get_stride()
        + value.get_size()
        + value.get_stride()
    )
    if not all(isinstance(value, (int, sympy.Integer)) for value in shapes):
        return None
    if not all(
        is_tensor_ir_node(buffer)
        for buffer in (*score_mod_other_buffers, *mask_mod_other_buffers)
    ):
        return None

    B, Hq, seq_len_q, _ = map(int, query.get_size())
    Hkv = int(value.get_size()[1])
    seq_len_kv = int(value.get_size()[2])
    gqa_shared_heads = int(kernel_options["GQA_SHARED_HEADS"])
    block_m = int(kernel_options["BLOCK_M"])
    num_block_m = ceildiv(seq_len_q * gqa_shared_heads, block_m)
    if num_block_m != 1:
        return None

    forward_mask_nodes = [
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks if has_full_blocks else None,
        full_kv_indices if has_full_blocks else None,
    ]
    tensor_nodes = [
        query,
        key,
        value,
        *(node for node in forward_mask_nodes if node is not None),
        *score_mod_other_buffers,
        *mask_mod_other_buffers,
    ]
    if not all(is_tensor_ir_node(node) for node in tensor_nodes):
        return None
    for node in tensor_nodes:
        node_shape = node.get_size() + node.get_stride()
        if not all(isinstance(value, (int, sympy.Integer)) for value in node_shape):
            return None
    if not _mask_batch_matches_query(B, forward_mask_nodes):
        return None

    sparse_kv_block_size = int(sparse_kv_block_size)
    partial_capacity = int(kv_indices.get_size()[-1])
    full_capacity = int(full_kv_indices.get_size()[-1]) if has_full_blocks else 0
    physical_sparse_blocks = ceildiv(seq_len_kv, sparse_kv_block_size)
    if has_full_blocks and partial_capacity + full_capacity > 0:
        partial_count = min(
            partial_capacity,
            ceildiv(
                physical_sparse_blocks * partial_capacity,
                partial_capacity + full_capacity,
            ),
        )
        full_count = min(full_capacity, physical_sparse_blocks - partial_count)
        partial_count = min(
            partial_capacity,
            physical_sparse_blocks - full_count,
        )
    else:
        partial_count = min(partial_capacity, physical_sparse_blocks)
        full_count = 0

    input_nodes = [query, key, value]
    mask_input_positions: list[int | None] = []
    input_gen_fns = {}
    for mask_index, node in enumerate(forward_mask_nodes):
        if node is None:
            mask_input_positions.append(None)
            continue
        mask_input_positions.append(len(input_nodes))
        input_nodes.append(node)
        if mask_index == 0:
            input_gen_fns[len(input_nodes) - 1] = _create_bucket_num_blocks_fake(
                partial_count
            )
        elif mask_index == 1:
            input_gen_fns[len(input_nodes) - 1] = _create_bounded_indices_fake(
                0, physical_sparse_blocks
            )
        elif mask_index == 2:
            input_gen_fns[len(input_nodes) - 1] = _create_bucket_num_blocks_fake(
                full_count
            )
        else:
            input_gen_fns[len(input_nodes) - 1] = _create_bounded_indices_fake(
                partial_count, physical_sparse_blocks
            )

    score_capture_start = len(input_nodes)
    input_nodes.extend(score_mod_other_buffers)
    mask_capture_start = len(input_nodes)
    input_nodes.extend(mask_mod_other_buffers)

    active_sparse_blocks = min(
        physical_sparse_blocks,
        partial_capacity + full_capacity,
    )
    max_kv_work = min(
        seq_len_kv,
        active_sparse_blocks * sparse_kv_block_size,
    )
    if "BLOCK_N" in kernel_options:
        min_block_n = min(int(kernel_options["BLOCK_N"]), sparse_kv_block_size)
    else:
        min_block_n = min(
            min(int(conf.block_n), sparse_kv_block_size) for conf in configs
        )
    max_useful_splits = max(1, ceildiv(max_kv_work, min_block_n))
    max_generated_split = min(_RUBIN_FLEX_DECODE_SPLIT_KV_CAP, max_useful_splits)

    current_split = int(kernel_options["SPLIT_KV"])
    properties = torch.cuda.get_device_properties(query.get_device())
    sm_count = int(properties.multi_processor_count)
    split_candidates = _get_split_kv_autotune_candidates(
        current_split,
        sm_count,
        num_block_m * B * Hkv,
        max_generated_split,
    )
    if len(split_candidates) == 1:
        return None

    live_outputs = _current_flex_decode_live_outputs()
    if live_outputs == (1,):
        output_shape = [B, Hq, seq_len_q]
        output_dtype = torch.float32
    else:
        output_shape = [B, Hq, seq_len_q, int(value.get_size()[-1])]
        output_dtype = query.get_dtype()
    output_layout = FixedLayout(
        query.get_device(),
        output_dtype,
        output_shape,
        FlexibleLayout.contiguous_strides(output_shape),
    )
    score_capture_count = len(score_mod_other_buffers)
    mask_capture_count = len(mask_mod_other_buffers)
    mask_graph = block_mask[-1].graph_module
    score_graph_hash = _graph_module_cache_key(score_mod_graph)
    mask_graph_hash = _graph_module_cache_key(mask_graph)
    if score_graph_hash is None or mask_graph_hash is None:
        return None
    choices = []

    from torch._dispatch.python import enable_python_dispatcher
    from torch.fx.experimental.proxy_tensor import make_fx

    from ...decomposition import select_decomp_table

    for split_kv in split_candidates:
        candidate_kernel_options = {
            name: int(value) if isinstance(value, sympy.Integer) else value
            for name, value in kernel_options.items()
        }
        candidate_kernel_options["BACKEND"] = "TRITON_DECODE"
        candidate_kernel_options["SPLIT_KV"] = split_kv

        def candidate(*args, kernel_options=candidate_kernel_options):
            candidate_query, candidate_key, candidate_value = args[:3]
            candidate_mask_nodes = [
                None if position is None else args[position]
                for position in mask_input_positions
            ]
            score_captures = args[
                score_capture_start : score_capture_start + score_capture_count
            ]
            mask_captures = args[
                mask_capture_start : mask_capture_start + mask_capture_count
            ]
            candidate_block_mask = (
                int(block_mask[0]),
                int(block_mask[1]),
                *candidate_mask_nodes,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                int(sparse_q_block_size),
                int(sparse_kv_block_size),
                mask_graph,
            )
            result = torch.ops.higher_order.flex_attention(
                candidate_query,
                candidate_key,
                candidate_value,
                score_mod_graph,
                candidate_block_mask,
                scale,
                kernel_options,
                tuple(score_captures),
                tuple(mask_captures),
            )
            if live_outputs == (0,):
                return result[0]
            if live_outputs == (1,):
                return result[1]
            return result[0], result[1]

        with enable_python_dispatcher():
            make_fx_candidate = make_fx(
                candidate,
                decomposition_table=select_decomp_table(),
                tracing_mode="symbolic",
            )

        def make_fx_graph(*args, make_fx_candidate=make_fx_candidate):
            graph_module = make_fx_candidate(*args)
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()
            return graph_module

        graph_hash = (
            f"sm_count={sm_count};split_kv={split_kv};live={live_outputs};"
            f"flex_search={config.max_autotune_flex_search_space};"
            f"coordinate_descent={config.coordinate_descent_tuning};"
            f"mask_proxy={_RUBIN_FLEX_DECODE_PROXY_VERSION};"
            f"kernel_options={sorted(candidate_kernel_options.items())};"
            f"score_mod={score_graph_hash};"
            f"mask_mod={mask_graph_hash}"
        )
        choice = flex_decode_split_template.generate(
            name=f"flex_decode_split_sm{sm_count}_s{split_kv}",
            input_nodes=input_nodes,
            layout=output_layout,
            make_fx_graph=make_fx_graph,
            description=f"SPLIT_KV={split_kv}, SM_COUNT={sm_count}",
            input_gen_fns=input_gen_fns,
            extra_hash_key=graph_hash,
            config_patches={"max_autotune": True},
        )
        # Explicit SPLIT_KV prevents recursive split tuning. The nested search
        # still picks the best decode-body config for this fixed split.
        choice.annotations["SPLIT_KV"] = split_kv
        choices.append(choice)

    _, selected_choice = autotune_select_algorithm(
        "flex_decode_split",
        choices,
        input_nodes,
        output_layout,
        input_gen_fns=input_gen_fns,
        return_choice_only=True,
    )
    return int(selected_choice.annotations["SPLIT_KV"])
