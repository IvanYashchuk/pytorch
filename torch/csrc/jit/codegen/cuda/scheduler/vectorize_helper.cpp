#include <torch/csrc/jit/codegen/cuda/scheduler/vectorize_helper.h>

#include <torch/csrc/jit/codegen/cuda/compute_at_map.h>
#include <torch/csrc/jit/codegen/cuda/contiguity.h>
#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/lower_divisible_split.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/registry.h>

#include <c10/util/irange.h>

#include <unordered_set>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace vectorize_helper {

// Grab all values and expressions used to make the merged_domain and remove
// them from the fusion
void cleanUpInnermostMergedDomains(
    const std::vector<IterDomain*>& root_domain,
    IterDomain* merged_domain) {
  TORCH_INTERNAL_ASSERT(merged_domain != nullptr);
  TORCH_INTERNAL_ASSERT(!root_domain.empty());

  std::unordered_set<Val*> root_set({root_domain.begin(), root_domain.end()});

  auto vals = DependencyCheck::getAllValsBetween(root_set, {merged_domain});

  for (auto it = vals.rbegin(); it != vals.rend(); ++it) {
    TORCH_INTERNAL_ASSERT((*it)->isA<IterDomain>());
    auto id = (*it)->as<IterDomain>();
    if (root_set.find(id) != root_set.end()) {
      continue;
    }
    Fusion* fusion = id->container()->as<Fusion>();
    auto id_def = id->definition();
    TORCH_INTERNAL_ASSERT(
        id_def->isA<Merge>(),
        "Invalid ID: ",
        id->toString(),
        ". Expected definition of a Merge expression: ",
        (id_def != nullptr ? id_def->toString() : "nullptr"));
    fusion->removeExpr(id_def);
    fusion->removeVal(id);
  }
}

// Merge innermost domains for finding the widest vectorizable
// size. Return the merged domain or nullptr if no merge is done.
IterDomain* mergeInnermostDomains(
    const std::vector<IterDomain*>& domain,
    int num_merged_domains) {
  const auto ndims = domain.size();
  IterDomain* merged_id = nullptr;
  bool is_merge_done = false;
  for (const auto i : c10::irange(num_merged_domains)) {
    auto id = domain.at(ndims - 1 - i);
    // broadcast and trivial reductions are ignored
    if (id->isBroadcast() || id->isTrivialReduction()) {
      continue;
    }
    if (merged_id == nullptr) {
      merged_id = id;
    } else {
      auto id_inner = merged_id;
      auto id_outer = id;
      merged_id = IterDomain::merge(id_outer, id_inner);
      is_merge_done = true;
    }
  }
  return is_merge_done ? merged_id : nullptr;
}

size_t collectMaxVectorizeSizeWithContigMerge(
    TensorView* tv,
    IterDomain* leaf_merged_domain,
    size_t max_vector_size_in_byte,
    ExpressionEvaluator& expression_evaluator,
    DataType index_type) {
  // Maybe too conservative, but only handles fully contiguous tensors
  // TODO: Relax the contiguity constraint to be similar to that in index
  // computing. Just looking for all merged root domains in the right order,
  // all merged root dimensions are contiguous, all merged root dimensions are
  // next to eachother (exlcuding broadcast).
  if (std::any_of(
          tv->domain()->contiguity().begin(),
          tv->domain()->contiguity().end(),
          [](const auto contig) { return !contig; })) {
    return 1;
  }

  auto dtype_size = dataTypeSize(tv->dtype(), index_type);
  const size_t max_vector_size = max_vector_size_in_byte / dtype_size;

  // Assume no halo-related expression appears in the fusion. No
  // broadcast is merged, so indexability can be assumed to be true.
  //
  // This is expensive, as ContigIDs builds other things like CAMap,
  // HaloInfo, and ConcreteBroadcast info. We should explicitly build and reuse
  // these as they're compile time information.
  ContigIDs contigIds(
      {leaf_merged_domain},
      tv->getMaybeRFactorDomain(),
      tv->domain()->contiguity(),
      {},
      getAllDivisibleSplits(tv->fusion()),
      {},
      true);

  auto innermost_root_id = tv->getMaybeRFactorDomain().back();
  auto indexed_id = contigIds.rootToIndexedID().at(innermost_root_id);

  size_t merged_size = 1;
  // If the indexed ID is a contig merged domain, i.e., it is
  // different from innermost_root_id, we accumulate the extents of
  // all the root domains covered by the contig indexed ID. Otherwise,
  // just look at the extent of the innermost root ID.
  if (indexed_id != innermost_root_id) {
    const auto& within_root = contigIds.withinContigIDs().at(indexed_id);
    for (auto root_id : tv->getMaybeRFactorDomain()) {
      if (within_root.find(root_id) == within_root.end()) {
        continue;
      }
      auto maybe_dimension_size =
          expression_evaluator.evaluate(root_id->extent());
      TORCH_INTERNAL_ASSERT(
          maybe_dimension_size.has_value(),
          "Unknown extent of tv: ",
          tv->toString(),
          ", id: ",
          root_id->toString());
      merged_size *= maybe_dimension_size->as<int64_t>();
    }
  } else {
    auto maybe_dimension_size =
        expression_evaluator.evaluate(innermost_root_id->extent());
    TORCH_INTERNAL_ASSERT(
        maybe_dimension_size.has_value(),
        "Unknown extent of tv: ",
        tv->toString(),
        ", id: ",
        innermost_root_id->toString());
    merged_size = maybe_dimension_size->as<int64_t>();
  }

  size_t vector_size = 1;
  size_t next_vector_size = vector_size * 2;

  // Try until vector size exceeds the max allowed size
  while (next_vector_size <= max_vector_size) {
    if (merged_size % next_vector_size != 0) {
      break;
    }
    vector_size = next_vector_size;
    next_vector_size *= 2;
  }

  return vector_size;
}

//! Attempt to expand vectorized domains to contig merged domains. Break point
//! identifies the point in which you can't propagate contiguous merges. For
//! example in pointwise this is the point where we want to split the
//! parallelization to take advantage of broadcast, and for reduction
//! schedulers it's the point where we switch from a reduction domain to an
//! iter domain (or vice versa).
size_t expandVectorizationToContigMergedDomains(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    const std::vector<TensorView*> vectorizable_inputs_outputs,
    TensorView* reference_tv,
    int break_point,
    size_t default_word_size) {
  size_t max_expand_size = SchedulerRuntimeInfo::max_alignment_size_in_byte;
  size_t common_alignment_size =
      SchedulerRuntimeInfo::max_alignment_size_in_byte;

  for (auto inp_out : vectorizable_inputs_outputs) {
    auto dtype_size = dataTypeSize(
        inp_out->dtype(), indexModeToDtype(runtime_info.getIndexMode()));

    max_expand_size = std::min(
        max_expand_size,
        SchedulerRuntimeInfo::max_alignment_size_in_byte / dtype_size);
    max_expand_size = std::min(
        max_expand_size, runtime_info.getMaxVectorizableWidth(inp_out));
    common_alignment_size =
        std::min(common_alignment_size, runtime_info.getAlignmentSize(inp_out));
  }

  // If there's no possibility to increase vector size of provided tensors,
  // then don't bother doing a more complex analysis to try and do so, just
  // return early.
  if (max_expand_size == default_word_size) {
    return default_word_size;
  }

  auto ca_map = ComputeAtMap(fusion);

  // Merge the domains right of the break point
  const auto& ref_root = reference_tv->getMaybeRFactorDomain();
  const int num_merged_domains =
      static_cast<int>(ref_root.size()) - static_cast<int>(break_point);

  // No expansion with no merged domain
  if (num_merged_domains == 0) {
    return default_word_size;
  }

  // Merge the domains but don't modify TensorDomain
  auto merged_domain = mergeInnermostDomains(ref_root, num_merged_domains);

  // No expansion is done if no merge is done.
  if (merged_domain == nullptr) {
    return default_word_size;
  }

  // Find the vectorizable word size with the merged domains
  size_t word_size = collectMaxVectorizeSizeWithContigMerge(
      reference_tv,
      merged_domain,
      common_alignment_size,
      runtime_info.expressionEvaluator(),
      indexModeToDtype(runtime_info.getIndexMode()));

  cleanUpInnermostMergedDomains(ref_root, merged_domain);

  // Stop if the reference doesn't get a larger word size.
  if (word_size <= default_word_size) {
    return default_word_size;
  }

  // Check the other TVs and take the minimum of the valid word sizes
  for (const auto tv : vectorizable_inputs_outputs) {
    if (tv == reference_tv) {
      continue;
    }

    const auto& tv_root = tv->getMaybeRFactorDomain();

    int tv_num_merged_domains = 0;
    for (const auto i : c10::irange(num_merged_domains)) {
      if (i == tv_root.size()) {
        break;
      }
      auto ref_id = ref_root.at(ref_root.size() - 1 - i);
      IterDomain* tv_id = tv_root.at(tv_root.size() - 1 - i);
      // If not mapped, stop expanding.
      if (!ca_map.areMapped(ref_id, tv_id, IdMappingMode::EXACT)) {
        break;
      } else {
        ++tv_num_merged_domains;
      }
    }

    size_t tv_word_size = 1;
    if (tv_num_merged_domains > 1) {
      auto tv_merged_domain =
          mergeInnermostDomains(tv_root, tv_num_merged_domains);
      if (tv_merged_domain == nullptr) {
        tv_word_size = runtime_info.getInnerDimVectorizableWidth(tv);
      } else {
        tv_word_size = collectMaxVectorizeSizeWithContigMerge(
            tv,
            tv_merged_domain,
            common_alignment_size,
            runtime_info.expressionEvaluator(),
            indexModeToDtype(runtime_info.getIndexMode()));
        cleanUpInnermostMergedDomains(tv_root, tv_merged_domain);
      }
    } else {
      tv_word_size = runtime_info.getInnerDimVectorizableWidth(tv);
    }

    word_size = std::min(word_size, tv_word_size);
  }

  return word_size;
}

ContiguousInnerDimensionsMapper::ContiguousInnerDimensionsMapper(
    TensorView* reference,
    std::vector<IterDomain*> reference_ids,
    std::shared_ptr<const ComputeAtMap> ca_map)
    : ca_map_(ca_map) {
  FusionGuard fg(reference->fusion());
  // Check which domain of tensor view we should be looking at. All IDs must be
  // found either in the root domain, or the rfactor domain**.
  bool reference_is_rfactor = reference->hasRFactor() &&
      std::all_of(reference_ids.begin(),
                  reference_ids.end(),
                  [reference](IterDomain* id) {
                    return (
                        std::find(
                            reference->getMaybeRFactorDomain().begin(),
                            reference->getMaybeRFactorDomain().end(),
                            id) != reference->getMaybeRFactorDomain().end());
                  });

  if (!reference_is_rfactor) {
    TORCH_INTERNAL_ASSERT(
        std::all_of(
            reference_ids.begin(),
            reference_ids.end(),
            [reference](IterDomain* id) {
              return (
                  std::find(
                      reference->getRootDomain().begin(),
                      reference->getRootDomain().end(),
                      id) != reference->getRootDomain().end());
            }),
        "\nIterDomains passed in to ContiguousInnerDimensionsMapper passed in to ",
        "ContiguousInnerDimensionsMapper must either all exist in the root domain, or all exist ",
        "in the rfactor domain.\nReference: ",
        reference->toString());
  }

  // Ordering of dimensions is important in this analysis, if an ordering is
  // contiguous in the reference, but not the target tensor views, then we
  // cannot consider that a contiguous merge dimension for vectorization.
  if (reference_is_rfactor) {
    std::vector<IterDomain*> reordered_rfactor;
    for (auto id : reference->getMaybeRFactorDomain()) {
      if (std::find(reference_ids.begin(), reference_ids.end(), id) !=
          reference_ids.end()) {
        reordered_rfactor.push_back(id);
      } else if (!id->isBroadcast()) {
        // Ignore broadcasts in the reference. Otherwise, remove non-contiguous
        // IDs in the reference tensor as this is the contiguous mapper.
        reordered_rfactor.clear();
      }
    }

    projected_rfactor_ids_[reference] = reordered_rfactor;
    // Project reference IDs to root
    projectIdToRoot(reference, reordered_rfactor);
  } else {
    std::vector<IterDomain*> reordered_root;
    for (auto id : reference->getRootDomain()) {
      if (std::find(reference_ids.begin(), reference_ids.end(), id) !=
          reference_ids.end()) {
        reordered_root.push_back(id);
      } else if (!id->isBroadcast()) {
        // Ignore broadcasts in the reference. Otherwise, remove non-contiguous
        // IDs in the reference tensor as this is the contiguous mapper.
        reordered_root.clear();
      }
    }
    projected_root_ids_[reference] = reordered_root;
    // Project reference IDs to rfactor if necessary, otherwise function will
    // just pass through root to rfactor.
    projectIdToRFactor(reference, reordered_root);
  }
}

std::unordered_map<TensorView*, std::vector<IterDomain*>>::iterator
ContiguousInnerDimensionsMapper::projectIdToRoot(
    TensorView* ref,
    std::vector<IterDomain*> ids) {
  auto transform_exprs = StmtSort::getExprs(
      ref->fusion(),
      {ref->getRFactorDomain().begin(), ref->getRFactorDomain().end()});

  // Mapping from rfactor to root, reverse expressions
  std::reverse(transform_exprs.begin(), transform_exprs.end());

  for (const auto* expr : transform_exprs) {
    if (const Split* split = dynamic_cast<const Split*>(expr)) {
      // Initialize state
      auto find_outer_it = ids.begin();
      auto outer_pos = ids.size();
      auto find_inner_it = ids.begin();
      auto inner_pos = ids.size();

      // Removes all entries to the left of provided `it`, if `it` is not
      // ids.begin(). Updates all state of finding outer and inner in the ids
      // vector after erasing.
      auto clear_left_of = [&find_outer_it,
                            &outer_pos,
                            &find_inner_it,
                            &inner_pos,
                            &ids,
                            &split](decltype(find_outer_it) it) {
        if (it != ids.begin()) {
          ids.erase(ids.begin(), it);
        }

        // Set outer it and position
        find_outer_it = std::find(ids.begin(), ids.end(), split->outer());
        outer_pos = find_outer_it == ids.end()
            ? ids.size()
            : std::distance(ids.begin(), find_outer_it);

        // Set inner it and position
        find_inner_it = std::find(ids.begin(), ids.end(), split->inner());
        inner_pos = find_inner_it == ids.end()
            ? ids.size()
            : std::distance(ids.begin(), find_inner_it);
      };

      // Dry run to fill state
      clear_left_of(ids.begin());

      // Check if the domains out of the split are contiguous in the mapped
      // domain.
      if (find_outer_it == ids.end() && find_inner_it != ids.end()) {
        // Outer dimension was not found, but inner dimension was. Must assume
        // everything to the left of inner is not contiguously merged.
        //
        // Clear left of inner
        clear_left_of(find_inner_it);
      } else if (find_outer_it != ids.end() && find_inner_it == ids.end()) {
        // Inner dimension was not found, outer and anything left of outer are
        // definitely not contiguous.
        //
        // Clear outer and left of outer
        clear_left_of(find_outer_it + 1);
        continue;
      } else if (find_outer_it == ids.end() && find_inner_it == ids.end()) {
        // Nothing mapped, just continue
        continue;
      }

      if (find_outer_it != ids.end() && find_inner_it != ids.end()) {
        // Both outer and inner mapped.
        if (outer_pos >= inner_pos) {
          // Make sure outer is outside inner, otherwise neither could be part
          // of a continuous mapping. There are cases where we could have
          // reversible operations e.g.:
          //    [id{3} id{5} id{6}] -> merge(1, 0)
          // -> [id{5*3} id{6}] -> split(0, 5)
          // -> [id{5} id{3} id{6}] -> transpose(0, 1)
          // -> [id{3} id{5} id{6}]
          // However we don't try and capture cases like this correcly, we'd
          // just reduce this down to only the iter domain of size 6 mapping.
          //
          // Clear outer and left of outer
          clear_left_of(find_outer_it + 1);
          continue;
        }
      }

      // Find the position inner would have to have to be considered ordered
      // relative to outer
      auto pos_after_outer = outer_pos + 1;
      for (; pos_after_outer < ids.size(); pos_after_outer++) {
        if (ids[pos_after_outer]->isBroadcast()) {
          // Skip broadcast axes as they must not have been concretized in the
          // reference. We remove dimensions that underwent a concretization as
          // well as the dimensions to the left of that.
          continue;
        }
        break;
      }

      if (inner_pos != pos_after_outer) {
        // Nothing to the left of inner could be continuous.
        //
        // Clear left of inner
        clear_left_of(find_inner_it);
      }

      if (hasPartialExtent(split->inner())) {
        // Nothing to the left of inner can map, clear left of inner
        clear_left_of(find_inner_it);
      }

      if (find_outer_it != ids.end() && find_inner_it != ids.end()) {
        // Both dimensions map, inner dimension maps fully. For more context see
        // the comment:
        //  projectIdToRFactor
        //    if (find_outer_it != ids.end() && find_inner_it != ids.end() &&
        //      !hasPartialExtent(merge->inner())) {
        //      Comment here.

        // Removed outer, so inner shifts left
        ids[inner_pos] = split->in();
        ids.erase(find_outer_it);
        auto in_pos = inner_pos - 1;
        if (hasPartialExtent(split->outer())) {
          // Outer dimension has a partial map, but inner dimension is full,
          // split->in should be partially mapped.
          partial_projected_extent_[split->in()] =
              SimplifyingIrBuilder::mulExpr(
                  getMaybePartialMappedExtent(split->outer()),
                  split->inner()->extent());
          // Clear to the left of split in since it partially maps
          if (inner_pos > 0) {
            ids.erase(ids.begin(), ids.begin() + in_pos);
          }
        }
      } else {
        // Only inner matches, mark a partial match
        ids[inner_pos] = split->in();
        // Mark the partial match, inner could already be a partial match
        partial_projected_extent_[split->in()] =
            getMaybePartialMappedExtent(split->inner());
        // Clear to the left of split in since it partially maps
        ids.erase(ids.begin(), ids.begin() + inner_pos);
      }
    } else if (const Merge* merge = dynamic_cast<const Merge*>(expr)) {
      auto find_out_it = std::find(ids.begin(), ids.end(), merge->out());
      if (find_out_it == ids.end()) {
        continue;
      }

      auto out_pos = std::distance(ids.begin(), find_out_it);

      if (!hasPartialExtent(merge->out())) {
        // No partial map in output, so simply map through to inputs.
        ids[out_pos] = merge->outer();
        ids.insert(ids.begin() + out_pos + 1, merge->inner());
      } else {
        // This could be done completely symbolically by adding more expressions
        // into the graph, for example the inner patial extent could be:
        //   auto inner_partial_extent =
        //   SimplifyingIrBuilder::minExpr(getMaybePartialMappedExtent(merge->out()),
        //   merge->inner()->extent());
        // and outer partial extent:
        //   auto outer_partial_extent =
        //   SimplifyingIrBuilder::divExpr(getMaybePartialMappedExtent(merge->out()),
        //   merge->inner()->extent())
        //
        // However, for right now we'll assume no partial outer dimension maps
        // unless the extent of merge->out and merge->inner are compile time
        // constant. If not compile time constant we'll take:
        //   min(merge->inner()->extent(), the partial extent of merge->out())
        // for the inner partial map, and assume outer doesn't partially map.
        //
        // TODO: Improve to do the above. The challenge here is adding correct
        // conditionals to evaluate (and add conditional evaluation in expr
        // evaluator) to check vectorization especially like (out_extent >
        // inner_extent).
        if (!getMaybePartialMappedExtent(merge->out())->isConstInt() ||
            !merge->inner()->extent()->isConstInt()) {
          // We don't know at compile time if the partial extent of merge->out
          // is bigger than merge->inner or not, so generate an expressions we
          // can evaluate at runtime for the inner partial mapping extent. Don't
          // attempt to map merge->outer, just assume it doesn't partially map.
          ids[out_pos] = merge->inner();
          partial_projected_extent_[merge->inner()] =
              SimplifyingIrBuilder::minExpr(
                  getMaybePartialMappedExtent(merge->out()),
                  merge->inner()->extent());
        } else {
          // Compile time constant analysis
          auto partial_out_extent =
              getMaybePartialMappedExtent(merge->out())->evaluateInt();
          auto inner_extent = merge->inner()->extent()->evaluateInt();
          if (partial_out_extent > inner_extent) {
            // Partial mapping on out is bigger than inner, so extend partial
            // mapping to outer and mark inner as fully mapped.
            //
            // TODO: Should this be ceilDiv? Normal div is more conservative, so
            // sticking with that for now.
            partial_projected_extent_[merge->outer()] =
                SimplifyingIrBuilder::divExpr(
                    getMaybePartialMappedExtent(merge->out()),
                    merge->inner()->extent());
            // Map inner and outer dimensions
            ids[out_pos] = merge->outer();
            ids.insert(ids.begin() + out_pos + 1, merge->inner());
          } else if (partial_out_extent == inner_extent) {
            // Only map inner dimension, but since it's the same extent as the
            // partial out extent, map all of it.
            ids[out_pos] = merge->inner();
          } else {
            // Partially map inner dimension
            ids[out_pos] = merge->inner();
            partial_projected_extent_[merge->inner()] =
                getMaybePartialMappedExtent(merge->out());
          }
        }
        // If any domain is partially mapped, we need to clear to the left of it
        ids.erase(ids.begin(), ids.begin() + out_pos);
      }
    } else {
      TORCH_INTERNAL_ASSERT(
          false,
          "ProjectDimensions does not support expr type: ",
          expr->toString());
    } // switch on expr type
  } // For loop on the transform expressions

  // Add to our tracking and return the iterator to the inserted entry
  return projected_root_ids_.emplace(ref, ids).first;
}

// This function is very similar to projectIdToRoot, we just generally swap the
// logic of split and merge as the reverse mapping of merge looks a lot like
// split and vice versa.
std::unordered_map<TensorView*, std::vector<IterDomain*>>::iterator
ContiguousInnerDimensionsMapper::projectIdToRFactor(
    TensorView* ref,
    std::vector<IterDomain*> ids) {
  auto transform_exprs = StmtSort::getExprs(
      ref->fusion(),
      {ref->getRFactorDomain().begin(), ref->getRFactorDomain().end()});

  // Map forward through transforms since we're going from root to rfactor
  for (const auto* expr : transform_exprs) {
    if (const Merge* merge = dynamic_cast<const Merge*>(expr)) {
      // Initialize state
      auto find_outer_it = ids.begin();
      auto outer_pos = ids.size();
      auto find_inner_it = ids.begin();
      auto inner_pos = ids.size();

      // Removes all entries to the left of provided `it`, if `it` is not
      // ids.begin(). Updates all state of finding outer and inner in the ids
      // vector after erasing.
      auto clear_left_of = [&find_outer_it,
                            &outer_pos,
                            &find_inner_it,
                            &inner_pos,
                            &ids,
                            &merge](decltype(find_outer_it) it) {
        if (it != ids.begin()) {
          ids.erase(ids.begin(), it);
        }
        find_outer_it = std::find(ids.begin(), ids.end(), merge->outer());
        outer_pos = find_outer_it == ids.end()
            ? ids.size()
            : std::distance(ids.begin(), find_outer_it);

        find_inner_it = std::find(ids.begin(), ids.end(), merge->inner());
        inner_pos = find_inner_it == ids.end()
            ? ids.size()
            : std::distance(ids.begin(), find_inner_it);
      };

      // Dry run to fill state
      clear_left_of(ids.begin());

      // Check if the input domains of the merge are contiguous in the mapped
      // domain.
      if (find_outer_it == ids.end() && find_inner_it != ids.end()) {
        // Outer dimension was not found, but inner dimension was. Must assume
        // everything to the left of inner is not contiguously merged.
        //
        // Clear left of inner
        clear_left_of(find_inner_it);
      } else if (find_outer_it != ids.end() && find_inner_it == ids.end()) {
        // Inner dimension was not found, outer and anything left of outer are
        // definitely not contiguous.
        //
        // Clear outer and left of outer
        clear_left_of(find_outer_it + 1);
        continue;
      } else if (find_outer_it == ids.end() && find_inner_it == ids.end()) {
        // Nothing mapped, just continue
        continue;
      }

      if (find_outer_it != ids.end() && find_inner_it != ids.end()) {
        // Both outer and inner mapped.
        if (outer_pos >= inner_pos) {
          // Make sure outer is outside inner, otherwise neither could be part
          // of a continuous mapping. There are cases where we could have
          // reversible operations e.g.:
          //    [id{3} id{5} id{6}] -> merge(1, 0)
          // -> [id{5*3} id{6}] -> split(0, 5)
          // -> [id{5} id{3} id{6}] -> transpose(0, 1)
          // -> [id{3} id{5} id{6}]
          // However we don't try and capture cases like this correcly, we'd
          // just reduce this down to only the iter domain of size 6 mapping.
          //
          // Clear outer and left of outer
          clear_left_of(find_outer_it + 1);
          continue;
        }

        // Find the position inner would have to have to be considered ordered
        // relative to outer
        auto pos_after_outer = outer_pos + 1;
        for (; pos_after_outer < ids.size(); pos_after_outer++) {
          if (ids[pos_after_outer]->isBroadcast()) {
            // Skip broadcast axes as they must not have been concretized in the
            // reference. We remove dimensions that underwent a concretization
            // as well as the dimensions to the left of that.
            continue;
          }
          break;
        }

        if (inner_pos != pos_after_outer) {
          // Nothing to the left of inner could be continuous.
          //
          // Clear left of inner
          clear_left_of(find_inner_it);
        }
      }

      if (hasPartialExtent(merge->inner())) {
        // Nothing to the left of inner can map, clear left of inner
        clear_left_of(find_inner_it);
      }

      if (find_outer_it != ids.end() && find_inner_it != ids.end()) {
        // Both dimensions map, inner dimension maps fully. We don't map the
        // outer dimension through if the inner dimension maps partially as we'd
        // have to support mapping a non-continuous dimension. i.e.:
        //
        // merge(I0*I1, I2*I3) -> I0*I1*I2*I3
        //
        // With partial mapping of I1 and I3, then there'd be I2 between them
        // so it wouldn't be a continuous segment that we map through this
        // merge. Therefore we'd only consider I3 partialy mapping through this
        // operation.
        //
        // If we have the same merge
        // merge(I0*I1, I2*I3) -> I0*I1*I2*I3
        // However, I2*I3 completely maps, and I1 partially maps, then we can
        // forward a partially mapped domain to the output of size I1*I2*I3

        ids[inner_pos] = merge->out();
        ids.erase(find_outer_it);
        auto out_pos = inner_pos - 1;
        if (hasPartialExtent(merge->outer())) {
          // Outer dimension has a partial map, but inner dimension is full,
          // merge->out should be partially mapped.
          partial_projected_extent_[merge->out()] =
              SimplifyingIrBuilder::mulExpr(
                  getMaybePartialMappedExtent(merge->outer()),
                  merge->inner()->extent());
          // Clear to the left of merge out since it partially maps
          ids.erase(ids.begin(), ids.begin() + out_pos);
        }
      } else {
        // Only inner matches, mark a partial match
        ids[inner_pos] = merge->out();
        // Mark the partial match
        partial_projected_extent_[merge->out()] =
            getMaybePartialMappedExtent(merge->inner());
        // Clear to the left of merge out since it partially maps
        if (inner_pos > 0) {
          ids.erase(ids.begin(), ids.begin() + inner_pos - 1);
        }
      }
    } else if (const Split* split = dynamic_cast<const Split*>(expr)) {
      auto find_in_it = std::find(ids.begin(), ids.end(), split->in());
      if (find_in_it == ids.end()) {
        continue;
      }

      auto in_pos = std::distance(ids.begin(), find_in_it);

      if (!hasPartialExtent(split->in())) {
        // No partial map in output, so simply map through to inputs.
        ids[in_pos] = split->outer();
        ids.insert(ids.begin() + in_pos + 1, split->inner());
      } else {
        // See comment in
        //   projectIdToRoot
        //     if (!hasPartialExtent(merge->out())) {
        //       ...
        //     } else {
        //   here
        if (!getMaybePartialMappedExtent(split->in())->isConstInt() ||
            !split->inner()->extent()->isConstInt()) {
          // We don't know at compile time if the partial extent of split->in is
          // bigger than split->inner or not, so generate an expressions we can
          // evaluate at runtime for the inner partial mapping extent. Don't
          // attempt to map split->outer, just assume it doesn't partially map.
          ids[in_pos] = split->inner();
          partial_projected_extent_[split->inner()] =
              SimplifyingIrBuilder::minExpr(
                  getMaybePartialMappedExtent(split->in()),
                  split->inner()->extent());
        } else {
          // Compile time constant analysis
          auto partial_in_extent =
              getMaybePartialMappedExtent(split->in())->evaluateInt();
          auto inner_extent = split->inner()->extent()->evaluateInt();
          if (partial_in_extent > inner_extent) {
            // Partial mapping on in is bigger than inner, so extend partial
            // mapping to outer and mark inner as fully mapped.
            //
            // TODO: Should this be ceilDiv? Normal div is more conservative, so
            // sticking with that for now.
            partial_projected_extent_[split->outer()] =
                SimplifyingIrBuilder::divExpr(
                    getMaybePartialMappedExtent(split->in()),
                    split->inner()->extent());
            // Map inner and outer dimensions
            ids[in_pos] = split->outer();
            ids.insert(ids.begin() + in_pos + 1, split->inner());
          } else if (partial_in_extent == inner_extent) {
            // Only map inner dimension, but since it's the same extent as the
            // partial out extent, map all of it.
            ids[in_pos] = split->inner();
          } else {
            // Partially map inner dimension
            ids[in_pos] = split->inner();
            partial_projected_extent_[split->inner()] =
                getMaybePartialMappedExtent(split->in());
          }
        }
        // If any domain is partially mapped, we need to clear to the left of it
        ids.erase(ids.begin(), ids.begin() + in_pos);
      }
    } else {
      TORCH_INTERNAL_ASSERT(
          false,
          "ProjectDimensions does not support expr type: ",
          expr->toString());
    } // switch on expr type
  } // For loop on the transform expressions

  // Add to our tracking and return the iterator to the inserted entry
  return projected_rfactor_ids_.emplace(ref, ids).first;
}

void ContiguousInnerDimensionsMapper::propagateC2P(
    TensorView* from,
    TensorView* to) {
  // If we have a case where we have a concretized broadcast that's being
  // tracked in a consumer but not concretized in the producer we should break
  // off the dimensions connected to the left of that dimension. So if we have:
  // T0[i0, i2]
  // T1[i0, b1, i2] = broadcast(T0)
  // T2[i0, i1, i2]
  // T3[i0, i1, i2] = T1 + T2
  // and we're propogating from T3 with {i0, i1, i2}
  // When we go from T3 to T0, we don't have any mechanism to understand that i0
  // and i2 are not contiguous in the original domain of T3. It's not ideal with
  // transpose, but when this happens we'll clear all dimensions mapped left of
  // the concretized broadcast.
  // So if we have:
  // T0[i1, i2]
  // T1[b0, i1, i2] = broadcast(T0)
  // T2[i1, b0, i2] = transpose(T1)
  // T3[i1, i0, i2]
  // T4[i1, i0, i2] = T2 + T3
  // T5[i0, i1, i2] = transpose(T4)
  // Then i1 and i2 are contiguous in both T0 and T5, but due to the realization
  // of the broadcast on T4 we will have removed i1 from the mapped set.
  auto from_ids = projected_root_ids_.at(from);
  PairwiseRootDomainMap root_map(to, from);
  auto c2p_map = root_map.mapConsumerToProducer(from->domain(), to->domain());

  // Id's in consumer to clear from the mapped set due to broadcast
  // concretization.
  std::unordered_set<IterDomain*> consumer_ids_to_clear;
  if (to->hasBroadcast()) {
    // Find the last broadcast dimension resolved in consumers root domain
    int clear_pos = -1;
    for (auto i : c10::irange(from->getRootDomain().size())) {
      auto c_id = from->getRootDomain()[i];
      auto c_it = c2p_map.find(c_id);
      if (c_it == c2p_map.end()) {
        continue;
      }
      auto p_id = c_it->second;
      if ((!c_id->isBroadcast() && !c_id->isTrivialReduction()) &&
          p_id->isBroadcast()) {
        clear_pos = i;
      }
    }
    // Clear everything to the left of the inner most resolved broadcast
    // dimension, including the broadcasted domain.
    if (clear_pos >= 0) {
      consumer_ids_to_clear.insert(
          from->getRootDomain().begin(),
          from->getRootDomain().begin() + clear_pos + 1);
    }
  }

  std::vector<IterDomain*> producer_rfactor_ids;
  for (auto from_id : from_ids) {
    auto c2p_it = c2p_map.find(from_id);
    if (c2p_it != c2p_map.end() &&
        consumer_ids_to_clear.find(c2p_it->first) ==
            consumer_ids_to_clear.end()) {
      producer_rfactor_ids.push_back(c2p_it->second);
      if (hasPartialExtent(c2p_it->first)) {
        partial_projected_extent_[c2p_it->second] =
            partial_projected_extent_.at(c2p_it->first);
      }
    }
  }
  projected_rfactor_ids_[to] = producer_rfactor_ids;
  projectIdToRoot(to, producer_rfactor_ids);
}

void ContiguousInnerDimensionsMapper::propagateP2C(
    TensorView* from,
    TensorView* to) {
  // If we have a case where we have a reduction that's being tracked in a
  // producer but not a consumer we should break off the dimensions connected to
  // the left of that reduction. So if we have:
  // T0[i0, i1, i2]
  // T1[i0, r1, i2] = sum(T0)
  // T2[i0, i2] = T1
  // and we're propogating from T0 with {i0, i1, i2}
  // When we go from T1 to T2, we don't have any mechanism to understand that i0
  // and i2 are not contiguous in the original domain of T0. It's not ideal with
  // transpose, but when this happens we'll clear all dimensions mapped left of
  // the reduction.
  // So if we have:
  // T0[i0, i1, i2]
  // T1[i1, i0, i2] = transpose(T0)
  // T2[i1, r0, i2] = sum(T1)
  // T3[i1, i2] = T2
  // Then i1 and i2 are contiguous in both T0 and T3, but due to the sum on T1
  // we will have removed i1.
  auto from_ids = projected_rfactor_ids_.at(from);
  PairwiseRootDomainMap root_map(from, to);
  auto p2c_map = root_map.mapProducerToConsumer(from->domain(), to->domain());
  std::vector<IterDomain*> consumer_root_ids;

  // Id's in producer to clear from the mapped set due to reductions.
  std::unordered_set<IterDomain*> producer_ids_to_clear;
  if (from->hasReduction()) {
    // Find the last reduction dimension in the rfactor domain.
    int clear_pos = -1;
    for (auto i : c10::irange(from->getMaybeRFactorDomain().size())) {
      if (from->getMaybeRFactorDomain()[i]->isReduction() &&
          !from->getMaybeRFactorDomain()[i]->isTrivialReduction()) {
        clear_pos = i;
      }
    }
    // Clear everything to the left of the inner most reduction dimension.
    if (clear_pos >= 0) {
      producer_ids_to_clear.insert(
          from->getMaybeRFactorDomain().begin(),
          from->getMaybeRFactorDomain().begin() + clear_pos + 1);
    }
  }

  for (auto from_id : from_ids) {
    auto p2c_it = p2c_map.find(from_id);
    if (p2c_it != p2c_map.end() &&
        producer_ids_to_clear.find(p2c_it->first) ==
            producer_ids_to_clear.end()) {
      consumer_root_ids.push_back(p2c_it->second);

      if (hasPartialExtent(p2c_it->first)) {
        partial_projected_extent_[p2c_it->second] =
            partial_projected_extent_.at(p2c_it->first);
      }
    }
  }
  projected_root_ids_[to] = consumer_root_ids;
  projectIdToRFactor(to, consumer_root_ids);
}

void ContiguousInnerDimensionsMapper::propagateSibling(
    TensorView* from,
    TensorView* to) {
  TORCH_INTERNAL_ASSERT(
      from->getRootDomain().size() == to->getRootDomain().size(),
      "Siblings of different root sizes not supported, but found:\n  ",
      from->toString(),
      "\n  and\n  ",
      to->toString(),
      "\nhave root sizes of ",
      from->getRootDomain().size(),
      " and ",
      to->getRootDomain().size());

  auto from_root_ids = projected_root_ids_.at(from);
  std::vector<IterDomain*> sibling_root_ids;

  for (auto from_root_id : from_root_ids) {
    auto from_it = std::find(
        from->getRootDomain().begin(),
        from->getRootDomain().end(),
        from_root_id);
    TORCH_INTERNAL_ASSERT(
        from_it != from->getRootDomain().end(),
        "Expected ",
        from_root_id->toString(),
        " to be in the root of ",
        from->toString());
    auto pos = std::distance(from->getRootDomain().begin(), from_it);
    sibling_root_ids.push_back(to->getRootDomain()[pos]);
  }

  projected_root_ids_[to] = sibling_root_ids;

  if (!from->hasRFactor()) {
    return;
  }

  TORCH_INTERNAL_ASSERT(
      from->getRFactorDomain().size() == to->getRFactorDomain().size(),
      "Siblings of different rfactor sizes not supported, but found:\n  ",
      from->toString(),
      "\n  and\n  ",
      to->toString(),
      "\nhave rfactor sizes of ",
      from->getRFactorDomain().size(),
      " and ",
      to->getRFactorDomain().size());

  auto from_rfactor_ids = projected_rfactor_ids_.at(from);
  std::vector<IterDomain*> sibling_rfactor_ids;

  for (auto from_rfactor_id : from_rfactor_ids) {
    auto from_it = std::find(
        from->getRFactorDomain().begin(),
        from->getRFactorDomain().end(),
        from_rfactor_id);
    TORCH_INTERNAL_ASSERT(
        from_it != from->getRFactorDomain().end(),
        "Expected ",
        from_rfactor_id->toString(),
        " to be in the rfactor of ",
        from->toString());
    auto pos = std::distance(from->getRFactorDomain().begin(), from_it);
    sibling_rfactor_ids.push_back(to->getRFactorDomain()[pos]);
  }

  projected_rfactor_ids_[to] = sibling_rfactor_ids;
}

// Returns Mappings of all dims in reference starting from inner most position
// to outer most position. e.g. T0[i0, r1, b2] will return 3 Mapper instances
// associated with:
// {{i0, r1, b1}, {r1, b1}, {b1}}
std::vector<ContiguousInnerDimensionsMapper> getAllVectorizedMapsOf(
    TensorView* ref) {
  std::vector<ContiguousInnerDimensionsMapper> mappers;
  auto root_dom = ref->hasReduction() && ref->hasRFactor()
      ? ref->getRootDomain()
      : ref->getMaybeRFactorDomain();
  while (!root_dom.empty()) {
    mappers.push_back(ContiguousInnerDimensionsMapper::map(ref, root_dom));
    root_dom.erase(root_dom.begin());
  }
  return mappers;
}

// Returns Val* entires that should be evaluated and multiplied based on
// contiguity of reference and dimensions mapped to ref in mapper.
std::vector<Val*> getContigVectorSizesOf(
    TensorView* of_tv,
    const ContiguousInnerDimensionsMapper& mapper) {
  // Logic copied to get root according to scheduler_utils::innerMostRootDim
  // also copied from SchedulerRuntimeInfo::getMaxVectorizableWidth
  bool use_root_dom = of_tv->hasReduction() && of_tv->hasRFactor();
  auto of_tv_root =
      use_root_dom ? of_tv->getRootDomain() : of_tv->getMaybeRFactorDomain();

  const auto& projected_dim_map =
      use_root_dom ? mapper.mappedRootIds() : mapper.mappedRFactorIds();

  std::vector<IterDomain*> null_dims;
  const std::vector<IterDomain*>& projected_dims =
      projected_dim_map.find(of_tv) == projected_dim_map.end()
      ? null_dims
      : projected_dim_map.at(of_tv);

  auto of_tv_root_no_reductions = TensorDomain::noReductions(of_tv_root);

  auto contiguity = of_tv->domain()->contiguity();
  // Appears after reductions the reduction domain often has a contiguity entry.
  // This only matters if the result of the reduction is an output
  if (contiguity.size() == of_tv_root.size() &&
      contiguity.size() != of_tv_root_no_reductions.size()) {
    std::vector<bool> new_contiguity;
    for (auto i : c10::irange(of_tv_root.size())) {
      if (!of_tv_root[i]->isReduction()) {
        new_contiguity.push_back(contiguity[i]);
      }
    }
    contiguity = new_contiguity;
  }
  of_tv_root = of_tv_root_no_reductions;

  auto of_tv_root_size = of_tv_root.size();

  // Filter out 0-dim tensors
  if (of_tv_root_size < 1) {
    return {};
  }

  // Filter out mismatched contiguity info
  if (of_tv_root_size != contiguity.size()) {
    return {};
  }

  std::vector<Val*> vectorizable_dim_sizes;

  // Order is important, need to make sure dimensions match up correctly with
  // what was propogated through the mapper. The mapper's dimensions is
  // propogated in the order of the reference, if that order doesn't match the
  // tensor we're mapping too then a transpose interfered with expanded the
  // vectorize dimension.
  size_t projected_dims_i = projected_dims.size();

  for (auto i : c10::irange(of_tv_root_size)) {
    if (projected_dims_i == 0) {
      break;
    }
    auto root_i = of_tv_root_size - i - 1;
    auto root_id = of_tv_root[root_i];

    if (root_id->extent()->isOneInt() || root_id->isBroadcast()) {
      if (projected_dims[projected_dims_i - 1]->sameAs(root_id)) {
        --projected_dims_i;
      }
      continue;
    }

    // Not contiguous
    if (!contiguity[root_i]) {
      break;
    }

    // Mapping order isn't correct, cannot expand vectorization dimension.
    if (!projected_dims[--projected_dims_i]->sameAs(root_id)) {
      break;
    }

    vectorizable_dim_sizes.insert(
        vectorizable_dim_sizes.begin(),
        mapper.getMaybePartialMappedExtent(root_id));
    if (mapper.hasPartialExtent(root_id)) {
      // If we have a partial map we cannot extend the mapping any further for
      // vectorization.
      break;
    }
  }
  return vectorizable_dim_sizes;
}

size_t getExpandedVectorization(
    const std::vector<ContiguousInnerDimensionsMapper>& reference_maps,
    SchedulerRuntimeInfo& runtime_info,
    const std::vector<TensorView*> vectorizable_inputs_outputs,
    TensorView* reference_tv,
    int break_point,
    size_t default_word_size) {
  if (vectorizable_inputs_outputs.empty()) {
    return 1;
  }

  size_t max_expand_size = SchedulerRuntimeInfo::max_alignment_size_in_byte;
  size_t common_alignment_size =
      SchedulerRuntimeInfo::max_alignment_size_in_byte;

  for (auto inp_or_out : vectorizable_inputs_outputs) {
    auto dtype_size = dataTypeSize(
        inp_or_out->dtype(), indexModeToDtype(runtime_info.getIndexMode()));

    max_expand_size = std::min(
        max_expand_size,
        SchedulerRuntimeInfo::max_alignment_size_in_byte / dtype_size);
    max_expand_size = std::min(
        max_expand_size, runtime_info.getMaxVectorizableWidth(inp_or_out));
    common_alignment_size = std::min(
        common_alignment_size, runtime_info.getAlignmentSize(inp_or_out));
  }

  // If there's no possibility to increase vector size of provided tensors,
  // then don't bother doing a more complex analysis to try and do so, just
  // return early.
  if (max_expand_size == default_word_size) {
    return default_word_size;
  }

  auto reference_map = reference_maps[break_point];
  // Initialize to max the tensors could support.
  size_t max_supported_vector_size = max_expand_size;
  for (auto inp_or_out : vectorizable_inputs_outputs) {
    auto contig_vec = getContigVectorSizesOf(inp_or_out, reference_map);

    auto inp_or_out_root =
        inp_or_out->hasReduction() && inp_or_out->hasRFactor()
        ? inp_or_out->getRootDomain()
        : inp_or_out->getMaybeRFactorDomain();

    // Accumulate the size of the dimensions that mapped and are contiguous.
    size_t contig_dim_size = 1;
    for (auto extent : contig_vec) {
      auto dim_size = runtime_info.expressionEvaluator().evaluate(extent);
      TORCH_INTERNAL_ASSERT(
          dim_size.has_value(),
          "Unknown extent of tv: ",
          inp_or_out->toString(),
          " size: ",
          extent->toInlineString());
      contig_dim_size *= (size_t)dim_size->as<int64_t>();
    }

    size_t local_max_vec_size = 1;
    while (contig_dim_size > 1 && contig_dim_size % 2 == 0 &&
           local_max_vec_size < max_expand_size) {
      contig_dim_size /= 2;
      local_max_vec_size *= 2;
    }

    max_supported_vector_size =
        std::min(local_max_vec_size, max_supported_vector_size);
  }
  max_supported_vector_size =
      std::min(max_supported_vector_size, max_expand_size);
  return max_supported_vector_size;
}

} // namespace vectorize_helper
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
