#pragma once

#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/maxinfo_propagator.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/registry.h>

#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace vectorize_helper {

// Grab all values and expressions used to make the merged_domain and remove
// them from the fusion
void cleanUpInnermostMergedDomains(
    const std::vector<IterDomain*>& root_domain,
    IterDomain* merged_domain);

// Merge innermost domains for finding the widest vectorizable
// size. Return the merged domain or nullptr if no merge is done.
IterDomain* mergeInnermostDomains(
    const std::vector<IterDomain*>& domain,
    int num_merged_domains);

//! Attempt to expand vectorized domains to contig merged domains. Break point
//! identifies the point in which you can't propagate contiguous merges. For
//! example in pointwise this is the point where we want to split the
//! parallelization to take advantage of broadcast, and for reduction schedulers
//! it's the point where we switch from a reduction domain to an iter domain (or
//! vice versa).
size_t expandVectorizationToContigMergedDomains(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    const std::vector<TensorView*> vectorizable_inputs_outputs,
    TensorView* reference_tv,
    int break_point,
    size_t default_word_size);

// Projects IterDomains that are next to eachother on the inner dimensions of
// the reference tensor through the fusion. Contiguous in the name of this class
// simply means dimensions that are next to eachother. This property is not
// enforced, but mapping can have some unpredictbale results if they are not.
// Tracks:
//   (1) The projected IterDomains as they're ordered in relative to the
//     TensorView's root and rfactor domain.
//   (2) The projected IterDomains as they're ordered relative to the original
//     reference.
//
// The tricky part of this class is what happens through combinations of view
// and transpose. IterDomains are projected on a "best effort partial basis".
// Partial as we can have partial dimensions that map in an iteration domain
// with view:
//   tv0[2*3, 5*7, 11]
//   tv1[2*3, 5, 7*11] = view(tv0)
// With tv1 and 7*11 as the reference and ids. When we project into tv0, we'd
// map the inner most 11, but we also want to partial map the 5*7 with 5*7's
// partial extent being 7. This gets tricky though:
//   tv0[2, 3*5*7, 11]
//   tv1[2*3, 5, 7*11] = view(tv0)
// with tv1 and [2*3, 7*11] as the reference and ids. tv0's 2 and 11 dim are
// easily identified as being mapped. The 3*5*7 dimension however, is partially
// mapped on the left and right side. Full partial dimension tracking is tricky,
// and this class does not aim to tackle it fully. Instead this class is really
// used to line up "inner dimensions" of tensors through out the graph for the
// purpose of unrolling and vectorization. Therefore we will only track partial
// dimensions as they track on the right hand side of iteration domains. For
// example in the last case we would only identify tv0's 3*5*7 dimension as
// being a partial mapping dimension with extent 7. If we further had:
//   tv0[5*7*11]
//   tv1[5*7, 11] = view(tv0)
//   tv2[5, 7*11] = view(tv1)
// with tv2 and [7*11] as the reference and ids (this could be a valid example
// from the pointwise scheduler).
// (1) tv1 would:
//     partial match on 5*7 with size 7
//     full match with dimension 11.
// (1) tv0 would:
//     partial match on 5*7*11 with size 7*11
//
//
// We track two orderings of the mapped dimensions as we propogate with this
// class. This is really due to transpose anlaysis which can get tricky
// especially when combined with view. For example:
//   tv0[8, 6*2]
//   tv1[8, 6, 2] = view(tv0)
//   tv2[6, 8, 2] = transpose(tv1)
// If we merge the last two dimensions in tv2 and split by a vectorize
// factor of 4
// i.e. tv2[6, 8, 2] -> tv2[6, 4, v{4}]
// would be propogated to tv0 as:
// tv0->split(1, 2)->merge(0, 2)->split(2, 4)
// It's clear the merge is not contiguous, and the split is larger than the
// dimension we want to vectorize. This would be invalid to vectorize. What we
// want to know from this analysis is that the maximum vectorize size of tv2 if
// we directly propogate it to tv0 is size 2.
class TORCH_CUDA_CU_API ContiguousInnerDimensionsMapper
    : public MaxInfoSpanningTree::Propagator {
 public:
  
  ContiguousInnerDimensionsMapper() = delete;

  static ContiguousInnerDimensionsMapper map(
      TensorView* reference,
      std::vector<IterDomain*> ids) {
    ContiguousInnerDimensionsMapper contig_inner_mapper(
        reference, std::move(ids));
    MaxRootDomainInfoSpanningTree tree(reference);
    tree.traverse(&contig_inner_mapper);
    return contig_inner_mapper;
  }

  bool hasPartialExtent(IterDomain* id) const {
    if (partial_mapped_extent_.find(id) == partial_mapped_extent_.end()) {
      return false;
    }
    return true;
  }

  Val* getExtent(IterDomain* id) const {
    if (hasPartialExtent(id)) {
      return partial_mapped_extent_.at(id);
    }
    return id->extent();
  }

  virtual void propagateC2P(TensorView* from, TensorView* to) override;
  virtual void propagateP2C(TensorView* from, TensorView* to) override;
  virtual void propagateSibling(TensorView* from, TensorView* to) override;

  const std::unordered_map<TensorView*, std::vector<IterDomain*>>&
  mappedRootIds() const {
    return mapped_root_ids_;
  }

  const std::unordered_map<TensorView*, std::vector<IterDomain*>>&
  mappedRFactorIds() const {
    return mapped_rfactor_ids_;
  }

  bool hasPartialMappedExtent(IterDomain* id) const {
    return partial_mapped_extent_.find(id) != partial_mapped_extent_.end();
  }

  Val* getMaybePartialMappedExtent(IterDomain* id) const {
    if (hasPartialMappedExtent(id)) {
      return partial_mapped_extent_.at(id);
    }
    return id->extent();
  }

  const std::unordered_map<IterDomain*, Val*>& partialMappedExtent() const {
    return partial_mapped_extent_;
  }

 private:
  ContiguousInnerDimensionsMapper(
      TensorView* reference,
      std::vector<IterDomain*> ids);

  std::unordered_map<TensorView*, std::vector<IterDomain*>>::iterator
  projectIdToRoot(TensorView* ref, std::vector<IterDomain*> ids);

  std::unordered_map<TensorView*, std::vector<IterDomain*>>::iterator
  projectIdToRFactor(TensorView* ref, std::vector<IterDomain*> ids);

  // Mapped root dimensions for each TensorView as we propogate. These mappings
  // are in the order of the reference.
  std::unordered_map<TensorView*, std::vector<IterDomain*>> mapped_root_ids_;
  std::unordered_map<TensorView*, std::vector<IterDomain*>> mapped_rfactor_ids_;

  std::unordered_map<IterDomain*, Val*> partial_mapped_extent_;
};

} // namespace vectorize_helper
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
