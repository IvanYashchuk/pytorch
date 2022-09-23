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

// Projects IterDomains through the fusion starting at provided reference. IDs
// in the reference are expected to be "contiguous", simply means dimensions
// that the iter domains are consecutive and next to eachother in the reference.
// This property is not enforced, but mapping can have some unpredictbale
// results if they are not. The reason we want contiguity here is this class is
// primarily used for vectorization analysis. Domains may be inserted or removed
// while propogating through the fusion and this class has to be senstitive to
// that.
//
// For example:
// Input: T0[i0, i2]
// Reference: T5[i0, i1, i2]
// If we want to base the vectorization size on the reference being contiguous
// in a 1D scheduler, we'd start the proces on the reference with {i0, i1, i2}.
// When we propogate to the input what we would still like is: {i0, i1, i2} to
// signify to us that the root domains in the input that map to the reference
// are not contiguous. So when we think of vector word, if we want the input to
// be included in the vectorized dimensions, we can only check multiples based
// on i2, not i0*i1*i2 like the reference would indicate.
//
// Another example:
// Input:[i1, i0, i2]
// Refrence [i0, i1, i2]
// Similarly as above when we propogate from the reference to the Input we'd
// like {i0, i1, i2}, which is the order of the reference, not the input. This
// is because we can compare that with the input domains to understand it's not
// ordered consistently, so once again we can only take into consideration
// vectorization based on i2.
//
// Another example:
// Input:[i1, i0, i2]
// Intermediate: [i1, i0, i2]
// Refrence [i0, i1, i2]
// Keeping the ordering relative to the reference also allows us to look though
// transpose operations without missing out in a case like this that the
// reference and input are consistently ordered so we can look at i0*i1*i2 for
// our vector multiple even though there are transposes in between them.
//
// This class primarily tracks:
//   The projected IterDomains from the reference through the fusion as
//   ordered relative to the reference. The projections can include dimensions
//   that are not in the local tensor view (See the first example in this
//   comment.)
//
// The tricky part of this class is what happens through combinations of view
// and transpose. IterDomains are projected on a "best effort partial basis".
// Partial as we can have partial dimensions that map in an iteration domain
// with view:
//   tv0[2*3, 5*7, 11]
//   tv1[2*3, 5, 7*11] = view(tv0)
// With tv1 and 7*11 as the reference and ids. When we project from tv1 to tv0,
// we'd map the inner most 11, but we also want to partial map the 5*7 with
// 5*7's partial extent being 7. This gets tricky though:
//   tv0[2, 3*5*7, 11]
//   tv1[2*3, 5, 7*11] = view(tv0)
// with tv1 and [2*3, 7*11] as the reference and ids. tv0's 2 and 11 dim are
// easily identified as being mapped. The 3*5*7 dimension however, is partially
// mapped on the left and right side. Full partial dimension tracking is tricky,
// and this class does not aim to tackle it fully. Instead this class is
// primarily intended to line up "inner dimensions" of tensors through out the
// graph for the purpose of unrolling and vectorization. Therefore we will only
// track partial dimensions as they track on the right hand side of iteration
// domains. For example in the last case we would only identify tv0's 3*5*7
// dimension as being a partial mapping dimension with extent 7. If we further
// had:
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
class TORCH_CUDA_CU_API ContiguousInnerDimensionsMapper
    : public MaxInfoSpanningTree::Propagator {
 public:
  ContiguousInnerDimensionsMapper() = delete;

  static ContiguousInnerDimensionsMapper map(
      TensorView* reference,
      std::vector<IterDomain*> ids,
      std::shared_ptr<const ComputeAtMap> ca_map) {
    ContiguousInnerDimensionsMapper contig_inner_mapper(
        reference, std::move(ids), ca_map);
    MaxRootDomainInfoSpanningTree tree(reference);
    tree.traverse(&contig_inner_mapper);
    return contig_inner_mapper;
  }

  static ContiguousInnerDimensionsMapper map(
      TensorView* reference,
      std::vector<IterDomain*> ids) {
    auto ca_map = std::make_shared<ComputeAtMap>(reference->fusion());
    return ContiguousInnerDimensionsMapper::map(reference, ids, ca_map);
  }

  bool hasPartialExtent(IterDomain* id) const {
    if (partial_projected_extent_.find(id) == partial_projected_extent_.end()) {
      return false;
    }
    return true;
  }

  Val* getExtent(IterDomain* id) const {
    if (hasPartialExtent(id)) {
      return partial_projected_extent_.at(id);
    }
    return id->extent();
  }

  virtual void propagateC2P(TensorView* from, TensorView* to) override;
  virtual void propagateP2C(TensorView* from, TensorView* to) override;
  virtual void propagateSibling(TensorView* from, TensorView* to) override;

  const std::unordered_map<TensorView*, std::vector<IterDomain*>>&
  mappedRootIds() const {
    return projected_root_ids_;
  }

  const std::unordered_map<TensorView*, std::vector<IterDomain*>>&
  mappedRFactorIds() const {
    return projected_rfactor_ids_;
  }

  bool hasPartialMappedExtent(IterDomain* id) const {
    return partial_projected_extent_.find(id) !=
        partial_projected_extent_.end();
  }

  Val* getMaybePartialMappedExtent(IterDomain* id) const {
    if (hasPartialMappedExtent(id)) {
      return partial_projected_extent_.at(id);
    }
    return id->extent();
  }

  const std::unordered_map<IterDomain*, Val*>& partialMappedExtent() const {
    return partial_projected_extent_;
  }

 private:
  ContiguousInnerDimensionsMapper(
      TensorView* reference,
      std::vector<IterDomain*> ids,
      std::shared_ptr<const ComputeAtMap> ca_map);

  std::unordered_map<TensorView*, std::vector<IterDomain*>>::iterator
  projectIdToRoot(TensorView* ref, std::vector<IterDomain*> ids);

  std::unordered_map<TensorView*, std::vector<IterDomain*>>::iterator
  projectIdToRFactor(TensorView* ref, std::vector<IterDomain*> ids);

  std::shared_ptr<const ComputeAtMap> ca_map_;

  // Mapped root dimensions for each TensorView as we propogate. These mappings
  // are in the order of the reference.
  std::unordered_map<TensorView*, std::vector<IterDomain*>> projected_root_ids_;
  std::unordered_map<TensorView*, std::vector<IterDomain*>>
      projected_rfactor_ids_;

  std::unordered_map<IterDomain*, Val*> partial_projected_extent_;
};

// Returns Mappings of all dims in reference starting from inner most position
// to outer most position.
//
// A tensor like T0[i0, r1, b2] will return 3 Mapper instances associated with:
// {{i0, r1, b1}, {r1, b1}, {b1}}
std::vector<ContiguousInnerDimensionsMapper> getAllVectorizedMapsOf(
    TensorView* ref);

// Returns Val* entires that should be evaluated and multiplied based on
// contiguity of reference and dimensions mapped to ref in mapper.
std::vector<Val*> getContigVectorSizesOf(
    TensorView* of_tv,
    const ContiguousInnerDimensionsMapper& mapper);

// TODO: More of this could be cached in the registry compile time cache beyond
// the reference_maps
//
// TODO: vectorizable_inputs_outputs is actually known based on the
// reference_maps. If nothing is mapped for a tensorview it's not vectorizable.
size_t getExpandedVectorization(
    const std::vector<ContiguousInnerDimensionsMapper>& reference_maps,
    SchedulerRuntimeInfo& runtime_info,
    const std::vector<TensorView*> vectorizable_inputs_outputs,
    TensorView* reference_tv,
    int break_point,
    size_t default_word_size);

} // namespace vectorize_helper
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
