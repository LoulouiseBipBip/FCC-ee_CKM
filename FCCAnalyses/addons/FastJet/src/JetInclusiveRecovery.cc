#include "FastJet/JetInclusiveRecovery.h"

#include <array>
#include <cmath>

namespace JetClustering {

  cluster_with_recovery::cluster_with_recovery(int   arg_n_seed,
                                                float arg_radius,
                                                float arg_E_min,
                                                float arg_exponent)
      : _n_seed(arg_n_seed),
        _radius(arg_radius),
        _E_min(arg_E_min),
        _exponent(arg_exponent),
        _jetAlgorithm(fastjet::JetAlgorithm::ee_genkt_algorithm),
        _def(_jetAlgorithm, _radius, _exponent) {}

  // Recursively walk a PseudoJet down to its PF-particle leaves and append
  // their user_index values to `out`. A "leaf" is any PseudoJet whose
  // user_index is >= 0 -- that's what JetClusteringUtils::set_pseudoJets
  // stamps on each input PF particle. FastJet's default user_index is -1,
  // which composite jets (clustered or fastjet::join results) inherit.
  //
  // Why recursion is needed:
  //   PseudoJet::constituents() returns only the *immediate* constituents.
  //   For a clustered jet (from ClusterSequence::inclusive_jets) those
  //   happen to be the PF leaves, so a single-level walk works. But for a
  //   fastjet::join() result they are the inputs to the join -- which can
  //   themselves be clustered jets with their own constituents. Our recovery
  //   uses fastjet::join, so a one- or two-level composite can hide the
  //   leaves. Recursing until user_index>=0 reaches them in every case.
  static void _walk_to_leaves(const fastjet::PseudoJet& jet,
                              std::vector<int>& out) {
    if (jet.user_index() >= 0) {
      out.push_back(jet.user_index());
      return;
    }
    if (!jet.has_constituents()) return;  // safety: composite with no children
    for (const auto& c : jet.constituents()) {
      _walk_to_leaves(c, out);
    }
  }

  // Per-jet constituent index list, flattened to PF-particle leaves. Mirrors
  // what JetClusteringUtils::build_jets stores in FCCAnalysesJet.constituents
  // for the standard clustering path, but works correctly for joined jets too.
  static std::vector<std::vector<int>>
  _build_constituent_indices(const std::vector<fastjet::PseudoJet>& jets) {
    std::vector<std::vector<int>> out;
    out.reserve(jets.size());
    for (const auto& jet : jets) {
      std::vector<int> idx;
      _walk_to_leaves(jet, idx);
      out.push_back(std::move(idx));
    }
    return out;
  }

  JetClustering::FCCAnalysesJet
  cluster_with_recovery::operator()(const std::vector<fastjet::PseudoJet>& input) {
    // 1. Inclusive clustering, no E cut at this stage -- the E_min cut comes
    //    after the recovery merging step.
    _cs = fastjet::ClusterSequence(input, _def);
    std::vector<fastjet::PseudoJet> all_jets =
        fastjet::sorted_by_E(_cs.inclusive_jets(0.0));

    // Fewer jets than seeds requested: nothing to recover, just apply E_min.
    if ((int)all_jets.size() <= _n_seed) {
      std::vector<fastjet::PseudoJet> out_jets;
      for (const auto& j : all_jets)
        if (j.E() > _E_min)
          out_jets.push_back(j);
      JetClustering::FCCAnalysesJet result;
      result.jets         = out_jets;
      result.constituents = _build_constituent_indices(out_jets);
      return result;
    }

    // 2. Take the n_seed hardest jets as recovery seeds; freeze their
    //    unit-vector directions BEFORE merging so the angle metric stays
    //    collinear-safe (per the talk's footnote).
    std::vector<fastjet::PseudoJet>    recovered(all_jets.begin(),
                                                  all_jets.begin() + _n_seed);
    std::vector<std::array<double, 3>> seed_dirs(_n_seed);
    for (int i = 0; i < _n_seed; ++i) {
      double p = std::sqrt(recovered[i].modp2());
      if (p > 0.0) {
        seed_dirs[i] = {recovered[i].px() / p,
                        recovered[i].py() / p,
                        recovered[i].pz() / p};
      } else {
        seed_dirs[i] = {0.0, 0.0, 0.0};
      }
    }

    // 3. For each leftover jet, find the seed with the smallest opening angle
    //    (largest cos against the frozen seed direction) and join.
    for (size_t iadd = _n_seed; iadd < all_jets.size(); ++iadd) {
      const fastjet::PseudoJet& j = all_jets[iadd];
      double pj = std::sqrt(j.modp2());
      if (pj <= 0.0)
        continue;

      double max_cos = -2.0;
      int    imax    = 0;
      for (int iseed = 0; iseed < _n_seed; ++iseed) {
        double cos_a = (j.px() * seed_dirs[iseed][0]
                      + j.py() * seed_dirs[iseed][1]
                      + j.pz() * seed_dirs[iseed][2]) / pj;
        if (cos_a > max_cos) {
          max_cos = cos_a;
          imax    = iseed;
        }
      }
      recovered[imax] = fastjet::join(recovered[imax], j);
    }

    // 4. Apply the final E_min cut, package as FCCAnalysesJet.
    std::vector<fastjet::PseudoJet> out_jets;
    out_jets.reserve(recovered.size());
    for (const auto& j : recovered)
      if (j.E() > _E_min)
        out_jets.push_back(j);

    JetClustering::FCCAnalysesJet result;
    result.jets         = out_jets;
    result.constituents = _build_constituent_indices(out_jets);
    return result;
  }

}  // namespace JetClustering
