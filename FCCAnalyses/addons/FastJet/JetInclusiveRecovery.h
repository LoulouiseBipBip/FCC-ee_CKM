#ifndef FastJet_JetInclusiveRecovery_h
#define FastJet_JetInclusiveRecovery_h

#include <vector>

#include "fastjet/ClusterSequence.hh"
#include "fastjet/JetDefinition.hh"
#include "fastjet/PseudoJet.hh"

#include "FastJet/JetClustering.h"  // for FCCAnalysesJet struct

/** Energy-recovery inclusive jet clustering for e+e- events.

  Implements the algorithm sketched by Soyez/Cacciari/Salam at the
  October 2022 FCC-ee workshop ("Reconstructing and using jets at the FCC-ee",
  slides 25 & 27). Motivation: inclusive clustering leaves leftover soft
  radiation outside the selected jets; for ee topologies with a known number
  of hard partons (2 for HH->bbnunu, 4 for WW->qqqq, 6 for ZH->bbWW->6j)
  this shows up as energy loss in the reconstructed resonance peak and as a
  pile-up of soft mini-jets at the pT/E threshold.

  Algorithm:
    1. Cluster all input particles inclusively with ee_genkt at a small R
       (recommended 0.3-0.4) and no E/pT cut.
    2. Sort jets by descending E. Take the n_seed hardest as seeds.
    3. For every remaining jet, find the seed with the smallest opening
       angle (using the ORIGINAL frozen seed directions, collinear-safe)
       and merge it in via fastjet::join.
    4. Apply an E_min cut on the final recovered jets.

  Exponent (FastJet ee_genkt convention):
    -1 -> anti-kT (default, recommended)
     0 -> Cambridge/Aachen
    +1 -> kT

  Recombination is hardcoded to E_scheme inside the JetDefinition (correct
  for ee invariant-mass reconstruction).

  Usage from RDataFrame:
    df = df.Define("recovered_jets",
                   "JetClustering::cluster_with_recovery(4, 0.4f, 10.0f, -1.0f)(pjetc_R5)")
*/

namespace JetClustering {

  struct cluster_with_recovery {
  public:
    cluster_with_recovery(int   arg_n_seed,
                          float arg_radius   = 0.4f,
                          float arg_E_min    = 10.0f,
                          float arg_exponent = -1.0f);
    /// Returns FCCAnalysesJet (same struct as clustering_ee_genkt) so the
    /// standard JetClusteringUtils::get_pseudoJets / get_constituents helpers
    /// work without any new utility code. exclusive_dmerge fields are left
    /// empty -- they don't apply to the recovery procedure.
    JetClustering::FCCAnalysesJet operator()(const std::vector<fastjet::PseudoJet>& input);

  private:
    int   _n_seed;     ///< number of expected hard jets at LO (defines the seed count)
    float _radius;     ///< clustering R, recommended 0.3-0.4
    float _E_min;      ///< minimum jet energy applied AFTER recovery [GeV]
    float _exponent;   ///< ee_genkt exponent: -1 anti-kT, 0 C/A, +1 kT
    fastjet::JetAlgorithm   _jetAlgorithm{fastjet::JetAlgorithm::undefined_jet_algorithm};
    fastjet::ClusterSequence _cs;
    fastjet::JetDefinition  _def;
  };

}  // namespace JetClustering

#endif
