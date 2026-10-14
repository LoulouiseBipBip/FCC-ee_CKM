'''
zg_recov.py -- energy-recovery inclusive clustering as the Z->qq jet finder.

WHAT THIS REPLACES
==================
The production chain clusters RP_no_hard_photon with Durham exclusive n=2:
every particle is forced into one of two jets.  That is mechanism M2 in the
correlation diagnostic -- a hard third parton has nowhere to go, and a
mis-assigned constituent leaves one jet AND joins the other, which correlates
the two taggers' inputs by construction.

This module swaps the jet finder for JetClustering::cluster_with_recovery
(addons/FastJet/JetInclusiveRecovery.{h,cc}, from the inclusive_recovery
package): cluster inclusively with ee_genkt at small R, keep the n_seed
hardest jets as seeds, and merge every leftover jet into the seed it is
closest to in ANGLE, with the seed directions frozen before any merging.

The difference that matters here is the assignment rule.  Durham exclusive
assigns soft radiation by the kT-like distance measure of a sequence that is
still running; the recovery assigns it by angle to a direction that is already
fixed.  Neither loses the soft energy -- both keep sum(E_jet) = E_visible --
but they can disagree about WHICH jet a given particle belongs to, and that is
precisely the input the flavour tagger sees.

HOW THE SWAP IS DONE
--------------------
ExclusiveJetClusteringHelper builds an ordered dict of Define strings and
exposes the output collection NAMES as .jets / .constituents.  attach_recovery()
rewrites exactly ONE entry of that dict -- the `_jet` clustering call -- before
helper.define() runs.  cluster_with_recovery returns the same FCCAnalysesJet
struct as clustering_ee_kt, so get_pseudoJets / get_constituents /
build_constituents_cluster and every per-jet observable the helper owns
(jet_p, jet_e, jet_mass, jet_phi, jet_theta, jet_nconst, event_njet) come out
of the recovered jets with no further edits, and so does the whole
JetFlavourHelper inference chain.

This is a stronger swap than the .jets/.constituents reassignment in the
package's analysis_snippet.py.  That snippet runs AFTER helper.define(), so
jet_p / jet_e / jet_nconst would still describe the EXCLUSIVE jets while
jets_p4 described the recovered ones -- zg.define_jets_and_scales compares the
two (jet_order_consistent) and reads jet_nconst, so it would have reported a
broken event every time.  Rewriting the dict entry keeps the helper internally
consistent.

WHAT BREAKS, AND THE FIX
------------------------
cluster_with_recovery leaves FCCAnalysesJet::exclusive_dmerge EMPTY -- d_23 is
a property of a Durham sequence and the recovery has none.  get_exclusive_dmerge
returns -1 rather than crashing, so the failure would have been silent: d_12,
d_23, d_34 and y_23 would all have been -1 in every event and the y23 cut the
whole tag-and-probe fit relies on would have selected nothing.

define_durham_scales() therefore runs a SECOND, Durham-exclusive clustering on
the same pseudojet column purely to supply the merge scales, and
zg.define_jets_and_scales is told to read them from there.  Reusing the
helper's own `pjetc` guarantees the two clusterings cannot see different input
particles, which is also what makes the excl_* baseline below a fair
comparison.

WHY E_min IS 0 HERE AND NOT THE 10-15 GeV THE GUIDE RECOMMENDS
--------------------------------------------------------------
With n_seed = 2 every leftover jet is merged into a seed, so after recovery
there are never more than 2 jets and E_min can only ever remove a SEED.  It is
therefore not a jet-cleaning cut in this topology, it is an event selection --
and an event selection applied inside the treemaker is invisible to the
downstream fit and is NOT applied to the exclusive-2 production chain this
run is meant to be compared against.  Any acceptance difference between the two
chains would then be indistinguishable from a real efficiency difference.

So the cut is left at 0 and the identical selection is available offline on
jet1_e / jet2_e, where it can be applied to both chains at once.  Set
RECOV_EMIN = 15.0 to restore the package default; the cutflow line "exactly 2
reco jets" is what changes.

E_min = 0 does NOT make the acceptance identical to the exclusive chain,
though.  If every particle in the event lands inside R there is only one
inclusive jet, hence only one seed, and the event has no second jet to tag;
Durham exclusive-2 splits such an event in two anyway.  On 500
wzp6_ee_ss_ecm163 events that is 5 events, all with 3-8 reconstructed
particles.  It is a property of inclusive clustering, not of the E_min choice.

VALIDATION, AND WHAT IT IS NOT ALLOWED TO BE
-------------------------------------------
Recovery and Durham exclusive-2 both partition the SAME particles into two
jets.  A partition of a fixed set has a fixed total 4-momentum however it is
split, so the visible energy, the dijet mass, the dijet momentum and every
other event-level sum are IDENTICAL between the two chains, event by event, to
the last bit.  None of them can be used to show that the swap did anything --
and any of them would look like a perfectly reasonable comparison to plot.

The entire difference is which jet each particle went into:

  recov_nconst_moved  particles the two clusterings assign to different jets,
  recov_frac_moved    and that as a fraction of the event.  This IS the effect.
                      0 everywhere means the swap changed nothing and there is
                      no point running this chain.
  recov_de_jet1       energy moved across the split, signed, on the harder jet.
  recov_dang_max      largest angle between a recovered jet axis and the
                      nearest Durham jet axis.

The energy-recovery claim in the guide is about plain inclusive clustering,
which DISCARDS the jets below its threshold, not about exclusive clustering,
which keeps everything.  So it is tested against that:

  recov_e_gain        > 0, and its mean is the energy plain inclusive
                      clustering at the same R throws away above
                      RECOV_EMIN_REF.  If it is ~0 the recovery is not doing
                      what it claims.
  recov_dm_vs_plain   the resonance-peak shift the guide predicts.

And the two bookkeeping checks the implementation guide asks for:

  recov_const_unique  1 in every event.  0 means a particle landed in two jets,
                      which means the user_index bookkeeping is wrong and the
                      tagger is being fed duplicated constituents.
  recov_const_kept    fraction of input particles that ended up in a jet.
                      Must be 1.0 for n_seed=2 with E_min=0.
  recov_dm_vs_excl    0 in every event, per the argument above.  Kept as the
  recov_de_vs_excl    closure test that the two clusterings really did see the
                      same particles, since that is the one way the
                      assignment comparison could be quietly meaningless.

USAGE
-----
    helper = ExclusiveJetClusteringHelper("RP_no_hard_photon", zgr.RECOV_N_SEED)
    zgr.attach_recovery(helper)          # BEFORE define(), rewrites one entry
    df = helper.define(df)

    df = zgr.define_durham_scales(df, helper)
    df = zg.define_jets_and_scales(df, helper, dmerge_clust=zgr.DURHAM_CLUST)
    df = zgr.define_recovery_diagnostics(df, helper)
    ...
    output = ... + zgr.recov_branches()

define_recovery_diagnostics must come after zg.define_jets_and_scales (it uses
jets_p4 and jet_esum) and define_durham_scales before it (define_jets_and_scales
reads the merge scales).

Reference: G. Soyez, M. Cacciari, G. Salam, "Reconstructing and using jets at
the FCC-ee", FCC-ee workshop October 2022, slides 25 & 27.
'''

import ROOT

import zg_common as zg


# ═════════════════════════════════════════════════════════════════════════════
# Configuration
# ═════════════════════════════════════════════════════════════════════════════

# Number of hard partons at LO.  Z -> qq is 2; this is also what makes E_min
# a pure event selection (see the docstring).
RECOV_N_SEED = 2

# Inclusive ee_genkt radius used for the seeding pass.  The guide recommends
# 0.3-0.4: small enough that the soft radiation is resolved as separate jets
# rather than swept up by the seed's own catchment area, which is what leaves
# the recovery something to do.
RECOV_R = 0.4

# ee_genkt exponent: -1 anti-kT, 0 Cambridge/Aachen, +1 kT.
RECOV_EXP = -1.0

# Jet-energy cut applied AFTER recovery.  0 on purpose -- see the docstring.
RECOV_EMIN = 0.0

# Threshold at which the "what would plain inclusive clustering have thrown
# away" comparison is made.  This is NOT applied to the jets that get tagged;
# it exists only so recov_e_gain / recov_dm_vs_plain measure something.  With
# RECOV_EMIN = 0 the recovered jets keep every particle, so comparing against a
# plain inclusive clustering at the same 0 threshold would compare an event
# with itself and give exactly 0 -- which is what the first version of this
# module did.  The package's recommended working point is used instead.
RECOV_EMIN_REF = 15.0

# Column holding the Durham exclusive clustering kept only for d_12/d_23/d_34.
DURHAM_CLUST = "_jet_durham"

# Durham exclusive-2 is what the production chain uses; matching its arguments
# exactly (exclusive=2, cut=njets, sorted=1 energy ordering, recomb=0 E-scheme)
# is what makes excl_* the right baseline rather than an approximation of it.
DURHAM_NJETS = 2


def describe():
    print("Jet finder: energy-recovery inclusive clustering "
          "(JetClustering::cluster_with_recovery)")
    print(f"  n_seed = {RECOV_N_SEED}, R = {RECOV_R}, "
          f"exponent = {RECOV_EXP:+g}, E_min = {RECOV_EMIN} GeV")
    if RECOV_EMIN <= 0.0:
        print("  E_min = 0: no event is lost to the jet finder, so the "
              "acceptance stays comparable to the exclusive-2 chain. "
              "Apply an energy cut offline on jet1_e / jet2_e.")
    print(f"  merge scales d_12/d_23/d_34 from a separate Durham exclusive-"
          f"{DURHAM_NJETS} clustering of the same particles ({DURHAM_CLUST})")
    print(f"  baselines: Durham exclusive-{DURHAM_NJETS} (excl_*, assignment) "
          f"and plain inclusive at E>{RECOV_EMIN_REF} GeV "
          "(recov_*_plain, energy recovery)")


# ═════════════════════════════════════════════════════════════════════════════
# C++ helpers
#
# The recovery header is included explicitly rather than relied on being
# auto-loaded: the FastJet addon ships no rootmap, so cling only knows
# cluster_with_recovery if something has pulled the declaration in.
# ═════════════════════════════════════════════════════════════════════════════

ROOT.gInterpreter.Declare(r"""
#ifndef ZG_RECOV_INCLUDE
#define ZG_RECOV_INCLUDE
#include "FastJet/JetInclusiveRecovery.h"
#endif
""")

ROOT.gInterpreter.Declare(r"""
#ifndef ZG_RECOV_HELPERS
#define ZG_RECOV_HELPERS

#include <algorithm>
#include <cmath>
#include <set>
#include <vector>

#include "TLorentzVector.h"
#include "ROOT/RVec.hxx"

namespace ZgRecov {

using RVecTLV = ROOT::VecOps::RVec<TLorentzVector>;

template <typename T>
int count_ge(const ROOT::VecOps::RVec<T>& v, double thr) {
    int n = 0;
    for (const auto& x : v) if ((double)x >= thr) ++n;
    return n;
}

template <typename T>
double sum_ge(const ROOT::VecOps::RVec<T>& v, double thr) {
    double s = 0.;
    for (const auto& x : v) if ((double)x >= thr) s += (double)x;
    return s;
}

// Total number of (jet, constituent) entries.  Compared against the unique
// count below: they differ exactly when one input particle was written into
// two jets, which for the recovery would mean the user_index walk double
// counted a fastjet::join branch.
int n_const_total(const std::vector<std::vector<int>>& idx) {
    int n = 0;
    for (const auto& jet : idx) n += (int)jet.size();
    return n;
}

int n_const_unique(const std::vector<std::vector<int>>& idx) {
    std::set<int> seen;
    for (const auto& jet : idx)
        for (int i : jet) seen.insert(i);
    return (int)seen.size();
}

// How many particles the two clusterings put in DIFFERENT jets.
//
// This is the whole difference between the two chains.  Both partition the
// same particles into two jets, so every event-level sum -- visible energy,
// dijet mass, total momentum -- is identical by construction and cannot show
// anything.  What changes is which tagger sees which particle, and that is
// exactly what this counts.
//
// The two jet lists are paired by maximum constituent overlap rather than by
// index: both are energy ordered, but the ordering can swap between the
// algorithms, and an index pairing would then call every particle moved.
int n_const_moved(const std::vector<std::vector<int>>& a,
                  const std::vector<std::vector<int>>& b) {
    if (a.size() != 2 || b.size() != 2) return -1;

    std::set<int> b0(b[0].begin(), b[0].end());
    std::set<int> b1(b[1].begin(), b[1].end());

    int ov_direct = 0, ov_swapped = 0;
    for (int i : a[0]) {
        if (b0.count(i)) ++ov_direct;
        if (b1.count(i)) ++ov_swapped;
    }
    const std::set<int>& m0 = (ov_direct >= ov_swapped) ? b0 : b1;
    const std::set<int>& m1 = (ov_direct >= ov_swapped) ? b1 : b0;

    int moved = 0;
    for (int i : a[0]) if (!m0.count(i)) ++moved;
    for (int i : a[1]) if (!m1.count(i)) ++moved;
    return moved;
}

// Invariant mass of the sum of every jet above thr.  This is what plain
// inclusive clustering would reconstruct for the resonance: the jets below the
// threshold are simply lost, which is the energy the recovery merges back in.
double sum_mass_ge(const RVecTLV& j, double thr) {
    TLorentzVector s;
    int n = 0;
    for (const auto& x : j) {
        if (x.E() >= thr) { s += x; ++n; }
    }
    return n > 0 ? s.M() : -1.;
}

// For every jet in `a`, the angle to the closest jet in `b`; returns the
// largest such angle.  Zero means the two clusterings found the same axes.
double max_min_angle(const RVecTLV& a, const RVecTLV& b) {
    if (a.empty() || b.empty()) return -1.;
    double worst = 0.;
    for (const auto& ja : a) {
        if (ja.Vect().Mag() <= 0.) continue;
        double best = 1e9;
        for (const auto& jb : b) {
            if (jb.Vect().Mag() <= 0.) continue;
            best = std::min(best, ja.Vect().Angle(jb.Vect()));
        }
        if (best < 1e8) worst = std::max(worst, best);
    }
    return worst;
}

// Invariant mass of the two hardest entries; -1 if there are not two.
double dijet_mass(const RVecTLV& j) {
    if (j.size() < 2) return -1.;
    return (j[0] + j[1]).M();
}

}  // namespace ZgRecov
#endif
""")


# ═════════════════════════════════════════════════════════════════════════════
# The swap
# ═════════════════════════════════════════════════════════════════════════════

def attach_recovery(helper, n_seed=None, radius=None, e_min=None, exponent=None):
    """Rewrite the helper's clustering Define to use cluster_with_recovery.

    MUST be called before helper.define(df).  Only the `_jet` entry changes;
    `pjetc` still comes from JetClusteringUtils::set_pseudoJets, which is what
    stamps the user_index the recovery walks down to when it rebuilds the
    per-jet constituent lists.  Hand-built pseudojets would silently produce
    empty constituent vectors and a tagger running on nothing.
    """
    n_seed   = RECOV_N_SEED if n_seed   is None else n_seed
    radius   = RECOV_R      if radius   is None else radius
    e_min    = RECOV_EMIN   if e_min    is None else e_min
    exponent = RECOV_EXP    if exponent is None else exponent

    clust = f"_jet{helper.tag}"
    pjets = f"pjetc{helper.tag}"

    if clust not in helper.definition:
        raise RuntimeError(
            f"{clust} is not in the helper's definition dict (keys: "
            f"{list(helper.definition)}). ExclusiveJetClusteringHelper has "
            "changed shape; the swap cannot be applied blindly.")
    if helper.definition[clust].startswith("JetClustering::cluster_with_recovery"):
        raise RuntimeError("attach_recovery() called twice on the same helper.")
    if helper.njets != n_seed:
        raise RuntimeError(
            f"helper was built with njets={helper.njets} but the recovery is "
            f"seeded with n_seed={n_seed}. These must agree: helper.njets is "
            "what the rest of the analysis assumes the jet multiplicity is.")

    helper.definition[clust] = (
        f"JetClustering::cluster_with_recovery({n_seed}, {radius}, "
        f"{e_min}, {exponent})({pjets})")
    return helper


def define_durham_scales(df, helper, njets=None):
    """The Durham exclusive clustering kept only for the merge scales.

    cluster_with_recovery leaves exclusive_dmerge empty, and get_exclusive_dmerge
    answers -1 instead of failing, so without this every d_ij and y_23 would be
    -1 and the y23 cut downstream would quietly select nothing.

    Runs on the helper's own `pjetc`, so the recovered jets and this reference
    are guaranteed to be two readings of the same particles.
    """
    njets = DURHAM_NJETS if njets is None else njets
    pjets = f"pjetc{helper.tag}"
    df = df.Define(DURHAM_CLUST,
                   f"JetClustering::clustering_ee_kt(2, {njets}, 1, 0)({pjets})")
    return df


# ═════════════════════════════════════════════════════════════════════════════
# Branch list
# ═════════════════════════════════════════════════════════════════════════════

def recov_branches():
    return [
        # what the recovery had to work with
        'recov_njet_incl', 'recov_nmerged',
        # energy recovery, against plain inclusive at the same R
        'recov_njet_plain', 'recov_esum_plain', 'recov_e_gain',
        'recov_m_plain', 'recov_dm_vs_plain',
        # constituent closure
        'recov_nconst_in', 'recov_nconst_tot', 'recov_nconst_uniq',
        'recov_const_unique', 'recov_const_kept',
        # head-to-head against the Durham exclusive-2 chain this replaces
        'excl_njet', 'excl_jet1_e', 'excl_jet2_e',
        'excl_dijet_m', 'excl_esum',
        'recov_nconst_moved', 'recov_frac_moved',
        'recov_de_jet1', 'recov_dang_max',
        # closure: must be identically 0
        'recov_dm_vs_excl', 'recov_de_vs_excl',
    ]


# ═════════════════════════════════════════════════════════════════════════════
# Analyser block
# ═════════════════════════════════════════════════════════════════════════════

def define_recovery_diagnostics(df, helper):
    """Everything needed to decide whether the swap did anything.

    Call after zg.define_jets_and_scales (uses jets_p4, jet_esum) and after
    define_durham_scales (uses DURHAM_CLUST).
    """
    df = zg.check_no_collisions(df, recov_branches(),
                                "define_recovery_diagnostics")

    tag   = helper.tag
    pjets = f"pjetc{tag}"
    jetc  = f"_jetc{tag}"          # the raw std::vector<std::vector<int>>

    # ── what the seeding pass saw ────────────────────────────────────────────
    # The same ee_genkt/R/exponent as the recovery, before any merging: the
    # jets the recovery had to redistribute.  ptmin = 0 because pT is measured
    # against the beam and says nothing about how hard a jet is in an e+e-
    # event -- the energy threshold below is the meaningful one.
    df = df.Define("_recov_plain_fcc",
                   f"JetClustering::clustering_ee_genkt({RECOV_R}, 0, 0.0, 1, 0, "
                   f"{RECOV_EXP})({pjets})")
    df = df.Define("_recov_plain_jets",
                   "JetClusteringUtils::get_pseudoJets(_recov_plain_fcc)")
    df = df.Define("_recov_plain_e",
                   "JetClusteringUtils::get_e(_recov_plain_jets)")
    df = df.Define("_recov_plain_p4",
                   "JetConstituentsUtils::compute_tlv_jets(_recov_plain_jets)")

    df = df.Define("recov_njet_incl", "(int)_recov_plain_e.size()")
    # Leftovers that had to be merged into a seed.  0 means the event was
    # already a clean 2-jet event and the recovery was a no-op for it.
    df = df.Define("recov_nmerged",
                   f"std::max(0, recov_njet_incl - {RECOV_N_SEED})")

    # ── energy recovery (guide check 2) ──────────────────────────────────────
    # Plain inclusive clustering keeps only the jets above the threshold and
    # discards the rest; the recovery merges them into the seeds instead.  The
    # difference is the energy the algorithm exists to recover, and
    # recov_dm_vs_plain is the resonance-peak shift the guide predicts.
    #
    # The threshold is RECOV_EMIN_REF, not RECOV_EMIN: at RECOV_EMIN = 0
    # nothing is ever discarded and this comparison would be an event against
    # itself, giving exactly 0 with no warning.
    df = df.Define("recov_njet_plain",
                   f"ZgRecov::count_ge(_recov_plain_e, {RECOV_EMIN_REF})")
    df = df.Define("recov_esum_plain",
                   f"ZgRecov::sum_ge(_recov_plain_e, {RECOV_EMIN_REF})")
    df = df.Define("recov_e_gain", "jet_esum - recov_esum_plain")
    df = df.Define("recov_m_plain",
                   f"ZgRecov::sum_mass_ge(_recov_plain_p4, {RECOV_EMIN_REF})")
    df = df.Define("recov_dm_vs_plain",
                   "recov_m_plain > 0. ? reco_dijet_m - recov_m_plain : -999.")

    # ── constituent closure (guide check 3) ──────────────────────────────────
    # A duplicate means one particle was handed to the tagger twice, in two
    # different jets; a loss means the E_min cut dropped a seed.  Both are
    # silent otherwise -- the scores would still look perfectly reasonable.
    df = df.Define("recov_nconst_in",  f"(int){pjets}.size()")
    df = df.Define("recov_nconst_tot", f"ZgRecov::n_const_total({jetc})")
    df = df.Define("recov_nconst_uniq", f"ZgRecov::n_const_unique({jetc})")
    df = df.Define("recov_const_unique",
                   "(int)(recov_nconst_tot == recov_nconst_uniq)")
    df = df.Define("recov_const_kept",
                   "recov_nconst_in > 0 ? "
                   "(double)recov_nconst_uniq / (double)recov_nconst_in : -1.")

    # ── the Durham exclusive-2 baseline ──────────────────────────────────────
    # Same particles, the clustering the production chain uses.  This is the
    # only way to tell an improvement from a relabelling.
    df = df.Define("_excl_jets",
                   f"JetClusteringUtils::get_pseudoJets({DURHAM_CLUST})")
    df = df.Define("_excl_p4",
                   "JetConstituentsUtils::compute_tlv_jets(_excl_jets)")
    df = df.Define("excl_njet",    "(int)_excl_p4.size()")
    df = df.Define("excl_jet1_e",
                   "excl_njet > 0 ? (double)_excl_p4[0].E() : -1.")
    df = df.Define("excl_jet2_e",
                   "excl_njet > 1 ? (double)_excl_p4[1].E() : -1.")
    df = df.Define("excl_esum",
                   "excl_njet > 1 ? excl_jet1_e + excl_jet2_e : -1.")
    df = df.Define("excl_dijet_m", "ZgRecov::dijet_mass(_excl_p4)")

    # ── how the two chains actually differ ───────────────────────────────────
    # Both partition the same particles into two jets, so the difference is
    # entirely in the SPLIT: which tagger sees which particle, how much energy
    # sits on each side, and how far the axes moved.  Nothing else can differ.
    df = df.Define("_excl_jetc",
                   f"JetClusteringUtils::get_constituents({DURHAM_CLUST})")
    df = df.Define("recov_nconst_moved",
                   f"ZgRecov::n_const_moved({jetc}, _excl_jetc)")
    df = df.Define("recov_frac_moved",
                   "recov_nconst_uniq > 0 && recov_nconst_moved >= 0 ? "
                   "(double)recov_nconst_moved / (double)recov_nconst_uniq : -1.")
    # Energy moved across the split, signed, on the harder jet.  Both lists are
    # energy ordered, so jet1 is the harder one in each.
    df = df.Define("recov_de_jet1",
                   "excl_jet1_e > 0. ? jet1_e - excl_jet1_e : -999.")
    df = df.Define("recov_dang_max",
                   "ZgRecov::max_min_angle(jets_p4, _excl_p4)")

    # ── closure, NOT a figure of merit ───────────────────────────────────────
    # These are zero in every event by construction, because a partition of a
    # fixed set of particles has a fixed total 4-momentum however it is split.
    # They are kept precisely because that is easy to forget: a nonzero value
    # means one of the two clusterings dropped or duplicated a particle, which
    # is the one way the comparison above could be quietly meaningless.
    df = df.Define("recov_dm_vs_excl",
                   "excl_dijet_m > 0. ? reco_dijet_m - excl_dijet_m : -999.")
    df = df.Define("recov_de_vs_excl",
                   "excl_esum > 0. ? jet_esum - excl_esum : -999.")
    return df
