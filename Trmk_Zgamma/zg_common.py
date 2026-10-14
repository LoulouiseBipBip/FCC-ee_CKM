'''
zg_common.py -- shared configuration, C++ helpers and analyser blocks for the
Z+gamma -> qq gamma treemakers.

WHY THIS FILE EXISTS
====================
The MC-PV diagnostic and the reconstructed-PV production treemaker are an A/B
pair: the difference between the two measured correlations C is only
interpretable if EVERYTHING except the primary vertex is identical.  They had
already drifted -- the ISR acceptance cut existed in one and not the other,
d_34 existed in one and not the other, and every C++ helper was copy-pasted.
Every shared definition now lives here and is imported by both, so a change
lands in both files or in neither.

INPUT-BRANCH STATUS FOR THIS CAMPAIGN
=====================================
Verified directly on
  winter2023/IDEA/wzp6_ee_bb_pqqmin40_ecm163/events_003875593.root
  EFlowTrack_1  present -- D0/phi/omega/Z0/tanLambda/referencePoint/covMatrix[21].
                 D0, Z0 and covMatrix compress at 1.07-1.38, i.e. real
                 per-track values.  (Contrast EFlowTrack_1.time at 32.9 or
                 Particle.endpoint at 122, which ARE constant placeholders.)
  EFlowTrack_2  present -- type/value/error.  value compresses at 1.21, so the
                 dNdx numbers are real.  error is constant but get_dndx only
                 consumes value.
  EFlowTrack_L  present (1.22).
  magFieldBz    present.
So the impact-parameter block and the dNdx block are both live here and
patch_missing_eflow_branches() is a no-op.  check_tagger_inputs() below
asserts that at job start instead of trusting it, and n_trk / n_trkstate are
written per event so a silent regression shows up in the output tree.
'''

import ROOT
import os
from treemaker_common import TAGGER_SCORES

# ═════════════════════════════════════════════════════════════════════════════
# Configuration -- SINGLE SOURCE OF TRUTH for both treemakers
# ═════════════════════════════════════════════════════════════════════════════

SQRTS         = 163.0
MZ_TARGET     = 91.1876

# ── ISR photon selection ─────────────────────────────────────────────────────
ISR_PMIN      = 5.0      # GeV, candidate threshold
ISR_CONE_DEG  = 15.0     # isolation half-angle
ISR_ISO_MAX   = 0.10     # sum |p| in cone / |p_gamma|
ISR_COSMAX    = 0.99     # |cos(theta)| acceptance -- was missing from the
                         # diagnostic version, present in production
ISR_MAXCAND   = 10       # hardest N candidates -> 2^N subset scan
ISR_MATCH_ANG = 0.05     # rad, reco-to-gen photon match for the validation

# The threshold the OLD sel_p() photon removal used.  The diagnostic counters
# that quantify the size of that bug must be evaluated HERE, not at ISR_PMIN --
# they were previously computed at 5 GeV while being named _gt10.
LEGACY_SELP_THRESHOLD = 10.0

# ── Heavy-flavour hadron content of the jets (mechanism M5) ──────────────────
# A gen b/c hadron is associated to the nearest jet within HF_JET_MAXANG.  The
# momentum floor keeps hadrons too soft to make a taggable vertex out of the
# count; both a loose and a tight count are written so the choice can be
# scanned offline.
HF_HADRON_PMIN_LOOSE = 1.0    # GeV
HF_HADRON_PMIN_TIGHT = 5.0    # GeV
HF_JET_MAXANG        = 0.7    # rad

# ── Tagger sanity ────────────────────────────────────────────────────────────
# A collapsed 7-class softmax sits at 1/7 = 0.143 for every class.
TAG_DEGENERATE_MAX = 0.2

# ── Branches the tagger needs.  EFlowTrack_1 is fatal if absent: without it
# every impact-parameter feature silently becomes zero and the scores collapse,
# which reads as an efficiency loss rather than as an error.
REQUIRED_TAGGER_BRANCHES = ("EFlowTrack_1", "EFlowTrack_L", "magFieldBz")
OPTIONAL_TAGGER_BRANCHES = ("EFlowTrack_2",)   # patched to zeros if missing

# Per-jet score labels.  treemaker_common's extract_jet_tag_scalars() builds
# jet{i}_{label} from these, and its labels already carry the "is" prefix
# ("isG", "isU", ...), so the column names are jet1_isG etc.  "isQ" is appended
# because recojet_isQ = isU + isD is built here, not by the model.
SCORE_LABELS = [s.replace("recojet_", "") for s in TAGGER_SCORES] + ["isQ"]

# (numerator, denominators) for the pairwise discriminants.  Identical to the
# set mlevere builds in the WW->4q treemaker, so the two analyses cut on the
# same quantity.
DISCRIMINANTS = (("B", "CSGQ"), ("C", "BSGQ"), ("S", "BCGQ"),
                 ("G", "BCSQ"), ("Q", "BCSG"))
# ── track-parameter smearing (closure-test distortion) ──────────────────────
# SmearedTracks re-draws the 5 track parameters from MC truth with the
# covariance scaled by s_j*s_k.  At scale 1.0 the result is NOT the original
# sample: it is an independent re-draw.  So the closure test's "nominal" must
# also go through this code at 1.0 -- never pair a smeared production against
# an unsmeared one.
DO_SMEAR      = True
SMEAR_D0      = float(os.environ.get("SMEAR_D0", 1.5))
SMEAR_Z0      = float(os.environ.get("SMEAR_Z0", 1.5))
SMEAR_PHI     = 1.0
SMEAR_OMEGA   = 1.0
SMEAR_TLAMBDA = 1.0
SMEAR_DNDX    = float(os.environ.get("SMEAR_DNDX", 1.0))   # 1.0 = untouched

_sm = f"d0{SMEAR_D0:g}_z0{SMEAR_Z0:g}_dndx{SMEAR_DNDX:g}"

outputDir = f'/eos/user/l/lberiet/CKM_Flav_smrd/Zg_163_recopv_smear_{_sm}/'

print(f"[smear] d0={SMEAR_D0:g} z0={SMEAR_Z0:g} dndx={SMEAR_DNDX:g}")
print(f"[smear] outputDir = {outputDir}")

ROOT.gInterpreter.Declare('#include "FCCAnalyses/SmearObjects.h"')

def isr_cos_cone():
    return ROOT.TMath.Cos(ISR_CONE_DEG * ROOT.TMath.Pi() / 180.)


# ═════════════════════════════════════════════════════════════════════════════
# C++ helpers
# ═════════════════════════════════════════════════════════════════════════════

ROOT.gInterpreter.Declare(r"""
#ifndef ZG_COMMON_HELPERS
#define ZG_COMMON_HELPERS

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <vector>
#include <utility>
#include "TVector3.h"
#include "TLorentzVector.h"
#include "Math/Vector4D.h"
#include "ROOT/RVec.hxx"
#include "edm4hep/MCParticleData.h"
#include "edm4hep/ReconstructedParticleData.h"
#include "edm4hep/TrackState.h"

namespace Zg {

using RVecRP  = ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>;
using RVecMC  = ROOT::VecOps::RVec<edm4hep::MCParticleData>;
using RVecTLV = ROOT::VecOps::RVec<TLorentzVector>;
using P4      = ROOT::Math::PxPyPzEVector;

// ═══════════════════════════════════════════════════════════════════════════
// Primitives
// ═══════════════════════════════════════════════════════════════════════════

inline double rp_p(const edm4hep::ReconstructedParticleData& p) {
    return std::sqrt((double)p.momentum.x*(double)p.momentum.x
                   + (double)p.momentum.y*(double)p.momentum.y
                   + (double)p.momentum.z*(double)p.momentum.z);
}

inline TVector3 rp_dir(const edm4hep::ReconstructedParticleData& p) {
    TVector3 v(p.momentum.x, p.momentum.y, p.momentum.z);
    return (v.Mag() > 0.) ? v.Unit() : TVector3(0., 0., 1.);
}

inline P4 rp_p4(const edm4hep::ReconstructedParticleData& p) {
    return P4(p.momentum.x, p.momentum.y, p.momentum.z, p.energy);
}

inline double mc_p(const edm4hep::MCParticleData& p) {
    return std::sqrt((double)p.momentum.x*(double)p.momentum.x
                   + (double)p.momentum.y*(double)p.momentum.y
                   + (double)p.momentum.z*(double)p.momentum.z);
}

// Massless 4-vector: correct for photons, which is all we use it for.
inline P4 mc_p4_massless(const edm4hep::MCParticleData& p) {
    return P4(p.momentum.x, p.momentum.y, p.momentum.z, mc_p(p));
}

inline P4 mc_p4(const edm4hep::MCParticleData& p) {
    double pp = mc_p(p);
    double m  = (double)p.mass;
    return P4(p.momentum.x, p.momentum.y, p.momentum.z,
              std::sqrt(pp*pp + m*m));
}

double sum_p_rp(const RVecRP& v) {
    double s = 0.; for (const auto& p : v) s += rp_p(p); return s;
}

double sum_p_mc(const RVecMC& v) {
    double s = 0.; for (const auto& p : v) s += mc_p(p); return s;
}

// Total reconstructed 4-momentum.  Used to test the (0,0,0,sqrt(s)) initial
// state that every recoil mass in this analysis assumes: a 30 mrad crossing
// angle would put ~2.4 GeV of px into the initial state at 163 GeV, which is
// the width of the recoil window being cut.
P4 sum_p4_rp(const RVecRP& v) {
    P4 s(0., 0., 0., 0.);
    for (const auto& p : v) s += rp_p4(p);
    return s;
}

// Same at gen level, over the full final state (neutrinos included), so this
// IS the initial state up to generator bookkeeping.
P4 gen_total_p4(const RVecMC& all) {
    P4 s(0., 0., 0., 0.);
    for (const auto& p : all) {
        if (p.generatorStatus != 1) continue;
        s += mc_p4(p);
    }
    return s;
}

// ═══════════════════════════════════════════════════════════════════════════
// Gen level -- hard process
// ═══════════════════════════════════════════════════════════════════════════

// Hard-process quarks: a beam e+- as a direct parent.
RVecMC sel_zg_quarks_ps(const RVecMC& all, const ROOT::VecOps::RVec<int>& mom) {
    RVecMC r;
    for (const auto& p : all) {
        int apdg = std::abs(p.PDG);
        if (apdg < 1 || apdg > 5) continue;
        for (int k = p.parents_begin; k < p.parents_end; ++k) {
            int idx = mom[k];
            if (idx < 0 || idx >= (int)all.size()) continue;
            if (std::abs(all[idx].PDG) == 11) { r.push_back(p); break; }
        }
    }
    return r;
}

// ISR photons carry a beam e+- as a direct parent; FSR photons come off the
// quarks.  Same parentage logic as the hard quarks above.
RVecMC sel_gen_isr_photons(const RVecMC& all,
                           const ROOT::VecOps::RVec<int>& mom) {
    RVecMC r;
    for (const auto& p : all) {
        if (p.PDG != 22 || p.generatorStatus != 1) continue;
        for (int k = p.parents_begin; k < p.parents_end; ++k) {
            int idx = mom[k];
            if (idx < 0 || idx >= (int)all.size()) continue;
            if (std::abs(all[idx].PDG) == 11) { r.push_back(p); break; }
        }
    }
    return r;
}

// Recoil 4-vector against the FULL truth ISR system: the best any photon-based
// method could do, including radiation that left no detector hit.
P4 gen_recoil_p4(const RVecMC& isr, double sqrts) {
    P4 rec(0., 0., 0., sqrts);
    for (const auto& p : isr) rec -= mc_p4_massless(p);
    return rec;
}

// Truth ISR restricted to what the detector could have seen.
int count_gen_isr_acc(const RVecMC& isr, double pmin, double abscos_max) {
    int n = 0;
    for (const auto& p : isr) {
        double pp = mc_p(p);
        if (pp < pmin || pp <= 0.) continue;
        if (std::abs((double)p.momentum.z / pp) >= abscos_max) continue;
        ++n;
    }
    return n;
}

// ═══════════════════════════════════════════════════════════════════════════
// Gen level -- heavy-flavour hadron content  [mechanism M5]
//
// The reason this exists: the tagger responds to b/c HADRONS among the jet
// constituents, not to the hard-process PDG code.  A wzp6_ee_uu event with
// g->bb contains two genuine b jets; labelling them "light" from the hard
// process puts real b jets in the light denominator (inflating the mistag)
// and removes them from the b denominator (biasing eps_b up).  Worse for the
// correlation study: g->bb makes TWO b hadrons that usually land in opposite
// hemispheres, so both jets tag -- a real event-level tag-probe correlation
// that an event-level parton label cannot see at all.
// ═══════════════════════════════════════════════════════════════════════════

// PDG digit test: does this hadron contain a quark of flavour q?
// Standard code n_r n_L n_q1 n_q2 n_q3 n_J; mesons have n_q1 == 0.
inline bool has_quark(int pdg, int q) {
    int a = std::abs(pdg);
    if (a < 100) return false;              // not a hadron
    if (a > 1000000000) return false;       // nucleus
    int nq3 = (a / 10)   % 10;
    int nq2 = (a / 100)  % 10;
    int nq1 = (a / 1000) % 10;
    return (nq1 == q) || (nq2 == q) || (nq3 == q);
}

// Last hadron in the flavour chain: contains flavour q and has no daughter
// that also contains q.  For q=5 these are the weakly-decaying B hadrons;
// taking the last one avoids counting every excited state on the way down.
// `dau` is the Particle#1.index (daughters) relation.
RVecMC sel_last_hf_hadrons(const RVecMC& all,
                           const ROOT::VecOps::RVec<int>& dau,
                           int q, double pmin) {
    RVecMC r;
    for (const auto& p : all) {
        if (!has_quark(p.PDG, q)) continue;
        if (mc_p(p) < pmin) continue;
        bool same_flavour_daughter = false;
        for (int k = p.daughters_begin; k < p.daughters_end; ++k) {
            int idx = dau[k];
            if (idx < 0 || idx >= (int)all.size()) continue;
            if (has_quark(all[idx].PDG, q)) { same_flavour_daughter = true; break; }
        }
        if (!same_flavour_daughter) r.push_back(p);
    }
    return r;
}

// Assign each hadron to its nearest jet in angle and count per jet.  Hadrons
// further than max_ang from every jet are dropped (they are not in either
// jet's catchment and cannot be what the tagger responded to).
ROOT::VecOps::RVec<int>
count_hf_in_jets(const RVecMC& hads, const RVecTLV& jets, double max_ang) {
    ROOT::VecOps::RVec<int> n(jets.size(), 0);
    for (const auto& h : hads) {
        TVector3 vh(h.momentum.x, h.momentum.y, h.momentum.z);
        if (vh.Mag() <= 0.) continue;
        int best = -1;
        double bestang = max_ang;
        for (size_t j = 0; j < jets.size(); ++j) {
            double a = vh.Angle(jets[j].Vect());
            if (a < bestang) { bestang = a; best = (int)j; }
        }
        if (best >= 0) n[best] += 1;
    }
    return n;
}

// Gluon-splitting cross-check: gen quarks of |PDG|==q with a gluon as a direct
// parent.  One g->qq gives 2.  Shower copies can inflate this, so it is a
// cross-check on the hadron counts above, not a replacement for them.
int count_gsplit_quarks(const RVecMC& all,
                        const ROOT::VecOps::RVec<int>& mom, int q) {
    int n = 0;
    for (const auto& p : all) {
        if (std::abs(p.PDG) != q) continue;
        for (int k = p.parents_begin; k < p.parents_end; ++k) {
            int idx = mom[k];
            if (idx < 0 || idx >= (int)all.size()) continue;
            if (all[idx].PDG == 21) { ++n; break; }
        }
    }
    return n;
}

// ═══════════════════════════════════════════════════════════════════════════
// ISR photon selection by recoil mass
// ═══════════════════════════════════════════════════════════════════════════

// Sum |p| of everything else inside a cone, normalised to |p_gamma|.  Small
// for a recoil photon, large for a pi0 daughter buried in a jet.  This is what
// stops the subset scan from deleting a jet's own energy to reach m_Z.
double photon_isolation(const edm4hep::ReconstructedParticleData& g,
                        const RVecRP& all, double cos_cone) {
    double pg = rp_p(g);
    if (pg <= 0.) return 1e9;
    TVector3 dg = rp_dir(g);
    double sum = 0.;
    for (const auto& p : all) {
        double pp = rp_p(p);
        if (pp <= 0.) continue;
        // skip the photon itself (same momentum vector to float precision)
        if (std::abs(pp - pg) < 1e-6 && rp_dir(p).Dot(dg) > 1. - 1e-9) continue;
        if (rp_dir(p).Dot(dg) > cos_cone) sum += pp;
    }
    return sum / pg;
}

// Isolated, in-acceptance photons above threshold, hardest first, truncated to
// maxcand so the scan stays at 2^maxcand.
RVecRP isr_candidates(const RVecRP& photons, const RVecRP& all,
                      double pmin, double cos_cone, double iso_max,
                      double abscos_max, int maxcand) {
    std::vector<std::pair<double, edm4hep::ReconstructedParticleData>> v;
    for (const auto& g : photons) {
        double pp = rp_p(g);
        if (pp < pmin) continue;
        if (std::abs((double)g.momentum.z / pp) > abscos_max) continue;
        if (photon_isolation(g, all, cos_cone) > iso_max) continue;
        v.emplace_back(pp, g);
    }
    std::sort(v.begin(), v.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    RVecRP out;
    for (size_t i = 0; i < v.size() && (int)i < maxcand; ++i)
        out.push_back(v[i].second);
    return out;
}

// Exhaustive scan over 2^N subsets.  Ties break toward FEWER removed photons,
// so the scan does not delete extra soft objects that leave the recoil mass
// unchanged.
int best_isr_subset(const RVecRP& cand, double sqrts, double mtarget) {
    int n = (int)cand.size();
    if (n == 0) return 0;
    std::vector<P4> p4(n);
    for (int i = 0; i < n; ++i) p4[i] = rp_p4(cand[i]);

    int best = 0, bestn = 999;
    double bestd = 1e18;
    for (int m = 0; m < (1 << n); ++m) {
        P4 s(0., 0., 0., 0.);
        int cnt = 0;
        for (int i = 0; i < n; ++i)
            if ((m >> i) & 1) { s += p4[i]; ++cnt; }
        P4 rec(0., 0., 0., sqrts);
        rec -= s;
        double m2 = rec.M2();
        double M  = (m2 > 0.) ? std::sqrt(m2) : -1.;
        double d  = std::abs(M - mtarget);
        if (d < bestd - 1e-9 || (std::abs(d - bestd) < 1e-9 && cnt < bestn)) {
            bestd = d; best = m; bestn = cnt;
        }
    }
    return best;
}

RVecRP subset_of(const RVecRP& cand, int mask) {
    RVecRP out;
    for (size_t i = 0; i < cand.size(); ++i)
        if ((mask >> i) & 1) out.push_back(cand[i]);
    return out;
}

double recoil_mass(const RVecRP& sel, double sqrts) {
    P4 rec(0., 0., 0., sqrts);
    for (const auto& g : sel) rec -= rp_p4(g);
    double m2 = rec.M2();
    return (m2 > 0.) ? std::sqrt(m2) : -1.;
}

// Counters over a photon-only collection.  Deliberately does NOT test
// ReconstructedParticleData::type == 22: the winter2023 Delphes files do not
// reliably fill `type` with a PDG code, so photon-ness comes from the
// collection the particles were drawn from (Photon#0), not from the struct.
int count_above(const RVecRP& coll, double pmin) {
    int n = 0;
    for (const auto& p : coll) if (rp_p(p) > pmin) ++n;
    return n;
}

double sum_energy_above(const RVecRP& coll, double pmin) {
    double s = 0.;
    for (const auto& p : coll) if (rp_p(p) > pmin) s += (double)p.energy;
    return s;
}

// How many SELECTED reco photons match a truth ISR photon in angle.
// Greedy nearest-first with the gen side marked used, so two reco photons can
// no longer both claim the same gen photon (which deflated n_sel_fake).
int count_matched_to_gen(const RVecRP& sel, const RVecMC& gen, double maxang) {
    std::vector<char> used(gen.size(), 0);
    int n = 0;
    for (const auto& r : sel) {
        TVector3 a(r.momentum.x, r.momentum.y, r.momentum.z);
        if (a.Mag() <= 0.) continue;
        int best = -1;
        double bestang = maxang;
        for (size_t g = 0; g < gen.size(); ++g) {
            if (used[g]) continue;
            TVector3 b(gen[g].momentum.x, gen[g].momentum.y, gen[g].momentum.z);
            if (b.Mag() <= 0.) continue;
            double ang = a.Angle(b);
            if (ang < bestang) { bestang = ang; best = (int)g; }
        }
        if (best >= 0) { used[best] = 1; ++n; }
    }
    return n;
}

// ═══════════════════════════════════════════════════════════════════════════
// Jet / tagger utilities
// ═══════════════════════════════════════════════════════════════════════════

ROOT::VecOps::RVec<int>
n_constituents(const ROOT::VecOps::RVec<RVecRP>& constits) {
    ROOT::VecOps::RVec<int> n;
    n.reserve(constits.size());
    for (const auto& c : constits) n.push_back((int)c.size());
    return n;
}

// ROOT::VecOps::Max reduces one RVec to a scalar; there is no element-wise
// binary Max, so the obvious spelling does not compile.  Templated because the
// scores come back as RVec<float> or RVec<double> depending on the build.
template <typename T>
ROOT::VecOps::RVec<T> per_jet_max(const ROOT::VecOps::RVec<T>& a,
                                  const ROOT::VecOps::RVec<T>& b,
                                  const ROOT::VecOps::RVec<T>& c,
                                  const ROOT::VecOps::RVec<T>& d,
                                  const ROOT::VecOps::RVec<T>& e) {
    ROOT::VecOps::RVec<T> o;
    o.reserve(a.size());
    for (size_t i = 0; i < a.size(); ++i)
        o.push_back(std::max(std::max(std::max(a[i], b[i]),
                                      std::max(c[i], d[i])), e[i]));
    return o;
}

// isB/(isB+isC) goes NaN when the tagger collapses; NaN then fails every
// working-point cut silently, which reads as an efficiency drop, not an error.
// Returns -1 instead, which is visible in a histogram.
template <typename T>
ROOT::VecOps::RVec<T> safe_ratio(const ROOT::VecOps::RVec<T>& a,
                                 const ROOT::VecOps::RVec<T>& b) {
    ROOT::VecOps::RVec<T> out;
    out.reserve(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        T d = a[i] + b[i];
        out.push_back(d > T(0) ? a[i] / d : T(-1));
    }
    return out;
}

// ═══════════════════════════════════════════════════════════════════════════
// Exchange-symmetric tag/probe role assignment  [mechanism M4]
//
// jets_p4[0] is the harder jet, so assigning roles by index biases the
// estimator.  A hash of the low mantissa bits of quantities INVARIANT under
// 1<->2 exchange: symmetric inputs carry no information about which jet is
// which, low floating-point bits carry none about the physics, and it is
// deterministic across reruns, chunks and thread counts -- unlike an RNG
// under IMT.
//
// Caveat: deterministic WITHIN a build.  A FastJet/ROOT upgrade can flip the
// last bits of jet_esum and hence the role of a given event.  That is
// statistically harmless but means event-by-event roles are not stable across
// stack versions; pin the stack for a data/MC comparison.
// ═══════════════════════════════════════════════════════════════════════════

inline uint64_t fnv1a(const void* data, size_t n, uint64_t h) {
    const unsigned char* p = (const unsigned char*)data;
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}

int symmetric_coin(double s1, double s2, double s3) {
    uint64_t h = 1469598103934665603ULL;
    h = fnv1a(&s1, sizeof(s1), h);
    h = fnv1a(&s2, sizeof(s2), h);
    h = fnv1a(&s3, sizeof(s3), h);
    return (int)((h >> 17) & 1ULL);
}

// Angular size of each jet from its own constituents.  Exclusive clustering
// has no radius parameter, so this is a MEASURED quantity, not a setting.
// Angle to the jet axis, not eta-phi dR: at 163 GeV the jets are not
// boosted and dR is the wrong metric near the beam axis.
//   mode 0 : max constituent-to-axis angle
//   mode 1 : |p|-weighted RMS angle (girth-like)
ROOT::VecOps::RVec<double>
jet_ang_size(const ROOT::VecOps::RVec<RVecRP>& constits,
             const RVecTLV& jets, int mode) {
    ROOT::VecOps::RVec<double> out;
    out.reserve(jets.size());
    for (size_t j = 0; j < jets.size(); ++j) {
        if (j >= constits.size()) { out.push_back(-1.); continue; }
        TVector3 axis = jets[j].Vect();
        if (axis.Mag() <= 0.) { out.push_back(-1.); continue; }
        double amax = 0., sw = 0., s2 = 0.;
        for (const auto& c : constits[j]) {
            double pp = rp_p(c);
            if (pp <= 0.) continue;
            double a = TVector3(c.momentum.x, c.momentum.y,
                                c.momentum.z).Angle(axis);
            amax = std::max(amax, a);
            sw += pp;  s2 += pp * a * a;
        }
        out.push_back(mode == 0 ? amax : (sw > 0. ? std::sqrt(s2 / sw) : -1.));
    }
    return out;
}

// Angle containing `frac` of the jet's |p|.  More robust than the max, which
// is set by a single soft constituent -- and under exclusive clustering that
// constituent can sit almost anywhere.
ROOT::VecOps::RVec<double>
jet_ang_containment(const ROOT::VecOps::RVec<RVecRP>& constits,
                    const RVecTLV& jets, double frac) {
    ROOT::VecOps::RVec<double> out;
    out.reserve(jets.size());
    for (size_t j = 0; j < jets.size(); ++j) {
        if (j >= constits.size()) { out.push_back(-1.); continue; }
        TVector3 axis = jets[j].Vect();
        std::vector<std::pair<double, double>> ap;   // (angle, |p|)
        double tot = 0.;
        for (const auto& c : constits[j]) {
            double pp = rp_p(c);
            if (pp <= 0.) continue;
            ap.emplace_back(TVector3(c.momentum.x, c.momentum.y,
                                     c.momentum.z).Angle(axis), pp);
            tot += pp;
        }
        if (tot <= 0.) { out.push_back(-1.); continue; }
        std::sort(ap.begin(), ap.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        double acc = 0., r = -1.;
        for (const auto& e : ap) {
            acc += e.second;
            if (acc >= frac * tot) { r = e.first; break; }
        }
        out.push_back(r);
    }
    return out;
}

// Tracks SmearedTracks could not match to an MC particle are stored as a
// zero-filled TrackState.  A named function, because an inline lambda in a
// string Define is compiled as a void function body.
int n_dummy_tracks(const ROOT::VecOps::RVec<edm4hep::TrackState>& tks) {
    int n = 0;
    for (const auto& t : tks)
        if (t.D0 == 0.f && t.Z0 == 0.f && t.omega == 0.f) ++n;
    return n;
}

}  // namespace Zg
#endif
""")


# ═════════════════════════════════════════════════════════════════════════════
# Startup checks
# ═════════════════════════════════════════════════════════════════════════════

def _branch_usable(df, name):
    """True only if the column exists AND has a resolvable C++ type.

    RDataFrame registers every TTree branch name, including empty podio
    placeholders that carry no entries and no resolvable EDM4Hep type; passing
    one of those to a JIT lambda gives 'use of undeclared identifier'.
    """
    if name not in {str(c) for c in df.GetColumnNames()}:
        return False
    try:
        df.GetColumnType(name)
        return True
    except Exception:
        return False


def check_tagger_inputs(df):
    """Fail loudly at job start if the tagger's inputs are not really there.

    EFlowTrack_1 missing is fatal: every impact-parameter feature would be
    zero-filled and the scores would collapse toward 1/7, which looks like an
    efficiency loss rather than a broken job.  EFlowTrack_2 missing is a loud
    warning: patch_missing_eflow_branches() will stub dNdx to zeros, the job
    will run, but the s-tagging inputs are then NOT comparable to any
    production that had real dNdx -- which matters directly for |Vcs|.
    """
    missing_required = [b for b in REQUIRED_TAGGER_BRANCHES
                        if not _branch_usable(df, b)]
    if missing_required:
        raise RuntimeError(
            "Tagger input branches missing or unresolvable: "
            f"{missing_required}. Refusing to continue: the job would run and "
            "produce plausible-looking scores computed from zero-filled "
            "impact-parameter features. Check the input campaign.")

    missing_optional = [b for b in OPTIONAL_TAGGER_BRANCHES
                        if not _branch_usable(df, b)]
    if missing_optional:
        print("  [tagger inputs] *** WARNING ***  missing: "
              f"{missing_optional} -> dNdx will be stubbed to zeros. "
              "s-tagging scores are NOT comparable to a production with real "
              "dNdx.")
    else:
        print("  [tagger inputs] all present: "
              f"{list(REQUIRED_TAGGER_BRANCHES) + list(OPTIONAL_TAGGER_BRANCHES)}")
    return df


# ═════════════════════════════════════════════════════════════════════════════
# Analyser blocks
# ═════════════════════════════════════════════════════════════════════════════

def check_no_collisions(df, names, where=""):
    """Raise before any work if we are about to redefine an existing column.

    RDataFrame's error for this arrives whenever the offending Define is
    reached, which on a batch job means after the chunk has already been
    opened.  Checking the whole block up front turns that into one readable
    message naming every clash at once.
    """
    existing = {str(c) for c in df.GetColumnNames()}
    clash = sorted(n for n in names if n in existing)
    if clash:
        raise RuntimeError(
            f"column name collision in {where or 'this block'}: {clash} "
            "already exist(s) on the dataframe. These are almost certainly "
            "owned by ExclusiveJetClusteringHelper (jet_p, jet_e, jet_mass, "
            "jet_phi, jet_theta, jet_nconst) or by JetFlavourHelper. Reuse "
            "the existing column instead of redefining it.")
    return df


def define_gen_zg(df, with_isr=True):
    """Hard-process quarks, truth Z, heavy-flavour content.

    with_isr=True (Z+gamma): also the truth ISR system and the recoil-vs-qq
    consistency check.  with_isr=False (Z -> qqbar): the Z is the qq system,
    no photon bookkeeping.

    Emits the flavour label but NOT a filter on it beyond the 2-quark
    requirement; check the cutflow report per flavour sample, because an
    unequal gen-selection efficiency between b/c/s/u/d biases every relative
    efficiency downstream.
    """
    df = df.Define("mc_particles", "Particle")
    df = df.Alias("Particle0", "Particle#0.index")   # parents
    df = df.Alias("Particle1", "Particle#1.index")   # daughters

    # ── hard-process quarks ──────────────────────────────────────────────────
    df = df.Define("gen_quarks_ps", "Zg::sel_zg_quarks_ps(mc_particles, Particle0)")
    df = df.Define("n_gen_quarks_ps",
                   "FCCAnalyses::MCParticle::get_n(gen_quarks_ps)")
    df = df.Filter("n_gen_quarks_ps == 2",
                   "gen: exactly 2 hard-process quarks from Z")

    df = df.Define("gen_quarks_ps_tlv",
                   "FCCAnalyses::MCParticle::get_tlv(gen_quarks_ps)")
    df = df.Define("gen_q1_p4", "gen_quarks_ps_tlv[0]")
    df = df.Define("gen_q2_p4", "gen_quarks_ps_tlv[1]")
    for q, src in (("q1", "gen_q1_p4"), ("q2", "gen_q2_p4")):
        df = df.Define(f"gen_{q}_p",     f"{src}.P()")
        df = df.Define(f"gen_{q}_theta", f"{src}.Theta()")
        df = df.Define(f"gen_{q}_phi",   f"{src}.Phi()")
    df = df.Define("gen_q1_pdg", "(int)gen_quarks_ps[0].PDG")
    df = df.Define("gen_q2_pdg", "(int)gen_quarks_ps[1].PDG")
    df = df.Define("gen_opang",  "gen_q1_p4.Angle(gen_q2_p4.Vect())")

    # PARTON-LEVEL, EVENT-LEVEL flavour label.  Correct for what the hard
    # process was; NOT the same thing as what the tagger responds to.  Use it
    # together with the per-jet hadron counts below, never alone.
    df = df.Define("true_flavour_pdg", "std::abs(gen_q1_pdg)")
    df = df.Define("gen_flavour_consistent",
                   "(int)(std::abs(gen_q1_pdg) == std::abs(gen_q2_pdg))")

    df = df.Define("gen_Zqq_p4",    "gen_q1_p4 + gen_q2_p4")
    df = df.Define("gen_Zqq_m",     "gen_Zqq_p4.M()")
    df = df.Define("gen_Zqq_p",     "gen_Zqq_p4.P()")
    df = df.Define("gen_Zqq_theta", "gen_Zqq_p4.Theta()")

    if with_isr:
        # ── truth ISR system ─────────────────────────────────────────────────
        df = df.Define("gen_isr_photons",
                       "Zg::sel_gen_isr_photons(mc_particles, Particle0)")
        df = df.Define("gen_n_isr",    "(int)gen_isr_photons.size()")
        df = df.Define("gen_isr_esum", "Zg::sum_p_mc(gen_isr_photons)")
        df = df.Define("gen_n_isr_acc",
                       f"Zg::count_gen_isr_acc(gen_isr_photons, {ISR_PMIN}, {ISR_COSMAX})")

        # Truth Z, two independent ways.
        #   gen_Z_*      : recoil against the truth ISR system (the ceiling on any
        #                  photon-based method)
        #   gen_Zqq_*    : the qq system itself, independent of ISR bookkeeping
        # Their difference is the check on whether sel_gen_isr_photons is complete.
        # (The previous get_hard_gen_photon_p4 took the hardest status-1 photon in
        # the WHOLE event with no parentage cut -- competing against every pi0
        # daughter in a hadronic event -- and disagreed with gen_m_recoil silently.)
        df = df.Define("gen_Z_p4",    f"Zg::gen_recoil_p4(gen_isr_photons, {SQRTS})")
        df = df.Define("gen_m_recoil", "gen_Z_p4.M()")
        df = df.Define("gen_dm_Z_Zqq",  "gen_Z_p4.M() - gen_Zqq_m")

        # Hardest truth ISR photon, for the single-photon plots.
        df = df.Define("gen_photon_p",
            "gen_n_isr > 0 ? ROOT::VecOps::Max("
            "FCCAnalyses::MCParticle::get_p(gen_isr_photons)) : -1.")
    else:
        # Z -> qqbar: the Z is the qq system.  No ISR recoil to construct.
        df = df.Define("gen_Z_p4", "gen_Zqq_p4")

    df = df.Define("gen_Z_m",      "gen_Z_p4.M()")
    df = df.Define("gen_Z_p",      "gen_Z_p4.P()")
    df = df.Define("gen_Z_theta",  "gen_Z_p4.Theta()")

    # ── crossing-angle / initial-state check ─────────────────────────────────
    # Every recoil mass here assumes the initial state is (0,0,0,sqrt(s)).
    # A 30 mrad crossing would put ~2.4 GeV of px in, comparable to the recoil
    # window.  If gen_tot_px is not centred on zero, recoil_mass needs the
    # boost.
    df = df.Define("gen_tot_p4", "Zg::gen_total_p4(mc_particles)")
    df = df.Define("gen_tot_px", "gen_tot_p4.Px()")
    df = df.Define("gen_tot_py", "gen_tot_p4.Py()")
    df = df.Define("gen_tot_pz", "gen_tot_p4.Pz()")
    df = df.Define("gen_tot_e",  "gen_tot_p4.E()")

    # ── heavy-flavour hadrons  [M5] ──────────────────────────────────────────
    for tag, pmin in (("loose", HF_HADRON_PMIN_LOOSE),
                      ("tight", HF_HADRON_PMIN_TIGHT)):
        df = df.Define(f"gen_bhad_{tag}",
            f"Zg::sel_last_hf_hadrons(mc_particles, Particle1, 5, {pmin})")
        df = df.Define(f"gen_chad_{tag}",
            f"Zg::sel_last_hf_hadrons(mc_particles, Particle1, 4, {pmin})")
        df = df.Define(f"n_bhad_{tag}", f"(int)gen_bhad_{tag}.size()")
        df = df.Define(f"n_chad_{tag}", f"(int)gen_chad_{tag}.size()")

    # Cross-check on the hadron counts: gen quarks with a gluon parent.
    # One g->bb gives 2.  Shower copies can inflate it, so treat the hadron
    # counts as primary and this as corroboration.
    df = df.Define("n_gsplit_b_quarks",
                   "Zg::count_gsplit_quarks(mc_particles, Particle0, 5)")
    df = df.Define("n_gsplit_c_quarks",
                   "Zg::count_gsplit_quarks(mc_particles, Particle0, 4)")
    return df


def define_isr_selection(df):
    """Recoil-mass ISR photon selection, its truth validation, and the cleaned
    reconstructed-particle collection the clustering runs on."""
    cos_cone = isr_cos_cone()

    df = df.Alias("Photon0", "Photon#0.index")
    df = df.Define("photons_all",
        "FCCAnalyses::ReconstructedParticle::get(Photon0, ReconstructedParticles)")

    # ── selection ────────────────────────────────────────────────────────────
    df = df.Define("isr_cand",
        f"Zg::isr_candidates(photons_all, ReconstructedParticles, {ISR_PMIN}, "
        f"{cos_cone}, {ISR_ISO_MAX}, {ISR_COSMAX}, {ISR_MAXCAND})")
    df = df.Define("n_gamma_cand", "(int)isr_cand.size()")
    df = df.Define("isr_mask",
        f"Zg::best_isr_subset(isr_cand, {SQRTS}, {MZ_TARGET})")
    df = df.Define("photon_isr",     "Zg::subset_of(isr_cand, isr_mask)")
    df = df.Define("n_gamma_sel",    "(int)photon_isr.size()")
    df = df.Define("has_isr_photon", "(int)(n_gamma_sel > 0)")
    df = df.Define("isr_psum",       "Zg::sum_p_rp(photon_isr)")

    # The selection variable.  This is simultaneously a photon ASSIGNMENT and a
    # Z SELECTION: events whose ISR went down the beam pipe cannot be rescued
    # and sit at m_recoil ~ sqrt(s).  Cut the window OFFLINE -- deliberately no
    # filter here.
    df = df.Define("m_recoil",  f"Zg::recoil_mass(photon_isr, {SQRTS})")
    df = df.Define("dm_recoil", f"m_recoil - {MZ_TARGET}")
    # Recoil if EVERY candidate were removed: shows how much the scan is
    # actually choosing rather than just taking everything.
    df = df.Define("m_recoil_allcand", f"Zg::recoil_mass(isr_cand, {SQRTS})")

    # ── truth validation ─────────────────────────────────────────────────────
    df = df.Define("n_sel_matched",
        f"Zg::count_matched_to_gen(photon_isr, gen_isr_photons, {ISR_MATCH_ANG})")
    df = df.Define("n_sel_fake", "n_gamma_sel - n_sel_matched")
    df = df.Define("isr_purity",
        "n_gamma_sel > 0 ? (double)n_sel_matched / (double)n_gamma_sel : -1.")
    # Headline: should peak sharply at zero for events with ISR in acceptance.
    df = df.Define("dm_recoil_gen", "m_recoil - gen_m_recoil")

    # ── cleaned collection ───────────────────────────────────────────────────
    df = df.Define("RP_no_hard_photon",
        "FCCAnalyses::ReconstructedParticle::remove("
        "ReconstructedParticles, photon_isr)")
    # remove() matches on field equality, so exactly-degenerate photons could
    # be double-removed.  This must equal n_gamma_sel in every event.
    df = df.Define("n_rp_removed",
        "(int)ReconstructedParticles.size() - (int)RP_no_hard_photon.size()")
    df = df.Define("rp_removal_consistent", "(int)(n_rp_removed == n_gamma_sel)")

    # ── size of the legacy sel_p() bug ───────────────────────────────────────
    # Evaluated at LEGACY_SELP_THRESHOLD, which is what the old cut actually
    # used.  These were previously computed at ISR_PMIN while carrying _gt10
    # names, so they did not measure what the docstring claimed.
    df = df.Define("photons_kept",
        "FCCAnalyses::ReconstructedParticle::remove(photons_all, photon_isr)")
    df = df.Define("n_gamma_legacy",
        f"Zg::count_above(photons_all, {LEGACY_SELP_THRESHOLD})")
    df = df.Define("n_gamma_legacy_kept",
        f"Zg::count_above(photons_kept, {LEGACY_SELP_THRESHOLD})")
    df = df.Define("e_gamma_legacy_kept",
        f"Zg::sum_energy_above(photons_kept, {LEGACY_SELP_THRESHOLD})")

    # ── reco initial-state check (pairs with gen_tot_*) ──────────────────────
    df = df.Define("rp_tot_p4", "Zg::sum_p4_rp(ReconstructedParticles)")
    df = df.Define("rp_tot_px", "rp_tot_p4.Px()")
    df = df.Define("rp_tot_py", "rp_tot_p4.Py()")
    df = df.Define("rp_tot_pz", "rp_tot_p4.Pz()")
    df = df.Define("rp_tot_e",  "rp_tot_p4.E()")
    return df

def define_smeared_tracks(df, track_collection="EFlowTrack_1",
                          mc_col="mc_particles"):
    """Smeared copy of the track states, parallel to `track_collection`.

    Same length and order as the input, so every index-based lookup
    downstream -- getRP2TRK for the vertex fit, and JetFlavourHelper's
    constituent-to-track association -- keeps working unchanged.

    Must be called after define_gen_zg (which defines mc_particles) and
    before define_track_diagnostics, whose RP2TRK feeds the vertex fit.
    """
    df = df.Alias("MCRecoAssociations0", "MCRecoAssociations#0.index")
    df = df.Alias("MCRecoAssociations1", "MCRecoAssociations#1.index")
    df = df.Define("RP2MC_idx",
        "FCCAnalyses::ReconstructedParticle2MC::getRP2MC_index("
        "MCRecoAssociations0, MCRecoAssociations1, ReconstructedParticles)")

    df = df.Define("EFlowTrack_1_sm",
        f"FCCAnalyses::SmearObjects::SmearedTracks("
        f"{SMEAR_D0}, {SMEAR_PHI}, {SMEAR_OMEGA}, {SMEAR_Z0}, "
        f"{SMEAR_TLAMBDA}, false)"
        f"(ReconstructedParticles, {track_collection}, RP2MC_idx, {mc_col})")

    # SmearedTracks writes a default-constructed TrackState when no MC match
    # is found: D0 = Z0 = 0 with a null covariance, i.e. a perfectly prompt
    # track.  This must be 0, or part of the "distortion" is an artefact.
    df = df.Define("n_dummy_trk", "Zg::n_dummy_tracks(EFlowTrack_1_sm)")

    if SMEAR_DNDX != 1.0:
        df = df.Define("EFlowTrack_2_sm",
            f"FCCAnalyses::SmearObjects::SmearedTracksdNdx({SMEAR_DNDX}, false)"
            f"(ReconstructedParticles, EFlowTrack_2, EFlowTrack_L, "
            f"RP2MC_idx, {mc_col})")
    return df


def smear_branches():
    return ['n_dummy_trk'] if DO_SMEAR else []

def tagger_collections_smeared(pf_collection):
    from treemaker_common import make_tagger_collections
    coll = make_tagger_collections(pf_collection)
    if DO_SMEAR:
        coll["TrackState"] = "EFlowTrack_1_sm"
        if SMEAR_DNDX != 1.0:
            coll["dNdx"] = "EFlowTrack_2_sm"
    return coll

def define_track_diagnostics(df, track_collection="EFlowTrack_1"):
    """Per-event evidence that the impact-parameter features had input.

    n_trkstate == 0 or n_trk == 0 across the sample means the track states are
    absent and nothing downstream of the tagger means anything.
    """
    df = df.Define("n_trkstate", f"(int){track_collection}.size()")
    df = df.Define("RP2TRK",
        "FCCAnalyses::ReconstructedParticle2Track::getRP2TRK("
        f"ReconstructedParticles, {track_collection})")
    df = df.Define("n_trk", "(int)RP2TRK.size()")
    return df


# Columns this block creates, checked up front against what the clustering
# helper already put on the dataframe.  jet_nconst is deliberately NOT here:
# it is handled separately below because the helper owns it.
_JET_BLOCK_COLUMNS = [
    "jets_p4", "n_reco_jets", "jet1", "jet2",
    "d_12", "d_23", "d_34", "y_23", "y_23_Q2",
    "jet1_nconst", "jet2_nconst",
    "jet_order_consistent", "jet_is_e_ordered",
    "jet1_e", "jet2_e", "jet_esum", "jet_easym", "jet_emin", "jet_emax",
    "reco_opang", "reco_dijet_m", "reco_dijet_p", "reco_dijet_theta",
    "jet_ang_max", "jet_ang_rms", "jet_ang_r90",
    "jet1_ang_max", "jet1_ang_rms", "jet1_ang_r90",
    "jet2_ang_max", "jet2_ang_rms", "jet2_ang_r90",
    "reco_dR", "reco_dR_eta", "reco_dR_y",
]


def define_jets_and_scales(df, helper, dmerge_clust=None):
    """jets_p4, the 2-jet requirement, merging scales [M2] and energies [M1].

    dmerge_clust names the FCCAnalysesJet column the d_ij merge scales are read
    from.  It defaults to the helper's own clustering, which is right whenever
    that clustering is a Durham sequence.  It is NOT right for a jet finder
    that has no exclusive sequence -- the energy-recovery clustering leaves
    exclusive_dmerge empty and get_exclusive_dmerge answers -1 instead of
    failing, so d_12/d_23/d_34/y_23 would be -1 in every event and the y23 cut
    downstream would silently select nothing.  Such a chain passes the column
    of a separate Durham clustering of the same particles (see zg_recov).
    """
    df = check_no_collisions(df, _JET_BLOCK_COLUMNS, "define_jets_and_scales")
    clust = dmerge_clust if dmerge_clust is not None else f"_jet{helper.tag}"

    df = df.Define("jets_p4",
                   f"JetConstituentsUtils::compute_tlv_jets({helper.jets})")
    df = df.Define("n_reco_jets", "(int)jets_p4.size()")
    df = df.Filter("n_reco_jets == 2", "exactly 2 reco jets")
    df = df.Define("jet1", "jets_p4[0]")
    df = df.Define("jet2", "jets_p4[1]")

    # ── M2: merging scales ───────────────────────────────────────────────────
    # d_23 is the 3->2 scale: large means a hard third parton was forcibly
    # absorbed, contaminating BOTH jets at once.
    df = df.Define("d_12", f"JetClusteringUtils::get_exclusive_dmerge({clust}, 1)")
    df = df.Define("d_23", f"JetClusteringUtils::get_exclusive_dmerge({clust}, 2)")
    df = df.Define("d_34", f"JetClusteringUtils::get_exclusive_dmerge({clust}, 3)")
    df = df.Define("y_23", "d_12 > 0 ? d_23 / d_12 : -1.")
    df = df.Define("y_23_Q2",
        "(double)jet1.E() + (double)jet2.E() > 0 ? "
        "d_23 / (((double)jet1.E() + (double)jet2.E()) * "
        "((double)jet1.E() + (double)jet2.E())) : -1.")

    # Per-jet constituent multiplicity: the direct handle on M2, since a
    # mis-assigned constituent leaves one jet and joins the other.
    #
    # ExclusiveJetClusteringHelper.define() ALREADY provides jet_nconst -- it
    # is one of the helper-owned columns alongside jet_p / jet_e / jet_mass /
    # jet_phi / jet_theta, even though treemaker_common's docstring omits it
    # from that list.  Defining it again is a hard RDataFrame error.  Reuse
    # the helper's column when it exists and only build our own (under the
    # same name, so everything downstream is identical either way) when it
    # does not.  The cast is because the helper may hand back floats.
    nconst = f"jet_nconst{helper.tag}"
    if not _branch_usable(df, nconst):
        df = df.Define(nconst, f"Zg::n_constituents({helper.constituents})")
    df = df.Define("jet1_nconst", f"(int){nconst}[0]")
    df = df.Define("jet2_nconst", f"(int){nconst}[1]")

    df = df.Define("jet_ang_max",
        f"Zg::jet_ang_size({helper.constituents}, jets_p4, 0)")
    df = df.Define("jet_ang_rms",
        f"Zg::jet_ang_size({helper.constituents}, jets_p4, 1)")
    df = df.Define("jet_ang_r90",
        f"Zg::jet_ang_containment({helper.constituents}, jets_p4, 0.90)")
    for i in (1, 2):
        for v in ("ang_max", "ang_rms", "ang_r90"):
            df = df.Define(f"jet{i}_{v}", f"jet_{v}[{i-1}]")
    # ── consistency flags ────────────────────────────────────────────────────
    df = df.Define("jet_order_consistent",
        "(int)(std::abs((double)jets_p4[0].P() - (double)jet_p[0]) < 1e-3 && "
        "std::abs((double)jets_p4[1].P() - (double)jet_p[1]) < 1e-3)")
    df = df.Define("jet_is_e_ordered",
        "(int)((double)jet1.E() >= (double)jet2.E())")

    # ── M1: energies ─────────────────────────────────────────────────────────
    # E1+E2 is pinned by the recoil against the photon, which is what makes the
    # two energies anti-correlated in the first place.
    df = df.Define("jet1_e", "(double)jet1.E()")
    df = df.Define("jet2_e", "(double)jet2.E()")
    df = df.Define("jet_esum",  "jet1_e + jet2_e")
    df = df.Define("jet_easym", "(jet1_e - jet2_e) / (jet1_e + jet2_e)")
    df = df.Define("jet_emin",  "std::min(jet1_e, jet2_e)")
    df = df.Define("jet_emax",  "std::max(jet1_e, jet2_e)")

    df = df.Define("reco_opang", "jet1.Vect().Angle(jet2.Vect())")
    df = df.Define("reco_dijet_m",     "(jet1 + jet2).M()")
    df = df.Define("reco_dijet_p",     "(jet1 + jet2).P()")
    df = df.Define("reco_dijet_theta", "(jet1 + jet2).Theta()")
    df = df.Define("reco_dR", "(double)jet1.DeltaR(jet2)")
        # pseudorapidity-based
    df = df.Define("reco_dR_eta",
        "std::sqrt(std::pow((double)(jet1.Eta() - jet2.Eta()), 2) + "
        "std::pow(TVector2::Phi_mpi_pi(jet1.Phi() - jet2.Phi()), 2))")

    # rapidity-based
    df = df.Define("reco_dR_y",
        "std::sqrt(std::pow((double)(jet1.Rapidity() - jet2.Rapidity()), 2) + "
        "std::pow(TVector2::Phi_mpi_pi(jet1.Phi() - jet2.Phi()), 2))")
    return df


def define_getflavour_pair_filter(df, helper, mc_col="mc_particles"):
    """True flavour from FCCAnalyses get_flavour; keep only events whose two
    exclusive jets reproduce the hard-process quark pair.

    JetTaggingUtils::get_flavour matches each clustered PseudoJet to status-2
    / 70–79 partons inside a 0.3 rad cone (quarks over gluons, heavier over
    lighter).  The event is kept if the unordered pair of |PDG| labels equals
    {|gen_q1_pdg|, |gen_q2_pdg|}:

      Z→bb  → both jets 5
      Z→cc  → both jets 4
      Z→ss/uu/dd → both jets 3/2/1
      WW Vcs → {4, 3}, Vud → {2, 1}, …

    Gluon (21), unmatched (0), or a gluon-split heavier flavour in either jet
    fail the pair and the event is dropped.  Call after exclusive n=2
    clustering (and the n_reco_jets==2 filter) and before the tagger.
    """
    df = df.Define(
        "jet_flavour_getflavour",
        f"FCCAnalyses::JetTaggingUtils::get_flavour({helper.jets}, {mc_col})")
    df = df.Define(
        "jet1_getflavour_pdg",
        "jet_flavour_getflavour.size() > 0 ? (int)jet_flavour_getflavour[0] : 0")
    df = df.Define(
        "jet2_getflavour_pdg",
        "jet_flavour_getflavour.size() > 1 ? (int)jet_flavour_getflavour[1] : 0")
    df = df.Define(
        "getflavour_pair_ok",
        "(int)("
        "(jet1_getflavour_pdg == std::abs(gen_q1_pdg) && "
        " jet2_getflavour_pdg == std::abs(gen_q2_pdg)) || "
        "(jet1_getflavour_pdg == std::abs(gen_q2_pdg) && "
        " jet2_getflavour_pdg == std::abs(gen_q1_pdg))"
        ")")
    df = df.Filter(
        "getflavour_pair_ok == 1",
        "get_flavour: exclusive jets match hard-process quark pair")
    return df


def define_hf_jet_content(df):
    """Per-jet heavy-flavour hadron counts [M5].

    This is the label the tagger actually responds to.  Compare it with
    true_flavour_pdg: in a uu/dd/ss sample any jet with n_bhad > 0 is a real
    b jet from gluon splitting, and both jets tagging in such an event is a
    genuine tag-probe correlation that the parton-level event label cannot
    represent.
    """
    for tag in ("loose", "tight"):
        df = df.Define(f"jet_nbhad_{tag}",
            f"Zg::count_hf_in_jets(gen_bhad_{tag}, jets_p4, {HF_JET_MAXANG})")
        df = df.Define(f"jet_nchad_{tag}",
            f"Zg::count_hf_in_jets(gen_chad_{tag}, jets_p4, {HF_JET_MAXANG})")
        for i in (1, 2):
            df = df.Define(f"jet{i}_nbhad_{tag}", f"jet_nbhad_{tag}[{i-1}]")
            df = df.Define(f"jet{i}_nchad_{tag}", f"jet_nchad_{tag}[{i-1}]")

    # Per-jet hadron-level flavour, b > c > light, in the same 5/4/0 coding as
    # true_flavour_pdg so the two can be compared directly.
    for i in (1, 2):
        df = df.Define(f"jet{i}_hf_flavour",
            f"jet{i}_nbhad_loose > 0 ? 5 : (jet{i}_nchad_loose > 0 ? 4 : 0)")

    # The M5 signature: heavy flavour in BOTH jets of an event whose hard
    # process was not that flavour.  This is the subpopulation to remove when
    # re-measuring C.
    df = df.Define("hf_both_jets",
        "(int)(jet1_hf_flavour > 0 && jet2_hf_flavour > 0)")
    df = df.Define("hf_label_disagrees",
        "(int)((jet1_hf_flavour != (int)true_flavour_pdg && jet1_hf_flavour > 0) || "
        "(jet2_hf_flavour != (int)true_flavour_pdg && jet2_hf_flavour > 0))")
    df = df.Define("hf_clean",
        "(int)(jet1_hf_flavour == 0 && jet2_hf_flavour == 0)")
    return df


def define_tagger_observables(df):
    """isQ, sanity flags and the NaN-guarded pairwise discriminants."""
    df = df.Define("recojet_isQ", "recojet_isU + recojet_isD")
    df = df.Define("jet1_isQ", "(float)recojet_isQ[0]")
    df = df.Define("jet2_isQ", "(float)recojet_isQ[1]")

    df = df.Define("tag_sum",
        "recojet_isG + recojet_isU + recojet_isD + recojet_isS + "
        "recojet_isC + recojet_isB + recojet_isTAU")
    df = df.Define("tag_max",
        "Zg::per_jet_max(recojet_isB, recojet_isC, recojet_isS,"
        " recojet_isQ, recojet_isG)")
    df = df.Define("tagger_degenerate",
        f"(int)(ROOT::VecOps::Min(tag_max) < {TAG_DEGENERATE_MAX})")

    for num, dens in DISCRIMINANTS:
        for den in dens:
            df = df.Define(f"D{num}_vs{den}",
                           f"Zg::safe_ratio(recojet_is{num}, recojet_is{den})")
    return df


def define_tag_probe(df):
    """Exchange-symmetric roles [M4], and every per-jet quantity projected onto
    them.

    The scores are projected here rather than left to downstream indexing.
    Downstream code that reaches for [0]/[1] silently reintroduces the
    energy-ordering bias this block exists to remove; with jetT_*/jetP_* it
    cannot.
    """
    df = df.Define("tp_flip",
        "Zg::symmetric_coin(jet_esum, jet1_e * jet2_e, std::cos(reco_opang))")
    df = df.Define("tag_idx",   "tp_flip")
    df = df.Define("probe_idx", "1 - tp_flip")

    # kinematics
    for name, expr1, expr2 in (
        ("e",     "jet1_e",                 "jet2_e"),
        ("p",     "(double)jet1.P()",       "(double)jet2.P()"),
        ("theta", "(double)jet1.Theta()",   "(double)jet2.Theta()"),
        ("mass",  "(double)jet1.M()",       "(double)jet2.M()"),
        ("nconst", "jet1_nconst",           "jet2_nconst"),
        ("ang_max", "jet1_ang_max",         "jet2_ang_max"),
        ("ang_rms", "jet1_ang_rms",         "jet2_ang_rms"),
        ("ang_r90", "jet1_ang_r90",         "jet2_ang_r90"),
    ):
        df = df.Define(f"jetT_{name}", f"tp_flip ? {expr2} : {expr1}")
        df = df.Define(f"jetP_{name}", f"tp_flip ? {expr1} : {expr2}")

    # truth flavour content
    for name in ("hf_flavour", "nbhad_loose", "nchad_loose",
                 "nbhad_tight", "nchad_tight"):
        df = df.Define(f"jetT_{name}", f"tp_flip ? jet2_{name} : jet1_{name}")
        df = df.Define(f"jetP_{name}", f"tp_flip ? jet1_{name} : jet2_{name}")

    # raw scores
    for lab in SCORE_LABELS:
        df = df.Define(f"jetT_{lab}", f"tp_flip ? jet2_{lab} : jet1_{lab}")
        df = df.Define(f"jetP_{lab}", f"tp_flip ? jet1_{lab} : jet2_{lab}")

    # discriminants
    for num, dens in DISCRIMINANTS:
        for den in dens:
            d = f"D{num}_vs{den}"
            df = df.Define(f"jetT_{d}", f"{d}[tag_idx]")
            df = df.Define(f"jetP_{d}", f"{d}[probe_idx]")
    return df


def define_mc_pv(df):
    """Generator primary vertex, in mm.

    Same columns as the reco-PV treemakers.  This chain does not fit a
    vertex, so pv_res* is not defined.
    Requires mc_particles (define_gen_zg).
    """
    df = df.Define(
        "mc_pv_p4",
        "FCCAnalyses::MCParticle::get_EventPrimaryVertexP4()(mc_particles)")
    df = df.Define("mc_pv_x", "(double)mc_pv_p4.X()")
    df = df.Define("mc_pv_y", "(double)mc_pv_p4.Y()")
    df = df.Define("mc_pv_z", "(double)mc_pv_p4.Z()")
    return df


def define_truth_matching(df):
    """dR jet<->hard-quark assignment with the swap margin."""
    df = df.Define("dR_j1q1", "(double)jet1.Vect().Angle(gen_q1_p4.Vect())")
    df = df.Define("dR_j1q2", "(double)jet1.Vect().Angle(gen_q2_p4.Vect())")
    df = df.Define("dR_j2q1", "(double)jet2.Vect().Angle(gen_q1_p4.Vect())")
    df = df.Define("dR_j2q2", "(double)jet2.Vect().Angle(gen_q2_p4.Vect())")
    df = df.Define("_jet_match_swap",
                   "(int)(dR_j1q1 + dR_j2q2 > dR_j1q2 + dR_j2q1)")
    df = df.Define("jet1_match_dR", "_jet_match_swap ? dR_j1q2 : dR_j1q1")
    df = df.Define("jet2_match_dR", "_jet_match_swap ? dR_j2q1 : dR_j2q2")
    df = df.Define("dR_swap_margin",
                   "std::abs((dR_j1q1 + dR_j2q2) - (dR_j1q2 + dR_j2q1))")
    return df


# ═════════════════════════════════════════════════════════════════════════════
# Branch lists -- kept here so the two treemakers cannot drift in output either
# ═════════════════════════════════════════════════════════════════════════════

def mc_pv_branches():
    return ["mc_pv_x", "mc_pv_y", "mc_pv_z"]


def gen_branches(with_isr=True):
    out = [
        'gen_q1_p', 'gen_q1_theta', 'gen_q1_phi', 'gen_q1_pdg',
        'gen_q2_p', 'gen_q2_theta', 'gen_q2_phi', 'gen_q2_pdg',
        'gen_opang', 'true_flavour_pdg', 'gen_flavour_consistent',
        'gen_Z_m', 'gen_Z_p', 'gen_Z_theta',
        'gen_Zqq_m', 'gen_Zqq_p', 'gen_Zqq_theta',
        'gen_tot_px', 'gen_tot_py', 'gen_tot_pz', 'gen_tot_e',
        'n_bhad_loose', 'n_chad_loose', 'n_bhad_tight', 'n_chad_tight',
        'n_gsplit_b_quarks', 'n_gsplit_c_quarks',
    ]
    if with_isr:
        out += [
            'gen_photon_p', 'gen_dm_Z_Zqq',
            'gen_n_isr', 'gen_n_isr_acc', 'gen_isr_esum', 'gen_m_recoil',
        ]
    return out


def isr_branches():
    return [
        'n_gamma_cand', 'n_gamma_sel', 'has_isr_photon', 'isr_psum',
        'm_recoil', 'dm_recoil', 'm_recoil_allcand',
        'n_sel_matched', 'n_sel_fake', 'isr_purity', 'dm_recoil_gen',
        'n_rp_removed', 'rp_removal_consistent',
        'n_gamma_legacy', 'n_gamma_legacy_kept', 'e_gamma_legacy_kept',
        'rp_tot_px', 'rp_tot_py', 'rp_tot_pz', 'rp_tot_e',
    ]


def track_branches():
    return ['n_trkstate', 'n_trk']


def jet_branches():
    return [
        'n_reco_jets',
        'reco_jet1_p', 'reco_jet1_theta', 'reco_jet1_phi', 'reco_jet1_mass',
        'reco_jet2_p', 'reco_jet2_theta', 'reco_jet2_phi', 'reco_jet2_mass',
        'jet1_e', 'jet2_e', 'jet_esum', 'jet_easym', 'jet_emin', 'jet_emax',
        'jet_ang_max', 'jet_ang_rms', 'jet_ang_r90',
        'jet1_ang_max', 'jet1_ang_rms', 'jet1_ang_r90',
        'jet2_ang_max', 'jet2_ang_rms', 'jet2_ang_r90',
        'jet1_nconst', 'jet2_nconst',
        'reco_opang', 'reco_dijet_m', 'reco_dijet_p', 'reco_dijet_theta',
        'd_12', 'd_23', 'd_34', 'y_23', 'y_23_Q2',
        'jet_order_consistent', 'jet_is_e_ordered',
        'jet_p', 'jet_theta', 'jet_e', 'jet_pt', 'jet_costheta',
        'reco_dR', 'reco_dR_eta', 'reco_dR_y',
    ]


def hf_branches():
    out = []
    for tag in ("loose", "tight"):
        for i in (1, 2):
            out += [f'jet{i}_nbhad_{tag}', f'jet{i}_nchad_{tag}']
    out += ['jet1_hf_flavour', 'jet2_hf_flavour',
            'hf_both_jets', 'hf_label_disagrees', 'hf_clean']
    return out


def tagger_branches():
    out = ['recojet_isQ', 'tag_sum', 'tag_max', 'tagger_degenerate']
    for i in (1, 2):
        out += [f'jet{i}_{lab}' for lab in SCORE_LABELS]
    for num, dens in DISCRIMINANTS:
        out += [f'D{num}_vs{den}' for den in dens]
    return out


def tag_probe_branches():
    out = ['tp_flip', 'tag_idx', 'probe_idx']
    for role in ("jetT", "jetP"):
        out += [f'{role}_{n}' for n in
                ("e", "p", "theta", "mass", "nconst", "hf_flavour",
                 "ang_max", "ang_rms", "ang_r90",
                 "nbhad_loose", "nchad_loose", "nbhad_tight", "nchad_tight")]
        out += [f'{role}_{lab}' for lab in SCORE_LABELS]
        for num, dens in DISCRIMINANTS:
            out += [f'{role}_D{num}_vs{den}' for den in dens]
    return out


def truth_matching_branches():
    return [
        'jet1_match_dR', 'jet2_match_dR', 'dR_swap_margin',
        'jet1_getflavour_pdg', 'jet2_getflavour_pdg', 'getflavour_pair_ok',
    ]