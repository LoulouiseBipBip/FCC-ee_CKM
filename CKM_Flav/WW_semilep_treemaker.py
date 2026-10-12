'''
WW → μν qq treemaker: gen-level truth quarks + reco jet kinematics.

Produces ntuples for all six CKM flavours at √s = 163 GeV (WW threshold).
Each event is required to have exactly one isolated lepton (semileptonic WW
selection) and exactly two reco jets (from the hadronic W).

Gen-level content
─────────────────
• Truth muons / neutrinos (genStatus=1, stable).
• Post-shower heavy quarks (c, b and their anti-quarks) identified by the
  Lund-string daughter criterion (see DUPLICATE QUARKS note below).
  For light-quark samples (Vud, Vus) the heavy-quark branches are empty
  (-999 fill value); use the reco jet kinematics for those.

Reco-level content
──────────────────
• Iso-lepton selection + FSR dressing (from treemaker_common).\'=
• Exclusive anti-kt → 2 jets on the lepton/FSR-cleaned RP collection.
• Scalar jet kinematics: p, pt, θ, φ, η, cos θ, mass for jet1 and jet2.
• reco_dijet_m = M(jet1+jet2) = hadronic-W mass proxy.
• Jet-pair kinematics matching zg_common: reco_opang, reco_dR / reco_dR_eta /
  reco_dR_y, jet energies (jet1_e, jet2_e, jet_esum, jet_easym), merging
  scales d_12 / d_23 / d_34, y_23 = d_23/d_12, y_23_Q2 = d_23/(E1+E2)^2,
  constituent counts and jet angular size.
• Per-jet flavour-tagger scores + isQ = isU+isD.
• Jet↔gen-quark ΔR matching (jet*_matched_pdg, jet*_match_dR): required
  because W→qq′ quarks have different flavours (unlike Z→qq̄).
• FCCAnalyses get_flavour truth (jet*_getflavour_pdg): stock angle-based
  matcher used by plot_tageff_ww_semilep.C for per-jet true flavour.
  Events are kept only when the unordered pair of get_flavour |PDG| labels
  equals the two hard-process W→qq′ quarks (Vud→{2,1}, Vcs→{4,3}, …).

DUPLICATE QUARKS in Whizard+Pythia6+Delphes EDM4Hep files
──────────────────────────────────────────────────────────
  Copy 1 — "pre-shower": parents = beam e±, daughters = same-flavour quark
            + ISR/FSR gluons.  WE SELECT THIS using the beam-e± parent criterion.
  Copy 2 — "post-shower": parent = pre-shower copy (same PDG), daughter =
            PDG 92 (Lund colour string).  Direction is closer to the jet, but
            the PDG-92 gate is not unique (gluon-split secondaries also pass).
  Selection: require |mother PDG| == 11 (beam e±).  This uniquely identifies
  Copy 1 without any ambiguity from shower-split quarks or heavy-hadron decay
  products.  The pre-shower direction is sufficient for dR-based jet assignment
  in the exclusive 2-jet topology used here.
'''
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ROOT
from addons.FastJet.jetClusteringHelper import ExclusiveJetClusteringHelper
from addons.ONNXRuntime.jetFlavourHelper import JetFlavourHelper
from treemaker_common import (
    select_isoleps, apply_channel_filter, cluster_jets,
    define_reco_jets_kinematics, define_jet_derived_observables,
    get_flav_tagger_paths, TAGGER_SCORES,
    make_tagger_collections, extract_jet_tag_scalars, patch_dndx_if_missing,
)

ROOT.gInterpreter.Declare("""
#ifndef SEL_BY_STRING_DAUGHTER
#define SEL_BY_STRING_DAUGHTER
// Filter a pre-selected quark collection to those that are string endpoints.
// Used for flavour-specific selections (gen_c_ps, gen_b_ps, …) where the
// caller has already filtered to a single PDG, so gluon-split impostors of
// a *different* flavour cannot enter.  If you need to select quarks of mixed
// flavour from the full particle list, use sel_ww_quarks_ps below which adds
// the same-flavour parent gate against gluon-split light contamination.
ROOT::VecOps::RVec<edm4hep::MCParticleData>
sel_by_string_daughter(const ROOT::VecOps::RVec<edm4hep::MCParticleData>& all,
                       const ROOT::VecOps::RVec<int>& dau,
                       const ROOT::VecOps::RVec<edm4hep::MCParticleData>& cands) {
    ROOT::VecOps::RVec<edm4hep::MCParticleData> r;
    for (const auto& p : cands)
        for (int j = p.daughters_begin; j < p.daughters_end; ++j)
            if (abs(all[dau[j]].PDG) == 92) { r.push_back(p); break; }
    return r;
}

// Select hard-process (pre-shower) quarks from W→qq′ (any flavour u–b).
//
// Criterion: quark has a beam e± (|PDG|=11) as a direct parent.
//
// In the Whizard+Pythia6 EDM4Hep record the two quarks from W decay
// ("Copy 1", pre-shower) appear as daughters of the beam e± particles.
// All secondary quarks — shower-split g→qq̄ pairs, heavy-hadron decay
// products, etc. — have non-lepton parents and are rejected automatically.
// No PDG-92 daughter check or status filter is needed.
//
// See sel_zg_quarks_ps in ZGamma_treemaker.py for the full explanation of
// why the previous PDG-92 criterion failed in bb̄/cc̄ events.
ROOT::VecOps::RVec<edm4hep::MCParticleData>
sel_ww_quarks_ps(const ROOT::VecOps::RVec<edm4hep::MCParticleData>& all,
                 const ROOT::VecOps::RVec<int>& mom) {
    ROOT::VecOps::RVec<edm4hep::MCParticleData> r;
    for (const auto& p : all) {
        int apdg = std::abs(p.PDG);
        if (apdg < 1 || apdg > 5) continue;  // u,d,s,c,b only
        for (int k = p.parents_begin; k < p.parents_end; ++k)
            if (std::abs(all[mom[k]].PDG) == 11) { r.push_back(p); break; }
    }
    return r;
}
#endif
""")

# Jet angular size / constituent helpers — same definitions as Zg:: in
# Trmk_Zgamma/zg_common.py (exclusive clustering has no radius, so size is
# a measured quantity). Kept local so this treemaker does not import the
# Z+γ ISR machinery.
ROOT.gInterpreter.Declare(r"""
#ifndef WW_JET_KIN_HELPERS
#define WW_JET_KIN_HELPERS

#include <cmath>
#include <algorithm>
#include <vector>
#include <utility>
#include "TVector3.h"
#include "TLorentzVector.h"
#include "ROOT/RVec.hxx"
#include "edm4hep/ReconstructedParticleData.h"

namespace WWKin {

using RVecRP  = ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>;
using RVecTLV = ROOT::VecOps::RVec<TLorentzVector>;

inline double rp_p(const edm4hep::ReconstructedParticleData& p) {
    return std::sqrt((double)p.momentum.x*(double)p.momentum.x
                   + (double)p.momentum.y*(double)p.momentum.y
                   + (double)p.momentum.z*(double)p.momentum.z);
}

// Angular size of each jet from its own constituents.
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

// Angle containing `frac` of the jet's |p|.
ROOT::VecOps::RVec<double>
jet_ang_containment(const ROOT::VecOps::RVec<RVecRP>& constits,
                    const RVecTLV& jets, double frac) {
    ROOT::VecOps::RVec<double> out;
    out.reserve(jets.size());
    for (size_t j = 0; j < jets.size(); ++j) {
        if (j >= constits.size()) { out.push_back(-1.); continue; }
        TVector3 axis = jets[j].Vect();
        std::vector<std::pair<double, double>> ap;
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

}  // namespace WWKin
#endif
""")

# ── sample list ──────────────────────────────────────────────────────────────
# All six CKM elements at √s = 163 GeV (WW threshold).
# Use fraction < 1 for quick tests; set fraction=1 (or omit) for full stats.
processList = {
    'wzp6_ee_munumuqq_Vud_ecm240': {'fraction': 1, 'chunks': 100},
    'wzp6_ee_munumuqq_Vus_ecm240': {'fraction': 1, 'chunks': 100},
    'wzp6_ee_munumuqq_Vub_ecm240': {'fraction': 1, 'chunks': 100},
    'wzp6_ee_munumuqq_Vcd_ecm240': {'fraction': 1, 'chunks': 100},
    'wzp6_ee_munumuqq_Vcs_ecm240': {'fraction': 1, 'chunks': 100},
    'wzp6_ee_munumuqq_Vcb_ecm240': {'fraction': 1, 'chunks': 100},
}

inputDir  = '/eos/experiment/fcc/ee/generation/DelphesEvents/winter2023/IDEA/'
outputDir = '/eos/user/l/lberiet/CKM_Flav/WW_semilep_240_newtagger/'

nCPUS    = 8
runBatch = True

# Flavour tagger: "default" = CKM_Flav/tagger (winter2023 wc),
#                 "gino"    = CKM_Flav/tagger_Gino (ZmWHmW_163 ParT).
# Override with env: FLAV_TAGGER=gino
FLAV_TAGGER = os.environ.get("FLAV_TAGGER", "new_tagger")
_FLAV_TAGGER_JSON, _FLAV_TAGGER_ONNX = get_flav_tagger_paths(FLAV_TAGGER)
print(f"Flavour tagger: {FLAV_TAGGER}  ({_FLAV_TAGGER_ONNX})")

# Module-level helpers — must survive until the framework has consumed them.
_jet_helper = None
_jet_flavour_helper = None


def define_getflavour_pair_filter(df, helper, mc_col="mc_particles"):
    """True flavour from FCCAnalyses get_flavour; keep only events whose two
    exclusive jets reproduce the hard-process W→qq′ pair.

    Unordered |PDG| match, e.g. Vud → {2,1}, Vcs → {4,3}, Vcb → {4,5}.
    Gluon (21), unmatched (0), or a swapped-in gluon-split flavour fails.
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


class RDFanalysis():

    def analysers(df):
        global _jet_helper, _jet_flavour_helper

        # ── Gen-level truth ───────────────────────────────────────────────────
        df = df.Define('mc_particles', 'Particle')
        df = df.Alias("Particle0", "Particle#0.index")   # mothers
        df = df.Alias("Particle1", "Particle#1.index")   # daughters

        # Truth muons (genStatus=1, stable; includes W-decay μ + secondaries)
        df = df.Define('gen_muons',       'FCCAnalyses::MCParticle::sel_pdgID(13, true)(mc_particles)')
        df = df.Define('gen_muons_st1',   'FCCAnalyses::MCParticle::sel_genStatus(1)(gen_muons)')
        df = df.Define('n_gen_muons',     'FCCAnalyses::MCParticle::get_n(gen_muons_st1)')
        df = df.Define('gen_muon_pt',     'FCCAnalyses::MCParticle::get_pt(gen_muons_st1)')
        df = df.Define('gen_muon_eta',    'FCCAnalyses::MCParticle::get_eta(gen_muons_st1)')
        df = df.Define('gen_muon_phi',    'FCCAnalyses::MCParticle::get_phi(gen_muons_st1)')
        df = df.Define('gen_muon_e',      'FCCAnalyses::MCParticle::get_e(gen_muons_st1)')
        df = df.Define('gen_muon_charge', 'FCCAnalyses::MCParticle::get_charge(gen_muons_st1)')
        df = df.Define('gen_muon_origin',
            'FCCAnalyses::MCParticle::get_leptons_origin(gen_muons_st1, mc_particles, Particle0)')
        df = df.Define('mu0_pt',     'n_gen_muons > 0 ? gen_muon_pt.at(0)     : -999.f')
        df = df.Define('mu0_eta',    'n_gen_muons > 0 ? gen_muon_eta.at(0)    : -999.f')
        df = df.Define('mu0_phi',    'n_gen_muons > 0 ? gen_muon_phi.at(0)    : -999.f')
        df = df.Define('mu0_e',      'n_gen_muons > 0 ? gen_muon_e.at(0)      : -999.f')
        df = df.Define('mu0_charge', 'n_gen_muons > 0 ? gen_muon_charge.at(0) : -999.f')

        # Truth muon-neutrinos (genStatus=1)
        df = df.Define('gen_neutrinos',     'FCCAnalyses::MCParticle::sel_pdgID(14, true)(mc_particles)')
        df = df.Define('gen_neutrinos_st1', 'FCCAnalyses::MCParticle::sel_genStatus(1)(gen_neutrinos)')
        df = df.Define('n_gen_neutrinos',   'FCCAnalyses::MCParticle::get_n(gen_neutrinos_st1)')
        df = df.Define('gen_nu_pt',         'FCCAnalyses::MCParticle::get_pt(gen_neutrinos_st1)')
        df = df.Define('gen_nu_eta',        'FCCAnalyses::MCParticle::get_eta(gen_neutrinos_st1)')
        df = df.Define('gen_nu_e',          'FCCAnalyses::MCParticle::get_e(gen_neutrinos_st1)')
        df = df.Define('nu0_pt',  'n_gen_neutrinos > 0 ? gen_nu_pt.at(0)  : -999.f')
        df = df.Define('nu0_eta', 'n_gen_neutrinos > 0 ? gen_nu_eta.at(0) : -999.f')
        df = df.Define('nu0_e',   'n_gen_neutrinos > 0 ? gen_nu_e.at(0)   : -999.f')

        # Truth quarks from hadronic W decay — all copies first, then post-shower
        df = df.Define('gen_c',    'FCCAnalyses::MCParticle::sel_pdgID(4,  false)(mc_particles)')
        df = df.Define('gen_cbar', 'FCCAnalyses::MCParticle::sel_pdgID(-4, false)(mc_particles)')
        df = df.Define('gen_b',    'FCCAnalyses::MCParticle::sel_pdgID(5,  false)(mc_particles)')
        df = df.Define('gen_bbar', 'FCCAnalyses::MCParticle::sel_pdgID(-5, false)(mc_particles)')
        df = df.Define('gen_c_st',    'FCCAnalyses::MCParticle::get_genStatus(gen_c)')
        df = df.Define('gen_cbar_st', 'FCCAnalyses::MCParticle::get_genStatus(gen_cbar)')

        # Post-shower string-endpoint quarks (USE THESE for tagger calibration)
        df = df.Define('gen_c_ps',    'sel_by_string_daughter(mc_particles, Particle1, gen_c)')
        df = df.Define('gen_cbar_ps', 'sel_by_string_daughter(mc_particles, Particle1, gen_cbar)')
        df = df.Define('gen_b_ps',    'sel_by_string_daughter(mc_particles, Particle1, gen_b)')
        df = df.Define('gen_bbar_ps', 'sel_by_string_daughter(mc_particles, Particle1, gen_bbar)')

        df = df.Define('n_gen_c',       'FCCAnalyses::MCParticle::get_n(gen_c)')
        df = df.Define('n_gen_b',       'FCCAnalyses::MCParticle::get_n(gen_b)')
        df = df.Define('n_gen_cbar',    'FCCAnalyses::MCParticle::get_n(gen_cbar)')
        df = df.Define('n_gen_bbar',    'FCCAnalyses::MCParticle::get_n(gen_bbar)')
        df = df.Define('n_gen_c_ps',    'FCCAnalyses::MCParticle::get_n(gen_c_ps)')
        df = df.Define('n_gen_b_ps',    'FCCAnalyses::MCParticle::get_n(gen_b_ps)')
        df = df.Define('n_gen_cbar_ps', 'FCCAnalyses::MCParticle::get_n(gen_cbar_ps)')
        df = df.Define('n_gen_bbar_ps', 'FCCAnalyses::MCParticle::get_n(gen_bbar_ps)')

        # Leading post-shower quark pT (-999 when absent for light-quark samples)
        df = df.Define('c_pt',
            'n_gen_c_ps    > 0 ? FCCAnalyses::MCParticle::get_pt(gen_c_ps).at(0)    : -999.f')
        df = df.Define('cbar_pt',
            'n_gen_cbar_ps > 0 ? FCCAnalyses::MCParticle::get_pt(gen_cbar_ps).at(0) : -999.f')
        df = df.Define('b_pt',
            'n_gen_b_ps    > 0 ? FCCAnalyses::MCParticle::get_pt(gen_b_ps).at(0)    : -999.f')
        df = df.Define('bbar_pt',
            'n_gen_bbar_ps > 0 ? FCCAnalyses::MCParticle::get_pt(gen_bbar_ps).at(0) : -999.f')

        # Leading post-shower quark |p| and θ (for gen–reco comparison)
        df = df.Define('c_p',
            'n_gen_c_ps    > 0 ? FCCAnalyses::MCParticle::get_p(gen_c_ps).at(0)     : -999.f')
        df = df.Define('c_theta',
            'n_gen_c_ps    > 0 ? FCCAnalyses::MCParticle::get_theta(gen_c_ps).at(0) : -999.f')
        df = df.Define('b_p',
            'n_gen_b_ps    > 0 ? FCCAnalyses::MCParticle::get_p(gen_b_ps).at(0)     : -999.f')
        df = df.Define('b_theta',
            'n_gen_b_ps    > 0 ? FCCAnalyses::MCParticle::get_theta(gen_b_ps).at(0) : -999.f')

        # Hard-process quarks from W→qq′: identified by beam-e± parentage.
        # The two quarks generally have different PDG (unlike Z→qq̄), so sample
        # name alone cannot assign per-jet flavour — we dR-match reco jets to
        # these gen quarks below.  Particle0 (mothers) is the only index needed.
        df = df.Define("gen_quarks_ps",
            "sel_ww_quarks_ps(mc_particles, Particle0)")
        df = df.Define("n_gen_quarks_ps",
            "FCCAnalyses::MCParticle::get_n(gen_quarks_ps)")
        df = df.Filter("n_gen_quarks_ps == 2",
                       "gen: exactly 2 hard-process quarks from W")
        df = df.Define("gen_quarks_ps_tlv",
            "FCCAnalyses::MCParticle::get_tlv(gen_quarks_ps)")
        df = df.Define("gen_q1_p4",  "gen_quarks_ps_tlv[0]")
        df = df.Define("gen_q2_p4",  "gen_quarks_ps_tlv[1]")
        for q, src in [("q1", "gen_q1_p4"), ("q2", "gen_q2_p4")]:
            df = df.Define(f"gen_{q}_p",        f"{src}.P()")
            df = df.Define(f"gen_{q}_pt",       f"{src}.Pt()")
            df = df.Define(f"gen_{q}_theta",    f"{src}.Theta()")
            df = df.Define(f"gen_{q}_phi",      f"{src}.Phi()")
            df = df.Define(f"gen_{q}_eta",      f"{src}.Eta()")
            df = df.Define(f"gen_{q}_costheta", f"{src}.CosTheta()")
        df = df.Define("gen_q1_pdg", "(int)gen_quarks_ps[0].PDG")
        df = df.Define("gen_q2_pdg", "(int)gen_quarks_ps[1].PDG")
        df = df.Define("gen_opang",  "gen_q1_p4.Angle(gen_q2_p4.Vect())")

        # ── Reco-level: semileptonic WW selection + jet kinematics ─────────────
        # 1. Build iso-lepton collections (muons + electrons, p > 12 GeV, cone iso)
        df = select_isoleps(df)
        # 2. Require exactly 1 isolated lepton; build FSR-dressed lepton + remove
        #    the lepton and FSR photons from the jet-clustering input.
        df = apply_channel_filter(df, "semihad")
        # 3. Exclusive anti-kt → exactly 2 jets on the cleaned RP collection.
        df, _jet_helper = cluster_jets(df, "semihad", algorithm="ee_kt", radius=0.5)
        # 3b. Keep only events where get_flavour on those two jets recovers
        #     the unordered W→qq′ pair (Vud→{u,d}, Vcs→{c,s}, …).
        df = define_getflavour_pair_filter(df, _jet_helper)
        # 4. Scalar jet kinematics (p, pt, θ, φ, η, cos θ, mass) + Whad_reco.
        df = define_reco_jets_kinematics(df)
        # 5. Dijet invariant mass (hadronic-W proxy, analogous to reco_dijet_m
        #    in Z+γ so analysis_final.py can use the same branch for both).
        df = df.Define("reco_dijet_m",     "Whad_reco.M()")
        df = df.Define("reco_dijet_p",     "Whad_reco.P()")
        df = df.Define("reco_dijet_theta", "Whad_reco.Theta()")
        # 6. Derived per-jet observables (cosθ, η, pT) from the clustering
        #    helper's jet_p / jet_theta RVecF columns.
        df = define_jet_derived_observables(df)
       # df = df.Filter("reco_jet1_eta < 2.6 && reco_jet2_eta < 2.6", "reco_jet1_eta > -2.6 && reco_jet2_eta > -2.6");
        # 7. Opening angle between the two reco jets (in radians), matching the
        #    definition used in plot_tageff_zgamma.C:
        #      cos(α) = sinθ1 sinθ2 cos(Δφ) + cosθ1 cosθ2,  α = acos(clamp(cosα))
        df = df.Define(
            "reco_opang",
            "TMath::ACos("
            "  TMath::Max(-1., TMath::Min(1.,"
            "    (double)TMath::Sin(reco_jet1_theta) * (double)TMath::Sin(reco_jet2_theta) * (double)TMath::Cos(reco_jet1_phi - reco_jet2_phi)"
            "  + (double)TMath::Cos(reco_jet1_theta) * (double)TMath::Cos(reco_jet2_theta)"
            "  ))"
            ")"
        )

        # ── Merging scales, ΔR, jet energies (same kinematics as zg_common) ─
        # cluster_jets already wrote d_12 and d_32 (= 3→2 scale).  Name the
        # 3→2 / 4→3 scales d_23 / d_34 to match Z+γ so y_23_Q2 and the
        # downstream counting scripts can use one branch list for both.
        _clust = f"_jet{_jet_helper.tag}"
        _nconst = f"jet_nconst{_jet_helper.tag}"
        df = df.Define("d_23", f"JetClusteringUtils::get_exclusive_dmerge({_clust}, 2)")
        df = df.Define("d_34", f"JetClusteringUtils::get_exclusive_dmerge({_clust}, 3)")
        df = df.Define("y_23", "d_12 > 0 ? d_23 / d_12 : -1.")
        df = df.Define("y_23_Q2",
            "(double)jet1.E() + (double)jet2.E() > 0 ? "
            "d_23 / (((double)jet1.E() + (double)jet2.E()) * "
            "((double)jet1.E() + (double)jet2.E())) : -1.")

        df = df.Define("jet1_nconst", f"(int){_nconst}[0]")
        df = df.Define("jet2_nconst", f"(int){_nconst}[1]")

        df = df.Define("jet_ang_max",
            f"WWKin::jet_ang_size({_jet_helper.constituents}, jets_p4, 0)")
        df = df.Define("jet_ang_rms",
            f"WWKin::jet_ang_size({_jet_helper.constituents}, jets_p4, 1)")
        df = df.Define("jet_ang_r90",
            f"WWKin::jet_ang_containment({_jet_helper.constituents}, jets_p4, 0.90)")
        for i in (1, 2):
            for v in ("ang_max", "ang_rms", "ang_r90"):
                df = df.Define(f"jet{i}_{v}", f"jet_{v}[{i-1}]")

        df = df.Define("jet_order_consistent",
            "(int)(std::abs((double)jets_p4[0].P() - (double)jet_p[0]) < 1e-3 && "
            "std::abs((double)jets_p4[1].P() - (double)jet_p[1]) < 1e-3)")
        df = df.Define("jet_is_e_ordered",
            "(int)((double)jet1.E() >= (double)jet2.E())")

        df = df.Define("jet1_e", "(double)jet1.E()")
        df = df.Define("jet2_e", "(double)jet2.E()")
        df = df.Define("jet_esum",  "jet1_e + jet2_e")
        df = df.Define("jet_easym", "(jet1_e - jet2_e) / (jet1_e + jet2_e)")
        df = df.Define("jet_emin",  "std::min(jet1_e, jet2_e)")
        df = df.Define("jet_emax",  "std::max(jet1_e, jet2_e)")

        df = df.Define("reco_dR", "(double)jet1.DeltaR(jet2)")
        df = df.Define("reco_dR_eta",
            "std::sqrt(std::pow((double)(jet1.Eta() - jet2.Eta()), 2) + "
            "std::pow(TVector2::Phi_mpi_pi(jet1.Phi() - jet2.Phi()), 2))")
        df = df.Define("reco_dR_y",
            "std::sqrt(std::pow((double)(jet1.Rapidity() - jet2.Rapidity()), 2) + "
            "std::pow(TVector2::Phi_mpi_pi(jet1.Phi() - jet2.Phi()), 2))")

        # ── Flavour tagger ────────────────────────────────────────────────────
        # Jets are clustered on ReconstructedParticlesNoMuNoElNoFSR (muons,
        # electrons and FSR photons removed), so the same collection is passed
        # as PFParticles to keep constituent indices consistent.
        _jet_flavour_helper = JetFlavourHelper(
            make_tagger_collections("ReconstructedParticlesNoMuNoElNoFSR"),
            _jet_helper.jets,        # "jet"
            _jet_helper.constituents, # "jetc"
        )
        # Patch: EFlowTrack_dNdx is absent in some Delphes campaigns (163 GeV
        # WW/ZGamma samples).  Replace the dNdxHandler definition with empty
        # collections so pfcand_dndx = 0 rather than crashing.
        patch_dndx_if_missing(df, _jet_flavour_helper)
        # Compute the ~30 constituent-level pfcand_* features required by the
        # ONNX model (track covariance, IP significance, ToF mass, dE/dx, …).
        df = _jet_flavour_helper.define(df)
        # Run the ParticleTransformer inference; produces recojet_isG/U/S/C/B/TAU/D
        # as RVecF with one score per jet per event (2 entries for 2-jet events).
        df = _jet_flavour_helper.inference(_FLAV_TAGGER_JSON, _FLAV_TAGGER_ONNX, df)
        # Unpack RVec scores to per-jet scalars: jet1_isG, jet1_isC, jet2_isB, …
        df = extract_jet_tag_scalars(df, n_jets=2)

        # Combined light-quark score: isQ = isU + isD
        df = df.Define("recojet_isQ", "recojet_isU + recojet_isD")
        df = df.Define("jet1_isQ", "(float)recojet_isQ[0]")
        df = df.Define("jet2_isQ", "(float)recojet_isQ[1]")

        # ── Jet ↔ gen-quark ΔR matching (per-jet truth for mixed-flavour W) ──
        # Choose the assignment that minimises ΔR(j1,q)+ΔR(j2,q′).  Unlike Z→qq̄
        # the two quarks have different flavours, so matched_pdg must be stored
        # per jet — sample name alone is not enough for the efficiency plotter.
        df = df.Define("_jet_match_swap",
            "(int)((double)jet1.DeltaR(gen_q1_p4) + (double)jet2.DeltaR(gen_q2_p4)"
            " > (double)jet1.DeltaR(gen_q2_p4) + (double)jet2.DeltaR(gen_q1_p4))")
        df = df.Define("jet1_matched_pdg",
            "_jet_match_swap ? gen_q2_pdg : gen_q1_pdg")
        df = df.Define("jet2_matched_pdg",
            "_jet_match_swap ? gen_q1_pdg : gen_q2_pdg")
        df = df.Define("jet1_match_dR",
            "(double)(_jet_match_swap"
            " ? jet1.DeltaR(gen_q2_p4) : jet1.DeltaR(gen_q1_p4))")
        df = df.Define("jet2_match_dR",
            "(double)(_jet_match_swap"
            " ? jet2.DeltaR(gen_q1_p4) : jet2.DeltaR(gen_q2_p4))")
        df = df.Define("dR_swap_margin",
            "std::abs("
            "((double)jet1.DeltaR(gen_q1_p4) + (double)jet2.DeltaR(gen_q2_p4))"
            " - ((double)jet1.DeltaR(gen_q2_p4) + (double)jet2.DeltaR(gen_q1_p4)))")

                    # b-tagger
        df = df.Define("DB_vsC", "recojet_isB / (recojet_isB + recojet_isC)")
        df = df.Define("DB_vsS", "recojet_isB / (recojet_isB + recojet_isS)")
        df = df.Define("DB_vsG", "recojet_isB / (recojet_isB + recojet_isG)")
        df = df.Define("DB_vsQ", "recojet_isB / (recojet_isB + recojet_isQ)")

            # c-tagger
        df = df.Define("DC_vsB", "recojet_isC / (recojet_isC + recojet_isB)")
        df = df.Define("DC_vsS", "recojet_isC / (recojet_isC + recojet_isS)")
        df = df.Define("DC_vsG", "recojet_isC / (recojet_isC + recojet_isG)")
        df = df.Define("DC_vsQ", "recojet_isC / (recojet_isC + recojet_isQ)")

            # s-tagger
        df = df.Define("DS_vsB", "recojet_isS / (recojet_isS + recojet_isB)")
        df = df.Define("DS_vsC", "recojet_isS / (recojet_isS + recojet_isC)")
        df = df.Define("DS_vsG", "recojet_isS / (recojet_isS + recojet_isG)")
        df = df.Define("DS_vsQ", "recojet_isS / (recojet_isS + recojet_isQ)")

            # g-tagger
        df = df.Define("DG_vsB", "recojet_isG / (recojet_isG + recojet_isB)")
        df = df.Define("DG_vsC", "recojet_isG / (recojet_isG + recojet_isC)")
        df = df.Define("DG_vsS", "recojet_isG / (recojet_isG + recojet_isS)")
        df = df.Define("DG_vsQ", "recojet_isG / (recojet_isG + recojet_isQ)")

            # q-tagger (light: u+d combined)
        df = df.Define("DQ_vsB", "recojet_isQ / (recojet_isQ + recojet_isB)")
        df = df.Define("DQ_vsC", "recojet_isQ / (recojet_isQ + recojet_isC)")
        df = df.Define("DQ_vsS", "recojet_isQ / (recojet_isQ + recojet_isS)")
        df = df.Define("DQ_vsG", "recojet_isQ / (recojet_isQ + recojet_isG)")


        return df

    def output():
        return [
            # ── Gen: truth muons ────────────────────────────────────────────
            'n_gen_muons',
            'gen_muon_pt', 'gen_muon_eta', 'gen_muon_phi',
            'gen_muon_e', 'gen_muon_charge', 'gen_muon_origin',
            'mu0_pt', 'mu0_eta', 'mu0_phi', 'mu0_e', 'mu0_charge',
            # ── Gen: truth neutrinos ────────────────────────────────────────
            'n_gen_neutrinos',
            'gen_nu_pt', 'gen_nu_eta', 'gen_nu_e',
            'nu0_pt', 'nu0_eta', 'nu0_e',
            # ── Gen: quark multiplicities (all copies) ──────────────────────
            'n_gen_c', 'n_gen_b', 'n_gen_cbar', 'n_gen_bbar',
            'gen_c_st', 'gen_cbar_st',
            # ── Gen: post-shower quarks (use for tagger calibration) ────────
            'n_gen_c_ps', 'n_gen_b_ps', 'n_gen_cbar_ps', 'n_gen_bbar_ps',
            'c_pt', 'cbar_pt', 'b_pt', 'bbar_pt',
            'c_p', 'c_theta', 'b_p', 'b_theta',
            # ── Gen: all-flavour W→qq′ quarks + PDG ─────────────────────────
            'n_gen_quarks_ps',
            'gen_q1_p', 'gen_q1_pt', 'gen_q1_theta',
            'gen_q1_phi', 'gen_q1_eta', 'gen_q1_costheta', 'gen_q1_pdg',
            'gen_q2_p', 'gen_q2_pt', 'gen_q2_theta',
            'gen_q2_phi', 'gen_q2_eta', 'gen_q2_costheta', 'gen_q2_pdg',
            'gen_opang',
            # ── Reco: jet 1 kinematics ───────────────────────────────────────
            'reco_jet1_p', 'reco_jet1_pt', 'reco_jet1_theta',
            'reco_jet1_phi', 'reco_jet1_eta', 'reco_jet1_costheta',
            'reco_jet1_mass',
            # ── Reco: jet 2 kinematics ───────────────────────────────────────
            'reco_jet2_p', 'reco_jet2_pt', 'reco_jet2_theta',
            'reco_jet2_phi', 'reco_jet2_eta', 'reco_jet2_costheta',
            'reco_jet2_mass',
            # ── Reco: dijet (hadronic-W proxy) ───────────────────────────────
            'n_reco_jets',
            'reco_dijet_m', 'reco_dijet_p', 'reco_dijet_theta',
            # ── Reco: jet-pair opening angle and ΔR ──────────────────────────
            'reco_opang', 'reco_dR', 'reco_dR_eta', 'reco_dR_y',
            # ── Reco: jet energies ───────────────────────────────────────────
            'jet1_e', 'jet2_e', 'jet_esum', 'jet_easym', 'jet_emin', 'jet_emax',
            # ── Reco: Durham merging scales (y_23_Q2 = d_23/(E1+E2)^2) ───────
            'd_12', 'd_23', 'd_34', 'y_23', 'y_23_Q2',
            # ── Reco: constituent multiplicity + angular size ────────────────
            'jet1_nconst', 'jet2_nconst',
            'jet_ang_max', 'jet_ang_rms', 'jet_ang_r90',
            'jet1_ang_max', 'jet1_ang_rms', 'jet1_ang_r90',
            'jet2_ang_max', 'jet2_ang_rms', 'jet2_ang_r90',
            'jet_order_consistent', 'jet_is_e_ordered',
            # ── Reco: per-jet vectors from clustering helper (2 entries/event) ─
            # jet_p, jet_theta, jet_mass, jet_phi, jet_e from helper.define()
            'jet_p', 'jet_theta', 'jet_mass', 'jet_phi', 'jet_e',
            # jet_costheta, jet_eta, jet_pt from define_jet_derived_observables()
            'jet_costheta', 'jet_eta', 'jet_pt',
            # ── Flavour tagger ───────────────────────────────────────────────
            # Score RVecs (2 entries/event): recojet_isG/U/S/C/B/TAU/D
            # plus jet-multiplicity counters: jet_nmu/nel/nchad/ngamma/nnhad
            *_jet_flavour_helper.outputBranches(),
            'recojet_isQ',
            # Scalar scores per jet for direct plotting (from extract_jet_tag_scalars)
            'jet1_isG', 'jet1_isU', 'jet1_isS', 'jet1_isC',
            'jet1_isB', 'jet1_isTAU', 'jet1_isD', 'jet1_isQ',
            'jet2_isG', 'jet2_isU', 'jet2_isS', 'jet2_isC',
            'jet2_isB', 'jet2_isTAU', 'jet2_isD', 'jet2_isQ',
            # ── Jet–quark ΔR matching (gen truth per reco jet) ───────────────
            'jet1_matched_pdg', 'jet2_matched_pdg',
            'jet1_match_dR',    'jet2_match_dR',
            'dR_swap_margin',
            # ── Jet–quark truth-matching: FCCAnalyses get_flavour ────────────
            'jet1_getflavour_pdg', 'jet2_getflavour_pdg', 'getflavour_pair_ok',
            # ── Flavour tagger scores ────────────────────────────────────────
            'DB_vsC', 'DB_vsS', 'DB_vsG', 'DB_vsQ',
            'DC_vsB', 'DC_vsS', 'DC_vsG', 'DC_vsQ',
            'DS_vsB', 'DS_vsC', 'DS_vsG', 'DS_vsQ',
            'DG_vsB', 'DG_vsC', 'DG_vsS', 'DG_vsQ',
            'DQ_vsB', 'DQ_vsC', 'DQ_vsS', 'DQ_vsG',
        ]
