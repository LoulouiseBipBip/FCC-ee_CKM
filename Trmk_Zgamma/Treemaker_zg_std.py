'''
Z+gamma -> qq gamma treemaker, standard selection
=================================================

Cutflow (each step is a named RDataFrame Filter, printed at the end of the job):

  0. all events
  1. >= 1 reconstructed photon with |p| > PHOTON_PMIN and |cos theta| < PHOTON_ABSCOS_MAX
     (optionally isolated, see PHOTON_ISO_MAX).  ALL such photons are taken as
     ISR candidates and removed from the clustering input; m_recoil is computed
     against their sum.
  2. Durham exclusive N=2 clustering on everything else -> exactly 2 jets
  3. Z window: |m_jj - m_Z| < Z_HALFWIDTH  (Z_SELECTION = "mjj", default)
             or |m_recoil - m_Z| < Z_HALFWIDTH  (Z_SELECTION = "recoil")
  4. (optional, truth) get_flavour labels of the two jets reproduce the
     hard-process quark pair -- APPLY_GETFLAVOUR_FILTER

Then flavour tagging (winter2023 ParT by default, MC primary vertex, no track
smearing), the pairwise discriminants, and the truth labels.

Clustering, the tagger collections and the PV follow the FCCAnalyses
weaver / ZH-flavour examples: the jet-clustering input is the same collection
passed to JetFlavourHelper as PFParticles.

Branch names match what Fit_TP_Multi/tp_counting.py and tag_eff_ww_zg.py read.

No tag/probe roles or working points are defined here: the tree is one entry
per event with both jets' scores, and the tag/probe pairing is done downstream
(as in the CMS DAS muon TnP, where pairs are built from a generic ntuple).

Run:  fccanalysis run Treemaker_zg_std.py
'''

import os
import atexit

import ROOT
from addons.FastJet.jetClusteringHelper import ExclusiveJetClusteringHelper
from addons.ONNXRuntime.jetFlavourHelper import JetFlavourHelper

# ═════════════════════════════════════════════════════════════════════════════
# Configuration
# ═════════════════════════════════════════════════════════════════════════════

ECM = int(os.environ.get("ZG_ECM", 163))          # GeV; also sets the samples
MZ  = 91.1876

# photon (ISR candidate)
PHOTON_PMIN       = 20.0     # GeV
PHOTON_ABSCOS_MAX = 0.99     # acceptance
PHOTON_ISO_MAX    = -1.      # <0 disables; else sum|p| in cone / |p_gamma| < this
PHOTON_ISO_CONE   = 0.3      # rad, half-angle of the isolation cone (if enabled)

# Z window
Z_SELECTION = "mjj"          # "mjj": dijet mass; "recoil": photon recoil mass
Z_HALFWIDTH = 15.0           # GeV

# truth purification (OFF = pure reco cutflow; ON = previous productions)
APPLY_GETFLAVOUR_FILTER = True

# flavour tagger
FLAV_TAGGER = os.environ.get("FLAV_TAGGER", "default")
_TAGGERS = {
    "default":    ("tagger", "fccee_flavtagging_edm4hep_wc.json",
                   "fccee_flavtagging_edm4hep_wc.onnx"),
    "gino":       ("tagger_Gino", "preprocess.json", "ZmWHmW_163_parT.onnx"),
    "new_tagger": ("TRAINING_70M_7labels_4GPUs_newKey4Hep_LongJob2",
                   "TRAINING_70M_7labels_4GPUs_newKey4Hep_LongJob2_preprocess.json",
                   "TRAINING_70M_7labels_4GPUs_newKey4Hep_LongJob2_best_epoch_state.onnx"),
}

# ── samples / output ─────────────────────────────────────────────────────────
_SAMPLE_TAG = {163: "pqqmin40", 240: "pqqmin80"}
if Z_SELECTION not in ("mjj", "recoil"):
    raise ValueError(f"Z_SELECTION={Z_SELECTION!r} must be 'mjj' or 'recoil'")
if ECM not in _SAMPLE_TAG:
    raise ValueError(f"ZG_ECM={ECM}: no samples configured (choose {list(_SAMPLE_TAG)})")

processList = {
    f"wzp6_ee_{f}_{_SAMPLE_TAG[ECM]}_ecm{ECM}": {"fraction": 1, "chunks": 50}
    for f in ("bb", "cc", "ss", "uu", "dd")
}
inputDir  = "/eos/experiment/fcc/ee/generation/DelphesEvents/winter2023/IDEA/"
outputDir = f"/eos/user/l/lberiet/CKM_Flav/Zg_{ECM}_std_{FLAV_TAGGER}_getflav_on/"
nCPUS     = 8
runBatch  = True

_tdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), _TAGGERS[FLAV_TAGGER][0])
TAGGER_JSON = os.path.join(_tdir, _TAGGERS[FLAV_TAGGER][1])
TAGGER_ONNX = os.path.join(_tdir, _TAGGERS[FLAV_TAGGER][2])
for _p in (TAGGER_JSON, TAGGER_ONNX):
    if not os.path.isfile(_p):
        raise FileNotFoundError(f"tagger file not found: {_p}")

print(f"[Zg std] ECM={ECM}  photon p>{PHOTON_PMIN} |cos|<{PHOTON_ABSCOS_MAX} "
      f"iso<{PHOTON_ISO_MAX if PHOTON_ISO_MAX > 0 else 'off'}  "
      f"|m_{Z_SELECTION}-mZ|<{Z_HALFWIDTH}  getflav filter={APPLY_GETFLAVOUR_FILTER}  "
      f"tagger={FLAV_TAGGER}")

# Tagger inputs: winter2023 IDEA Delphes storage names (see treemaker_common.py)
TAGGER_COLLECTIONS = {
    "GenParticles":     "Particle",
    "PFParticles":      "RP_no_photon",       # == clustering input
    "PFTracks":         "EFlowTrack",
    "PFPhotons":        "EFlowPhoton",
    "PFNeutralHadrons": "EFlowNeutralHadron",
    "TrackState":       "EFlowTrack_1",
    "TrackerHits":      "TrackerHits",
    "CalorimeterHits":  "CalorimeterHits",
    "dNdx":             "EFlowTrack_2",
    "PathLength":       "EFlowTrack_L",
    "Bz":               "magFieldBz",
}

SCORES = ["G", "U", "D", "S", "C", "B", "TAU", "Q"]      # Q = U + D (built here)
DISCRIMINANTS = (("B", "CSGQ"), ("C", "BSGQ"), ("S", "BCGQ"),
                 ("G", "BCSQ"), ("Q", "BCSG"))

# ═════════════════════════════════════════════════════════════════════════════
# C++ helpers
# ═════════════════════════════════════════════════════════════════════════════

ROOT.gInterpreter.Declare(r"""
#ifndef ZG_STD_HELPERS
#define ZG_STD_HELPERS
#include <cmath>
#include <algorithm>
#include "TVector3.h"
#include "Math/Vector4D.h"
#include "ROOT/RVec.hxx"
#include "edm4hep/ReconstructedParticleData.h"
#include "edm4hep/MCParticleData.h"

namespace ZgStd {
using RVecRP = ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>;
using RVecMC = ROOT::VecOps::RVec<edm4hep::MCParticleData>;

inline double p3(const edm4hep::ReconstructedParticleData& p) {
    return std::sqrt((double)p.momentum.x*p.momentum.x + (double)p.momentum.y*p.momentum.y
                   + (double)p.momentum.z*p.momentum.z);
}

// sum |p| of all other particles within an angular cone, over |p_gamma|
double cone_iso(const edm4hep::ReconstructedParticleData& g, const RVecRP& all,
                double cos_cone) {
    TVector3 dg(g.momentum.x, g.momentum.y, g.momentum.z);
    double pg = dg.Mag();
    if (pg <= 0.) return 1e9;
    dg *= 1. / pg;
    double sum = 0.;
    for (const auto& p : all) {
        TVector3 v(p.momentum.x, p.momentum.y, p.momentum.z);
        double pp = v.Mag();
        if (pp <= 0.) continue;
        double c = v.Dot(dg) / pp;
        if (c > 1. - 1e-9 && std::abs(pp - pg) < 1e-6) continue;   // the photon itself
        if (c > cos_cone) sum += pp;
    }
    return sum / pg;
}

// photons above pmin, inside |cos theta| < abscos_max, optionally isolated
// (iso_max < 0 disables), sorted hardest first
RVecRP sel_photons(const RVecRP& photons, const RVecRP& all, double pmin,
                   double abscos_max, double cos_cone, double iso_max) {
    RVecRP out;
    for (const auto& g : photons) {
        double pp = p3(g);
        if (pp < pmin) continue;
        if (std::abs(g.momentum.z / pp) >= abscos_max) continue;
        if (iso_max > 0. && cone_iso(g, all, cos_cone) >= iso_max) continue;
        out.push_back(g);
    }
    std::sort(out.begin(), out.end(),
              [](const auto& a, const auto& b) { return p3(a) > p3(b); });
    return out;
}

// recoil mass against the selected photon(s), initial state (0,0,0,sqrt(s))
double recoil_mass(const RVecRP& g, double sqrts) {
    ROOT::Math::PxPyPzEVector r(0., 0., 0., sqrts);
    for (const auto& p : g)
        r -= ROOT::Math::PxPyPzEVector(p.momentum.x, p.momentum.y, p.momentum.z, p.energy);
    double m2 = r.M2();
    return m2 > 0. ? std::sqrt(m2) : -1.;
}

// hard-process quarks: u..b with a beam e+- as direct parent (Whizard+Pythia6)
RVecMC hard_quarks(const RVecMC& all, const ROOT::VecOps::RVec<int>& parents) {
    RVecMC r;
    for (const auto& p : all) {
        int a = std::abs(p.PDG);
        if (a < 1 || a > 5) continue;
        for (int k = p.parents_begin; k < p.parents_end; ++k) {
            int i = parents[k];
            if (i >= 0 && i < (int)all.size() && std::abs(all[i].PDG) == 11) {
                r.push_back(p); break;
            }
        }
    }
    return r;
}
}  // namespace ZgStd
#endif
""")

# ═════════════════════════════════════════════════════════════════════════════
# Cutflow: Report() booked on the final node fills during the Snapshot loop;
# printed when the job exits (one block per processed sample / chunk).
# ═════════════════════════════════════════════════════════════════════════════

_CUTFLOWS = []


@atexit.register
def _print_cutflows():
    for i, rep in enumerate(_CUTFLOWS):
        print(f"\n===== Zg std cutflow [{i}] =====")
        rep.Print()


_jet_helper = None
_flav_helper = None


class RDFanalysis():

    def analysers(df):
        global _jet_helper, _flav_helper

        df = df.Alias("Photon0",   "Photon#0.index")
        df = df.Alias("Particle0", "Particle#0.index")

        # ── 1. ISR photon candidate ──────────────────────────────────────────
        cos_cone = ROOT.TMath.Cos(PHOTON_ISO_CONE)
        df = df.Define("photons_all",
            "FCCAnalyses::ReconstructedParticle::get(Photon0, ReconstructedParticles)")
        df = df.Define("photons_sel",
            f"ZgStd::sel_photons(photons_all, ReconstructedParticles, {PHOTON_PMIN}, "
            f"{PHOTON_ABSCOS_MAX}, {cos_cone}, {PHOTON_ISO_MAX})")
        df = df.Define("n_photons_sel", "(int)photons_sel.size()")
        df = df.Filter("n_photons_sel >= 1",
                       f">=1 photon p>{PHOTON_PMIN:g} |cos|<{PHOTON_ABSCOS_MAX:g}")

        # all selected photons are ISR candidates (hardest first)
        df = df.Define("photon_p",        "FCCAnalyses::ReconstructedParticle::get_p(photons_sel)")
        df = df.Define("photon_e",        "FCCAnalyses::ReconstructedParticle::get_e(photons_sel)")
        df = df.Define("photon_theta",    "FCCAnalyses::ReconstructedParticle::get_theta(photons_sel)")
        df = df.Define("photon_esum",     "(double)ROOT::VecOps::Sum(photon_e)")
        df = df.Define("m_recoil", f"ZgStd::recoil_mass(photons_sel, {float(ECM)})")
        df = df.Define("RP_no_photon",
            "FCCAnalyses::ReconstructedParticle::remove(ReconstructedParticles, photons_sel)")

        # ── 2. Durham exclusive N=2 ──────────────────────────────────────────
        helper = ExclusiveJetClusteringHelper("RP_no_photon", 2)
        df = helper.define(df)
        _jet_helper = helper

        df = df.Define("jets_p4", f"JetConstituentsUtils::compute_tlv_jets({helper.jets})")
        df = df.Define("n_reco_jets", "(int)jets_p4.size()")
        df = df.Filter("n_reco_jets == 2", "exactly 2 jets")

        df = df.Define("jet1", "jets_p4[0]")          # E-ordered
        df = df.Define("jet2", "jets_p4[1]")
        for i in (1, 2):
            for v, expr in (("p", "P()"), ("theta", "Theta()"), ("phi", "Phi()"),
                            ("costheta", "CosTheta()"), ("mass", "M()")):
                df = df.Define(f"reco_jet{i}_{v}", f"(double)jet{i}.{expr}")
            df = df.Define(f"jet{i}_e", f"(double)jet{i}.E()")

        df = df.Define("dijet_p4",         "jet1 + jet2")
        df = df.Define("reco_dijet_m",     "dijet_p4.M()")
        df = df.Define("reco_dijet_p",     "dijet_p4.P()")
        df = df.Define("reco_dijet_theta", "dijet_p4.Theta()")
        df = df.Define("reco_opang",       "jet1.Vect().Angle(jet2.Vect())")
        df = df.Define("reco_dR",          "(double)jet1.DeltaR(jet2)")
        df = df.Define("jet_esum",         "jet1_e + jet2_e")

        df = df.Define("d_12", f"JetClusteringUtils::get_exclusive_dmerge(_jet{helper.tag}, 1)")
        df = df.Define("d_23", f"JetClusteringUtils::get_exclusive_dmerge(_jet{helper.tag}, 2)")
        df = df.Define("y_23_Q2", "jet_esum > 0 ? d_23 / (jet_esum * jet_esum) : -1.")

        # ── 3. Z window ──────────────────────────────────────────────────────
        zvar = {"mjj": "reco_dijet_m", "recoil": "m_recoil"}[Z_SELECTION]
        df = df.Filter(f"{zvar} > 0 && std::abs({zvar} - {MZ}) < {Z_HALFWIDTH}",
                       f"|{zvar} - mZ| < {Z_HALFWIDTH:g}")

        # ── truth labels ─────────────────────────────────────────────────────
        df = df.Define("gen_quarks", "ZgStd::hard_quarks(Particle, Particle0)")
        df = df.Define("n_gen_quarks", "(int)gen_quarks.size()")
        df = df.Define("gen_q1_pdg", "n_gen_quarks > 0 ? std::abs(gen_quarks[0].PDG) : 0")
        df = df.Define("gen_q2_pdg", "n_gen_quarks > 1 ? std::abs(gen_quarks[1].PDG) : 0")
        df = df.Define("true_flavour_pdg", "n_gen_quarks == 2 ? gen_q1_pdg : 0")

        df = df.Define("jet_getflavour",
            f"FCCAnalyses::JetTaggingUtils::get_flavour({helper.jets}, Particle)")
        df = df.Define("jet1_getflavour_pdg", "(int)jet_getflavour[0]")
        df = df.Define("jet2_getflavour_pdg", "(int)jet_getflavour[1]")
        df = df.Define("getflavour_pair_ok",
            "(int)(n_gen_quarks == 2 && "
            "((jet1_getflavour_pdg == gen_q1_pdg && jet2_getflavour_pdg == gen_q2_pdg) || "
            " (jet1_getflavour_pdg == gen_q2_pdg && jet2_getflavour_pdg == gen_q1_pdg)))")

        # ── 4. optional truth purification ───────────────────────────────────
        if APPLY_GETFLAVOUR_FILTER:
            df = df.Filter("getflavour_pair_ok == 1", "get_flavour pair = hard-process pair")

        # ── flavour tagging (MC PV, unsmeared tracks) ────────────────────────
        _flav_helper = JetFlavourHelper(TAGGER_COLLECTIONS, helper.jets, helper.constituents)
        df = _flav_helper.define(df)
        df = _flav_helper.inference(TAGGER_JSON, TAGGER_ONNX, df)

        df = df.Define("recojet_isQ", "recojet_isU + recojet_isD")
        for i in (1, 2):
            for s in SCORES:
                df = df.Define(f"jet{i}_is{s}", f"(float)recojet_is{s}[{i-1}]")

        # X / (X + Y), -1 when both scores vanish (instead of NaN)
        for num, dens in DISCRIMINANTS:
            for den in dens:
                a, b = f"recojet_is{num}", f"recojet_is{den}"
                df = df.Define(f"D{num}_vs{den}",
                               f"ROOT::VecOps::Where(({a} + {b}) > 0, {a} / ({a} + {b}), -1.f)")

        _CUTFLOWS.append(df.Report())
        return df

    def output():
        out = [
            # photon
            "n_photons_sel", "photon_p", "photon_e", "photon_theta", "photon_esum", "m_recoil",
            # jets
            "n_reco_jets",
            "reco_jet1_p", "reco_jet1_theta", "reco_jet1_phi", "reco_jet1_costheta", "reco_jet1_mass",
            "reco_jet2_p", "reco_jet2_theta", "reco_jet2_phi", "reco_jet2_costheta", "reco_jet2_mass",
            "jet1_e", "jet2_e", "jet_esum",
            "reco_dijet_m", "reco_dijet_p", "reco_dijet_theta", "reco_opang", "reco_dR",
            "d_12", "d_23", "y_23_Q2",
            # truth
            "n_gen_quarks", "gen_q1_pdg", "gen_q2_pdg", "true_flavour_pdg",
            "jet1_getflavour_pdg", "jet2_getflavour_pdg", "getflavour_pair_ok",
            # tagger
            "recojet_isQ",
        ]
        out += [f"jet{i}_is{s}" for i in (1, 2) for s in SCORES]
        out += [f"D{num}_vs{den}" for num, dens in DISCRIMINANTS for den in dens]
        out += list(_flav_helper.outputBranches())
        return out
