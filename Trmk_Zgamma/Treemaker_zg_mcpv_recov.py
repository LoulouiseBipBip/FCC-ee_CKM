'''
Z+gamma -> qq gamma treemaker -- Durham exclusive n=2, fitted PV
================================================================

Same event selection as Treemaker_zg_mcpv.py: Durham exclusive n=2, the same
ISR selection, the same get_flavour pair filter.  The one change is the vertex
the flavour tagger uses.

  * VertexFitterSimple::VertexFitter_Tk on primary tracks (displaced tracks
    removed by get_PrimaryTracks) with the beam-spot constraint — the
    "pri+BSC" configuration of CKM_Flav/WW_semilep_treemaker_recopv.py.
  * That vertex, as a TLorentzVector in mm, replaces the MC primary vertex
    inside JetFlavourHelper.  The impact-parameter features are then computed
    with respect to the fitted PV.
  * Beam-spot sigmas are the 163 GeV numbers from that script (15 / 0.02 /
    650 um).  CKM_Flav/ZGamma_PV_treemker.py quotes the Z-pole set and marks
    sigma_z = 300 um as wrong at this energy.

PV_FOR_TAGGER = False reruns the identical chain on the MC PV.
'''

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ROOT
from addons.FastJet.jetClusteringHelper import ExclusiveJetClusteringHelper
from addons.ONNXRuntime.jetFlavourHelper import JetFlavourHelper
from treemaker_common import (
    define_jet_kinematics_basic, define_jet_derived_observables,
    get_flav_tagger_paths, make_tagger_collections,
    extract_jet_tag_scalars, patch_dndx_if_missing,
)
import zg_common as zg
import zg_incl as zgi

# Fitted primary vertex.  Same VertexFitter_Tk / get_PrimaryTracks contract as
# CKM_Flav/WW_semilep_treemaker_recopv.py.  Position is in mm
# (PV_POS_SCALE_TO_MM); beam-spot sigmas and the beam-spot centre are in
# microns, which is what VertexFitter_Tk expects.
ROOT.gInterpreter.Declare(r"""
#ifndef ZG_RECOPV_HELPERS
#define ZG_RECOPV_HELPERS

#include <cmath>
#include "TLorentzVector.h"
#include "ROOT/RVec.hxx"
#include "edm4hep/VertexData.h"
#include "edm4hep/TrackState.h"

namespace ZgPV {

double vtx_x   (const edm4hep::VertexData& v) { return v.position.x; }
double vtx_y   (const edm4hep::VertexData& v) { return v.position.y; }
double vtx_z   (const edm4hep::VertexData& v) { return v.position.z; }
double vtx_chi2(const edm4hep::VertexData& v) { return v.chi2; }

// edm4hep::VertexData::covMatrix is lower triangular:
//   [0]=xx  [1]=xy  [2]=yy  [3]=xz  [4]=yz  [5]=zz
double vtx_cov(const edm4hep::VertexData& v, int i) {
    if (i < 0 || i > 5) return 0.;
    return v.covMatrix[i];
}

double vtx_sigma(const edm4hep::VertexData& v, int diag_index) {
    double c = vtx_cov(v, diag_index);
    return (c > 0.) ? std::sqrt(c) : -1.;
}

bool vtx_valid(const edm4hep::VertexData& v) {
    return vtx_cov(v, 0) > 0. && vtx_cov(v, 2) > 0. && vtx_cov(v, 5) > 0.;
}

// TLorentzVector position in mm, the type JetFlavourHelper's pv column
// expects.  A failed fit falls back to the beam-spot centre; pv_valid
// flags those events so they can be cut offline.
TLorentzVector pv_as_tlv(const edm4hep::VertexData& v, double scale_to_mm,
                         double fb_x_mm, double fb_y_mm, double fb_z_mm) {
    if (!vtx_valid(v)) return TLorentzVector(fb_x_mm, fb_y_mm, fb_z_mm, 0.);
    return TLorentzVector(v.position.x * scale_to_mm,
                          v.position.y * scale_to_mm,
                          v.position.z * scale_to_mm, 0.);
}

// VertexFitter_Tk and get_PrimaryTracks are not safe below two tracks.
FCCAnalyses::VertexingUtils::FCCAnalysesVertex
safe_vertex_fit(const ROOT::VecOps::RVec<edm4hep::TrackState>& tks,
                bool bsc, double sx, double sy, double sz,
                double bx, double by, double bz) {
    FCCAnalyses::VertexingUtils::FCCAnalysesVertex out{};
    if (tks.size() < 2) {
        out.vertex.chi2 = -1.f;
        return out;
    }
    return FCCAnalyses::VertexFitterSimple::VertexFitter_Tk(
               1, tks, bsc, sx, sy, sz, bx, by, bz);
}

ROOT::VecOps::RVec<edm4hep::TrackState>
safe_primary_tracks(const ROOT::VecOps::RVec<edm4hep::TrackState>& tks,
                    bool bsc, double sx, double sy, double sz,
                    double bx, double by, double bz) {
    if (tks.size() < 2) return tks;
    return FCCAnalyses::VertexFitterSimple::get_PrimaryTracks(
               tks, bsc, sx, sy, sz, bx, by, bz);
}

}  // namespace ZgPV
#endif
""")

# ═════════════════════════════════════════════════════════════════════════════
# Samples
# ═════════════════════════════════════════════════════════════════════════════

processList = {
    'wzp6_ee_bb_pqqmin40_ecm163': {'fraction': 1, 'chunks': 50},
    'wzp6_ee_cc_pqqmin40_ecm163': {'fraction': 1, 'chunks': 50},
    'wzp6_ee_ss_pqqmin40_ecm163': {'fraction': 1, 'chunks': 50},
    'wzp6_ee_uu_pqqmin40_ecm163': {'fraction': 1, 'chunks': 50},
    'wzp6_ee_dd_pqqmin40_ecm163': {'fraction': 1, 'chunks': 50},
}

inputDir  = '/eos/experiment/fcc/ee/generation/DelphesEvents/winter2023/IDEA/'
outputDir = zg.outputDir

nCPUS    = 8
runBatch = True

# Beam-spot sigmas in MICRONS, as VertexFitter_Tk expects them.  163 GeV
# numbers from WW_semilep_treemaker_recopv.py (not the Z-pole set in
# ZGamma_PV_treemker.py).
BSC_SIGMA_X = 15       # um
BSC_SIGMA_Y = 20e-3    # um
BSC_SIGMA_Z = 650.     # um
BSC_POS     = (0., 0., 0.)   # um

TRACK_COLLECTION = "EFlowTrack_1"
# VertexFitterSimple returns VertexData.position in mm.
PV_POS_SCALE_TO_MM = 1.0

# False reruns this chain with the MC primary vertex still in the tagger.
PV_FOR_TAGGER = False
# Extra fit without the beam-spot constraint, as a resolution control.
DO_PV_CONTROL = False


def retarget_tagger_to_pv(flav_helper, pv_column, verbose=True):
    """Replace JetFlavourHelper's MC-PV column with pv_column.

    pv_column must already exist and be a TLorentzVector position in mm.
    Overwriting this one definition redirects pfcand_dxy, pfcand_dz,
    pfcand_phi0 and pfcand_mtof, i.e. the whole impact-parameter block.
    """
    key = "pv{}".format(flav_helper.tag)
    defs = flav_helper.definition
    if key not in defs:
        raise RuntimeError(
            f"JetFlavourHelper has no '{key}' definition. Refusing to "
            "continue: the job would run on the MC PV and look correct "
            "while answering a different question.")
    if verbose:
        print(f"  [PV swap] {key}: {defs[key]}")
        print(f"  [PV swap] {key}: -> {pv_column}")
    defs[key] = pv_column
    return key


def pv_branches():
    out = [
        'n_trk_pv', 'f_trk_pv',
        'pv_valid', 'pv_chi2',
        'pv_x', 'pv_y', 'pv_z',
        'pv_resx', 'pv_resy', 'pv_resz',
        'pv_sigx', 'pv_sigy', 'pv_sigz',
        'pv_pullx', 'pv_pully', 'pv_pullz',
        'mc_pv_x', 'mc_pv_y', 'mc_pv_z',
    ]
    if DO_PV_CONTROL:
        out += ['pvc_valid', 'pvc_chi2']
        out += [f'pvc_{c}'    for c in "xyz"]
        out += [f'pvc_res{c}' for c in "xyz"]
        out += [f'pvc_sig{c}' for c in "xyz"]
    return out


FLAV_TAGGER = os.environ.get("FLAV_TAGGER", "default")
_FLAV_TAGGER_JSON, _FLAV_TAGGER_ONNX = get_flav_tagger_paths(FLAV_TAGGER)

print(f"Flavour tagger: {FLAV_TAGGER}")
print(f"  preprocess: {_FLAV_TAGGER_JSON}")
print(f"  model:      {_FLAV_TAGGER_ONNX}")
print(f"Tagger primary vertex: "
      f"{'RECONSTRUCTED (pri+BSC)' if PV_FOR_TAGGER else 'MC'}")
print("Jet finder: Durham exclusive n=2")
print(f"ISR selection: pmin={zg.ISR_PMIN} GeV, cone={zg.ISR_CONE_DEG} deg, "
      f"iso<{zg.ISR_ISO_MAX}, |cos|<{zg.ISR_COSMAX}, maxcand={zg.ISR_MAXCAND}")

_jet_helper = None
_jet_flavour_helper = None

zgi.describe()


class RDFanalysis():

    def analysers(df):
        global _jet_helper, _jet_flavour_helper

        # Fatal if the impact-parameter inputs are not really present.
        df = zg.check_tagger_inputs(df)

        # ── gen level: quarks, truth ISR, truth Z, heavy-flavour content ─────
        df = zg.define_gen_zg(df)

        if zg.DO_SMEAR:                                   
            df = zg.define_smeared_tracks(df, TRACK_COLLECTION)
        TRK = "EFlowTrack_1_sm" if zg.DO_SMEAR else TRACK_COLLECTION



        # ── track-state evidence ─────────────────────────────────────────────
        # Also builds RP2TRK and n_trk, which the vertex fit reuses.
        df = zg.define_track_diagnostics(df, TRK)        

        # ── fitted PV (pri+BSC), MC PV kept as the residual reference ───────
        df = df.Define("mc_pv_p4",
            "FCCAnalyses::MCParticle::get_EventPrimaryVertexP4()(mc_particles)")
        df = df.Define("mc_pv_x", "(double)mc_pv_p4.X()")   # mm
        df = df.Define("mc_pv_y", "(double)mc_pv_p4.Y()")
        df = df.Define("mc_pv_z", "(double)mc_pv_p4.Z()")

        _bx, _by, _bz = BSC_POS
        _fb = tuple(v * 1e-3 for v in BSC_POS)   # um -> mm, fallback position
        df = df.Define("PrimaryTracks",
            f"ZgPV::safe_primary_tracks(RP2TRK, true, {BSC_SIGMA_X}, "
            f"{BSC_SIGMA_Y}, {BSC_SIGMA_Z}, {_bx}, {_by}, {_bz})")
        df = df.Define("n_trk_pv", "(int)PrimaryTracks.size()")
        df = df.Define("f_trk_pv",
            "n_trk > 0 ? (double)n_trk_pv / (double)n_trk : -1.")
        df = df.Define("vtxobj_pv",
            f"ZgPV::safe_vertex_fit(PrimaryTracks, true, {BSC_SIGMA_X}, "
            f"{BSC_SIGMA_Y}, {BSC_SIGMA_Z}, {_bx}, {_by}, {_bz})")
        df = df.Define("vtx_pv", "vtxobj_pv.vertex")
        df = df.Define("pv_valid", "(int)ZgPV::vtx_valid(vtx_pv)")
        df = df.Define("pv_chi2",  "ZgPV::vtx_chi2(vtx_pv)")
        for c, acc in (("x", "ZgPV::vtx_x"), ("y", "ZgPV::vtx_y"),
                       ("z", "ZgPV::vtx_z")):
            df = df.Define(f"pv_{c}", f"{PV_POS_SCALE_TO_MM} * {acc}(vtx_pv)")
        for c in "xyz":
            df = df.Define(f"pv_res{c}", f"1000. * (pv_{c} - mc_pv_{c})")  # um
        for c, idx in (("x", 0), ("y", 2), ("z", 5)):
            df = df.Define(f"pv_sig{c}",
                f"1000. * {PV_POS_SCALE_TO_MM} * ZgPV::vtx_sigma(vtx_pv, {idx})")
        for c in "xyz":
            df = df.Define(f"pv_pull{c}",
                f"pv_sig{c} > 0 ? pv_res{c} / pv_sig{c} : -999.")
        df = df.Define("reco_pv_p4",
            f"ZgPV::pv_as_tlv(vtx_pv, {PV_POS_SCALE_TO_MM}, "
            f"{_fb[0]}, {_fb[1]}, {_fb[2]})")

        if DO_PV_CONTROL:
            df = df.Define("vtxobj_pv_nobsc",
                f"ZgPV::safe_vertex_fit(PrimaryTracks, false, {BSC_SIGMA_X}, "
                f"{BSC_SIGMA_Y}, {BSC_SIGMA_Z}, {_bx}, {_by}, {_bz})")
            df = df.Define("vtx_pv_nobsc", "vtxobj_pv_nobsc.vertex")
            df = df.Define("pvc_valid", "(int)ZgPV::vtx_valid(vtx_pv_nobsc)")
            df = df.Define("pvc_chi2",  "ZgPV::vtx_chi2(vtx_pv_nobsc)")
            for c, acc, idx in (("x", "ZgPV::vtx_x", 0),
                                ("y", "ZgPV::vtx_y", 2),
                                ("z", "ZgPV::vtx_z", 5)):
                df = df.Define(f"pvc_{c}",
                    f"{PV_POS_SCALE_TO_MM} * {acc}(vtx_pv_nobsc)")
                df = df.Define(f"pvc_res{c}", f"1000. * (pvc_{c} - mc_pv_{c})")
                df = df.Define(f"pvc_sig{c}",
                    f"1000. * {PV_POS_SCALE_TO_MM} * "
                    f"ZgPV::vtx_sigma(vtx_pv_nobsc, {idx})")

        # ── M3: ISR photon selection ─────────────────────────────────────────
        df = zg.define_isr_selection(df)

        # ── clustering: Durham exclusive n=2, same as Treemaker_zg_mcpv.py ──
        helper = ExclusiveJetClusteringHelper("RP_no_hard_photon", 2)
        df = helper.define(df)
        _jet_helper = helper

        df = zg.define_jets_and_scales(df, helper)
        df = zg.define_getflavour_pair_filter(df, helper)
        df = zgi.define_inclusive_crosscheck(df)
        df = define_jet_kinematics_basic(df, jet_indices=(1, 2))
        df = define_jet_derived_observables(df)

        # ── M5: which jets actually contain heavy-flavour hadrons ────────────
        df = zg.define_hf_jet_content(df)

        # ── flavour tagger; PV column swapped to the fitted vertex ───────────
        _jet_flavour_helper = JetFlavourHelper(
            zg.tagger_collections_smeared("RP_no_hard_photon"),
            helper.jets,
            helper.constituents,
        )
        if PV_FOR_TAGGER:
            retarget_tagger_to_pv(_jet_flavour_helper, "reco_pv_p4")
        # No-op on this campaign (EFlowTrack_2 is present and real); kept as
        # the guard for samples where it is not.  It mutates
        # helper.definition in place and returns None -- there is no return
        # value to capture.
        patch_dndx_if_missing(df, _jet_flavour_helper)

        df = _jet_flavour_helper.define(df)
        df = _jet_flavour_helper.inference(_FLAV_TAGGER_JSON, _FLAV_TAGGER_ONNX, df)
        df = extract_jet_tag_scalars(df, n_jets=2)

        df = zg.define_tagger_observables(df)

        # ── M4: roles, and every per-jet quantity projected onto them ────────
        df = zg.define_tag_probe(df)

        # ── truth matching ───────────────────────────────────────────────────
        df = zg.define_truth_matching(df)

        return df

    def output():
        return (
            zg.track_branches()
            + zg.gen_branches()
            + zg.isr_branches()
            + zg.jet_branches()
            + zg.hf_branches()
            + zg.tagger_branches()
            + zg.tag_probe_branches()
            + zg.truth_matching_branches()
            + zgi.incl_branches()
            + pv_branches()
            + zg.smear_branches()
            + list(_jet_flavour_helper.outputBranches())
        )
