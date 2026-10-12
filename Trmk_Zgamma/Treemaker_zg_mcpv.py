'''
Z+gamma -> qq gamma treemaker -- CORRELATION-SOURCE DIAGNOSTIC (MC PV)
======================================================================

MECHANISMS INSTRUMENTED
-----------------------
M1  ENERGY ANTI-CORRELATION.  After the ISR photon is removed E1+E2 is nearly
    fixed event by event, so the two jet energies are anti-correlated by
    construction.  Any energy dependence of the efficiency turns that into a
    tag-probe correlation with no detector pathology behind it.
    -> jet_esum, jet_easym, jet_emin/emax, jetT_e/jetP_e.

M2  FORCED MERGING.  Exclusive njets=2 partitions every particle into exactly
    one jet.  A hard third parton is absorbed, and a mis-assigned constituent
    leaves one jet AND joins the other -- a direct anti-correlation of the two
    taggers' inputs.
    -> d_12, d_23, d_34, y_23, jet1/2_nconst.

M3  PHOTON SELECTION.  The isolated-subset recoil scan, with truth validation.
    -> isr_* branches; n_gamma_legacy* quantify the old sel_p(10) bug at the
       threshold it actually used.

M4  JET ORDERING.  Roles assigned by a deterministic exchange-symmetric bit.
    Every per-jet quantity is projected onto jetT_*/jetP_* here, so downstream
    code never has to index a score RVec by hand.

M5  HEAVY FLAVOUR FROM GLUON SPLITTING.  [NEW]
    The tagger responds to b/c HADRONS, not to the hard-process PDG code.  A
    wzp6_ee_uu event with g->bb contains two genuine b jets: true_flavour_pdg
    calls them light, so they land in the light mistag numerator and are
    missing from the b denominator.  And because g->bb makes TWO b hadrons
    that usually land in opposite hemispheres, BOTH jets tag -- a real
    event-level correlation invisible to an event-level parton label.
    Rates per hadronic Z decay are roughly 0.25% for g->bb and 3% for g->cc,
    against tight-WP light mistags of O(1e-3..1e-4), so at tight working
    points the b-tag rate measured in the light samples is expected to be
    dominated by genuine gluon-split b hadrons rather than by mistags.
    -> jet1/2_nbhad_*, jet1/2_nchad_*, jet1/2_hf_flavour, hf_both_jets,
       hf_clean, n_gsplit_*.
    THE TEST: recompute C on hf_clean == 1.  If the off-diagonal C drops
    toward 1, M5 is the mechanism.

FIXES IN THIS VERSION
---------------------
  * ISR acceptance cut (|cos theta| < 0.99) restored -- it was in production
    and missing here, so the two chains were not comparable
  * n_gamma_legacy* evaluated at 10 GeV, the threshold the old cut used
    (previously computed at 5 GeV while named _gt10)
  * gen_Z_* built from the parentage-selected ISR system, not from the hardest
    status-1 photon in the whole event (which competes against every pi0
    daughter and disagreed with gen_m_recoil silently); gen_Zqq_* added as an
    independent handle and gen_dm_Z_Zqq as the consistency check
  * sqrt(s) no longer hardcoded inside the gen recoil
  * count_matched_to_gen no longer lets two reco photons claim the same gen
    photon, which deflated n_sel_fake
  * dead helpers removed (make_pseudojets_from_p4 used fastjet::PseudoJet with
    no FastJet header included and was never called; angle_to_gen_photon was
    unused)
  * tagger input branches asserted at startup instead of assumed; n_trkstate
    and n_trk written per event
  * n_rp_removed / rp_removal_consistent verify that remove() deleted exactly
    the selected photons
  * gen_tot_p* and rp_tot_p* test the (0,0,0,sqrt(s)) initial state that every
    recoil mass assumes
  * reco_jet{1,2}_mass written out, to settle empirically whether this helper's
    recombination scheme produces massless pseudojets
  * exclusive n=2 jets are labelled with JetTaggingUtils::get_flavour and the
    event is kept only if that unordered pair equals the two hard-process
    quarks (bb→{5,5}, cc→{4,4}, …).  Gluon / unmatched / g→QQ jets drop out.

WHAT TO CHECK IN THE OUTPUT BEFORE TRUSTING A NUMBER
----------------------------------------------------
  n_trk, n_trkstate     nonzero, or the IP features had no input
  tagger_degenerate     ~0, or the softmax collapsed to 1/7
  rp_removal_consistent  1 in every event
  gen_dm_Z_Zqq          narrow around 0, or the ISR parentage selection is
                        incomplete
  gen_tot_px            centred on 0, or recoil_mass needs the crossing-angle
                        boost
  reco_jet1_mass        if ~0, the clustering really is massless-recombination
  getflavour_pair_ok    1 in every stored event (filter applied in the treemaker)
  cutflow report        the gen 2-quark efficiency must agree across the five
                        flavour samples; an unequal loss there biases every
                        relative efficiency downstream.  The get_flavour pair
                        cut is allowed to differ by flavour — that is the
                        purification.
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

# ═════════════════════════════════════════════════════════════════════════════
# Samples
# ═════════════════════════════════════════════════════════════════════════════

processList = {
    'wzp6_ee_bb_pqqmin80_ecm240': {'fraction': 1, 'chunks': 50},
    'wzp6_ee_cc_pqqmin80_ecm240': {'fraction': 1, 'chunks': 50},
    'wzp6_ee_ss_pqqmin80_ecm240': {'fraction': 1, 'chunks': 50},
    'wzp6_ee_uu_pqqmin80_ecm240': {'fraction': 1, 'chunks': 50},
    'wzp6_ee_dd_pqqmin80_ecm240': {'fraction': 1, 'chunks': 50},
}

inputDir  = '/eos/experiment/fcc/ee/generation/DelphesEvents/winter2023/IDEA/'
outputDir = '/eos/user/l/lberiet/CKM_Flav/Zg_240_mcpv_new_tagger_getflav/'

nCPUS    = 8
runBatch = True

TRACK_COLLECTION = "EFlowTrack_1"

FLAV_TAGGER = os.environ.get("FLAV_TAGGER", "new_tagger")
_FLAV_TAGGER_JSON, _FLAV_TAGGER_ONNX = get_flav_tagger_paths(FLAV_TAGGER)

print(f"Flavour tagger: {FLAV_TAGGER}")
print(f"  preprocess: {_FLAV_TAGGER_JSON}")
print(f"  model:      {_FLAV_TAGGER_ONNX}")
print("Tagger primary vertex: MC (diagnostic / reconstruction floor)")
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

        # ── track-state evidence ─────────────────────────────────────────────
        df = zg.define_track_diagnostics(df, TRACK_COLLECTION)

        # ── M3: ISR photon selection ─────────────────────────────────────────
        df = zg.define_isr_selection(df)

        # ── clustering ───────────────────────────────────────────────────────
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

        # ── flavour tagger, MC PV retained on purpose ────────────────────────
        _jet_flavour_helper = JetFlavourHelper(
            make_tagger_collections("RP_no_hard_photon"),
            helper.jets,
            helper.constituents,
        )
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
            + list(_jet_flavour_helper.outputBranches())
        )