#!/usr/bin/env python3
"""
MC-true flavour-tagging efficiency: WW (semileptonic) vs Z+gamma.

Purpose
-------
Compare eps_b, eps_c, eps_s at the analysis working points in 3D bins of
|p|, |cos theta| and jet-jet opening angle, for WW->munu qq and
Z+gamma->qq gamma.  The working-point definition is the same on both sides.  The truth label is
the same definition tp_counting.py uses for Zgamma (sample / hard-process
PDG) and the WW treemaker's FCCAnalyses get_flavour per-jet PDG, because
W->qq' is mixed-flavour.  No tag-and-probe here: this is the MC truth
efficiency, so every jet with a truth label enters the denominator.

Truth labels
------------
WW  jet{i}_getflavour_pdg  FCCAnalyses JetTaggingUtils::get_flavour
                           (angle match to a status-2 parton).  Same branch
                           WW_semilep_treemaker.py writes and that
                           plot_tageff_ww_semilep.C / plot_ww_flav_2d.py
                           use.  0 = unmatched; those jets do not enter a
                           b/c/s denominator.  NOT jet{i}_matched_pdg
                           (dR to the two hard-process W->qq' quarks).

Zg  true_flavour_pdg       event-level hard-process |PDG|, identical to how
                           tp_counting.py labels the bb/cc/ss files (every
                           jet in the bb sample is true b).  NOT jet_hf_flavour:
                           gluon-split heavy flavour stays with the sample.

Fiducial  (copied from tp_counting.py)
-------------------------------------
Both processes: both jets p > JET_P_MIN and |cos theta| < JET_ABSCOS_MAX.

y_23_Q2 = d_23/(E1+E2)^2, independent WW / Zgamma windows (Y23Q2).
A bound on either side also drops the sentinel y_23_Q2 = -1.

Zgamma only, matching jet_fiducial_mask + Z_SELECTION:
    reco_dR >= DR_MIN,  DIJET_P_MIN < reco_dijet_p < DIJET_P_MAX,
    and, if Z_SELECTION='dijet', MASS_MIN < reco_dijet_m < MASS_MAX.
Those three are the recoiling-Z window and are not applied to WW.

Discriminant conventions
------------------------
WW  writes DB_vsC = isB/(isB+isC) with no guard  -> NaN when both are zero.
Zg  writes the same quantity through Zg::safe_ratio -> -1 when both are zero.
sfv() maps both onto -1; logit_sfv() then clips to [1e-8, 1-1e-8] before
logit, matching tp_counting._logit, so a collapsed tagger fails every cut
instead of silently NaN-ing out of the numerator.

The working point is the NOMINAL cascade from Fit_TP_Multi/tp_counting.py
(NOMINAL_WP_LOGIT + CASCADE_DISC).  There is no loose/tight pair here:
loose/tight in the previous version of this script were placeholders on
the wrong planes (DB_vsQ, DS_vsG, …) and cannot be compared to T&P.

    pass b : logit(DB_vsC) > logit(0.99717)  AND  logit(DB_vsS) > logit(0.99564)
    pass c : fails b, AND logit(DC_vsB) > logit(0.00006) AND logit(DC_vsS) > logit(0.53319)
    pass s : fails b and c, AND logit(DS_vsC) > logit(0.46681) AND logit(DS_vsQ) > logit(0.20117)

These are disjoint: a jet sits in exactly one of {b, c, s, q}.  The MC-true
efficiency for flavour f is P(cascade category = f | true flavour = f).
That is the same quantity as mixture.json's bins[*].response[f][f]["truth"],
on the same Zgamma fiducial and the same sample-level truth label.

Output
------
  tageff_ww_vs_zg.root   3D num/den (p, |cos theta|, opang), 1D projections,
                         per-cell and mean Zg/WW ratios
  tageff_ww_vs_zg.pdf    per flavour: 1D projected eps + 1D ratio with the
                         3D cells drawn as a swarm; then (p, |cos theta|)
                         COLZ slices in each opang bin.  Last page: b/c/s
                         1D means.
  stdout                 inclusive efficiencies and Zg/WW (inclusive and 3D)
"""

import glob
import math
import array
import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.EnableImplicitMT()

# ═════════════════════════════════════════════════════════════════════════════
# Configuration
# ═════════════════════════════════════════════════════════════════════════════

TREE = "events"

# ROOT files live one sample-directory down, not in the treemaker outputDir
# itself.  Restrict WW to munumuqq so the old Zgamma chunks that share
# kin_jet163_preshower/ are not mixed in.
# WW_GLOB = "/eos/user/l/lberiet/CKM_Flav/WW_semilep_240_newtagger/wzp6_ee_munumuqq_V*_ecm240/*.root"
# ZG_GLOB = "/eos/user/l/lberiet/CKM_Flav/Zg_240_mcpv_new_tagger_getflav/wzp6_ee_*_pqqmin80_ecm240/*.root"

WW_GLOB = "/eos/user/l/lberiet/CKM_Flav/WW_semilep_163_newtagger/wzp6_ee_munumuqq_V*_ecm163/*.root"
ZG_GLOB = "/eos/user/l/lberiet/CKM_Flav/Zg_163_mcpv_new_tagger_getflav/wzp6_ee_*_pqqmin40_ecm163/*.root"
# ── fiducial / Z selection, copied from tp_counting.py (do not drift) ────────
Z_SELECTION = "dijet"
MZ = 91.1876
RECOIL_HALFWIDTH = 20.0
MASS_MIN, MASS_MAX = 75.0, 110.0

JET_P_MIN, JET_P_MAX           = None, None
JET_ABSCOS_MIN, JET_ABSCOS_MAX = None, 0.99
DR_MIN, DR_MAX                 = None, None
DIJET_P_MIN, DIJET_P_MAX       = None, None

# Independent y_23_Q2 = d_23/(E1+E2)^2 windows: (min, max).  None = open.
# Upper bound rejects 3-jet-like events; WW and Zgamma spectra differ, so
# the two sides are not forced to share a number.
Y23Q2 = {
    "ww": (None, None),
    "zg": (None, 0.05),
}


def _window_cut(expr, lo, hi):
    parts = []
    if lo is not None:
        parts.append(f"({expr}) >= {lo}")
    if hi is not None:
        parts.append(f"({expr}) <= {hi}")
    return " && ".join(parts)


def y23_expr(kind):
    """Event-level y_23_Q2 window for this sample, or None if both sides open.

    y_23_Q2 = -1 is a failed-kinematics sentinel, not a small merging scale;
    it is dropped whenever either bound is set (same as tp_counting.get_y23).
    """
    lo, hi = Y23Q2[kind]
    if lo is None and hi is None:
        return None
    parts = ["y_23_Q2 >= 0"]
    w = _window_cut("y_23_Q2", lo, hi)
    if w:
        parts.append(w)
    return " && ".join(parts)


def fiducial_expr(kind):
    """Event-level selection.  WW gets only |p| / |cos theta|; Zg gets the
    full tp_counting jet_fiducial_mask plus the configured Z window."""
    parts = []
    for i in (1, 2):
        w = _window_cut(f"reco_jet{i}_p", JET_P_MIN, JET_P_MAX)
        if w:
            parts.append(w)
        w = _window_cut(f"std::abs(std::cos(reco_jet{i}_theta))",
                        JET_ABSCOS_MIN, JET_ABSCOS_MAX)
        if w:
            parts.append(w)
    if kind != "zg":
        return " && ".join(parts)
    w = _window_cut("reco_dR", DR_MIN, DR_MAX)
    if w:
        parts.append(w)
    w = _window_cut("reco_dijet_p", DIJET_P_MIN, DIJET_P_MAX)
    if w:
        parts.append(w)
    if Z_SELECTION == "dijet":
        w = _window_cut("reco_dijet_m", MASS_MIN, MASS_MAX)
        if w:
            parts.append(w)
    elif Z_SELECTION == "recoil":
        parts.append(
            f"m_recoil > 0 && std::abs(m_recoil - {MZ}) < {RECOIL_HALFWIDTH}")
    return " && ".join(parts)


SAMPLES = [
    dict(name="WW semilep",
         kind="ww",
         tag="WW",
         files=WW_GLOB,
         colour=ROOT.kBlack,
         marker=20),
    dict(name="Z#gamma",
         kind="zg",
         tag="Zg",
         files=ZG_GLOB,
         colour=ROOT.kAzure + 1,
         marker=21),
]

# ── nominal cascade, copied from tp_counting.py (do not drift) ──────────────
# Planes: both scores in (0,1), cut FROM BELOW after logit.
SCORE_EPS = 1e-8
CASCADE_DISC = {
    "b": ("DB_vsC", "DB_vsS"),
    "c": ("DC_vsB", "DC_vsS"),
    "s": ("DS_vsC", "DS_vsQ"),
}


def _lg(p):
    """score -> logit, identical to tp_counting._lg."""
    p = min(max(float(p), SCORE_EPS), 1.0 - SCORE_EPS)
    return math.log(p / (1.0 - p))


NOMINAL_WP_LOGIT = {
    "b": {"x": _lg(0.99717), "y": _lg(0.99564)},
    "c": {"x": _lg(0.00006), "y": _lg(0.53319)},
    "s": {"x": _lg(0.46681), "y": _lg(0.20117)},
}

FLAV_PDG = {"b": 5, "c": 4, "s": 3}
FLAV_STYLE = {
    "b": dict(colour=ROOT.kBlack,     marker=20),
    "c": dict(colour=ROOT.kRed + 1,   marker=21),
    "s": dict(colour=ROOT.kAzure + 1, marker=22),
}

# Binning: tp_counting.EDGES_COARSE.  Efficiencies are filled in the 3D grid
# (p, |cos theta|, opang); 1D displays are projections / cell averages of that.
#For 240 GeV 
# P_BINS = array.array('d', [20, 30, 40, 60, 80, 100])
# COS_BINS = array.array('d', [0.0, 0.74, 0.88, 0.92, 0.96, 0.99])
# OPA_BINS = array.array('d', [1,1.5,1.7,1.9,2.1,2.3])
#For 163 GeV 
P_BINS = array.array('d', [15, 17.5, 20, 25, 40, 55,70, 85])
COS_BINS = array.array('d', [0.0, 0.74, 0.88, 0.92, 0.96, 0.99])
OPA_BINS = array.array('d', [1.5,1.8,2,2.25,2.5,3])
AXES_1D = (("p", "|p| [GeV]"),
           ("cos", "|cos#theta|"),
           ("opa", "opening angle [rad]"))

OUT_ROOT = "tageff_ww_vs_zg_dijet163_p20.root"
OUT_PDF  = "tageff_ww_vs_zg_dijet163_p20.pdf"

# Ratio-axis range.  Inclusive Zg/WW is ~1.00, so 0.7–1.3 hides the 3D
# structure; this window is wide enough for outliers and tight enough to see.
RATIO_YMIN, RATIO_YMAX = 0.92, 1.08


# ═════════════════════════════════════════════════════════════════════════════
# C++ helpers
# ═════════════════════════════════════════════════════════════════════════════

ROOT.gInterpreter.Declare(r"""
#ifndef TAGEFF_CMP_HELPERS
#define TAGEFF_CMP_HELPERS
#include <cmath>
#include "ROOT/RVec.hxx"

// Safe per-jet discriminant access.  WW leaves isB/(isB+isC) as NaN when the
// denominator vanishes, Zgamma's safe_ratio already returns -1.  Map both to
// -1 so a working-point expression means the same thing on both sides, and so
// a collapsed tagger is visible in a histogram instead of silently failing
// every cut.
template <typename T>
double sfv(const ROOT::VecOps::RVec<T>& v, int k) {
    if (k < 0 || k >= (int)v.size()) return -1.;
    double x = (double)v[k];
    return std::isfinite(x) ? x : -1.;
}

// logit of sfv, clipped to [eps, 1-eps] exactly as tp_counting._logit.
template <typename T>
double logit_sfv(const ROOT::VecOps::RVec<T>& v, int k) {
    const double eps = 1e-8;
    double x = sfv(v, k);
    if (x < eps) x = eps;
    if (x > 1.0 - eps) x = 1.0 - eps;
    return std::log(x / (1.0 - x));
}
#endif
""")


# ═════════════════════════════════════════════════════════════════════════════
# Per-jet column construction
# ═════════════════════════════════════════════════════════════════════════════

def _columns(df):
    return {str(c) for c in df.GetColumnNames()}


def _require(cols, names, sample_name):
    missing = [n for n in names if n not in cols]
    if missing:
        raise RuntimeError(
            f"[{sample_name}] missing branches: {missing}")


def _corrupt_basket(branch):
    """Return a reason if a non-final basket is a truncated EOS write.

    A mid-write chunk leaves one basket much smaller than the others, with
    fNevBuf = 0 after decompression.  RDataFrame then segfaults in
    RDefineReader instead of reporting the file.  The final basket of a
    healthy file is short because it is only partly filled, so it is not
    treated as suspect.
    """
    nb = int(branch.GetWriteBasket())
    if nb <= 0:
        return f"{branch.GetName()} has no baskets"
    sizes = [int(branch.GetBasketBytes()[k]) for k in range(nb)]
    med = sorted(sizes)[nb // 2]
    if med <= 0:
        return f"{branch.GetName()} median basket is empty"
    for k, sz in enumerate(sizes[:-1]):
        if sz >= 0.5 * med:
            continue
        bk = branch.GetBasket(k)
        if not bk or bk.GetNevBuf() <= 0 or bk.GetObjlen() <= 0:
            return (f"{branch.GetName()} basket {k} unreadable "
                    f"({sz} bytes, median {med})")
    return None


def usable_root_files(paths, branches):
    """Drop chunks that would crash the multi-thread reader."""
    good = []
    old_level = ROOT.gErrorIgnoreLevel
    ROOT.gErrorIgnoreLevel = ROOT.kFatal
    try:
        for path in paths:
            f = ROOT.TFile.Open(path)
            reason = None
            if not f or f.IsZombie():
                reason = "file does not open"
            else:
                tree = f.Get(TREE)
                if not tree:
                    reason = f"no TTree '{TREE}'"
                elif tree.GetEntries() <= 0:
                    reason = "empty tree"
                else:
                    for name in branches:
                        branch = tree.GetBranch(name)
                        if branch:
                            reason = _corrupt_basket(branch)
                            if reason:
                                break
                f.Close()
            if reason:
                print(f"  [skip] {path}")
                print(f"         {reason}")
                continue
            good.append(path)
    finally:
        ROOT.gErrorIgnoreLevel = old_level
    if not good:
        raise RuntimeError(f"no readable ROOT files among {len(paths)} matched")
    if len(good) < len(paths):
        print(f"  using {len(good)}/{len(paths)} readable file(s)")
    return good


def flavour_expr(kind, i):
    """C++ expression for the per-jet truth PDG used as the efficiency label.

    WW : |jet{i}_getflavour_pdg| from FCCAnalyses get_flavour (mixed-flavour
         W, sample name is not enough).  Same flag WW_semilep_treemaker.py
         writes.  0 means unmatched and is not a b/c/s label.
    Zg : true_flavour_pdg, the hard-process |PDG|.  Same as tp_counting.py,
         which labels every jet by which flavour sample the event came from.
    """
    return f"std::abs(jet{i}_getflavour_pdg)"


def build(sample):
    """Return an RDataFrame carrying harmonised per-jet columns j{i}_*."""
    files = sorted(glob.glob(sample["files"]))
    if not files:
        raise RuntimeError(f"no files matched {sample['files']}")
    # Branches the event loop actually deserialises.  One truncated basket
    # in any of these crashes EnableImplicitMT inside RDefineReader.
    read_branches = [
        "reco_jet1_p", "reco_jet2_p", "reco_jet1_theta", "reco_jet2_theta",
        "reco_opang",
        "DB_vsC", "DB_vsS", "DC_vsB", "DC_vsS", "DS_vsC", "DS_vsQ",
        "jet1_getflavour_pdg", "jet2_getflavour_pdg",
    ]
    files = usable_root_files(files, read_branches)
    df = ROOT.RDataFrame(TREE, files)
    cols = _columns(df)

    if sample["kind"] == "ww":
        need = ["jet1_getflavour_pdg", "jet2_getflavour_pdg", "reco_opang"]
    elif sample["kind"] == "zg":
        need = ["true_flavour_pdg", "reco_dR", "reco_dijet_p", "reco_dijet_m",
                "reco_opang",
                "DB_vsC", "DB_vsS", "DC_vsB", "DC_vsS", "DS_vsC", "DS_vsQ"]
        if Z_SELECTION == "recoil":
            need.append("m_recoil")
    else:
        raise ValueError(f"unknown sample kind {sample['kind']!r}")
    if y23_expr(sample["kind"]) is not None:
        need.append("y_23_Q2")
    _require(cols, need, sample["name"])

    df = df.Filter(fiducial_expr(sample["kind"]), "fiducial")
    y23 = y23_expr(sample["kind"])
    if y23 is not None:
        df = df.Filter(y23, "y23_Q2")

    for i in (1, 2):
        k = i - 1

        df = df.Define(f"j{i}_p",    f"(double)reco_jet{i}_p")
        df = df.Define(f"j{i}_acos", f"std::abs(std::cos(reco_jet{i}_theta))")
        df = df.Define(f"j{i}_flav", flavour_expr(sample["kind"], i))

        # Diagnostic only: how often an unused alternative label would have
        # overridden the efficiency label.  Not used in the numerator/denominator.
        if sample["kind"] == "zg" and f"jet{i}_hf_flavour" in cols:
            df = df.Define(
                f"j{i}_label_disagree",
                f"(int)(jet{i}_hf_flavour > 0 && "
                f"jet{i}_hf_flavour != (int)true_flavour_pdg)")
        elif sample["kind"] == "ww" and f"jet{i}_matched_pdg" in cols:
            df = df.Define(
                f"j{i}_label_disagree",
                f"(int)(std::abs(jet{i}_matched_pdg) != "
                f"std::abs(jet{i}_getflavour_pdg))")
        else:
            df = df.Define(f"j{i}_label_disagree", "0")

        # Nominal cascade, same planes and logit cuts as tp_counting.py.
        for fl, (xb, yb) in CASCADE_DISC.items():
            df = df.Define(f"j{i}_lg_{fl}x", f"logit_sfv({xb}, {k})")
            df = df.Define(f"j{i}_lg_{fl}y", f"logit_sfv({yb}, {k})")
        bx, by = NOMINAL_WP_LOGIT["b"]["x"], NOMINAL_WP_LOGIT["b"]["y"]
        cx, cy = NOMINAL_WP_LOGIT["c"]["x"], NOMINAL_WP_LOGIT["c"]["y"]
        sx, sy = NOMINAL_WP_LOGIT["s"]["x"], NOMINAL_WP_LOGIT["s"]["y"]
        df = df.Define(
            f"j{i}_pass_b",
            f"j{i}_lg_bx > {bx:.16g} && j{i}_lg_by > {by:.16g}")
        df = df.Define(
            f"j{i}_pass_c",
            f"!j{i}_pass_b && j{i}_lg_cx > {cx:.16g} && j{i}_lg_cy > {cy:.16g}")
        df = df.Define(
            f"j{i}_pass_s",
            f"!j{i}_pass_b && !j{i}_pass_c && "
            f"j{i}_lg_sx > {sx:.16g} && j{i}_lg_sy > {sy:.16g}")

    return df


# ═════════════════════════════════════════════════════════════════════════════
# Booking
# ═════════════════════════════════════════════════════════════════════════════

def book(df, tag):
    """Book 3D num/den histograms for every flavour at the nominal cascade WP.

    Axes: |p|, |cos theta|, jet-jet opening angle (coarse edges).  Both jets
    of an event are filled into the same histogram, so a 2-jet event
    contributes two entries.  That is correct for an MC-truth efficiency: the
    per-jet correlations only matter for a tag-and-probe estimator, which this
    deliberately is not.  Jets outside any of the three axis ranges go to
    overflow and are excluded from the 1D displays.
    """
    out = {}
    title = ";|p| [GeV];|cos#theta|;opening angle [rad]"
    for fl, pdg in FLAV_PDG.items():
        acc = {}
        for i in (1, 2):
            dfl = df.Filter(f"j{i}_flav == {pdg}")
            dnm = dfl.Filter(f"j{i}_pass_{fl}")
            for lvl, d in (("den", dfl), ("num", dnm)):
                n = f"{tag}_{fl}_{lvl}_j{i}_3d"
                acc.setdefault(f"{lvl}3d", []).append(d.Histo3D(
                    ROOT.RDF.TH3DModel(
                        n, title,
                        len(P_BINS) - 1, P_BINS,
                        len(COS_BINS) - 1, COS_BINS,
                        len(OPA_BINS) - 1, OPA_BINS),
                    f"j{i}_p", f"j{i}_acos", "reco_opang"))
        out[fl] = acc
    return out


def merge(acc):
    """Collapse the per-jet-index lists into single histograms."""
    res = {}
    for k, hs in acc.items():
        h = hs[0].GetValue().Clone(hs[0].GetName().replace("_j1", ""))
        h.SetDirectory(0)
        for extra in hs[1:]:
            h.Add(extra.GetValue())
        res[k] = h
    return res


def project_axis(h3, axis, name):
    """1D projection of a TH3 onto p / |cos theta| / opang, excluding overflow."""
    nx, ny, nz = h3.GetNbinsX(), h3.GetNbinsY(), h3.GetNbinsZ()
    if axis == "p":
        h = h3.ProjectionX(name, 1, ny, 1, nz, "e")
    elif axis == "cos":
        h = h3.ProjectionY(name, 1, nx, 1, nz, "e")
    elif axis == "opa":
        h = h3.ProjectionZ(name, 1, nx, 1, ny, "e")
    else:
        raise ValueError(axis)
    h.SetDirectory(0)
    return h


def attach_projections(res, tag):
    """Add 1D projections of the 3D num/den onto the results dict."""
    for fl in FLAV_PDG:
        for lvl in ("num", "den"):
            h3 = res[fl][f"{lvl}3d"]
            for var, _ in AXES_1D:
                res[fl][f"{lvl}_{var}"] = project_axis(
                    h3, var, f"{tag}_{fl}_{lvl}_{var}")
    return res


# ═════════════════════════════════════════════════════════════════════════════
# Efficiencies and ratios
# ═════════════════════════════════════════════════════════════════════════════

def efficiency(num, den, name):
    """Clopper-Pearson efficiency as a TGraphAsymmErrors (1D) so empty bins
    are dropped rather than drawn at zero."""
    if not ROOT.TEfficiency.CheckConsistency(num, den):
        raise RuntimeError(f"inconsistent num/den for {name}")
    te = ROOT.TEfficiency(num, den)
    te.SetStatisticOption(ROOT.TEfficiency.kFCP)
    te.SetName(name)
    g = te.CreateGraph() if hasattr(te, "CreateGraph") else None
    return te, g


def _cell_ratio(a, A, b, B):
    """(eps1, eps2, ratio, ratio_err, harmonic occupancy) or None if empty."""
    if A <= 0 or B <= 0 or a <= 0 or b <= 0:
        return None
    e1, e2 = a / A, b / B
    s1 = math.sqrt(max(e1 * (1 - e1) / A, 0.0))
    s2 = math.sqrt(max(e2 * (1 - e2) / B, 0.0))
    r = e1 / e2
    er = r * math.sqrt((s1 / e1) ** 2 + (s2 / e2) ** 2)
    w = (A * B) / (A + B)
    return e1, e2, r, er, w


def _iter_3d_slice(h3, axis, ibin):
    """Yield (ix, iy, iz) for the 3D cells that sit in 1D bin `ibin` of axis."""
    nx, ny, nz = h3.GetNbinsX(), h3.GetNbinsY(), h3.GetNbinsZ()
    if axis == "p":
        return ((ibin, j, k) for j in range(1, ny + 1) for k in range(1, nz + 1))
    if axis == "cos":
        return ((i, ibin, k) for i in range(1, nx + 1) for k in range(1, nz + 1))
    if axis == "opa":
        return ((i, j, ibin) for i in range(1, nx + 1) for j in range(1, ny + 1))
    raise ValueError(axis)


def _axis_of(h3, axis):
    return {"p": h3.GetXaxis(), "cos": h3.GetYaxis(), "opa": h3.GetZaxis()}[axis]


def ratio_1d_bundle(n1, d1, n2, d2, axis, name):
    """1D display of the 3D-cell Zg/WW ratios.

    gscat : every occupied 3D cell, drawn vs this axis (jittered inside the
            1D bin so cells with different |cos| / opang do not sit on top
            of each other).
    genv  : occupancy-weighted RMS of those cells around the mean (the
            cell-to-cell spread, not the statistical error).
    gmean : occupancy-weighted mean, with the statistical error on that mean.
    """
    ax = _axis_of(n1, axis)
    ny, nz = n1.GetNbinsY(), n1.GetNbinsZ()
    nx = n1.GetNbinsX()
    n_other = {"p": ny * nz, "cos": nx * nz, "opa": nx * ny}[axis]

    gmean = ROOT.TGraphAsymmErrors()
    gmean.SetName(name + "_mean")
    genv = ROOT.TGraphAsymmErrors()
    genv.SetName(name + "_rms")
    gscat = ROOT.TGraph()
    gscat.SetName(name + "_cells")
    ip = isp = 0

    for b in range(1, ax.GetNbins() + 1):
        lo, hi = ax.GetBinLowEdge(b), ax.GetBinUpEdge(b)
        x = ax.GetBinCenter(b)
        sw = sr = srr = 0.0
        cells = []
        i_other = 0
        for ix, iy, iz in _iter_3d_slice(n1, axis, b):
            cell = _cell_ratio(n1.GetBinContent(ix, iy, iz),
                               d1.GetBinContent(ix, iy, iz),
                               n2.GetBinContent(ix, iy, iz),
                               d2.GetBinContent(ix, iy, iz))
            i_other += 1
            if cell is None:
                continue
            _, _, r, er, w = cell
            sw += w
            sr += w * r
            srr += (w * er) ** 2
            cells.append((r, w, er))
            # Spread cells across 80% of the 1D bin so the 3D occupancy is visible.
            u = (i_other - 0.5) / max(n_other, 1)
            xj = lo + 0.10 * (hi - lo) + 0.80 * u * (hi - lo)
            gscat.SetPoint(isp, xj, r)
            isp += 1
        if sw <= 0:
            continue
        rmean = sr / sw
        ermean = math.sqrt(srr) / sw
        rms = math.sqrt(max(sum(w * (r - rmean) ** 2 for r, w, _ in cells) / sw,
                            0.0))
        exl, exh = x - lo, hi - x
        gmean.SetPoint(ip, x, rmean)
        gmean.SetPointError(ip, exl, exh, ermean, ermean)
        genv.SetPoint(ip, x, rmean)
        genv.SetPointError(ip, exl, exh, rms, rms)
        ip += 1
    return gmean, gscat, genv


def ratio_th2_slice(n1, d1, n2, d2, iz, name, title):
    """TH2 of occupancy-weighted Zg/WW vs (|p|, |cos theta|).

    iz >= 1: that opang bin only.  iz == 0: average over all opang bins.
    Empty cells are left at zero so COLZ0 skips them.
    """
    nx, ny, nz = n1.GetNbinsX(), n1.GetNbinsY(), n1.GetNbinsZ()
    h = ROOT.TH2D(name, title, nx, P_BINS, ny, COS_BINS)
    h.SetDirectory(0)
    zrange = range(1, nz + 1) if iz == 0 else (iz,)
    for ix in range(1, nx + 1):
        for iy in range(1, ny + 1):
            sw = sr = 0.0
            for z in zrange:
                cell = _cell_ratio(n1.GetBinContent(ix, iy, z),
                                   d1.GetBinContent(ix, iy, z),
                                   n2.GetBinContent(ix, iy, z),
                                   d2.GetBinContent(ix, iy, z))
                if cell is None:
                    continue
                _, _, r, _, w = cell
                sw += w
                sr += w * r
            if sw <= 0:
                continue
            h.SetBinContent(ix, iy, sr / sw)
    h.SetMinimum(RATIO_YMIN)
    h.SetMaximum(RATIO_YMAX)
    h.GetZaxis().SetTitle("Z#gamma / WW")
    h.GetZaxis().SetTitleOffset(1.2)
    return h


def ratio_inclusive_3d(n1, d1, n2, d2):
    """Occupancy-weighted mean of eps1/eps2 over every occupied 3D cell."""
    sw = sr = 0.0
    n_used = 0
    for ix in range(1, n1.GetNbinsX() + 1):
        for iy in range(1, n1.GetNbinsY() + 1):
            for iz in range(1, n1.GetNbinsZ() + 1):
                cell = _cell_ratio(n1.GetBinContent(ix, iy, iz),
                                   d1.GetBinContent(ix, iy, iz),
                                   n2.GetBinContent(ix, iy, iz),
                                   d2.GetBinContent(ix, iy, iz))
                if cell is None:
                    continue
                _, _, r, _, w = cell
                sw += w
                sr += w * r
                n_used += 1
    if sw <= 0:
        return float("nan"), 0
    return sr / sw, n_used


def n_occupied_3d(den):
    n = 0
    for ix in range(1, den.GetNbinsX() + 1):
        for iy in range(1, den.GetNbinsY() + 1):
            for iz in range(1, den.GetNbinsZ() + 1):
                if den.GetBinContent(ix, iy, iz) > 0:
                    n += 1
    return n


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    frames, booked, reports, disagree = {}, {}, {}, {}

    for s in SAMPLES:
        df = build(s)
        frames[s["tag"]] = df
        booked[s["tag"]] = book(df, s["tag"])
        reports[s["tag"]] = df.Report()
        disagree[s["tag"]] = [df.Mean(f"j{i}_label_disagree") for i in (1, 2)]

    # Everything above is lazy; this is where the event loops run.
    results = {}
    for s in SAMPLES:
        tag = s["tag"]
        results[tag] = attach_projections(
            {k: merge(v) for k, v in booked[tag].items()}, tag)

    fout = ROOT.TFile(OUT_ROOT, "RECREATE")

    print("\n" + "=" * 78)
    print("cutflow and label consistency")
    print("=" * 78)
    for kind, (lo, hi) in Y23Q2.items():
        print(f"  y_23_Q2[{kind}]: {lo} <= y_23_Q2 <= {hi}")
    for s in SAMPLES:
        tag = s["tag"]
        print(f"\n--- {s['name']}")
        reports[tag].Print()
        frac = 0.5 * sum(m.GetValue() for m in disagree[tag])
        if s["kind"] == "zg":
            print(f"    truth label: true_flavour_pdg  "
                  f"(sample-level, as tp_counting)")
            print(f"    hf_flavour would have disagreed for {frac:.3%} of jets "
                  f"(not used)")
        else:
            print(f"    truth label: |jet*_getflavour_pdg|  "
                  f"(FCCAnalyses get_flavour)")
            print(f"    matched_pdg would have disagreed for {frac:.3%} of jets "
                  f"(not used)")

    print("\n" + "=" * 78)
    print("inclusive MC-true efficiencies  (nominal cascade, 3D grid)")
    print("  pass f  = cascade category f  |  den = true flavour f")
    print("  WW: |jet*_getflavour_pdg| (get_flavour)   "
          "Zg: true_flavour_pdg (as tp_counting)")
    print("  Zg/WW incl. = ratio of totals;  Zg/WW 3D = occupancy-weighted")
    print("  mean of per-cell ratios on the (p, |cos theta|, opang) grid")
    print("=" * 78)
    ww = next(s for s in SAMPLES if s["kind"] == "ww")
    zg = next(s for s in SAMPLES if s["kind"] == "zg")
    n_cells = ((len(P_BINS) - 1) * (len(COS_BINS) - 1) * (len(OPA_BINS) - 1))
    print(f"  3D grid: {len(P_BINS)-1} p x {len(COS_BINS)-1} |cos| x "
          f"{len(OPA_BINS)-1} opang = {n_cells} cells")
    header = (f"{'flavour':<10}"
              + "".join(f"{s['name']:>14}" for s in SAMPLES)
              + f"{'Zg/WW incl.':>14}{'Zg/WW 3D':>12}{'cells':>8}")
    print(header)
    for fl in FLAV_PDG:
        row = f"{fl:<10}"
        vals = {}
        for s in SAMPLES:
            h = results[s["tag"]][fl]
            n = h["num3d"].Integral()
            d = h["den3d"].Integral()
            vals[s["kind"]] = (n / d) if d else float("nan")
            row += f"{vals[s['kind']]:>13.4f} "
        ratio_incl = (vals["zg"] / vals["ww"]
                      if vals.get("ww") else float("nan"))
        hw = results[ww["tag"]][fl]
        hz = results[zg["tag"]][fl]
        ratio_3d, n_used = ratio_inclusive_3d(
            hz["num3d"], hz["den3d"], hw["num3d"], hw["den3d"])
        row += f"{ratio_incl:>13.4f} {ratio_3d:>11.4f} {n_used:>7d} "
        print(row)
        print(f"{'':10}  occupied den cells:  WW "
              f"{n_occupied_3d(hw['den3d'])}/{n_cells}  Zg "
              f"{n_occupied_3d(hz['den3d'])}/{n_cells}")

    # ── plots ────────────────────────────────────────────────────────────────
    def _ratio_frame(href, name, title):
        fr = href.Clone(name)
        fr.SetDirectory(0)
        fr.Reset()
        fr.SetTitle(title)
        fr.SetMinimum(RATIO_YMIN)
        fr.SetMaximum(RATIO_YMAX)
        fr.SetStats(0)
        return fr

    def _draw_ratio_1d(pad, href, fl, var, xlab, hw, hz, keep):
        """Grey 3D-cell swarm + RMS band + occupancy-weighted mean."""
        pad.SetGridy()
        pad.SetLeftMargin(0.14)
        fr = _ratio_frame(
            href, f"frame_{fl}_{var}",
            f"#varepsilon_{{{fl}}}^{{Z#gamma}} / #varepsilon_{{{fl}}}^{{WW}}"
            f";{xlab};ratio")
        fr.Draw("AXIS")
        line = ROOT.TLine(fr.GetXaxis().GetXmin(), 1.0,
                          fr.GetXaxis().GetXmax(), 1.0)
        line.SetLineStyle(2)
        line.SetLineColor(ROOT.kGray + 2)
        line.Draw("same")
        gmean, gscat, genv = ratio_1d_bundle(
            hz["num3d"], hz["den3d"], hw["num3d"], hw["den3d"],
            var, f"ratio_{zg['tag']}_over_{ww['tag']}_{fl}_{var}")
        st = FLAV_STYLE[fl]
        genv.SetFillColor(ROOT.TColor.GetColorTransparent(st["colour"], 0.25))
        genv.SetFillStyle(1001)
        genv.SetLineColor(st["colour"])
        genv.SetMarkerSize(0)
        genv.Draw("3 same")
        gscat.SetMarkerStyle(24)
        gscat.SetMarkerSize(0.45)
        gscat.SetMarkerColor(ROOT.kGray + 1)
        gscat.SetLineColor(0)
        if gscat.GetN() > 0:
            gscat.Draw("P same")
        gmean.SetLineColor(st["colour"])
        gmean.SetMarkerColor(st["colour"])
        gmean.SetMarkerStyle(st["marker"])
        gmean.SetMarkerSize(1.1)
        gmean.Draw("P same")
        keep.extend([fr, line, genv, gscat, gmean])
        fout.cd()
        gmean.Write()
        gscat.Write()
        genv.Write()
        return gmean, gscat, genv

    ROOT.gStyle.SetPalette(ROOT.kBird)
    ROOT.gStyle.SetNumberContours(20)

    c = ROOT.TCanvas("c", "c", 1500, 900)
    c.Print(OUT_PDF + "[")

    for fl in FLAV_PDG:
        hw = results[ww["tag"]][fl]
        hz = results[zg["tag"]][fl]
        fout.cd()
        hw["num3d"].Write()
        hw["den3d"].Write()
        hz["num3d"].Write()
        hz["den3d"].Write()

        # Page 1: 1D projected efficiency (top) and 1D ratio of 3D cells (bottom)
        c.Clear()
        c.Divide(3, 2)
        keep = []
        for ipad, (var, xlab) in enumerate(AXES_1D, start=1):
            pad = c.cd(ipad)
            pad.SetGridy()
            pad.SetLeftMargin(0.14)
            leg = ROOT.TLegend(0.18, 0.18, 0.55, 0.40)
            leg.SetBorderSize(0)
            leg.SetFillStyle(0)
            first = True
            for s in SAMPLES:
                h = results[s["tag"]][fl]
                te, _ = efficiency(h[f"num_{var}"], h[f"den_{var}"],
                                   f"eff_{s['tag']}_{fl}_{var}")
                te.SetLineColor(s["colour"])
                te.SetMarkerColor(s["colour"])
                te.SetMarkerStyle(s["marker"])
                te.SetTitle(f"#varepsilon_{{{fl}}};{xlab};efficiency")
                te.Draw("" if first else "same")
                ROOT.gPad.Update()
                g = te.GetPaintedGraph()
                if g:
                    g.SetMinimum(0.0)
                    g.SetMaximum(1.05)
                leg.AddEntry(te, s["name"], "lp")
                keep.append(te)
                first = False
            leg.Draw()
            keep.append(leg)
            fout.cd()
            for s in SAMPLES:
                h = results[s["tag"]][fl]
                h[f"num_{var}"].Write()
                h[f"den_{var}"].Write()

            pad = c.cd(ipad + 3)
            gmean, gscat, genv = _draw_ratio_1d(
                pad, hz[f"den_{var}"], fl, var, xlab, hw, hz, keep)
            leg_r = ROOT.TLegend(0.18, 0.18, 0.62, 0.40)
            leg_r.SetBorderSize(0)
            leg_r.SetFillStyle(0)
            if gscat.GetN() > 0:
                leg_r.AddEntry(gscat, "3D cells", "p")
            leg_r.AddEntry(genv, "cell RMS", "f")
            leg_r.AddEntry(gmean, "occupancy-weighted mean", "lp")
            leg_r.Draw()
            keep.append(leg_r)
        c.Print(OUT_PDF)

        # Page 2: (p, |cos theta|) maps, one pad per opang bin + inclusive
        c.Clear()
        c.Divide(3, 2)
        keep2 = []
        nopa = len(OPA_BINS) - 1
        slices = [(iz, f"{OPA_BINS[iz-1]:.1f} #leq #Delta#theta < "
                   f"{OPA_BINS[iz]:.1f} rad")
                  for iz in range(1, nopa + 1)]
        slices.append((0, "all opang (occupancy-weighted)"))
        for ipad, (iz, slab) in enumerate(slices, start=1):
            pad = c.cd(ipad)
            pad.SetRightMargin(0.16)
            pad.SetLeftMargin(0.12)
            h2 = ratio_th2_slice(
                hz["num3d"], hz["den3d"], hw["num3d"], hw["den3d"],
                iz,
                f"ratio2d_{fl}_opa{iz}",
                f"#varepsilon_{{{fl}}}^{{Z#gamma}} / "
                f"#varepsilon_{{{fl}}}^{{WW}}   {slab}"
                f";|p| [GeV];|cos#theta|")
            h2.SetStats(0)
            h2.Draw("COLZ0")
            keep2.append(h2)
            fout.cd()
            h2.Write()
        c.Print(OUT_PDF)

    # Summary page: Zg/WW means for b, c, s on each kinematic axis
    c.Clear()
    c.Divide(3, 1)
    keep_sum = []
    for ipad, (var, xlab) in enumerate(AXES_1D, start=1):
        pad = c.cd(ipad)
        pad.SetGridy()
        pad.SetLeftMargin(0.16)
        pad.SetBottomMargin(0.14)
        fr = _ratio_frame(
            results[ww["tag"]]["b"][f"den_{var}"],
            f"frame_all_{var}",
            f"#varepsilon_{{Z#gamma}} / #varepsilon_{{WW}}"
            f";{xlab};ratio")
        fr.Draw("AXIS")
        line = ROOT.TLine(fr.GetXaxis().GetXmin(), 1.0,
                          fr.GetXaxis().GetXmax(), 1.0)
        line.SetLineStyle(2)
        line.SetLineColor(ROOT.kGray + 2)
        line.Draw("same")
        keep_sum.extend([fr, line])
        leg = ROOT.TLegend(0.20, 0.18, 0.55, 0.48)
        leg.SetBorderSize(0)
        leg.SetFillStyle(0)
        for fl in FLAV_PDG:
            hw = results[ww["tag"]][fl]
            hz = results[zg["tag"]][fl]
            gmean, gscat, genv = ratio_1d_bundle(
                hz["num3d"], hz["den3d"], hw["num3d"], hw["den3d"],
                var, f"ratio_all_{fl}_{var}")
            st = FLAV_STYLE[fl]
            genv.SetFillColor(ROOT.TColor.GetColorTransparent(st["colour"], 0.20))
            genv.SetFillStyle(1001)
            genv.SetLineColor(st["colour"])
            genv.SetMarkerSize(0)
            genv.Draw("3 same")
            gmean.SetLineColor(st["colour"])
            gmean.SetMarkerColor(st["colour"])
            gmean.SetMarkerStyle(st["marker"])
            gmean.Draw("P same")
            keep_sum.extend([genv, gmean, gscat])
            leg.AddEntry(gmean, fl, "lp")
            fout.cd()
            gmean.Write()
        leg.Draw()
        keep_sum.append(leg)

    c.Print(OUT_PDF)
    c.Print(OUT_PDF + "]")
    fout.Close()
    print(f"\nwrote {OUT_ROOT} and {OUT_PDF}")


if __name__ == "__main__":
    main()