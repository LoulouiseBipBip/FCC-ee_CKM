"""
mixture_tp.py

Mixture tag & probe with FOUR disjoint tag regions, closure against the true
per-jet efficiency.


THE REGIONS
-----------
The tag jet is put through the another cascade as the probe, at the tight
working points:

    tag b  : passes tight b
    tag c  : fails b, passes tight c
    tag s  : fails b and c, passes tight s
    tag q  : fails all three          

Disjoint and exhaustive, so every jet pair lands in exactly one region and
the region totals add up to the untagged total. 

THE MODEL
---------
    mu[k,t,Y] = sum_f  N[k,f] * eps[f,t] * resp[k,f,Y]

  N[k,f]       weighted probe jets of true flavour f in cell k. external, from lumi x cs 
               or MC sample size
  eps[f,t]     probability a flavour-f jet lands in tag region t.
               sum_t eps[f,t] = 1 by construction.  FITTED.
  resp[k,f,Y]  probability a flavour-f probe lands in cascade category Y.
               sum_Y resp[k,f,Y] = 1.  THE MEASUREMENT.

  1  fit eps[f,t] to the region totals            (12 parameters)
  2  invert  resp[k,:,Y] = A[k]^-1 data[k,:,Y]    (4x4, exact)
          with A[k,t,f] = N[k,f] eps[f,t]


CLOSURE
-------
Against resp_true[k,f,Y], the response with no tag requirement

    bias = fitted - truth      
    C    = fitted / truth        


Usage:
    python tp_counting.py --binning p,abscos
    python tp_counting.py --binning inclusive --unweighted
    python tp_counting.py --binning p,abscos --y23-max 0.01
    python tp_counting.py --binning y23
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import uproot
import awkward as ak
from iminuit import Minuit


# ═════════════════════════════════════════════════════════════════════════════
# CONFIG 
# ═════════════════════════════════════════════════════════════════════════════

TREE = "events"

INPUT_BASE = Path("/eos/user/l/lberiet/CKM_Flav/Zg_163_mcpv_incl_R08_getflav/")
CHUNK_GLOB = "chunk*.root"

# True-flavour MC. Light quarks (q) are uu and dd combined.
SAMPLES = {
    "b":  [INPUT_BASE / "wzp6_ee_bb_pqqmin40_ecm163" / CHUNK_GLOB],
    "c":  [INPUT_BASE / "wzp6_ee_cc_pqqmin40_ecm163" / CHUNK_GLOB],
    "s":  [INPUT_BASE / "wzp6_ee_ss_pqqmin40_ecm163" / CHUNK_GLOB],
    "ud": [INPUT_BASE / "wzp6_ee_uu_pqqmin40_ecm163" / CHUNK_GLOB,
           INPUT_BASE / "wzp6_ee_dd_pqqmin40_ecm163" / CHUNK_GLOB],
}


def set_input_base(path):
    """Repoint SAMPLES at another ntuple directory."""
    global INPUT_BASE, SAMPLES
    INPUT_BASE = Path(path)
    SAMPLES = {
        "b":  [INPUT_BASE / "wzp6_ee_bb_pqqmin40_ecm163" / CHUNK_GLOB],
        "c":  [INPUT_BASE / "wzp6_ee_cc_pqqmin40_ecm163" / CHUNK_GLOB],
        "s":  [INPUT_BASE / "wzp6_ee_ss_pqqmin40_ecm163" / CHUNK_GLOB],
        "ud": [INPUT_BASE / "wzp6_ee_uu_pqqmin40_ecm163" / CHUNK_GLOB,
               INPUT_BASE / "wzp6_ee_dd_pqqmin40_ecm163" / CHUNK_GLOB],
    }

# The 2D plane for each flavour: (x_branch, y_branch). Both are scores in
# (0,1) 
CASCADE_DISC = {
    "b": ("DB_vsC", "DB_vsS"),
    "c": ("DC_vsB", "DC_vsS"),
    "s": ("DS_vsC", "DS_vsQ"),
}

SCORE_EPS = 1e-4          # scores are clipped to [EPS, 1-EPS] before logit

# ── Z selection ──────────────────────────────────────────────────────────────
# "recoil"  |m_recoil - MZ| < RECOIL_HALFWIDTH       
# "dijet"   MASS_MIN < reco_dijet_m < MASS_MAX        
# "none"    no Z selection at all
# careful here that the recoil selection leave only ~30% of the events
Z_SELECTION = "none"
MZ = 91.1876
RECOIL_HALFWIDTH = 20.0        # GeV, half-width on |m_recoil - MZ|

# Mass dijet window 
MASS_MIN, MASS_MAX = 80.0, 120.0

# ── fiducial jet kinematics ──────────────────────────────────────────────────

JET_P_MIN, JET_P_MAX         = None, None    # GeV
JET_ABSCOS_MIN, JET_ABSCOS_MAX = None, 0.99  # |cos theta| of each jet
DR_MIN, DR_MAX               = None, None    # reco_dR(j1, j2)
DIJET_P_MIN, DIJET_P_MAX     = None, None   # reco_dijet_p [GeV]


def _lg(p):
    """score -> logit."""
    p = min(max(float(p), SCORE_EPS), 1.0 - SCORE_EPS)
    return float(np.log(p / (1.0 - p)))


# Probe working points:
NOMINAL_WP_LOGIT = {
    "b": {"x": _lg(0.99717), "y": _lg(0.99564)},
    "c": {"x": _lg(0.00006), "y": _lg(0.53319)},
    "s": {"x": _lg(0.46681), "y": _lg(0.20117)},
}


# Tag working points:
TIGHT_WP_LOGIT = {
    "b": {"x": _lg(0.99717), "y": _lg(0.99564)},
    "c": {"x": _lg(0.7), "y": _lg(0.7)},
    "s": {"x": _lg(0.8), "y": _lg(0.9)},
}



FLAVOURS = ["b", "c", "s", "ud"]     # true flavour of the MC sample
CATS     = ["b", "c", "s", "q"]      # probe cascade categories
REGIONS  = ["b", "c", "s", "q"]      # tag regions, same cascade, tight WPs
NF, NC, NR = len(FLAVOURS), len(CATS), len(REGIONS)
FI = {f: i for i, f in enumerate(FLAVOURS)}
CI = {c: i for i, c in enumerate(CATS)}
DIAG = {"b": "b", "c": "c", "s": "s", "ud": "q"}   # "correct" probe category

# 240 GeV Z(gamma)->qq.  163 GeV: b=20.32 c=17.94 s=20.97 ud=39.62
# 240 GeV XSEC_PB = {"b": 7.48, "c": 6.78, "s": 7.73, "ud": 15.08}
XSEC_PB = {"b": 20.32, "c": 17.94, "s": 20.97, "ud": 39.62}
LUMI_PB = 1.08e7

# Uncertainty on the number of jets in the sample, so cross section (for now I keep 0 because only look at MC)
SIG_NJET = 0.0

KIN_BRANCHES = ["reco_jet1_p", "reco_jet2_p",
                "reco_jet1_theta", "reco_jet2_theta",
                "reco_opang", "reco_dijet_m",
                "m_recoil"]

OPT_BRANCHES = ["y_23_Q2", "y_23", "d_12", "d_23",
                "jet_esum", "jet1_e", "jet2_e",
                "reco_dR", "reco_dijet_p"]

EDGES_FINE = {
    "p":        np.array([12.5, 15, 17.5, 20, 22.5, 25, 27.5, 30,
                          35, 40, 45, 50, 55, 60,65, 70,75, 80, 85]),
    "theta":    np.array([0.15, 0.3, 0.5, 0.7, 1.1, 1.6, 2.1, 2.4, 2.75, 2.99]),
    "abscos":   np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                          0.75, 0.8, 0.85, 0.9, 0.94, 1.0]),
    "costheta": np.array([-1.0, -0.9, -0.75, -0.6, -0.4, -0.2, 0.0,
                          0.2, 0.4, 0.6, 0.75, 0.9, 1.0]),
    "opang":    np.array([0.15, 0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 2.4, 2.75, 2.99]),
    "y23":      np.array([0.0, 0.002, 0.004, 0.007, 0.01, 0.015,
                          0.02, 0.03, 0.05, 0.08, 0.15]),
}
EDGES_COARSE = {
    "p":        np.array([12.5, 21.5, 24, 26, 28, 30.5, 33.5, 37, 69.5, 78]),
    "theta":    np.array([0.15, 0.4, 0.7, 1.1, 1.6, 2.0, 2.4, 2.99]),
    "abscos":   np.array([0, 0.74, 0.88, 0.92, 0.96, 0.99]),
    "costheta": np.array([-1.0, -0.87, -0.6, -0.3, 0.0, 0.3, 0.6, 0.87, 1.0]),
    "opang":    np.array([2.0, 2.2, 2.4, 2.6, 2.8, 3.0]),
    "y23":      np.array([0.0, 0.005, 0.01, 0.02, 0.05, 0.15]),
}
# which fixed kinematic cut an axis replaces 
AXIS_CUT = {"p": "p", "theta": "theta", "abscos": "theta",
            "costheta": "theta", "opang": "opang", "y23": "y23"}

OUTDIR = Path("/eos/user/l/lberiet/Tag_and_Probe/Mix_test_fit")

# event-level merging-scale window, set from the command line
Y23_MIN, Y23_MAX = None, None
SIG_THETA = 1.0 # default value for the closure test
TOL_COEF  = 1e-6 # default value for the closure test

# ── counting ─────────────────────────────────────────────────────────────────

def parse_binning(s):
    if s is None or s.strip().lower() in ("", "inclusive", "none"):
        return []
    axes = [v.strip() for v in s.replace(":", ",").split(",") if v.strip()]
    bad = [v for v in axes if v not in EDGES_FINE]
    if bad:
        raise ValueError(f"unknown axis {bad}; choose from {list(EDGES_FINE)}")
    cuts = [AXIS_CUT[v] for v in axes]
    if len(set(cuts)) != len(cuts):
        raise ValueError(f"{axes} binds two axes to the same quantity {cuts}")
    return axes


def select_edges(axis_vars, choice="auto"):
    if choice == "auto":
        choice = "coarse" if len(axis_vars) > 1 else "fine"
    table = {"fine": EDGES_FINE, "coarse": EDGES_COARSE}[choice]
    return [(v, np.asarray(table[v], float)) for v in axis_vars], choice


def expand_sample_globs(paths):
    """Resolve each glob to actual chunk*.root files."""
    files = []
    for p in paths:
        matched = sorted(glob.glob(str(p)))
        if not matched:
            raise FileNotFoundError(f"No files matching {p}")
        print(f"  {p}: {len(matched)} file(s)")
        files.extend(matched)
    return files


def readable_root_files(files, tree):
    """Drop truncated / corrupt ROOT files before concatenate."""
    good = []
    for path in files:
        try:
            with uproot.open(path) as f:
                if tree not in f:
                    raise KeyError(f"no TTree '{tree}'")
                if f[tree].num_entries <= 0:
                    raise ValueError("empty tree")
        except Exception as exc:
            print(f"  [skip] {path}")
            print(f"         {type(exc).__name__}: {exc}")
            continue
        good.append(path)
    if not good:
        raise FileNotFoundError(
            f"no readable ROOT files among {len(files)} matched")
    if len(good) < len(files):
        print(f"  using {len(good)}/{len(files)} readable file(s)")
    return good


def load_sample(paths):
    pairwise = [b for pair in CASCADE_DISC.values() for b in pair]
    files = readable_root_files(expand_sample_globs(paths), TREE)
    tagged = [f"{f}:{TREE}" for f in files]
    arrs = uproot.concatenate(
        tagged, filter_name=pairwise + KIN_BRANCHES + OPT_BRANCHES, library="ak")
    return arrs[ak.num(arrs[pairwise[0]], axis=1) == 2]


def _logit(arrs, branch):
    v = np.clip(ak.to_numpy(arrs[branch]), SCORE_EPS, 1.0 - SCORE_EPS)
    return np.log(v / (1.0 - v))


def _flat(arrs, branch):
    return np.asarray(ak.to_numpy(arrs[branch]), dtype=float)


def cascade_codes(d, jet, wp):
    """
    The cascade, as a disjoint and exhaustive 0..3 code. 
    """
    b = (d["bx"][:, jet] > wp["b"]["x"]) & (d["by"][:, jet] > wp["b"]["y"])
    c = ~b & (d["cx"][:, jet] > wp["c"]["x"]) & (d["cy"][:, jet] > wp["c"]["y"])
    s = ~b & ~c & (d["sx"][:, jet] > wp["s"]["x"]) & (d["sy"][:, jet] > wp["s"]["y"])
    return np.select([b, c, s], [0, 1, 2], default=3)


def get_y23(arrs):
    """y_23_Q2 = d_23 / (E1+E2)^2."""
    fields = set(arrs.fields)
    if "y_23_Q2" in fields:
        return _flat(arrs, "y_23_Q2")
    if "d_23" not in fields:
        return None
    d23 = _flat(arrs, "d_23")
    if "jet_esum" in fields:
        q = _flat(arrs, "jet_esum")
    elif {"jet1_e", "jet2_e"} <= fields:
        q = _flat(arrs, "jet1_e") + _flat(arrs, "jet2_e")
    else:
        return None
    q2 = q * q
    return np.where(q > 0, d23 / np.where(q > 0, q2, 1.0), -1.0)


def z_selection_mask(arrs):
    """Event-level Z selection."""
    n = len(arrs[KIN_BRANCHES[0]])
    if Z_SELECTION == "none":
        return np.ones(n, dtype=bool)

    if Z_SELECTION == "recoil":
        if "m_recoil" not in set(arrs.fields):
            raise RuntimeError(
                "Z_SELECTION='recoil' needs the m_recoil branch, which is not "
                "in the tree. Rerun the treemaker.")
        mr = _flat(arrs, "m_recoil")
        return (mr > 0) & (np.abs(mr - MZ) < RECOIL_HALFWIDTH)

    if Z_SELECTION == "dijet":
        m = _flat(arrs, "reco_dijet_m")
        sel = np.ones(n, dtype=bool)
        if MASS_MIN is not None:
            sel &= m >= MASS_MIN
        if MASS_MAX is not None:
            sel &= m <= MASS_MAX
        return sel

    raise ValueError(f"Z_SELECTION={Z_SELECTION!r} must be "
                     "'recoil', 'dijet' or 'none'")


def _window(v, lo, hi):
    ok = np.ones(v.shape, dtype=bool)
    if lo is not None:
        ok &= v >= lo
    if hi is not None:
        ok &= v <= hi
    return ok


def _fmt_window(lo, hi, unit=""):
    if lo is None and hi is None:
        return "none"
    left = f">= {lo}{unit}" if lo is not None else ""
    right = f"<= {hi}{unit}" if hi is not None else ""
    return " and ".join(x for x in (left, right) if x)


def jet_fiducial_active():
    return any(x is not None for x in (
        JET_P_MIN, JET_P_MAX, JET_ABSCOS_MIN, JET_ABSCOS_MAX,
        DR_MIN, DR_MAX, DIJET_P_MIN, DIJET_P_MAX))


def jet_fiducial_mask(arrs, jet_p, jet_theta):
    """Both jets in the fiducial |p| / |cos theta| window, plus event dR and dijet momentum."""
    sel = np.ones(jet_p.shape[0], dtype=bool)
    sel &= _window(jet_p[:, 0], JET_P_MIN, JET_P_MAX)
    sel &= _window(jet_p[:, 1], JET_P_MIN, JET_P_MAX)
    abscos = np.abs(np.cos(jet_theta))
    sel &= _window(abscos[:, 0], JET_ABSCOS_MIN, JET_ABSCOS_MAX)
    sel &= _window(abscos[:, 1], JET_ABSCOS_MIN, JET_ABSCOS_MAX)
    if DR_MIN is not None or DR_MAX is not None:
        if "reco_dR" not in set(arrs.fields):
            raise RuntimeError(
                "DR_MIN/DR_MAX is set but reco_dR is not in the tree. ")
        sel &= _window(_flat(arrs, "reco_dR"), DR_MIN, DR_MAX)
    if DIJET_P_MIN is not None or DIJET_P_MAX is not None:
        if "reco_dijet_p" not in set(arrs.fields):
            raise RuntimeError(
                "DIJET_P_MIN/DIJET_P_MAX is set but reco_dijet_p is not in "
                "the tree. Rerun the treemaker that writes reco_dijet_p.")
        sel &= _window(_flat(arrs, "reco_dijet_p"), DIJET_P_MIN, DIJET_P_MAX)
    return sel


def axis_value(var, jet_p, jet_theta, opang, y23, role):
    if var == "p":
        return jet_p[:, role]
    if var == "theta":
        return jet_theta[:, role]
    if var == "costheta":
        return np.cos(jet_theta[:, role])
    if var == "abscos":
        return np.abs(np.cos(jet_theta[:, role]))
    if var == "opang":
        return opang
    if var == "y23":
        return y23
    raise ValueError(var)


def count_regions(arrs, axes):
    """
    counts[k, t, Y] for ONE sample: probe in cell k, tag jet in region t,
    probe in category Y.
    """
    d = {}
    for key, (xb, yb) in CASCADE_DISC.items():
        d[key + "x"] = _logit(arrs, xb)
        d[key + "y"] = _logit(arrs, yb)

    jet_p = np.stack([_flat(arrs, "reco_jet1_p"), _flat(arrs, "reco_jet2_p")], 1)
    jet_theta = np.stack([_flat(arrs, "reco_jet1_theta"),
                          _flat(arrs, "reco_jet2_theta")], 1)
    opang = _flat(arrs, "reco_opang")
    y23 = get_y23(arrs)
    z_sel = z_selection_mask(arrs)
    jet_sel = jet_fiducial_mask(arrs, jet_p, jet_theta)
    pre = z_sel & jet_sel

    shape = tuple(len(e) - 1 for _, e in axes)
    n_cells = int(np.prod(shape)) if shape else 1
    binned = {AXIS_CUT[v] for v, _ in axes}

    counts = np.zeros((n_cells, NR, NC))

    for r in (0,1):
        role = 1 - r
        sel = pre.copy()
        if "y23" not in binned and (Y23_MIN is not None or Y23_MAX is not None):
            if y23 is None:
                raise RuntimeError(
                    "a y23 cut was requested but neither y_23_Q2 nor d_23 and "
                    "the jet energies are in the tree; rerun the treemaker "
                    "that writes y_23_Q2 (zg_common.define_jets_and_scales)")
            sel &= y23 >= 0.0
            if Y23_MIN is not None:
                sel &= y23 >= Y23_MIN
            if Y23_MAX is not None:
                sel &= y23 <= Y23_MAX

        if axes:
            per_axis = []
            for var, edges in axes:
                val = axis_value(var, jet_p, jet_theta, opang, y23, role)
                idx = np.digitize(val, edges) - 1
                sel &= (idx >= 0) & (idx < len(edges) - 1)
                per_axis.append(np.clip(idx, 0, len(edges) - 2))
            flat_idx = np.ravel_multi_index(tuple(per_axis), shape)
        else:
            flat_idx = np.zeros(sel.shape, dtype=np.int64)

        if not sel.any():
            continue
        tag_code = cascade_codes(d, r, TIGHT_WP_LOGIT)
        probe_code = cascade_codes(d, role, NOMINAL_WP_LOGIT)
        code = (flat_idx[sel] * NR + tag_code[sel]) * NC + probe_code[sel]
        counts += np.bincount(
            code, minlength=n_cells * NR * NC).reshape(n_cells, NR, NC)

    return counts, z_sel, jet_sel


def load_all(binning, edges_choice, lumi, unweighted):
    axis_vars = parse_binning(binning)
    axes, edges_used = select_edges(axis_vars, edges_choice)
    shape = tuple(len(e) - 1 for _, e in axes)
    n_cells = int(np.prod(shape)) if shape else 1
    key = "_".join(axis_vars) if axis_vars else "inclusive"

    print(f"binning : {key}"
          + (f" ({edges_used} edges, {' x '.join(map(str, shape))}"
             f" = {n_cells} cells)" if axis_vars else " (1 inclusive cell)"))
    if Y23_MIN is not None or Y23_MAX is not None:
        print(f"y23 window : {Y23_MIN} <= y_23_Q2 <= {Y23_MAX}  (event level, "
              "so it applies to the tag and probe side identically)")
    if Z_SELECTION == "recoil":
        print(f"Z selection : |m_recoil - {MZ:g}| < {RECOIL_HALFWIDTH:g} GeV "
              "(photon-only, no jet information)")
    elif Z_SELECTION == "dijet":
        print(f"Z selection : {MASS_MIN} < reco_dijet_m < {MASS_MAX} GeV "
              "[!] couples the two jets -- see the module docstring")
    else:
        print("Z selection : none")
    print("jet fiducial : both jets  "
          f"|p| {_fmt_window(JET_P_MIN, JET_P_MAX, ' GeV')}  "
          f"|cos theta| {_fmt_window(JET_ABSCOS_MIN, JET_ABSCOS_MAX)}  "
          f"dR {_fmt_window(DR_MIN, DR_MAX)}  "
          f"reco_dijet_p {_fmt_window(DIJET_P_MIN, DIJET_P_MAX, ' GeV')}"
          "  (always, independent of binning)")

    raw = np.zeros((n_cells, NR, NC, NF))
    scales, n_gen, z_eff, kin_eff = {}, {}, {}, {}
    for f in FLAVOURS:
        print(f"\n[load] true flavour = {f}")
        arrs = load_sample(SAMPLES[f])
        n_gen[f] = int(len(arrs))
        scales[f] = 1.0 if (unweighted or not n_gen[f]) \
            else XSEC_PB[f] * lumi / n_gen[f]
        print(f"  N_gen={n_gen[f]}  scale={scales[f]:.6g}")
        cnt, zsel, jetsel = count_regions(arrs, axes)
        raw[..., FI[f]] = cnt
        z_eff[f] = float(zsel.mean())
        kin_eff[f] = float(jetsel.mean())
        print(f"  Z selection efficiency = {z_eff[f]:.4f}")
        print(f"  jet fiducial efficiency = {kin_eff[f]:.4f}  "
              f"(both = {float((zsel & jetsel).mean()):.4f})")

    # n_gen is the PRE-selection count and the scale is sigma/n_gen, so
    # N[k,f] = sigma_f * lumi * (n_passing / n_gen_f) already carries the
    # acceptance. The efficiency below is therefore NOT a correction to apply
    # -- it is a diagnostic of how flavour-uniform the selection is, which is
    # the whole reason for preferring the recoil.
    lo, hi = min(z_eff.values()), max(z_eff.values())
    print(f"\nZ selection efficiency: {lo:.4f} .. {hi:.4f}  "
          f"(spread {100 * (hi / lo - 1):.1f}%)")
    if hi / lo - 1 > 0.10:
        print("  [!] >10% spread. The selection keeps a different slice of "
              "each flavour,")
        print("      so the calibrated efficiencies apply to that slice and "
              "not to an")
        print("      inclusive jet -- a transferability problem when these "
              "are carried")
        print("      over to WW.")

    return {"raw": raw, "scales": scales, "n_gen": n_gen, "axes": axes,
            "axis_vars": axis_vars, "edges_used": edges_used,
            "n_cells": n_cells, "binning_key": key,
            "unweighted": unweighted, "z_eff": z_eff, "kin_eff": kin_eff}


# ── model ────────────────────────────────────────────────────────────────────

def build(c):
    """
    Stack the samples and derive everything the two steps need.

      data[k,t,Y]     weighted counts -- what data would look like
      var[k,t,Y]      its variance, sum_f s_f^2 n_f. NOT data: a weighted
                      sum is not Poisson, and using data as its own variance
                      underestimates the error by ~sqrt(s)
      N[k,f]          weighted probe jets per flavour -- the external input
      eps_mc[k,f,t]   MC tag efficiency PER CELL. The tag jet's kinematics
                      correlate with the probe cell (back-to-back, E1+E2
                      pinned), so this varies strongly with k and a global
                      value cannot describe the region totals.
      eps_incl[f,t]   the same, integrated over cells -- reporting only
      resp_true       true per-jet response: summed over ALL regions, so no
                      tag requirement anywhere. The closure reference.
    """
    raw, scales = c["raw"], c["scales"]
    s = np.array([scales[f] for f in FLAVOURS])

    data = np.einsum("f,ktyf->kty", s, raw)
    var = np.einsum("f,ktyf->kty", s ** 2, raw)

    per_flav = raw.sum(axis=2)                     # (k, t, f)
    N = np.einsum("f,ktf->kf", s, per_flav)        # weighted jets per flavour

    # per-cell tag efficiency, (k, f, t)
    pf = np.transpose(per_flav, (0, 2, 1))         # (k, f, t)
    tot_kf = pf.sum(axis=2, keepdims=True)         # (k, f, 1)
    eps_mc = np.divide(pf, tot_kf, out=np.zeros_like(pf), where=tot_kf > 0)

    # inclusive shape, for reporting and as the fallback in empty cells
    incl = pf.sum(axis=0)                          # (f, t)
    incl_tot = incl.sum(axis=1, keepdims=True)
    eps_incl = np.divide(incl, incl_tot, out=np.zeros_like(incl),
                         where=incl_tot > 0)
    for k, f in zip(*np.nonzero(tot_kf[:, :, 0] <= 0)):
        eps_mc[k, f] = eps_incl[f]

    untag = raw.sum(axis=1)                        # (k, Y, f), all regions
    tot_ky = untag.sum(axis=1, keepdims=True)
    resp_true = np.divide(untag, tot_ky, out=np.zeros_like(untag),
                          where=tot_ky > 0)
    resp_true = np.transpose(resp_true, (0, 2, 1))  # -> (k, f, Y)

    # Region totals and their effective counts. A weighted sum has variance
    # sum s^2 n, so rescale by alpha = S1/S2 to make a Poisson term valid:
    # fit Poisson(alpha*mu) against alpha*S1. Exactly 1 when unweighted.
    T1 = data.sum(axis=2)
    T2 = var.sum(axis=2)
    alpha = np.where(T2 > 0, T1 / np.where(T2 > 0, T2, 1.0), 1.0)

    return {"data": data, "var": var, "N": N, "eps_mc": eps_mc,
            "eps_incl": eps_incl, "resp_true": resp_true, "T1": T1,
            "alpha": alpha, "scales": s, "raw": raw}


# ── step 1 : fit eps to the region totals ────────────────────────────────────

def apply_theta(eps_mc, theta):
    """
    eps[k,f,t] proportional to eps_mc[k,f,t] * exp(theta[f,t]), renormalised
    over t so the rows still sum to 1.

    theta has NR-1 columns: the last region is the reference, because a
    common shift across t cancels in the renormalisation and would otherwise
    leave four exactly flat directions in the fit.
    """
    th = np.concatenate([np.asarray(theta, float),
                         np.zeros((NF, 1))], axis=1)        # (NF, NR)
    w = eps_mc * np.exp(th)[None, :, :]                     # (k, NF, NR)
    return w / np.clip(w.sum(axis=2, keepdims=True), 1e-300, None)


def fit_eps(m, verbose=True):
    """
    Poisson fit of the region totals for the data/MC tag scale factors.
    12 free parameters whatever the number of cells, because the response
    has been profiled out exactly.

    On stacked MC the minimum is at theta = 0 by construction, so this is
    also the closure test of the composition model: if the fit does not land
    there, N and the counts disagree.
    """
  
    N, T1, alpha, eps_mc = m["N"], m["T1"], m["alpha"], m["eps_mc"]
    n_eff = alpha * T1
    names = [f"th_{f}_{r}" for f in FLAVOURS for r in REGIONS[:-1]]

    def _mu(theta):
        eps = apply_theta(eps_mc, theta)
        return np.clip(alpha * np.einsum("kf,kft->kt", N, eps), 1e-12, None), eps

    def nll(p):
        p = np.asarray(p)
        mu, _ = _mu(p.reshape(NF, NR - 1))
        return float(np.sum(mu - n_eff * np.log(mu))
                    + 0.5 * np.sum(np.square(p / SIG_THETA)))

    def grad(p):
        p = np.asarray(p)
        theta = p.reshape(NF, NR - 1)
        mu, eps = _mu(theta)
        cc = alpha * (1.0 - n_eff / mu)
        term1 = np.einsum("kt,kf,kft->ft", cc, N, eps)
        S = np.einsum("kt,kf,kft->kf", cc, N, eps)
        term2 = np.einsum("kf,kft->ft", S, eps)
        return (term1 - term2)[:, :-1].ravel() + p / SIG_THETA**2

    start = np.zeros(NF * (NR - 1))
    mi = Minuit(nll, start, grad=grad, name=names)
    mi.errordef = Minuit.LIKELIHOOD
    # MIGRAD's EDM goal is ABSOLUTE (edm_goal = 0.002 * tol * errordef) while
    # this NLL is O(1e7), so the default tol demands ~1e-11 relative
    # precision and is unreachable. Scale the goal to the problem instead.
    mi.tol = TOL_COEF 
    mi.migrad()
    mi.hesse()

    theta = np.array(mi.values).reshape(NF, NR - 1)
    if verbose:
        print(f"\n[eps fit] valid={mi.valid} "
              f"accurate_cov={mi.fmin.has_accurate_covar} "
              f"edm={mi.fmin.edm:.2e} (goal {mi.fmin.edm_goal:.2e})  "
              f"{len(names)} parameters, {N.shape[0] * NR} region totals")
        worst = np.max(np.abs(theta))
        print(f"  largest |theta| = {worst:.4f}  "
              f"(scale factor {np.exp(worst):.4f})")
        if worst > 0.05:
            print("  [!] on stacked MC theta should be ~0. A large value "
                  "means N and the counts are inconsistent, NOT that the "
                  "tagger is mismodelled.")

    return apply_theta(eps_mc, theta), theta, np.array(mi.covariance), mi


# ── step 2 : solve for the response ──────────────────────────────────────────

def solve(N, eps, data):
    """
    resp[k,:,Y] = A[k]^-1 data[k,:,Y]  with  A[k,t,f] = N[k,f] eps[k,f,t].

    Exactly determined, so no fit and no constraint. Row sums are NOT
    imposed: they come out at 1 when eps reproduces the region totals, and
    their deviation is the goodness-of-fit.
    """
    n_cells = data.shape[0]
    A = np.einsum("kf,kft->ktf", N, eps)
    resp = np.full((n_cells, NF, NC), np.nan)
    Ainv = np.full((n_cells, NF, NR), np.nan)
    cond = np.full(n_cells, np.nan)
    for k in range(n_cells):
        cond[k] = np.linalg.cond(A[k])
        if not np.isfinite(cond[k]) or cond[k] > 1e12:
            continue
        Ainv[k] = np.linalg.inv(A[k])
        resp[k] = Ainv[k] @ data[k]
    return resp, Ainv, cond, A


def stat_cov(Ainv, var, multinomial=False):
    n_cells = Ainv.shape[0]
    T2 = var.sum(axis=2)                       # per (k, t)
    cov = np.full((n_cells, NC, NF, NF), np.nan)
    for k in range(n_cells):
        if not np.isfinite(Ainv[k]).all():
            continue
        for y in range(NC):
            v = var[k, :, y]
            if multinomial:
                v = v * (1.0 - np.divide(v, T2[k],
                                         out=np.zeros_like(v),
                                         where=T2[k] > 0))
            cov[k, y] = Ainv[k] @ np.diag(v) @ Ainv[k].T
    return cov



def impurity_error(m, theta, cov_theta, base):
    """
    Propagate the fitted scale-factor covariance by shifting along its
    EIGENVECTORS and re-solving. Eigen-directions rather than one parameter
    at a time: the 12 parameters come from one fit and are correlated, so
    independent shifts would both double count and miss the correlated
    direction.
    """
    w, V = np.linalg.eigh(cov_theta)
    shifts = []
    for i in range(len(w)):
        if w[i] <= 0:
            continue
        dth = (np.sqrt(w[i]) * V[:, i]).reshape(NF, NR - 1)
        alt, _, _, _ = solve(m["N"], apply_theta(m["eps_mc"], theta + dth),
                             m["data"])
        shifts.append(alt - base)
    if not shifts:
        return np.zeros_like(base)
    return np.sqrt(np.nansum(np.square(shifts), axis=0))


def composition_error(m, base, verbose=False):
    """N[k,f] scaled by SIG_NJET, eps refitted, response re-solved."""
    shifts = []
    for f in range(NF):
        mm = dict(m)
        N = m["N"].copy()
        N[:, f] *= (1.0 + SIG_NJET)
        mm["N"] = N
        eps_alt, _, _, _ = fit_eps(mm, verbose=verbose)
        alt, _, _, _ = solve(N, eps_alt, m["data"])
        shifts.append(alt - base)
    return np.sqrt(np.nansum(np.square(shifts), axis=0))


def purity(A):
    tot = A.sum(axis=2)
    out = np.full_like(tot, np.nan)
    for t, r in enumerate(REGIONS):
        f = FI["ud"] if r == "q" else FI[r]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[:, t] = np.where(tot[:, t] > 0, A[:, t, f] / tot[:, t], np.nan)
    return out


def pure_error(m, f, k):
    """Error a PURE sample of the same size would have given in the matched
    region. The ratio to the stat error is the price of the unfolding."""
    t = REGIONS.index(DIAG[f]) if f == "ud" else REGIONS.index(f)
    i = FI[f]
    n = m["raw"][k, t, :, i].sum()
    if n <= 0:
        return np.nan
    e = m["raw"][k, t, CI[DIAG[f]], i] / n
    return float(np.sqrt(max(e * (1.0 - e), 0.0) / n))


def region_response_check(m):
    """
    delta[t,Y,f] = resp(region t) - resp(untagged), cell-integrated. This is
    the tag/probe correlation resolved by region -- the thing the model
    assumes is zero.

    Because the regions are disjoint and exhaustive,
    sum_t eps[f,t] delta[f,t,Y] = 0 identically. The residual returned here
    is therefore a CODE CHECK (should be ~1e-15) and not a physics result:
    the physics is the spread of delta across t, which is what the closure
    bias is built from.
    """
    raw = m["raw"]                                   # (k, t, Y, f)
    per_region = raw.sum(axis=0)                     # (t, Y, f)
    den_t = per_region.sum(axis=1, keepdims=True)    # (t, 1, f)
    resp_t = np.divide(per_region, den_t, out=np.zeros_like(per_region),
                       where=den_t > 0)
    untag = raw.sum(axis=(0, 1))                     # (Y, f)
    den_a = untag.sum(axis=0, keepdims=True)         # (1, f)
    resp_all = np.divide(untag, den_a, out=np.zeros_like(untag),
                         where=den_a > 0)
    delta = resp_t - resp_all[None, :, :]            # (t, Y, f)
    resid = np.einsum("ft,tyf->fy", m["eps_incl"], delta)
    return delta, resid


# ── reporting ────────────────────────────────────────────────────────────────

def report(m, eps, theta, resp, cov, imp, comp, cond, A, max_cells=8):
    print("\nfitted data/MC tag scale factors  exp(theta[f,t])")
    print("  (1.0 everywhere is the expected answer on stacked MC)")
    print(f"  {'':<5}" + "".join(f"{'tag ' + r:>10}" for r in REGIONS))
    sf = np.exp(np.concatenate([theta, np.zeros((NF, 1))], axis=1))
    for f in FLAVOURS:
        print(f"  {f:<5}" + "".join(f"{sf[FI[f], t]:>10.4f}"
                                    for t in range(NR)))

    print("\ntag efficiency eps[f,t], cell 0   (cell-integrated in brackets)")
    print(f"  {'':<5}" + "".join(f"{'tag ' + r:>18}" for r in REGIONS)
          + f"{'sum_t':>9}")
    for f in FLAVOURS:
        i = FI[f]
        print(f"  {f:<5}" + "".join(
            f"{eps[0, i, t]:>10.4f} [{m['eps_incl'][i, t]:.4f}]"
            for t in range(NR))
            + f"{m['eps_incl'][i].sum():>9.4f}")
    print("  the two differ because the tag jet's kinematics correlate with "
          "the\n  probe cell -- that difference is why a global eps could "
          "not work.\n  sum_t must be 1.0000: the regions are disjoint and "
          "exhaustive.")

    pur = purity(A)
    print("\nregion purity (matched flavour / all), cell 0")
    print("  " + "".join(f"{'tag ' + r:>10}" for r in REGIONS))
    print("  " + "".join(f"{pur[0, t]:>10.4f}" for t in range(NR)))
    finite = cond[np.isfinite(cond)]
    if finite.size:
        n_bad = int((cond > 1e12).sum() + (~np.isfinite(cond)).sum())
        print(f"  cond(A): median {np.median(finite):.3g}  "
              f"max {finite.max():.3g}   ({n_bad} cells skipped)")

    delta, resid = region_response_check(m)
    print("\ntag/probe correlation by region: "
          "delta[f,t,Y] = resp(tag t) - resp(untagged)")
    print("  (diagonal category only; the model assumes this is 0)")
    print(f"  {'':<5}" + "".join(f"{'tag ' + r:>10}" for r in REGIONS)
          + f"{'eps-wgt':>12}")
    for f in FLAVOURS:
        i, y = FI[f], CI[DIAG[f]]
        print(f"  {f:<5}" + "".join(f"{delta[t, y, i]:>10.4f}"
                                    for t in range(NR))
              + f"{resid[i, y]:>12.2e}")
    print("  the last column is sum_t eps[f,t] delta[f,t,Y]; it is 0 by "
          "construction for\n  disjoint regions, so it is a code check and "
          "not a result. The bias comes\n  from the SPREAD across t, weighted "
          "by the mismatch between A^-1 and eps.")

    n_show = min(max_cells, resp.shape[0])
    print(f"\nresponse, first {n_show} cell(s)"
          "   value +- stat +- impurity +- compos | truth  C=fit/truth")
    for k in range(n_show):
        print(f"  cell {k}   rowsum: " + "  ".join(
            f"{f}={np.nansum(resp[k, FI[f]]):.4f}" for f in FLAVOURS))
        for f in FLAVOURS:
            i, y = FI[f], CI[DIAG[f]]
            v, tr = resp[k, i, y], m["resp_true"][k, i, y]
            st = np.sqrt(max(cov[k, y, i, i], 0.0)) \
                if np.isfinite(cov[k, y, i, i]) else np.nan
            pe = pure_error(m, f, k)
            infl = st / pe if (np.isfinite(pe) and pe > 0) else np.nan
            cval = v / tr if tr > 0 else np.nan
            print(f"    {f:<3}->{DIAG[f]:<3} {v:8.4f} +-{st:7.4f} "
                  f"+-{imp[k, i, y]:7.4f} +-{comp[k, i, y]:7.4f} | "
                  f"{tr:8.4f}  C={cval:7.4f}  infl={infl:5.2f}")


def bins_meta(axes):
    if not axes:
        return [{"index": [], "lo": [], "hi": [], "center": []}]
    shape = tuple(len(e) - 1 for _, e in axes)
    out = []
    for idx in np.ndindex(*shape):
        lo = [float(axes[j][1][idx[j]]) for j in range(len(axes))]
        hi = [float(axes[j][1][idx[j] + 1]) for j in range(len(axes))]
        out.append({"index": [int(i) for i in idx], "lo": lo, "hi": hi,
                    "center": [0.5 * (a + b) for a, b in zip(lo, hi)]})
    return out

def delta_diag_check(m, eps, Ainv, resp=None, flavour="b", max_cells=8):
    """
    Per-cell tag/probe correlation in the matched entry,
    delta[k, f, t=DIAG(f), Y=DIAG(f)], and the bias it predicts.

    The exact bias of the solve, with no approximation, is

        bias[k,f,Y] = sum_t Ainv[k,f,t] sum_g N[k,g] eps[k,g,t] delta[k,t,Y,g]

    Splitting off g = f and using sum_t P[k,f,t] = 1 with
    P[k,f,t] = Ainv[k,f,t] N[k,f] eps[k,f,t] gives

        bias_own[k,f,Y] = sum_t ( P - eps )[k,f,t] delta[k,f,t,Y]

    and when the matched region is pure, P -> (0..1..0) and that collapses
    to bias = delta[f, matched region, diagonal category]. `pred` below is
    the exact expression, `own` the same-flavour part, `diag` the matched
    term alone; the gap between them is the cross-flavour leak, which is
    small for b and large for s.

    The matched delta is NOT free: sum_t eps[f,t] delta[f,t,Y] = 0 forces

        delta[f,diag] = - (1/eps[f,diag]) sum_{t != diag} eps[f,t] delta[f,t]

    so tightening a tag hurts twice -- it moves weight into the off-diagonal
    regions AND shrinks the denominator. --breakdown prints both factors.
    """
    raw, N = m["raw"], m["N"]                        # raw: (k, t, Y, f)
    n_cells = raw.shape[0]

    den_t = raw.sum(axis=2, keepdims=True)           # (k, t, 1, f)
    resp_t = np.divide(raw, den_t, out=np.zeros_like(raw), where=den_t > 0)
    untag = raw.sum(axis=1)                          # (k, Y, f)
    den_a = untag.sum(axis=1, keepdims=True)
    resp_a = np.divide(untag, den_a, out=np.zeros_like(untag), where=den_a > 0)
    delta = resp_t - resp_a[:, None, :, :]           # (k, t, Y, f)

    P = np.einsum("kft,kf,kft->kft", Ainv, N, eps)   # (k, f, t), sums to 1 over t

    n_show = min(max_cells, n_cells)
    print("\ndiagonal tag/probe correlation, per cell")
    print("  delta = delta[f, tag DIAG(f), Y=DIAG(f)];  eps_d = eps[f, DIAG(f)]")
    print("  pred = exact bias;  own = same-flavour part;  bias = fit - truth")
    print(f"  {'':<4}{'':<6}" + "".join(f"{h:>10}" for h in
          ("eps_d", "delta", "own", "pred", "bias", "C")))
    for k in range(n_show):
        print(f"  cell {k}")
        for f in FLAVOURS:
            i = FI[f]
            t = REGIONS.index(DIAG[f])
            y = CI[DIAG[f]]
            d_diag = delta[k, t, y, i]
            own = float(np.sum((P[k, i] - eps[k, i]) * delta[k, :, y, i]))
            # exact: sum_t Ainv[f,t] sum_g N[g] eps[g,t] delta[t,y,g]
            pred = float(np.einsum("t,g,gt,tg->",
                                   Ainv[k, i], N[k], eps[k], delta[k, :, y, :]))
            tr = m["resp_true"][k, i, y]
            if resp is not None and np.isfinite(resp[k, i, y]):
                bias = resp[k, i, y] - tr
                cval = resp[k, i, y] / tr if tr > 0 else np.nan
            else:
                bias, cval = np.nan, np.nan
            print(f"    {f:<6}" + "".join(f"{v:>10.4f}" for v in
                  (eps[k, i, t], d_diag, own, pred, bias, cval)))

    # sum-rule decomposition for one flavour: which region forces delta[f,diag]
    i = FI[flavour]
    t_d = REGIONS.index(DIAG[flavour])
    y = CI[DIAG[flavour]]
    print(f"\nsum rule for {flavour}: delta[{flavour}, tag {DIAG[flavour]}]"
          f" = -(1/eps_d) sum_(t != d) eps[t] delta[t]")
    print(f"  {'cell':<6}{'region':<8}" + "".join(f"{h:>12}" for h in
          ("eps", "delta", "eps*delta", "share")))
    for k in range(n_show):
        terms = np.array([eps[k, i, t] * delta[k, t, y, i] for t in range(NR)])
        off = terms.copy()
        off[t_d] = 0.0
        tot = off.sum()
        for t, r in enumerate(REGIONS):
            if t == t_d:
                continue
            share = terms[t] / tot if tot != 0 else np.nan
            print(f"  {k:<6}{'tag ' + r:<8}"
                  f"{eps[k, i, t]:>12.4f}{delta[k, t, y, i]:>12.4f}"
                  f"{terms[t]:>12.5f}{100 * share:>11.1f}%")
        print(f"  {'':<6}{'-> sum':<8}{'':>12}{'':>12}{tot:>12.5f}")
        print(f"  {'':<6}{'/eps_d':<8}{eps[k, i, t_d]:>12.4f}{'':>12}"
              f"{-tot / eps[k, i, t_d]:>12.5f}"
              f"   (= delta[{flavour}, tag {DIAG[flavour]}])")

    return delta

def write_json(path, theta, c, m, eps, mi, resp, cov, imp, comp, cond, A):
    bins = bins_meta(c["axes"])
    pur = purity(A)
    delta, resid = region_response_check(m)
    cells = []
    SIG_THETA = 1.0
    for k in range(c["n_cells"]):
        flav = {}
        for f in FLAVOURS:
            i = FI[f]
            row = {}
            for y, cat in enumerate(CATS):
                v = resp[k, i, y]
                ok = np.isfinite(v)
                st = float(np.sqrt(max(cov[k, y, i, i], 0.0))) \
                    if np.isfinite(cov[k, y, i, i]) else None
                tr = float(m["resp_true"][k, i, y])
                row[cat] = {
                    "value": float(v) if ok else None,
                    "stat": st,
                    "impurity": float(imp[k, i, y]),
                    "composition": float(comp[k, i, y]),
                    "truth": tr,
                    "bias": float(v - tr) if ok else None,
                    "C": float(v / tr) if (ok and tr > 0) else None,
                }
            row["rowsum"] = float(np.nansum(resp[k, i]))
            flav[f] = row
        entry = {"cell": k, "response": flav,
                 "purity": {r: float(pur[k, t]) for t, r in enumerate(REGIONS)},
                 "cond_A": float(cond[k]) if np.isfinite(cond[k]) else None}
        if k < len(bins):
            entry.update(bins[k])
        cells.append(entry)

    payload = {
        "method": "4 disjoint tag regions; eps fitted to region totals, "
                  "response solved exactly per cell",
        "sig_theta": SIG_THETA,
        "closure_reference": "true per-jet efficiency (all regions summed)",
        "binning": c["binning_key"], "axis_vars": c["axis_vars"],
        "axes": [{"var": v, "edges": np.asarray(e, float).tolist(),
                  "n_bins": int(len(e) - 1)} for v, e in c["axes"]],
        "n_cells": c["n_cells"],
        "flavours": FLAVOURS, "probe_categories": CATS,
        "tag_regions": REGIONS, "diag_probe": DIAG,
        "config": {
            "xsec_pb": XSEC_PB, "lumi_pb": LUMI_PB,
            "unweighted": c["unweighted"], "n_gen": c["n_gen"],
            "scales": {f: float(c["scales"][f]) for f in FLAVOURS},
            "sig_njet": SIG_NJET,
            "y23_definition": "y_23_Q2 = d_23 / (jet1_e + jet2_e)^2 "
                              "(zg_common.define_jets_and_scales)",
            "y23_min": Y23_MIN, "y23_max": Y23_MAX,
            "z_selection": Z_SELECTION,
            "recoil_halfwidth": RECOIL_HALFWIDTH,
            "mz": MZ,
            "mass_window": [MASS_MIN, MASS_MAX],
            "z_efficiency": c.get("z_eff", {}),
            "jet_fiducial": {
                "p_min": JET_P_MIN, "p_max": JET_P_MAX,
                "abscos_min": JET_ABSCOS_MIN, "abscos_max": JET_ABSCOS_MAX,
                "dR_min": DR_MIN, "dR_max": DR_MAX,
                "dijet_p_min": DIJET_P_MIN, "dijet_p_max": DIJET_P_MAX,
                "dijet_p_definition": "reco_dijet_p = (jet1+jet2).P() "
                                      "(zg_common.define_jets_and_scales)",
                "note": "both jets; always applied, independent of binning",
            },
            "jet_fiducial_efficiency": c.get("kin_eff", {}),
            "tree": TREE,
            "cascade_disc": {k: list(v) for k, v in CASCADE_DISC.items()},
            "nominal_wp_logit": NOMINAL_WP_LOGIT,
            "tight_wp_logit": TIGHT_WP_LOGIT,
        },
        "eps_fit": {
            "valid": bool(mi.valid),
            "accurate_cov": bool(mi.fmin.has_accurate_covar),
            "edm": float(mi.fmin.edm),
            "scale_factors": {
                f: {r: float(np.exp(theta[FI[f], t])) if t < NR - 1 else 1.0
                    for t, r in enumerate(REGIONS)} for f in FLAVOURS},
            "eps_cell0": {f: {r: float(eps[0, FI[f], t])
                              for t, r in enumerate(REGIONS)}
                          for f in FLAVOURS},
            "eps_inclusive": {f: {r: float(m["eps_incl"][FI[f], t])
                                  for t, r in enumerate(REGIONS)}
                              for f in FLAVOURS},
        },
        "region_response": {
            "note": "delta[f][t][Y] = resp(tag region t) - resp(untagged), "
                    "cell-integrated. sum_t eps[f,t] delta = 0 identically "
                    "for disjoint regions; closure_residual is that sum and "
                    "should be ~1e-15.",
            "delta": {f: {r: {cat: float(delta[t, y, FI[f]])
                              for y, cat in enumerate(CATS)}
                          for t, r in enumerate(REGIONS)} for f in FLAVOURS},
            "closure_residual": {f: {cat: float(resid[FI[f], y])
                                     for y, cat in enumerate(CATS)}
                                 for f in FLAVOURS},
        },
        "bins": cells,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"\nwrote {path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    global Y23_MIN, Y23_MAX, Z_SELECTION, RECOIL_HALFWIDTH
    global DIJET_P_MIN, DIJET_P_MAX
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binning", default="p,abscos")
    ap.add_argument("--edges", default="coarse", choices=["auto", "fine", "coarse"])
    ap.add_argument("--unweighted", action="store_true",
                    help="stack raw counts. Composition follows generated "
                         "statistics rather than the Z->qq widths, but the "
                         "variance is then exactly Poisson -- the right first "
                         "closure test.")
    ap.add_argument("--lumi", type=float, default=LUMI_PB)
    ap.add_argument("--y23-min", type=float, default=None)
    ap.add_argument("--y23-max", type=float, default=None,
                    help="upper bound on the event-level merging scale "
                         "y_23_Q2 = d_23/(E1+E2)^2. Removes events where "
                         "exclusive njets=2 clustering forcibly absorbed a "
                         "hard third parton.")
    ap.add_argument("--dijet-p-min", type=float, default=DIJET_P_MIN,
                    help="lower bound on reco_dijet_p = (jet1+jet2).P() [GeV]. "
                         "Event-level, so both roles see the same cut.")
    ap.add_argument("--dijet-p-max", type=float, default=DIJET_P_MAX,
                    help="upper bound on reco_dijet_p = (jet1+jet2).P() [GeV].")
    ap.add_argument("--z-selection", default=Z_SELECTION,
                    choices=["recoil", "dijet", "none"],
                    help="'recoil' uses |m_recoil - mZ| and is photon-only, so "
                         "it cannot couple the two jets. 'dijet' is the legacy "
                         "reco_dijet_m window, kept for the A/B comparison.")
    ap.add_argument("--recoil-halfwidth", type=float, default=RECOIL_HALFWIDTH,
                    help="half-width in GeV. The Z Breit-Wigner (Gamma=2.495) "
                         "dominates over the ~1 GeV resolution, so 5 GeV holds "
                         "~84%% of the peak.")
    ap.add_argument("--max-print", type=int, default=8)
    ap.add_argument("--out", default=None, help="JSON path, '-' to skip")
    ap.add_argument("--input-base", default=None,
                    help="ntuple directory (default: the INPUT_BASE in the CONFIG block)")
    args = ap.parse_args()

    if args.input_base:
        set_input_base(args.input_base)
    Y23_MIN, Y23_MAX = args.y23_min, args.y23_max
    DIJET_P_MIN, DIJET_P_MAX = args.dijet_p_min, args.dijet_p_max
    Z_SELECTION, RECOIL_HALFWIDTH = args.z_selection, args.recoil_halfwidth

    c = load_all(args.binning, args.edges, args.lumi, args.unweighted)
    m = build(c)

    n_obs = c["n_cells"] * NR
    print("\nparameter counting for the eps fit")
    print(f"  region totals      : {n_obs}  ({c['n_cells']} cells x {NR} regions)")
    print(f"  free eps parameters: {NF * (NR - 1)}")
    if n_obs <= NF * (NR - 1):
        print("  [!] not enough cells to determine eps from the data; the fit "
              "will sit at the start value")

    eps, theta, cov_theta, mi = fit_eps(m)
    resp, Ainv, cond, A = solve(m["N"], eps, m["data"])
    delta_diag_check(m, eps, Ainv, resp=resp, flavour="b",
                     max_cells=args.max_print)
    cov = stat_cov(Ainv, m["var"])
    imp = impurity_error(m, theta, cov_theta, resp)
    comp = composition_error(m, resp)
    w, V = np.linalg.eigh(cov_theta)
    print("eigenvalues:", np.sort(w))
    top = V[:, np.argmax(w)]
    for n, v in zip(mi.parameters, top):
        if abs(v) > 0.2: print(f"  {n}: {v:+.3f}")
    report(m, eps, theta, resp, cov, imp, comp, cond, A,
           max_cells=args.max_print)

    if args.out != "-":
        y23_tag = ""
        if Y23_MIN is not None or Y23_MAX is not None:
            lo = "" if Y23_MIN is None else f"gt{Y23_MIN:g}"
            hi = "" if Y23_MAX is None else f"lt{Y23_MAX:g}"
            y23_tag = "_y23" + lo + hi
        ztag = ("_recoil%g" % RECOIL_HALFWIDTH if Z_SELECTION == "recoil"
                else "_" + Z_SELECTION)
        fid_tag = ""
        if jet_fiducial_active():
            bits = []
            if JET_P_MIN is not None: bits.append(f"pgt{JET_P_MIN:g}")
            if JET_P_MAX is not None: bits.append(f"plt{JET_P_MAX:g}")
            if JET_ABSCOS_MIN is not None: bits.append(f"acgt{JET_ABSCOS_MIN:g}")
            if JET_ABSCOS_MAX is not None: bits.append(f"aclt{JET_ABSCOS_MAX:g}")
            if DR_MIN is not None: bits.append(f"drgt{DR_MIN:g}")
            if DR_MAX is not None: bits.append(f"drlt{DR_MAX:g}")
            if DIJET_P_MIN is not None: bits.append(f"pjjgt{DIJET_P_MIN:g}")
            if DIJET_P_MAX is not None: bits.append(f"pjjlt{DIJET_P_MAX:g}")
            fid_tag = "_fid" + "".join(bits)
        out = args.out or (
            OUTDIR / f"mixture4_{c['binning_key']}{ztag}{y23_tag}{fid_tag}.json")
        write_json(out, theta, c, m, eps, mi, resp, cov, imp, comp, cond, A)


if __name__ == "__main__":
    main()