#!/usr/bin/env python3
"""
tp_counting_b.py

Pure counting on MC. No fit, no template, no role doubling.

For every event ONE jet is chosen at random as the tag and the other is the
probe. Then, in bins of one PROBE variable:

  eff_notag      true-b probes passing the b cascade / all true-b probes
                 -> the truth: what a b jet does, no tag requirement

  eff_tp_b       true-b probes passing / true-b probes, TAG PASSED
                 -> T&P restricted to true b: isolates the tag/probe
                    CORRELATION.  C = eff_tp_b / eff_notag

  eff_tp_naive   ALL probes passing / ALL probes, TAG PASSED
                 -> what naive T&P would read off data: contamination AND
                    correlation together

The gap  eff_tp_naive - eff_tp_b  is the impurity effect;
the gap  eff_tp_b - eff_notag     is the correlation effect.

  purity_tag     true-b fraction of the probes, tag passed
  purity_notag   true-b fraction of all probes (the sample composition)

The probe category is the SAME cascade as the nominal analysis
(mixture_tp.NOMINAL_WP_LOGIT, priority b > c > s); "passes b" means cascade
code 0. The tag is a tight box in the b plane (DB_vsC, DB_vsS).

Weighting: cross section x lumi / N_gen per flavour by default, so the
purities are physical. --unweighted gives equal-statistics MC fractions
instead (fine for efficiencies, meaningless for purity).

Usage:
    python tp_counting_b.py --var p
    python tp_counting_b.py --var abscos --tag-wp-x 0.999 --tag-wp-y 0.999
    python tp_counting_b.py --var opang --y23-max 0.01 --z-selection recoil
"""

import argparse
import json
import numpy as np

import tp_counting as M

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


TARGET = "b"
PER_JET_VARS = ("p", "theta", "abscos", "costheta")
EVENT_VARS = ("opang", "y23")


def jet_kinematics(arrs):
    p = np.stack([M._flat(arrs, "reco_jet1_p"), M._flat(arrs, "reco_jet2_p")], 1)
    th = np.stack([M._flat(arrs, "reco_jet1_theta"),
                   M._flat(arrs, "reco_jet2_theta")], 1)
    return p, th


def probe_value(arrs, var, jet_p, jet_theta, probe_idx):
    rows = np.arange(len(probe_idx))
    if var == "p":
        return jet_p[rows, probe_idx]
    if var == "theta":
        return jet_theta[rows, probe_idx]
    if var == "costheta":
        return np.cos(jet_theta[rows, probe_idx])
    if var == "abscos":
        return np.abs(np.cos(jet_theta[rows, probe_idx]))
    if var == "opang":
        return M._flat(arrs, "reco_opang")
    if var == "y23":
        y = M.get_y23(arrs)
        if y is None:
            raise RuntimeError("no y_23_Q2 (or d_23 + jet energies) in the tree")
        return y
    raise ValueError(var)


def count_flavour(arrs, var, edges, tag_wp, y23_max, rng):
    """Raw counts per bin for one true-flavour sample."""
    d = {}
    for key, (xb, yb) in M.CASCADE_DISC.items():
        d[key + "x"] = M._logit(arrs, xb)
        d[key + "y"] = M._logit(arrs, yb)

    jet_p, jet_theta = jet_kinematics(arrs)
    n = jet_p.shape[0]

    # ── random role assignment: one entry per event ─────────────────────
    tag_idx = rng.integers(0, 2, size=n)
    probe_idx = 1 - tag_idx

    # ── event selection: identical to the fit ───────────────────────────
    sel = M.z_selection_mask(arrs) & M.jet_fiducial_mask(arrs, jet_p, jet_theta)
    if y23_max is not None:
        y23 = M.get_y23(arrs)
        if y23 is None:
            raise RuntimeError("--y23-max set but no y23 information in the tree")
        sel &= (y23 >= 0.0) & (y23 <= y23_max)

    # ── probe: nominal cascade, b = code 0 ──────────────────────────────
    code0 = M.cascade_codes(d, 0, M.NOMINAL_WP_LOGIT)
    code1 = M.cascade_codes(d, 1, M.NOMINAL_WP_LOGIT)
    probe_pass = np.where(probe_idx == 0, code0, code1) == 0

    # ── tag: tight box in the b plane ───────────────────────────────────
    tx, ty = tag_wp
    tb0 = (d["bx"][:, 0] > tx) & (d["by"][:, 0] > ty)
    tb1 = (d["bx"][:, 1] > tx) & (d["by"][:, 1] > ty)
    tag_pass = np.where(tag_idx == 0, tb0, tb1)

    # ── bin on the probe ────────────────────────────────────────────────
    val = probe_value(arrs, var, jet_p, jet_theta, probe_idx)
    nb = len(edges) - 1
    idx = np.digitize(val, edges) - 1
    sel &= (idx >= 0) & (idx < nb)
    i = idx[sel]

    return {
        "n_all": np.bincount(i, minlength=nb).astype(float),
        "p_all": np.bincount(i, weights=probe_pass[sel].astype(float), minlength=nb),
        "n_tag": np.bincount(i, weights=tag_pass[sel].astype(float), minlength=nb),
        "p_tag": np.bincount(i, weights=(tag_pass & probe_pass)[sel].astype(float),
                             minlength=nb),
        "n_events": int(n), "n_selected": int(sel.sum()),
    }


def ratio(P, F, VP, VF):
    """P/(P+F) with its error, treating P and F as independent weighted sums."""
    T = P + F
    with np.errstate(invalid="ignore", divide="ignore"):
        eff = np.where(T > 0, P / T, np.nan)
        err = np.where(T > 0, np.sqrt(np.clip(F ** 2 * VP + P ** 2 * VF, 0, None))
                       / T ** 2, np.nan)
    return eff, err


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--var", default="p",
                    choices=list(PER_JET_VARS) + list(EVENT_VARS))
    ap.add_argument("--edges", default=None,
                    help="comma-separated bin edges; default mixture_tp.EDGES_FINE[var]")
    ap.add_argument("--tag-wp-x", type=float, default=None,
                    help="tag cut on DB_vsC, raw score; default TIGHT_WP_LOGIT['b']")
    ap.add_argument("--tag-wp-y", type=float, default=None,
                    help="tag cut on DB_vsS, raw score")
    ap.add_argument("--z-selection", default=None, choices=["recoil", "dijet", "none"],
                    help="default: whatever mixture_tp.Z_SELECTION is set to")
    ap.add_argument("--y23-max", type=float, default=None)
    ap.add_argument("--unweighted", action="store_true")
    ap.add_argument("--lumi", type=float, default=M.LUMI_PB)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="tp_counting_b")
    args = ap.parse_args()

    if args.z_selection is not None:
        M.Z_SELECTION = args.z_selection
    rng = np.random.default_rng(args.seed)

    edges = (np.array([float(x) for x in args.edges.split(",")])
             if args.edges else np.asarray(M.EDGES_FINE[args.var], float))
    nb = len(edges) - 1

    if args.tag_wp_x is not None and args.tag_wp_y is not None:
        tag_wp = (M._lg(args.tag_wp_x), M._lg(args.tag_wp_y))
    else:
        tag_wp = (M.TIGHT_WP_LOGIT["b"]["x"], M.TIGHT_WP_LOGIT["b"]["y"])
    nom = (M.NOMINAL_WP_LOGIT["b"]["x"], M.NOMINAL_WP_LOGIT["b"]["y"])
    if np.allclose(tag_wp, nom):
        print("[!] tag cut is IDENTICAL to the nominal probe b cut. The tag is then "
              "no tighter than the probe; pass --tag-wp-x/--tag-wp-y to tighten it.")

    print(f"variable: probe {args.var}   {nb} bins   Z selection: {M.Z_SELECTION}"
          + (f"   y23 <= {args.y23_max}" if args.y23_max is not None else ""))

    counts, scales = {}, {}
    for f in M.FLAVOURS:
        print(f"\n[load] {f}")
        arrs = M.load_sample(M.SAMPLES[f])
        ngen = len(arrs)
        scales[f] = 1.0 if args.unweighted else M.XSEC_PB[f] * args.lumi / ngen
        counts[f] = count_flavour(arrs, args.var, edges, tag_wp, args.y23_max, rng)
        c = counts[f]
        print(f"  events {c['n_events']}  selected {c['n_selected']}  "
              f"tag-pass fraction {c['n_tag'].sum() / max(c['n_all'].sum(), 1):.4f}")

    b = counts[TARGET]
    wb = scales[TARGET]

    # truth, no tag: true b only
    eff_notag, err_notag = ratio(wb * b["p_all"], wb * (b["n_all"] - b["p_all"]),
                                 wb ** 2 * b["p_all"], wb ** 2 * (b["n_all"] - b["p_all"]))

    # T&P, true b only: the correlation
    eff_tpb, err_tpb = ratio(wb * b["p_tag"], wb * (b["n_tag"] - b["p_tag"]),
                             wb ** 2 * b["p_tag"], wb ** 2 * (b["n_tag"] - b["p_tag"]))

    # T&P, everything that passed the tag: what naive counting on data gives
    P = sum(scales[f] * counts[f]["p_tag"] for f in M.FLAVOURS)
    F = sum(scales[f] * (counts[f]["n_tag"] - counts[f]["p_tag"]) for f in M.FLAVOURS)
    VP = sum(scales[f] ** 2 * counts[f]["p_tag"] for f in M.FLAVOURS)
    VF = sum(scales[f] ** 2 * (counts[f]["n_tag"] - counts[f]["p_tag"]) for f in M.FLAVOURS)
    eff_naive, err_naive = ratio(P, F, VP, VF)

    # purities
    def purity(key):
        Pb = wb * b[key]
        Fo = sum(scales[f] * counts[f][key] for f in M.FLAVOURS if f != TARGET)
        return ratio(Pb, Fo, wb ** 2 * b[key],
                     sum(scales[f] ** 2 * counts[f][key]
                         for f in M.FLAVOURS if f != TARGET))
    pur_tag, epur_tag = purity("p_tag")
    pur_notag, epur_notag = purity("p_all")

    with np.errstate(invalid="ignore", divide="ignore"):
        C = eff_tpb / eff_notag
        eC = C * np.sqrt((err_tpb / eff_tpb) ** 2 + (err_notag / eff_notag) ** 2)
        R = eff_naive / eff_notag
        eR = R * np.sqrt((err_naive / eff_naive) ** 2 + (err_notag / eff_notag) ** 2)

    # ── table ───────────────────────────────────────────────────────────
    print(f"\n{'bin':>16}{'eff_notag':>11}{'eff_tp_b':>11}{'eff_naive':>11}"
          f"{'C':>9}{'naive/true':>12}{'pur_tag':>10}{'pur_all':>10}")
    for k in range(nb):
        lab = f"[{edges[k]:.3g},{edges[k + 1]:.3g}]"
        print(f"{lab:>16}{eff_notag[k]:>11.4f}{eff_tpb[k]:>11.4f}{eff_naive[k]:>11.4f}"
              f"{C[k]:>9.4f}{R[k]:>12.4f}{pur_tag[k]:>10.4f}{pur_notag[k]:>10.4f}")

    tot = lambda key: b[key].sum()
    C_incl = (b["p_tag"].sum() / tot("n_tag")) / (b["p_all"].sum() / tot("n_all"))
    print(f"\ninclusive C (true b) = {C_incl:.4f}")
    print("  C carries the error of both samples added in quadrature; the tagged\n"
          "  sample is a subset of the untagged one, so this overestimates it.")
    if args.unweighted:
        print("[!] --unweighted: purities are equal-statistics MC fractions, "
              "not physical ones.")

    # ── plot ────────────────────────────────────────────────────────────
    xc = 0.5 * (edges[:-1] + edges[1:])
    xe = 0.5 * np.diff(edges)
    fig, ax = plt.subplots(3, 1, figsize=(8, 10), sharex=True,
                           gridspec_kw={"height_ratios": [3, 1.6, 1.6]})

    kw = dict(fmt="o", ms=4, capsize=2, lw=1)
    ax[0].errorbar(xc, eff_notag, xerr=xe, yerr=err_notag, color="black",
                   label="true b, no tag (truth)", **kw)
    ax[0].errorbar(xc, eff_tpb, xerr=xe, yerr=err_tpb, color="crimson",
                   label="T&P, true b only (correlation)", **kw)
    ax[0].errorbar(xc, eff_naive, xerr=xe, yerr=err_naive, color="steelblue",
                   label="T&P, all tagged probes (naive)", **kw)
    ax[0].set_ylabel("b-cascade efficiency")
    ax[0].legend(fontsize=9)
    ax[0].grid(alpha=0.3)

    ax[1].errorbar(xc, C, xerr=xe, yerr=eC, color="crimson",
                   label="C = T&P(true b) / truth", **kw)
    ax[1].errorbar(xc, R, xerr=xe, yerr=eR, color="steelblue",
                   label="naive / truth", **kw)
    ax[1].axhline(1.0, color="k", ls="--", lw=1)
    ax[1].set_ylabel("ratio to truth")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    ax[2].errorbar(xc, pur_tag, xerr=xe, yerr=epur_tag, color="crimson",
                   label="b purity, tag passed", **kw)
    ax[2].errorbar(xc, pur_notag, xerr=xe, yerr=epur_notag, color="gray",
                   label="b fraction, no tag", **kw)
    ax[2].set_ylabel("b purity of probes")
    ax[2].set_ylim(0, 1.05)
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3)

    xlabel = {"p": "probe |p| [GeV]", "theta": r"probe $\theta$ [rad]",
              "abscos": r"probe $|\cos\theta|$", "costheta": r"probe $\cos\theta$",
              "opang": "opening angle [rad]", "y23": r"$y_{23}$"}[args.var]
    ax[2].set_xlabel(xlabel)
    tsx, tsy = [1 / (1 + np.exp(-v)) for v in tag_wp]
    fig.suptitle(f"b tagging, pure counting, random tag/probe roles\n"
                 f"tag: DB_vsC > {tsx:.5f}, DB_vsS > {tsy:.5f}   "
                 f"Z sel: {M.Z_SELECTION}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(f"{args.out}_{args.var}.png", dpi=140)
    print(f"\nwrote {args.out}_{args.var}.png")

    out = {"var": args.var, "edges": edges.tolist(), "seed": args.seed,
           "tag_wp_score": [float(tsx), float(tsy)],
           "z_selection": M.Z_SELECTION, "y23_max": args.y23_max,
           "unweighted": bool(args.unweighted), "C_inclusive": float(C_incl),
           "eff_notag": eff_notag.tolist(), "err_notag": err_notag.tolist(),
           "eff_tp_b": eff_tpb.tolist(), "err_tp_b": err_tpb.tolist(),
           "eff_tp_naive": eff_naive.tolist(), "err_tp_naive": err_naive.tolist(),
           "C": C.tolist(), "purity_tag": pur_tag.tolist(),
           "purity_notag": pur_notag.tolist()}
    with open(f"{args.out}_{args.var}.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {args.out}_{args.var}.json")


if __name__ == "__main__":
    main()