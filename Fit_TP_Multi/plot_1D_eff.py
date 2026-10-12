"""
plot_mixture_1d.py

Plot the 1D-binned response from a mixture_tp.py JSON:

  points with error bars  the value the mixture measurement returns
  open squares            the TRUE per-jet efficiency from MC, measured with
                          no tag requirement anywhere

The gap between them is the tag/probe correlation. The lower panel shows it
as C = fitted / truth, which is the multiplicative bias of the method.

Error bars: the outer cap is stat + impurity + composition in quadrature,
the inner (thicker) bar is statistics alone. The difference is what the
mixture costs. MC truth is drawn without an error bar -- it has none in the
JSON, and it is the reference, not a measurement.

Bins whose `rowsum` is not 1 within --rowsum-tol are drawn hollow and greyed:
in those the composition model failed and the value is not a measurement of
anything. They are shown rather than dropped so the failure stays visible.

Usage:
    python plot_mixture_1d.py mixture4_p.json
    python plot_mixture_1d.py mixture4_p.json --category all
    python plot_mixture_1d.py mixture4_p.json --out plots/ --logx
    python plot_mixture_1d.py mixture4_p.json --ylim 0.4 1.0 --clim 0.88 1.2
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

XLABEL = {
    "p":        r"probe $|p|$ [GeV]",
    "theta":    r"probe $\theta$ [rad]",
    "abscos":   r"probe $|\cos\theta|$",
    "costheta": r"probe $\cos\theta$",
    "opang":    r"opening angle [rad]",
    "y23":      r"$y_{23} = d_{23}/d_{12}$",
}
COLOR = {"b": "#C0392B", "c": "#1F618D", "s": "#1E8449", "ud": "#5D6D7E"}
LABEL = {"b": r"$Z\to b\bar b$", "c": r"$Z\to c\bar c$",
         "s": r"$Z\to s\bar s$", "ud": r"$Z\to u\bar u,\,d\bar d$"}


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(1.2)
        ax.spines[sp].set_color("0.15")
    ax.tick_params(which="major", labelsize=13, length=6, width=1.1,
                   colors="0.15")
    ax.tick_params(which="minor", length=3, width=0.8, colors="0.35")
    ax.minorticks_on()
    ax.grid(True, which="major", axis="y", color="0.88", lw=0.8)
    ax.set_axisbelow(True)


def series(payload, flav, cat, rowsum_tol):
    """Pull one (flavour, category) series out of the JSON."""
    n = payload["n_cells"]
    x = np.full(n, np.nan)
    xerr = np.full(n, np.nan)
    y = np.full(n, np.nan)
    e_stat = np.full(n, np.nan)
    e_tot = np.full(n, np.nan)
    truth = np.full(n, np.nan)
    good = np.zeros(n, dtype=bool)

    for b in payload["bins"]:
        k = b["cell"]
        if b.get("center"):
            x[k] = b["center"][0]
            xerr[k] = 0.5 * (b["hi"][0] - b["lo"][0])
        entry = b["response"].get(flav)
        if entry is None:
            continue
        truth[k] = entry[cat]["truth"]
        v = entry[cat]["value"]
        if v is None or not np.isfinite(v):
            continue
        y[k] = v
        st = entry[cat].get("stat") or 0.0
        im = entry[cat].get("impurity") or 0.0
        cp = entry[cat].get("composition") or 0.0
        e_stat[k] = st
        e_tot[k] = float(np.sqrt(st ** 2 + im ** 2 + cp ** 2))
        good[k] = abs(entry.get("rowsum", 1.0) - 1.0) <= rowsum_tol

    return x, xerr, y, e_stat, e_tot, truth, good


def plot_one(payload, flav, cat, outdir, rowsum_tol, logx, ylim, clim):
    var = payload["axis_vars"][0]
    x, xerr, y, e_stat, e_tot, truth, good = series(
        payload, flav, cat, rowsum_tol)

    ok = np.isfinite(y) & np.isfinite(x)
    if not ok.any():
        print(f"  {flav} -> {cat}: nothing to plot")
        return

    fig, (ax, rax) = plt.subplots(
        2, 1, figsize=(8.2, 6.6), sharex=True,
        gridspec_kw={"height_ratios": [2.6, 1.0], "hspace": 0.08})

    # MC truth first, so the measurement sits on top of it
    tok = np.isfinite(truth) & np.isfinite(x)
    ax.errorbar(x[tok], truth[tok], xerr=xerr[tok], fmt="s", ms=6.5,
                mfc="white", mew=1.5, color="0.35", ecolor="0.35",
                elinewidth=1.2, capsize=0, zorder=2,
                label="MC truth")

    for mask, filled, lab in ((ok & good, True, "mixture fit"),
                              (ok & ~good, False, "rowsum $\\neq$ 1, unusable")):
        if not mask.any():
            continue
        col = COLOR[flav] if filled else "0.7"
        # outer: total uncertainty
        ax.errorbar(x[mask], y[mask], yerr=e_tot[mask], fmt="none",
                    ecolor=col, elinewidth=1.4, capsize=3.5, capthick=1.3,
                    alpha=1.0 if filled else 0.6, zorder=3)
        # inner: statistics only
        ax.errorbar(x[mask], y[mask], xerr=xerr[mask], yerr=e_stat[mask],
                    fmt="o", ms=7.5, mew=1.1, mec="white",
                    color=col, ecolor=col, elinewidth=2.8, capsize=0,
                    mfc=col if filled else "white",
                    alpha=1.0 if filled else 0.6, zorder=4, label=lab)

    ax.set_ylabel(rf"$\varepsilon^{{{flav}}}_{{{cat}}}$", fontsize=17,
                  labelpad=8)
    kind = "efficiency" if cat == payload["diag_probe"][flav] else "mistag"
    ax.set_title(
        rf"{kind.upper()}  $\cdot$  true {flav} $\to$ probe {cat}"
        f"\n({LABEL[flav]})",
        loc="left", fontsize=14, pad=10, color="0.12")
    # ax.text(1.0, 1.02, "FCC-ee simulation", transform=ax.transAxes,
    #         ha="right", va="bottom", fontsize=11.5, color="0.4",
    #         style="italic")
    ax.legend(frameon=False, fontsize=11.5, loc="best")
    style(ax)

    # ── ratio: C = fitted / truth ───────────────────────────────────────────
    # MC truth carries no uncertainty in the JSON, so the band here is the
    # measurement error alone.
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.where(truth > 0, y / truth, np.nan)
        c_err = np.where(truth > 0, e_tot / truth, np.nan)
    rax.axhline(1.0, color="0.55", lw=1.1, ls="--", zorder=1)
    m = np.isfinite(c) & ok & good
    if m.any():
        rax.errorbar(x[m], c[m], xerr=xerr[m], yerr=c_err[m], fmt="o",
                     ms=6.5, mew=1.0, mec="white", color=COLOR[flav],
                     ecolor=COLOR[flav], elinewidth=1.4, capsize=3, zorder=3)
        if clim is None:
            dev = np.nanmax(np.abs(np.concatenate([c[m] - 1.0, c_err[m]])))
            pad = max(1.5 * dev, 0.02)
            rax.set_ylim(1.0 - pad, 1.0 + pad)
    else:
        rax.text(0.5, 0.5, "no usable bins", transform=rax.transAxes,
                 ha="center", va="center", fontsize=12, color="0.45")
    rax.set_ylabel(r"$C$ = fit / truth", fontsize=13, labelpad=8)
    rax.set_xlabel(XLABEL.get(var, var), fontsize=16, labelpad=8)
    if logx:
        ax.set_xscale("log")
        rax.set_xscale("log")
    if ylim is not None:
        ax.set_ylim(*ylim)
    if clim is not None:
        rax.set_ylim(*clim)
    style(rax)

    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"mix_{var}_true{flav}_to_{cat}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("json", help="JSON written by mixture_tp.py")
    ap.add_argument("--out", default=None,
                    help="output directory (default: alongside the JSON)")
    ap.add_argument("--category", default="diag", choices=["diag", "all"],
                    help="'diag' plots the correct-classification category "
                         "for each flavour; 'all' adds the mistags")
    ap.add_argument("--rowsum-tol", type=float, default=0.02,
                    help="how far rowsum may sit from 1 before a bin is "
                         "greyed out as a composition failure")
    ap.add_argument("--logx", action="store_true")
    ap.add_argument("--ylim", nargs=2, type=float, metavar=("YMIN", "YMAX"),
                    default=None,
                    help="fixed y-axis range on the main (efficiency) panel")
    ap.add_argument("--clim", nargs=2, type=float, metavar=("CMIN", "CMAX"),
                    default=None,
                    help="fixed y-axis range on the C-ratio panel")
    args = ap.parse_args()

    jpath = Path(args.json)
    with open(jpath) as fh:
        payload = json.load(fh)

    if len(payload.get("axis_vars", [])) != 1:
        raise SystemExit(
            f"this script plots 1D binnings; the file has "
            f"axis_vars={payload.get('axis_vars')}. Rerun mixture_tp.py with "
            f"a single --binning variable.")

    outdir = Path(args.out) if args.out else jpath.parent / f"plots_{jpath.stem}"
    print(f"binning: {payload['binning']}   cells: {payload['n_cells']}")

    for flav in payload["flavours"]:
        cats = ([payload["diag_probe"][flav]] if args.category == "diag"
                else payload["probe_categories"])
        for cat in cats:
            plot_one(payload, flav, cat, outdir, args.rowsum_tol, args.logx,
                     args.ylim, args.clim)


if __name__ == "__main__":
    main()