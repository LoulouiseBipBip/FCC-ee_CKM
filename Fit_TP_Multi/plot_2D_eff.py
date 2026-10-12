"""
plot_mixture_2d.py

Display the 2D-binned output of mixture_tp.py: the fitted response, the truth
MC response, and the gap between them.

Reads the JSON only -- nothing is refitted here, so this is cheap to rerun
while you iterate on the presentation.

WHAT IS PLOTTED
---------------
For each true flavour f, the DIAGONAL probe category (b->b, c->c, s->s,
ud->q) unless --category says otherwise.

  fig 1 (maps)     fitted | truth | bias = fitted-truth | C = fitted/truth
  fig 2 (errors)   stat | impurity | compos | pull = bias / total error
  fig 3 (slices)   C against axis 0, one line per axis-1 bin
  fig 4 (diag)     cond(A), rowsum-1, and the per-region tag purity
                   (--diagnostics)

READING THEM
------------
`bias` and `C` are the tag/probe correlation: the method measures the probe
response GIVEN a tagged partner, the truth is the response with no tag
requirement, and the model assumes those are equal. They are not, and the
maps are the size of the difference.

`pull` asks whether the gap is covered by the quoted uncertainty. |pull| >~ 2
means the closure bias is real and not a fluctuation. Note this is MC against
itself, so `stat` is the statistical power of the MC sample, not of any data.

CAVEAT ON `impurity`
--------------------
The impurity error is built entirely from the eigendecomposition of the
covariance of the eps scale factors. If that fit had unconstrained
directions -- eps entries so small that the data carry no information on
them, e.g. the light-flavour contamination of a near-pure tag-b region --
the covariance is near-singular and the impurity error is meaningless
rather than merely large. Check `accurate_cov` in the JSON header, which
this script prints. Use --err stat,compos to exclude it from the pull while
that is unresolved.

Usage:
    python plot_mixture_2d.py mixture4_p_costheta.json
    python plot_mixture_2d.py mixture4_p_costheta.json --outdir plots --pdf
    python plot_mixture_2d.py f.json --category all --err stat,compos
    python plot_mixture_2d.py f.json --diagnostics --annotate
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


AXIS_LABEL = {
    "p":        r"jet $p$ [GeV]",
    "theta":    r"jet $\theta$ [rad]",
    "abscos":   r"$|\cos\theta|$",
    "costheta": r"$\cos\theta$",
    "opang":    r"opening angle [rad]",
    "y23":      r"$y_{23}$",
}

ERR_KEYS = ["stat", "impurity", "composition"]
ERR_ALIAS = {"stat": "stat", "impurity": "impurity",
             "compos": "composition", "composition": "composition"}


# ── loading ──────────────────────────────────────────────────────────────────

def load(path):
    with open(path) as fh:
        p = json.load(fh)
    if len(p["axes"]) != 2:
        raise SystemExit(
            f"this script draws 2D maps, but the file has {len(p['axes'])} "
            f"axis/axes ({p['binning']}). Rerun mixture_tp.py with two axes, "
            "e.g. --binning p,costheta")
    return p


def axis_info(p):
    """(var, edges, n_bins) for each of the two axes, in binning order."""
    return [(a["var"], np.asarray(a["edges"], float), int(a["n_bins"]))
            for a in p["axes"]]


def grid(p, getter):
    """
    Scatter the per-cell values onto the (n0, n1) rectangular grid.

    Uses each cell's stored `index`, not its position in the list, so this
    stays correct whatever order the cells were written in. Cells the solve
    skipped (singular A) stay NaN and are drawn as blanks.
    """
    shape = tuple(a["n_bins"] for a in p["axes"])
    g = np.full(shape, np.nan)
    for cell in p["bins"]:
        idx = cell.get("index")
        if not idx or len(idx) != 2:
            continue
        try:
            v = getter(cell)
        except (KeyError, TypeError):
            v = None
        if v is not None and np.isfinite(v):
            g[tuple(idx)] = float(v)
    return g


def resp_getter(flav, cat, key):
    return lambda cell: cell["response"][flav][cat][key]


def total_err(p, flav, cat, keys):
    """Quadrature sum of the requested error components, as a grid."""
    tot = np.zeros(tuple(a["n_bins"] for a in p["axes"]))
    any_finite = np.zeros_like(tot, dtype=bool)
    for k in keys:
        g = grid(p, resp_getter(flav, cat, k))
        any_finite |= np.isfinite(g)
        tot += np.nan_to_num(g, nan=0.0) ** 2
    out = np.sqrt(tot)
    out[~any_finite] = np.nan
    return out


# ── drawing ──────────────────────────────────────────────────────────────────

def draw(ax, p, g, title, cmap="viridis", center=None, robust=98,
         annotate=False, cbar_label=None):
    """
    One pcolormesh panel. `center` gives a diverging map symmetric about that
    value (0 for bias, 1 for C); otherwise a sequential map.

    Limits come from a percentile rather than min/max so that one pathological
    cell -- and with a near-singular eps covariance there will be some -- does
    not flatten the whole map to a single colour.
    """
    (v0, e0, _), (v1, e1, _) = axis_info(p)
    finite = g[np.isfinite(g)]
    if finite.size == 0:
        ax.text(0.5, 0.5, "no finite cells", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="0.4")
        ax.set_title(title, fontsize=9)
        return None

    if center is None:
        vmin = np.nanpercentile(finite, 100 - robust)
        vmax = np.nanpercentile(finite, robust)
        if vmin == vmax:
            vmin, vmax = vmin - 1e-9, vmax + 1e-9
        norm = None
        kw = dict(vmin=vmin, vmax=vmax)
    else:
        d = np.nanpercentile(np.abs(finite - center), robust)
        d = max(float(d), 1e-9)
        norm = TwoSlopeNorm(vmin=center - d, vcenter=center, vmax=center + d)
        kw = dict(norm=norm)

    # pcolormesh wants C indexed [y, x]; axis 0 of the binning is x
    mesh = ax.pcolormesh(e0, e1, np.ma.masked_invalid(g).T, cmap=cmap,
                         shading="flat", **kw)
    ax.set_xlabel(AXIS_LABEL.get(v0, v0), fontsize=8)
    ax.set_ylabel(AXIS_LABEL.get(v1, v1), fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)

    if annotate:
        cx = 0.5 * (e0[:-1] + e0[1:])
        cy = 0.5 * (e1[:-1] + e1[1:])
        for i in range(len(cx)):
            for j in range(len(cy)):
                if np.isfinite(g[i, j]):
                    ax.text(cx[i], cy[j], f"{g[i, j]:.3f}", ha="center",
                            va="center", fontsize=5.5, color="w")

    cb = plt.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=7)
    if cbar_label:
        cb.set_label(cbar_label, fontsize=7)
    return mesh


def fig_maps(p, pairs, annotate, robust):
    n = len(pairs)
    fig, axes = plt.subplots(n, 4, figsize=(17, 3.6 * n), squeeze=False)
    for r, (f, cat) in enumerate(pairs):
        val = grid(p, resp_getter(f, cat, "value"))
        tru = grid(p, resp_getter(f, cat, "truth"))
        bias = grid(p, resp_getter(f, cat, "bias"))
        cc = grid(p, resp_getter(f, cat, "C"))
        draw(axes[r][0], p, val, f"{f}->{cat}  fitted", annotate=annotate,
             robust=robust)
        draw(axes[r][1], p, tru, f"{f}->{cat}  truth (untagged MC)",
             annotate=annotate, robust=robust)
        draw(axes[r][2], p, bias, f"{f}->{cat}  bias = fit - truth",
             cmap="RdBu_r", center=0.0, annotate=annotate, robust=robust)
        draw(axes[r][3], p, cc, f"{f}->{cat}  C = fit / truth",
             cmap="RdBu_r", center=1.0, annotate=annotate, robust=robust)
    fig.suptitle(f"response and gap to truth   [{p['binning']}]", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    return fig


def fig_errors(p, pairs, err_keys, annotate, robust):
    n = len(pairs)
    fig, axes = plt.subplots(n, 4, figsize=(17, 3.6 * n), squeeze=False)
    for r, (f, cat) in enumerate(pairs):
        for c, key in enumerate(ERR_KEYS):
            g = grid(p, resp_getter(f, cat, key))
            draw(axes[r][c], p, g, f"{f}->{cat}  {key}", cmap="magma",
                 annotate=annotate, robust=robust)
        bias = grid(p, resp_getter(f, cat, "bias"))
        tot = total_err(p, f, cat, err_keys)
        with np.errstate(invalid="ignore", divide="ignore"):
            pull = np.where(tot > 0, bias / tot, np.nan)
        draw(axes[r][3], p, pull,
             f"{f}->{cat}  pull = bias / ({'+'.join(err_keys)})",
             cmap="RdBu_r", center=0.0, annotate=annotate, robust=robust)
    fig.suptitle(f"uncertainties and pull   [{p['binning']}]", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    return fig


def fig_slices(p, pairs, err_keys):
    """
    C against axis 0, one line per axis-1 bin. This is where a trend that the
    2D maps only hint at becomes obvious -- e.g. C rising at low p while the
    other flavours stay flat.
    """
    (v0, e0, n0), (v1, e1, n1) = axis_info(p)
    x = 0.5 * (e0[:-1] + e0[1:])
    n = len(pairs)
    ncol = min(n, 2)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.5 * ncol, 4.2 * nrow),
                             squeeze=False)
    cmap = plt.get_cmap("viridis")
    for a, (f, cat) in enumerate(pairs):
        ax = axes[a // ncol][a % ncol]
        cc = grid(p, resp_getter(f, cat, "C"))
        tru = grid(p, resp_getter(f, cat, "truth"))
        tot = total_err(p, f, cat, err_keys)
        # error on the RATIO, at fixed truth
        with np.errstate(invalid="ignore", divide="ignore"):
            ecc = np.where(tru > 0, tot / tru, np.nan)
        for j in range(n1):
            col = cmap(j / max(n1 - 1, 1))
            lab = f"{e1[j]:g} to {e1[j+1]:g}"
            ax.errorbar(x, cc[:, j], yerr=ecc[:, j], marker="o", ms=3, lw=1,
                        capsize=2, color=col, label=lab)
        ax.axhline(1.0, color="k", lw=0.8, ls="--")
        ax.set_xlabel(AXIS_LABEL.get(v0, v0), fontsize=9)
        ax.set_ylabel("C = fitted / truth", fontsize=9)
        ax.set_title(f"{f} -> {cat}", fontsize=10)
        ax.grid(alpha=0.25)
        ax.legend(title=AXIS_LABEL.get(v1, v1), fontsize=6, title_fontsize=7,
                  ncol=2)
    for a in range(len(pairs), nrow * ncol):
        axes[a // ncol][a % ncol].axis("off")
    fig.suptitle(f"closure against {AXIS_LABEL.get(v0, v0)}   "
                 f"[{p['binning']}]", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def fig_diagnostics(p, annotate, robust):
    """cond(A), the rowsum goodness-of-fit, and the tag region purities."""
    regions = p["tag_regions"]
    ncol = 2 + len(regions)
    fig, axes = plt.subplots(1, ncol, figsize=(4.2 * ncol, 4.0), squeeze=False)
    ax = axes[0]

    draw(ax[0], p, grid(p, lambda c: c["cond_A"]), "cond(A)", cmap="magma",
         annotate=annotate, robust=robust)
    # rowsum should be 1 wherever the eps fit reproduced the region totals
    rs = grid(p, lambda c: max(
        (abs(c["response"][f]["rowsum"] - 1.0) for f in p["flavours"]),
        default=None))
    draw(ax[1], p, rs, "max |rowsum - 1| over flavours", cmap="magma",
         annotate=annotate, robust=robust)
    for i, r in enumerate(regions):
        draw(ax[2 + i], p, grid(p, lambda c, r=r: c["purity"][r]),
             f"purity, tag {r}", annotate=annotate, robust=robust)
    fig.suptitle(f"diagnostics   [{p['binning']}]", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


# ── text summary ─────────────────────────────────────────────────────────────

def summarise(p, pairs, err_keys):
    (v0, e0, n0), (v1, e1, n1) = axis_info(p)
    ef = p.get("eps_fit", {})
    print(f"file binning : {p['binning']}  "
          f"({n0} x {n1} = {p['n_cells']} cells)")
    print(f"eps fit      : valid={ef.get('valid')} "
          f"accurate_cov={ef.get('accurate_cov')} edm={ef.get('edm'):.2e}"
          if ef.get("edm") is not None else "")
    if ef.get("accurate_cov") is False:
        print("  [!] the eps covariance is NOT accurate, so `impurity` is "
              "unreliable\n      and the pull that includes it is not "
              "meaningful. See --err.")
    print(f"error components in the pull: {', '.join(err_keys)}")

    print(f"\n{'flav':<8}{'median C':>10}{'min C':>10}{'max C':>10}"
          f"{'|C-1|>5%':>10}{'worst cell':>28}")
    for f, cat in pairs:
        cc = grid(p, resp_getter(f, cat, "C"))
        fin = np.isfinite(cc)
        if not fin.any():
            print(f"{f + '->' + cat:<8}{'-':>10}")
            continue
        d = np.abs(cc - 1.0)
        i, j = np.unravel_index(np.nanargmax(np.where(fin, d, -1)), cc.shape)
        loc = (f"{v0} {e0[i]:g}-{e0[i+1]:g}, {v1} {e1[j]:g}-{e1[j+1]:g}")
        print(f"{f + '->' + cat:<8}{np.nanmedian(cc):>10.4f}"
              f"{np.nanmin(cc):>10.4f}{np.nanmax(cc):>10.4f}"
              f"{int((d[fin] > 0.05).sum()):>10d}"
              f"{'  C=' + format(cc[i, j], '.3f') + '  ' + loc:>28}")

    print("\nnote: `truth` is the untagged MC response and `C` is the "
          "tag/probe\ncorrelation the method does not correct for. A C away "
          "from 1 is a real\nbias of the method, not a fit failure.")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json", help="output of mixture_tp.py, 2D binning")
    ap.add_argument("--outdir", default=None,
                    help="default: alongside the JSON")
    ap.add_argument("--category", default="diag",
                    help="'diag' (default), 'all' for every probe category, "
                         "or an explicit f:Y list like b:b,b:c,ud:q")
    ap.add_argument("--err", default="stat,impurity,compos",
                    help="error components in the pull and the slice bars")
    ap.add_argument("--annotate", action="store_true",
                    help="print the value in each cell; only readable on a "
                         "coarse grid")
    ap.add_argument("--robust", type=float, default=98.0,
                    help="colour-scale percentile, to stop one pathological "
                         "cell flattening the map (default 98)")
    ap.add_argument("--diagnostics", action="store_true",
                    help="also draw cond(A), rowsum and the tag purities")
    ap.add_argument("--pdf", action="store_true", help="also write a PDF")
    ap.add_argument("--dpi", type=int, default=140)
    args = ap.parse_args()

    p = load(args.json)
    diag = p["diag_probe"]

    if args.category == "diag":
        pairs = [(f, diag[f]) for f in p["flavours"]]
    elif args.category == "all":
        pairs = [(f, y) for f in p["flavours"] for y in p["probe_categories"]]
    else:
        pairs = []
        for tok in args.category.split(","):
            f, _, y = tok.partition(":")
            f, y = f.strip(), y.strip()
            if f not in p["flavours"] or y not in p["probe_categories"]:
                raise SystemExit(f"bad --category entry {tok!r}")
            pairs.append((f, y))

    err_keys = []
    for tok in args.err.split(","):
        tok = tok.strip()
        if tok and tok not in ERR_ALIAS:
            raise SystemExit(f"unknown error component {tok!r}; "
                             f"choose from {sorted(set(ERR_ALIAS))}")
        if tok:
            err_keys.append(ERR_ALIAS[tok])
    if not err_keys:
        raise SystemExit("--err needs at least one component")

    summarise(p, pairs, err_keys)

    src = Path(args.json)
    outdir = Path(args.outdir) if args.outdir else src.parent
    outdir.mkdir(parents=True, exist_ok=True)
    stem = src.stem

    figs = [("maps", fig_maps(p, pairs, args.annotate, args.robust)),
            ("errors", fig_errors(p, pairs, err_keys, args.annotate,
                                  args.robust)),
            ("slices", fig_slices(p, pairs, err_keys))]
    if args.diagnostics:
        figs.append(("diagnostics",
                     fig_diagnostics(p, args.annotate, args.robust)))

    print()
    for name, fig in figs:
        out = outdir / f"{stem}_{name}.png"
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
        print(f"wrote {out}")
        if args.pdf:
            outp = out.with_suffix(".pdf")
            fig.savefig(outp, bbox_inches="tight")
            print(f"wrote {outp}")
        plt.close(fig)


if __name__ == "__main__":
    main()