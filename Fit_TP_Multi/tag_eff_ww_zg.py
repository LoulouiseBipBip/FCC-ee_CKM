#!/usr/bin/env python3
"""
MC-true tagging efficiency, WW (semileptonic) vs Z+gamma, in bins of jet |p| and |cos theta|.

eps_f = (true-f jets in cascade category f) / (true-f jets).  Both jets of an
event enter; truth = |jet{i}_getflavour_pdg| on both samples.  No tag-and-probe.

Cascade (same planes / WPs as tp_counting.NOMINAL_WP_LOGIT), disjoint:
    b : DB_vsC > 0.99717 && DB_vsS > 0.99564
    c : !b && DC_vsB > 0.00006 && DC_vsS > 0.53319
    s : !b && !c && DS_vsC > 0.46681 && DS_vsQ > 0.20117
Cutting on the raw score is identical to cutting on its logit.  NaN (WW, 0/0)
and -1 (Zg safe_ratio) both fail every cut.

Output: OUT.root (TH2 num/den per sample and flavour) and OUT.pdf
(per flavour: eps vs |p| and vs |cos theta| with Zg/WW ratio, then the 2D Zg/WW map).
"""

import ctypes
import glob
import array
import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetPaintTextFormat("4.3f")
ROOT.EnableImplicitMT()

# ── configuration ────────────────────────────────────────────────────────────
SAMPLES = {   # name: (glob, extra event-level cut or "", colour, marker)
    "WW":  ("/eos/user/l/lberiet/CKM_Flav/WW_semilep_163_newtagger/wzp6_ee_munumuqq_V*_ecm163/*.root",
            "", ROOT.kBlack, 20),
    "Zg":  ("/eos/user/l/lberiet/CKM_Flav/Zg_163_std_default/wzp6_ee_*_pqqmin40_ecm163/*.root",
            "", ROOT.kAzure + 1, 21),
}
JET_ABSCOS_MAX = 1.0            # both jets, both samples

WP = {
    "b": ("DB_vsC", 0.99717, "DB_vsS", 0.99564),
    "c": ("DC_vsB", 0.00006, "DC_vsS", 0.53319),
    "s": ("DS_vsC", 0.46681, "DS_vsQ", 0.20117),
}
FLAV_PDG = {"b": 5, "c": 4, "s": 3}

P_EDGES = [15, 17.5, 20, 25, 30, 35, 40, 47.5, 55, 62.5, 70, 85]

# Same |cos theta| boundaries as the old theta edges, folded forward/backward.
ABSCOS_EDGES = [0.0, 0.74, 0.88, 0.92, 0.96, 0.99]

OUT = "tageff_ww_vs_zg_p_costheta_163"
RATIO_RANGE = (0.9, 1.1)

# Columns book() actually reads. A zero-length basket header on one of these
# is what makes TBranch::GetBasket print "byte:0" and then segfault.
_READ_BRANCHES = (
    "reco_jet1_p", "reco_jet2_p", "reco_jet1_theta", "reco_jet2_theta",
    "jet1_getflavour_pdg", "jet2_getflavour_pdg",
    "DB_vsC", "DB_vsS", "DC_vsB", "DC_vsS", "DS_vsC", "DS_vsQ",
)


def _corrupt_reason(path):
    """None if every read branch has a readable basket header, else a short reason."""
    f = None
    try:
        f = ROOT.TFile.Open(path)
        if not f or f.IsZombie() or not f.IsOpen():
            return "cannot open"
        tree = f.Get("events")
        if not tree:
            return "no events tree"
        buf = ctypes.create_string_buffer(256)
        nbytes, objlen, keylen = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        for bname in _READ_BRANCHES:
            br = tree.GetBranch(bname)
            if not br:
                return f"missing branch {bname}"
            for i in range(br.GetWriteBasket()):
                seek = br.GetBasketSeek(i)
                if seek <= 0:
                    return f"{bname} basket {i} has seek 0"
                ret = f.GetRecordHeader(buf, seek, 256, nbytes, objlen, keylen)
                if ret <= 0 or nbytes.value <= 0 or keylen.value <= 0:
                    return f"{bname} basket {i} unreadable"
        return None
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    finally:
        if f:
            f.Close()


# ── event loop ───────────────────────────────────────────────────────────────
def book(name, pattern, extra):
    files = sorted(glob.glob(pattern))
    if not files:
        raise RuntimeError(f"[{name}] no files match {pattern}")
    good = []
    for path in files:
        reason = _corrupt_reason(path)
        if reason:
            print(f"[{name}] skip {path}: {reason}", flush=True)
        else:
            good.append(path)
    if not good:
        raise RuntimeError(f"[{name}] every file matching {pattern} is unreadable")
    files = good
    df = ROOT.RDataFrame("events", files)
    df = df.Filter(" && ".join(f"std::abs(std::cos(reco_jet{i}_theta)) < {JET_ABSCOS_MAX}"
                               for i in (1, 2)), "fiducial")
    if extra:
        df = df.Filter(extra, "extra")

    # per-jet vectors (index 0 = jet1, 1 = jet2), same order as the score RVecs
    df = df.Define("jp",   "ROOT::RVecD{reco_jet1_p, reco_jet2_p}")
    df = df.Define("jct",  "ROOT::RVecD{std::abs(std::cos(reco_jet1_theta)), "
                           "std::abs(std::cos(reco_jet2_theta))}")
    df = df.Define("jfl",  "ROOT::RVecI{std::abs((int)jet1_getflavour_pdg), "
                           "std::abs((int)jet2_getflavour_pdg)}")
    cut = lambda f: f"({WP[f][0]} > {WP[f][1]}) && ({WP[f][2]} > {WP[f][3]})"
    df = df.Define("pass_b", cut("b"))
    df = df.Define("pass_c", f"!pass_b && {cut('c')}")
    df = df.Define("pass_s", f"!pass_b && !pass_c && {cut('s')}")

    model = lambda n: ROOT.RDF.TH2DModel(
        n, ";|p| [GeV];|cos#theta|",
        len(P_EDGES) - 1, array.array("d", P_EDGES),
        len(ABSCOS_EDGES) - 1, array.array("d", ABSCOS_EDGES))
    h = {}
    for fl, pdg in FLAV_PDG.items():
        den, num = f"isf_{fl}", f"isfp_{fl}"
        df = df.Define(den, f"jfl == {pdg}")
        df = df.Define(num, f"{den} && pass_{fl}")
        df = df.Define(f"p_{den}",  f"jp[{den}]").Define(f"c_{den}",  f"jct[{den}]")
        df = df.Define(f"p_{num}",  f"jp[{num}]").Define(f"c_{num}",  f"jct[{num}]")
        h[fl, "den"] = df.Histo2D(model(f"{name}_{fl}_den"), f"p_{den}", f"c_{den}")
        h[fl, "num"] = df.Histo2D(model(f"{name}_{fl}_num"), f"p_{num}", f"c_{num}")
    return h, df.Report()


# ── helpers ──────────────────────────────────────────────────────────────────
def proj(h2, axis):
    """1D projection onto |p| or |cos theta|, summing the other axis (no overflow)."""
    if axis == "p":
        return h2.ProjectionX(f"{h2.GetName()}_p", 1, h2.GetNbinsY(), "e")
    return h2.ProjectionY(f"{h2.GetName()}_abscos", 1, h2.GetNbinsX(), "e")


def ratio_hist(nz, dz, nw, dw, name):
    """Zg/WW efficiency ratio, bin by bin, same dimension as the inputs."""
    ez, ew = nz.Clone(name + "_ez"), nw.Clone(name + "_ew")
    ez.Divide(nz, dz, 1, 1, "B")
    ew.Divide(nw, dw, 1, 1, "B")
    r = ez.Clone(name)
    r.Divide(ew)
    r.SetDirectory(0)
    return r


def cell_mean_ratio(nz, dz, nw, dw):
    """Occupancy-weighted mean of the per-cell Zg/WW ratio over the 2D grid."""
    sw = sr = 0.
    for ix in range(1, nz.GetNbinsX() + 1):
        for iy in range(1, nz.GetNbinsY() + 1):
            a, A, b, B = (h.GetBinContent(ix, iy) for h in (nz, dz, nw, dw))
            if min(a, A, b, B) <= 0:
                continue
            w = A * B / (A + B)
            sw += w
            sr += w * (a / A) / (b / B)
    return sr / sw if sw else float("nan")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    booked = {n: book(n, g, x) for n, (g, x, _, _) in SAMPLES.items()}
    H = {n: {k: v.GetValue() for k, v in h.items()} for n, (h, _) in booked.items()}
    for n, (_, rep) in booked.items():
        print(f"\n--- {n}"); rep.Print()

    print(f"\n{'flav':<6}{'eps WW':>10}{'eps Zg':>10}{'Zg/WW':>10}{'Zg/WW cells':>14}")
    for fl in FLAV_PDG:
        e = {n: H[n][fl, "num"].Integral() / max(H[n][fl, "den"].Integral(), 1) for n in H}
        cm = cell_mean_ratio(H["Zg"][fl, "num"], H["Zg"][fl, "den"],
                             H["WW"][fl, "num"], H["WW"][fl, "den"])
        print(f"{fl:<6}{e['WW']:>10.4f}{e['Zg']:>10.4f}{e['Zg'] / e['WW']:>10.4f}{cm:>14.4f}")

    fout = ROOT.TFile(OUT + ".root", "RECREATE")
    c = ROOT.TCanvas("c", "", 1400, 900)
    c.Print(OUT + ".pdf[")
    keep = []
    for fl in FLAV_PDG:
        for n in H:
            for lvl in ("num", "den"):
                H[n][fl, lvl].Write()

        c.Clear(); c.Divide(2, 2)
        for col, (axis, xlab) in enumerate((("p", "|p| [GeV]"), ("abscos", "|cos#theta|"))):
            pad = c.cd(col + 1); pad.SetGridy()
            leg = ROOT.TLegend(0.15, 0.15, 0.45, 0.32); leg.SetBorderSize(0)
            pr = {n: (proj(H[n][fl, "num"], axis), proj(H[n][fl, "den"], axis)) for n in H}
            for i, (n, (_, _, colr, mk)) in enumerate(SAMPLES.items()):
                te = ROOT.TEfficiency(*pr[n])
                te.SetTitle(f"#varepsilon_{{{fl}}};{xlab};efficiency")
                te.SetLineColor(colr); te.SetMarkerColor(colr); te.SetMarkerStyle(mk)
                te.Draw("AP" if i == 0 else "P same")
                leg.AddEntry(te, n, "lp"); keep.append(te)
            leg.Draw(); keep.append(leg)

            pad = c.cd(col + 3); pad.SetGridy()
            r = ratio_hist(*pr["Zg"], *pr["WW"], f"ratio_{fl}_{axis}")
            r.SetTitle(f"#varepsilon_{{{fl}}}^{{Z#gamma}} / #varepsilon_{{{fl}}}^{{WW}};{xlab};ratio")
            r.SetMinimum(RATIO_RANGE[0]); r.SetMaximum(RATIO_RANGE[1])
            r.SetMarkerStyle(20); r.Draw("E1"); r.Write(); keep.append(r)
        c.Print(OUT + ".pdf")

        c.Clear()
        r2 = ratio_hist(H["Zg"][fl, "num"], H["Zg"][fl, "den"],
                        H["WW"][fl, "num"], H["WW"][fl, "den"], f"ratio2d_{fl}")
        r2.SetTitle(f"#varepsilon_{{{fl}}}^{{Z#gamma}} / #varepsilon_{{{fl}}}^{{WW}};|p| [GeV];|cos#theta|")
        r2.SetMinimum(RATIO_RANGE[0]); r2.SetMaximum(RATIO_RANGE[1])
        c.SetRightMargin(0.15)
        r2.Draw("COLZ0 TEXT"); r2.Write(); keep.append(r2)
        c.Print(OUT + ".pdf")
        c.SetRightMargin(0.1)

    c.Print(OUT + ".pdf]")
    fout.Close()
    print(f"\nwrote {OUT}.root and {OUT}.pdf")


if __name__ == "__main__":
    main()