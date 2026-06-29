/*
  Plot 1D or 2D distributions from LHE skimmer ROOT output.

  Compare WW and Z+gamma (1D):
    root -l -q 'plot_lhe.C("quarkPt", "ww.root", "zgamma.root", "quark_pt.pdf", 0, 100)'
    root -l -q 'plot_lhe.C("quarkP",  "ww.root", "zgamma.root", "quark_p.pdf",  0, 150)'

  Z+gamma only:
    root -l -q 'plot_lhe.C("photonP", "", "zgamma.root", "photon_p.pdf", 0, 120)'
    root -l -q 'plot_lhe.C("photonP_vs_quarkP", "", "zgamma.root", "out.pdf", 10, 20, 0, 130)'

  Event selection (optional last three args: selVariable, selMin, selMax):
    root -l -q 'plot_lhe.C("quarkP", "ww.root", "zgamma.root", "out.pdf", 0, 140, 0, 130, 200, "qPairP", 50, 120)'
  Omit an axis limit (pass -999) to auto-range from the plotted distribution with 5% margin.
  For WW vs Z+gamma comparisons, the cut applies to Z+gamma only; WW stays unfiltered.

  Aliases: pt/quarkpt, p/quarkp, qpairp/quarkpairp, mzqq, theta/quarktheta, eta/quarketa,
           deltar/quarkdeltar, openangle/quarkopenangle, photonp, photon_vs_quarkp
  Selection branches: any lheEvents branch (e.g. qPairP, q1P, photonP, mZqq, mWhad)
*/

#include "TCanvas.h"
#include "TFile.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TLatex.h"
#include "TLegend.h"
#include "TPad.h"
#include "TString.h"
#include "TStyle.h"
#include "TTree.h"
#include "TBranch.h"

#include <algorithm>
#include <cmath>
#include <iostream>

struct LHEPlotConfig {
    TString histName;
    TString xAxisTitle;
    TString yAxisTitle;
    bool compareWW;
    bool logY;
    bool is2D;
    TString treeDrawExpr;
    TString canvasTitle;
};

bool normalizeVariable(TString variable, LHEPlotConfig& cfg) {
    variable.ToLower();
    if (variable == "quarkpt" || variable == "pt") {
        cfg.histName = "quarkPt";
        cfg.xAxisTitle = "parton p_{T} [GeV]";
        cfg.yAxisTitle = "(1/N) dN/dp_{T} [GeV^{-1}]";
        cfg.compareWW = true;
        cfg.logY = true;
        cfg.is2D = false;
        cfg.canvasTitle = "quark p_{T}";
        return true;
    }
    if (variable == "quarkp" || variable == "p") {
        cfg.histName = "quarkP";
        cfg.xAxisTitle = "parton |p| [GeV]";
        cfg.yAxisTitle = "(1/N) dN/d|p| [GeV^{-1}]";
        cfg.compareWW = true;
        cfg.logY = true;
        cfg.is2D = false;
        cfg.canvasTitle = "quark |p|";
        return true;
    }
    if (variable == "quarkpairp" || variable == "qpairp" || variable == "qqpairp") {
        cfg.histName = "quarkPairP";
        cfg.xAxisTitle = "|#vec{p}_{1}+#vec{p}_{2}| [GeV]";
        cfg.yAxisTitle = "(1/N) dN/d|p| [GeV^{-1}]";
        cfg.compareWW = true;
        cfg.logY = true;
        cfg.is2D = false;
        cfg.canvasTitle = "quark pair |p|";
        return true;
    }
    if (variable == "photonp") {
        cfg.histName = "photonP";
        cfg.xAxisTitle = "photon |p| [GeV]";
        cfg.yAxisTitle = "(1/N) dN/d|p| [GeV^{-1}]";
        cfg.compareWW = false;
        cfg.logY = false;
        cfg.is2D = false;
        cfg.canvasTitle = "photon |p|";
        return true;
    }
    if (variable == "mzqq") {
        cfg.histName = "mZqq";
        cfg.xAxisTitle = "m_{q#bar{q}} [GeV]";
        cfg.yAxisTitle = "(1/N) dm [GeV^{-1}]";
        cfg.compareWW = true;
        cfg.logY = false;
        cfg.is2D = false;
        cfg.canvasTitle = "m_{q#bar{q}}";
        return true;
    }
    if (variable == "quarktheta" || variable == "theta") {
        cfg.histName = "quarkTheta";
        cfg.xAxisTitle = "parton #theta [rad]";
        cfg.yAxisTitle = "(1/N) d#theta [rad^{-1}]";
        cfg.compareWW = true;
        cfg.logY = false;
        cfg.is2D = false;
        cfg.canvasTitle = "quark #theta";
        return true;
    }
    if (variable == "quarketa" || variable == "eta") {
        cfg.histName = "quarkEta";
        cfg.xAxisTitle = "parton #eta";
        cfg.yAxisTitle = "(1/N) d#eta";
        cfg.compareWW = true;
        cfg.logY = false;
        cfg.is2D = false;
        cfg.canvasTitle = "quark #eta";
        return true;
    }
    if (variable == "quarkdeltar" || variable == "deltar" || variable == "qdr") {
        cfg.histName = "quarkDeltaR";
        cfg.xAxisTitle = "#Delta R";
        cfg.yAxisTitle = "(1/N) d#Delta R";
        cfg.compareWW = true;
        cfg.logY = false;
        cfg.is2D = false;
        cfg.canvasTitle = "quark pair #Delta R";
        return true;
    }
    if (variable == "quarkopenangle" || variable == "openangle"
        || variable == "angularseparation" || variable == "dtheta") {
        cfg.histName = "quarkOpenAngle";
        cfg.xAxisTitle = "opening angle [rad]";
        cfg.yAxisTitle = "(1/N) d#theta [rad^{-1}]";
        cfg.compareWW = true;
        cfg.logY = false;
        cfg.is2D = false;
        cfg.canvasTitle = "quark pair opening angle";
        return true;
    }
    if (variable == "photonp_vs_quarkp" || variable == "photon_vs_quarkp") {
        cfg.histName = "";
        cfg.xAxisTitle = "photon |p| [GeV]";
        cfg.yAxisTitle = "quark |p| [GeV]";
        cfg.compareWW = false;
        cfg.logY = false;
        cfg.is2D = true;
        cfg.treeDrawExpr = "q1P+q2P:photonP";
        cfg.canvasTitle = "photon |p| vs quark |p|";
        return true;
    }
    std::cerr << "ERROR: unknown variable '" << variable
              << "'. Choose quarkPt, quarkP, quarkPairP, mZqq, quarkTheta, quarkEta,"
              << " quarkDeltaR, quarkOpenAngle, photonP, or photonP_vs_quarkP." << std::endl;
    return false;
}

bool hasSelection(const TString& selVariable, double selMin, double selMax) {
    TString v = selVariable;
    v.ToLower();
    v.ReplaceAll(" ", "");
    if (v == "" || v == "none" || v == "off") return false;
    return selMin <= selMax;
}

TString normalizeSelectionBranch(TString branch) {
    branch.ReplaceAll(" ", "");
    TString key = branch;
    key.ToLower();
    if (key == "qpairp" || key == "qqpairp" || key == "quarkpairp") return "qPairP";
    if (key == "q1theta") return "q1Theta";
    if (key == "q2theta") return "q2Theta";
    if (key == "q1eta") return "q1Eta";
    if (key == "q2eta") return "q2Eta";
    if (key == "qdeltar" || key == "quarkdeltar") return "qDeltaR";
    if (key == "qopenangle" || key == "quarkopenangle") return "qOpenAngle";
    if (key == "q1p") return "q1P";
    if (key == "q2p") return "q2P";
    if (key == "q1pt") return "q1Pt";
    if (key == "q2pt") return "q2Pt";
    if (key == "qjetplead") return "qJetPLead";
    if (key == "qjetpsub") return "qJetPSub";
    if (key == "qjetptlead") return "qJetPtLead";
    if (key == "qjetptsub") return "qJetPtSub";
    if (key == "photonp") return "photonP";
    if (key == "photonpt") return "photonPt";
    if (key == "photoneta") return "photonEta";
    if (key == "photonabs-costheta" || key == "photonabscostheta") return "photonAbsCosTheta";
    if (key == "mzqq") return "mZqq";
    if (key == "mwhad") return "mWhad";
    if (key == "mwlep") return "mWlep";
    if (key == "leppt") return "lepPt";
    if (key == "lepeta") return "lepEta";
    if (key == "nupt") return "nuPt";
    if (key == "mqq") return "mqq";
    return branch;
}

TString resolvePlotHistName(const LHEPlotConfig& cfg, const char* label) {
    if (cfg.histName == "mZqq" && TString(label) == "ww") return "mWhad";
    return cfg.histName;
}

bool branchExists(TTree* tree, const TString& branch) {
    return tree && tree->GetBranch(branch.Data()) != 0;
}

TString resolveBranchName(TTree* tree, const TString& requested) {
    if (!tree) return normalizeSelectionBranch(requested);

    TString canonical = normalizeSelectionBranch(requested);
    if (branchExists(tree, canonical)) return canonical;
    if (branchExists(tree, requested)) return requested;

    TString reqLower = requested;
    reqLower.ToLower();
    reqLower.ReplaceAll(" ", "");
    TObjArray* branches = tree->GetListOfBranches();
    for (int i = 0; i < branches->GetEntries(); ++i) {
        TBranch* b = (TBranch*)branches->At(i);
        TString name = b->GetName();
        TString nameLower = name;
        nameLower.ToLower();
        if (nameLower == reqLower) return name;
    }
    return canonical;
}

TString buildSelectionCut(const TString& branch, double selMin, double selMax) {
    return Form("%s>=%g && %s<=%g", branch.Data(), selMin, branch.Data(), selMax);
}

long countSelectedEvents(TTree* tree, const TString& cut) {
    if (!tree) return 0;
    if (cut == "") return tree->GetEntries();
    return static_cast<long>(tree->GetEntries(cut.Data()));
}

void drawSelectionAnnotation(const LHEPlotConfig& cfg,
                             const TString& selBranch,
                             double selMin,
                             double selMax,
                             long nWW,
                             long nZG,
                             bool compareWW) {
    TLatex lat;
    lat.SetNDC();
    lat.SetTextSize(0.032);
    lat.SetTextAlign(11);

    double y = 0.88;
    if (compareWW) {
        lat.DrawLatex(0.15, y,
                      Form("Z+#gamma cut: %s #in [%.2g, %.2g]",
                           selBranch.Data(), selMin, selMax));
    } else {
        lat.DrawLatex(0.15, y,
                      Form("%s #in [%.2g, %.2g]", selBranch.Data(), selMin, selMax));
    }
    y -= 0.045;
    if (compareWW) {
        lat.DrawLatex(0.15, y, Form("N_{evt} (WW, unfiltered): %ld", nWW));
        y -= 0.045;
        lat.DrawLatex(0.15, y, Form("N_{evt} (Z+#gamma, selected): %ld", nZG));
    } else {
        lat.DrawLatex(0.15, y, Form("N_{evt}: %ld", nZG));
    }
}

const double kAutoAxis = -999.0;

bool isAutoAxisLimit(double v) {
    return v <= kAutoAxis + 1.0;
}

bool computeHistContentRange(const TH1* h, double& lo, double& hi, double padFrac = 0.05) {
    if (!h) return false;

    int binLow = h->GetNbinsX() + 1;
    int binHigh = 0;
    for (int i = 1; i <= h->GetNbinsX(); ++i) {
        if (h->GetBinContent(i) > 0.0) {
            binLow = std::min(binLow, i);
            binHigh = std::max(binHigh, i);
        }
    }
    if (binLow > binHigh) return false;

    lo = h->GetBinLowEdge(binLow);
    hi = h->GetBinLowEdge(binHigh + 1);
    double pad = padFrac * (hi - lo);
    if (pad <= 0.0) pad = 0.05 * std::max(std::fabs(hi), 1.0);
    lo -= pad;
    hi += pad;
    return true;
}

bool computeTreeBranchRange(TTree* tree, const char* branch, const char* cut,
                            double& lo, double& hi, double padFrac = 0.05) {
    if (!tree || !branch || branch[0] == '\0') return false;

    const char* cutArg = (cut && cut[0] != '\0') ? cut : "";
    double vmin = tree->GetMinimum(branch);
    double vmax = tree->GetMaximum(branch);
    if (vmax <= vmin || (vmin == 0.0 && vmax == 0.0)) {
        tree->Draw(Form("%s>>hAutoRangeTmp", branch), cutArg, "goff");
        TH1* hTmp = (TH1*)gDirectory->Get("hAutoRangeTmp");
        if (!hTmp) return false;
        bool ok = computeHistContentRange(hTmp, lo, hi, padFrac);
        delete hTmp;
        return ok;
    }

    lo = vmin;
    hi = vmax;
    double pad = padFrac * (hi - lo);
    if (pad <= 0.0) pad = 0.05 * std::max(std::fabs(hi), 1.0);
    lo -= pad;
    hi += pad;
    return true;
}

void resolveAxisLimit(double& limit, double autoValue) {
    if (isAutoAxisLimit(limit)) limit = autoValue;
}

void resolveXRangeFromHists(double& xMin, double& xMax, TH1D* h1, TH1D* h2 = 0) {
    if (!isAutoAxisLimit(xMin) && !isAutoAxisLimit(xMax)) return;

    double lo = 0.0;
    double hi = 0.0;
    bool ok = computeHistContentRange(h1, lo, hi);
    if (h2) {
        double lo2 = 0.0;
        double hi2 = 0.0;
        if (computeHistContentRange(h2, lo2, hi2)) {
            if (ok) {
                lo = std::min(lo, lo2);
                hi = std::max(hi, hi2);
            } else {
                lo = lo2;
                hi = hi2;
                ok = true;
            }
        }
    }
    if (!ok) return;

    resolveAxisLimit(xMin, lo);
    resolveAxisLimit(xMax, hi);
}

void resolveYRangeFromHists(double& yMin, double& yMax, TH1D* h1, TH1D* h2 = 0,
                          bool logY = false) {
    if (!isAutoAxisLimit(yMin) && !isAutoAxisLimit(yMax)) return;

    double ymax = h1 ? h1->GetMaximum() : 0.0;
    if (h2) ymax = std::max(ymax, h2->GetMaximum());

    if (logY) {
        if (isAutoAxisLimit(yMin)) yMin = std::max(ymax * 1.0e-4, 1.0e-12);
        if (isAutoAxisLimit(yMax)) yMax = ymax * 10.0;
        return;
    }

    if (isAutoAxisLimit(yMin)) yMin = 0.0;
    if (isAutoAxisLimit(yMax)) {
        double pad = 0.05 * ymax;
        if (pad <= 0.0) pad = 0.01;
        yMax = ymax + pad;
    }
}

void applyXRange(TH1D* h, double xMin, double xMax) {
    if (!h || xMin >= xMax) return;
    // Zoom the display without rebinning: SetBins would remap bin contents and shift the shape.
    h->GetXaxis()->SetRangeUser(xMin, xMax);
}

TH1D* loadNormalizedHist(TFile* f, const char* histName, const char* label) {
    TH1D* h = (TH1D*)f->Get(histName);
    if (!h) {
        std::cerr << "ERROR: histogram '" << histName << "' not found in "
                  << f->GetName() << std::endl;
        return 0;
    }
    TH1D* hDraw = (TH1D*)h->Clone(Form("%s_%s", histName, label));
    hDraw->SetDirectory(0);
    if (hDraw->Integral() <= 0) {
        std::cerr << "ERROR: histogram '" << histName << "' is empty in "
                  << f->GetName() << std::endl;
        delete hDraw;
        return 0;
    }
    hDraw->Scale(1.0 / hDraw->Integral(), "width");
    hDraw->SetLineWidth(2);
    return hDraw;
}

TH1D* fillHistFromTree(TTree* tree, TH1D* hTemplate, const LHEPlotConfig& cfg,
                       const TString& cut, const char* label, long& nEvents) {
    if (!tree || !hTemplate) return 0;

    nEvents = countSelectedEvents(tree, cut);

    TH1D* h = (TH1D*)hTemplate->Clone(Form("%s_%s", cfg.histName.Data(), label));
    h->Reset();

    const char* cutArg = cut == "" ? "" : cut.Data();
    const char* hname = h->GetName();
    if (cfg.histName == "quarkPt") {
        tree->Draw(Form("q1Pt>>%s", hname), cutArg, "goff");
        tree->Draw(Form("q2Pt>>+%s", hname), cutArg, "goff");
    } else if (cfg.histName == "quarkP") {
        tree->Draw(Form("q1P>>%s", hname), cutArg, "goff");
        tree->Draw(Form("q2P>>+%s", hname), cutArg, "goff");
    } else if (cfg.histName == "quarkPairP") {
        tree->Draw(Form("qPairP>>%s", hname), cutArg, "goff");
    } else if (cfg.histName == "mZqq") {
        tree->Draw(Form("mZqq>>%s", hname), cutArg, "goff");
    } else if (cfg.histName == "quarkTheta") {
        tree->Draw(Form("q1Theta>>%s", hname), cutArg, "goff");
        tree->Draw(Form("q2Theta>>+%s", hname), cutArg, "goff");
    } else if (cfg.histName == "quarkEta") {
        tree->Draw(Form("q1Eta>>%s", hname), cutArg, "goff");
        tree->Draw(Form("q2Eta>>+%s", hname), cutArg, "goff");
    } else if (cfg.histName == "quarkDeltaR") {
        tree->Draw(Form("qDeltaR>>%s", hname), cutArg, "goff");
    } else if (cfg.histName == "quarkOpenAngle") {
        tree->Draw(Form("qOpenAngle>>%s", hname), cutArg, "goff");
    } else if (cfg.histName == "photonP") {
        tree->Draw(Form("photonP>>%s", hname), cutArg, "goff");
    } else {
        std::cerr << "ERROR: cannot fill histogram '" << cfg.histName
                  << "' from tree." << std::endl;
        delete h;
        return 0;
    }
    h->SetDirectory(0);

    if (h->Integral() <= 0) {
        std::cerr << "WARNING: histogram '" << cfg.histName
                  << "' is empty after selection (N_{evt}=" << nEvents << ")." << std::endl;
        h->SetLineWidth(2);
        return h;
    }
    h->Scale(1.0 / h->Integral(), "width");
    h->SetLineWidth(2);
    return h;
}

TH1D* loadPlotHist(TFile* f, const LHEPlotConfig& cfg, const char* label,
                   const TString& cut, long& nEvents) {
    const TString histName = resolvePlotHistName(cfg, label);
    if (cut != "") {
        TTree* tree = (TTree*)f->Get("lheEvents");
        TH1D* hTemplate = (TH1D*)f->Get(histName.Data());
        if (!tree) {
            std::cerr << "ERROR: tree 'lheEvents' not found in " << f->GetName() << std::endl;
            return 0;
        }
        if (!hTemplate) {
            std::cerr << "ERROR: histogram '" << histName << "' not found in "
                      << f->GetName() << std::endl;
            return 0;
        }
        return fillHistFromTree(tree, hTemplate, cfg, cut, label, nEvents);
    }
    nEvents = 0;
    TTree* tree = (TTree*)f->Get("lheEvents");
    if (tree) nEvents = tree->GetEntries();
    return loadNormalizedHist(f, histName.Data(), label);
}

TH2D* draw2DFromTree(TTree* tree, const char* expr, const char* name,
                     int nxb, double xlo, double xhi,
                     int nyb, double ylo, double yhi,
                     const char* title,
                     const char* cut = "") {
    tree->Draw(Form("%s>>%s(%d,%g,%g,%d,%g,%g)", expr, name, nxb, xlo, xhi, nyb, ylo, yhi),
               cut, "goff");
    TH2D* h = (TH2D*)gDirectory->Get(name);
    if (!h) return 0;
    h->SetDirectory(0);
    h->SetTitle(title);
    return h;
}

bool plot1DCompare(const LHEPlotConfig& cfg,
                   TFile* fWW, TFile* fZG,
                   const TString& outPdf,
                   double xMin, double xMax,
                   double yMin, double yMax,
                   const TString& cut,
                   const TString& selBranch,
                   double selMin, double selMax) {
    long nWW = 0;
    long nZG = 0;
    TH1D* hWW = loadPlotHist(fWW, cfg, "ww", "", nWW);
    TH1D* hZG = loadPlotHist(fZG, cfg, "zgamma", cut, nZG);
    if (!hWW || !hZG) return false;

    resolveXRangeFromHists(xMin, xMax, hWW, hZG);

    hWW->SetLineColor(kRed + 1);
    hZG->SetLineColor(kBlue + 1);
    hWW->SetTitle(Form(";%s;%s", cfg.xAxisTitle.Data(), cfg.yAxisTitle.Data()));

    applyXRange(hWW, xMin, xMax);
    applyXRange(hZG, xMin, xMax);

    resolveYRangeFromHists(yMin, yMax, hWW, hZG, cfg.logY);
    hWW->SetMinimum(yMin);
    hWW->SetMaximum(yMax);
    hZG->SetMinimum(yMin);
    hZG->SetMaximum(yMax);

    TCanvas* c = new TCanvas(Form("c_%s", cfg.histName.Data()), cfg.canvasTitle, 800, 600);
    if (cfg.logY) c->SetLogy();
    gStyle->SetOptStat(0);

    hWW->Draw("HIST");
    hZG->Draw("HIST SAME");

    TLegend* leg = new TLegend(0.66, 0.83, 0.84, 0.89); // smaller box
    leg->SetTextSize(0.025); // smaller text
    leg->AddEntry(hWW, "W #rightarrow q#bar{q}' (WW)", "l");
    leg->AddEntry(hZG, "Z #rightarrow q#bar{q} (Z+#gamma)", "l");
    leg->Draw();


    if (cut != "") {
        drawSelectionAnnotation(cfg, selBranch, selMin, selMax, nWW, nZG, true);
    }

    c->SaveAs(outPdf);
    std::cout << "Saved " << outPdf << std::endl;
    return true;
}

bool plot1DSingle(const LHEPlotConfig& cfg,
                  TFile* fZG,
                  const TString& outPdf,
                  double xMin, double xMax,
                  double yMin, double yMax,
                  const TString& cut,
                  const TString& selBranch,
                  double selMin, double selMax) {
    long nEvents = 0;
    TH1D* h = loadPlotHist(fZG, cfg, "zgamma", cut, nEvents);
    if (!h) return false;

    resolveXRangeFromHists(xMin, xMax, h);

    h->SetLineColor(kBlue + 1);
    h->SetTitle(Form(";%s;%s", cfg.xAxisTitle.Data(), cfg.yAxisTitle.Data()));
    applyXRange(h, xMin, xMax);

    resolveYRangeFromHists(yMin, yMax, h, 0, cfg.logY);
    h->SetMinimum(yMin);
    h->SetMaximum(yMax);

    TCanvas* c = new TCanvas(Form("c_%s", cfg.histName.Data()), cfg.canvasTitle, 800, 600);
    if (cfg.logY) c->SetLogy();
    gStyle->SetOptStat(0);

    h->Draw("HIST");

    if (cut != "") {
        drawSelectionAnnotation(cfg, selBranch, selMin, selMax, 0, nEvents, false);
    }

    c->SaveAs(outPdf);
    std::cout << "Saved " << outPdf << std::endl;
    return true;
}

bool plot2DPhotonVsQuark(TFile* fZG,
                         const LHEPlotConfig& cfg,
                         const TString& outPdf,
                         double xMin, double xMax,
                         double yMin, double yMax,
                         int nBins,
                         const TString& cut,
                         const TString& selBranch,
                         double selMin, double selMax) {
    TTree* tree = (TTree*)fZG->Get("lheEvents");
    if (!tree) {
        std::cerr << "ERROR: tree 'lheEvents' not found in " << fZG->GetName() << std::endl;
        return false;
    }

    const char* cutArg = cut == "" ? "" : cut.Data();
    if (isAutoAxisLimit(xMin) || isAutoAxisLimit(xMax)) {
        double lo = 0.0;
        double hi = 0.0;
        if (computeTreeBranchRange(tree, "photonP", cutArg, lo, hi)) {
            resolveAxisLimit(xMin, lo);
            resolveAxisLimit(xMax, hi);
        }
    }
    if (isAutoAxisLimit(yMin) || isAutoAxisLimit(yMax)) {
        double lo1 = 0.0;
        double hi1 = 0.0;
        double lo2 = 0.0;
        double hi2 = 0.0;
        bool ok1 = computeTreeBranchRange(tree, "q1P", cutArg, lo1, hi1);
        bool ok2 = computeTreeBranchRange(tree, "q2P", cutArg, lo2, hi2);
        if (ok1 || ok2) {
            double lo = ok1 ? lo1 : lo2;
            double hi = ok1 ? hi1 : hi2;
            if (ok1 && ok2) {
                lo = std::min(lo1, lo2);
                hi = std::max(hi1, hi2);
            }
            resolveAxisLimit(yMin, lo);
            resolveAxisLimit(yMax, hi);
        }
    }

    TH2D* hQ1 = draw2DFromTree(tree, "q1P:photonP", "h2_photonP_vs_q1P",
                                nBins, xMin, xMax, nBins, yMin, yMax,
                                Form("Z #rightarrow q#bar{q};%s;%s",
                                     cfg.xAxisTitle.Data(), cfg.yAxisTitle.Data()),
                                cutArg);
    TH2D* hQ2 = draw2DFromTree(tree, "q2P:photonP", "h2_photonP_vs_q2P",
                                nBins, xMin, xMax, nBins, yMin, yMax,
                                Form("Z #rightarrow q#bar{q};%s;%s",
                                     cfg.xAxisTitle.Data(), cfg.yAxisTitle.Data()),
                                cutArg);
    TH2D* hBoth = hQ1 ? (TH2D*)hQ1->Clone("h2_photonP_vs_qBoth") : 0;
    if (hBoth && hQ2) hBoth->Add(hQ2);

    if (!hBoth) {
        std::cerr << "ERROR: failed to fill 2D histogram (need photonP, q1P, q2P branches)" << std::endl;
        return false;
    }

    TCanvas* c = new TCanvas("c_photonP_vs_quarkP", cfg.canvasTitle, 800, 600);
    gStyle->SetOptStat(0);
    gPad->SetRightMargin(0.15);
    hBoth->Draw("COLZ");

    if (cut != "") {
        long nEvents = countSelectedEvents(tree, cut);
        drawSelectionAnnotation(cfg, selBranch, selMin, selMax, 0, nEvents, false);
    }

    c->SaveAs(outPdf);
    std::cout << "Saved " << outPdf << std::endl;
    return true;
}

void plot_lhe(TString variable,
              TString rootFileWW = "",
              TString rootFileZG = "",
              TString outPdf = "",
              double xMin = kAutoAxis,
              double xMax = kAutoAxis,
              double yMin = kAutoAxis,
              double yMax = kAutoAxis,
              int nBins2D = 200,
              TString selVariable = "",
              double selMin = -1.0,
              double selMax = -1.0) {
    LHEPlotConfig cfg;
    if (!normalizeVariable(variable, cfg)) return;

    if (outPdf == "") {
        outPdf = Form("%s.pdf", variable.Data());
    }

    TString cut = "";
    TString selBranch = "";
    if (hasSelection(selVariable, selMin, selMax)) {
        selBranch = normalizeSelectionBranch(selVariable);
    }

    auto applySelection = [&](TTree* refTree) -> bool {
        if (cut != "" || selBranch == "") return true;
        selBranch = resolveBranchName(refTree, selVariable);
        if (!branchExists(refTree, selBranch)) {
            std::cerr << "ERROR: selection branch '" << selVariable
                      << "' (resolved as '" << selBranch
                      << "') not found in lheEvents tree." << std::endl;
            return false;
        }
        cut = buildSelectionCut(selBranch, selMin, selMax);
        std::cout << "Event selection (Z+gamma only): " << cut << std::endl;
        return true;
    };

    if (cfg.is2D) {
        if (rootFileZG == "" && rootFileWW != "") {
            rootFileZG = rootFileWW;
            rootFileWW = "";
        }
        TFile* fZG = TFile::Open(rootFileZG);
        if (!fZG || fZG->IsZombie()) {
            std::cerr << "ERROR: cannot open " << rootFileZG << std::endl;
            return;
        }
        if (!applySelection((TTree*)fZG->Get("lheEvents"))) {
            delete fZG;
            return;
        }
        plot2DPhotonVsQuark(fZG, cfg, outPdf, xMin, xMax, yMin, yMax, nBins2D,
                            cut, selBranch, selMin, selMax);
        delete fZG;
        return;
    }

    if (!cfg.compareWW) {
        if (rootFileZG == "" && rootFileWW != "") {
            rootFileZG = rootFileWW;
        }
        TFile* fZG = TFile::Open(rootFileZG);
        if (!fZG || fZG->IsZombie()) {
            std::cerr << "ERROR: cannot open " << rootFileZG << std::endl;
            return;
        }
        if (!applySelection((TTree*)fZG->Get("lheEvents"))) {
            delete fZG;
            return;
        }
        plot1DSingle(cfg, fZG, outPdf, xMin, xMax, yMin, yMax, cut, selBranch, selMin, selMax);
        delete fZG;
        return;
    }

    if (rootFileWW == "" || rootFileZG == "") {
        std::cerr << "ERROR: quarkPt/quarkP comparison requires WW and Z+gamma ROOT files." << std::endl;
        return;
    }

    TFile* fWW = TFile::Open(rootFileWW);
    if (!fWW || fWW->IsZombie()) {
        std::cerr << "ERROR: cannot open " << rootFileWW << std::endl;
        return;
    }
    TFile* fZG = TFile::Open(rootFileZG);
    if (!fZG || fZG->IsZombie()) {
        std::cerr << "ERROR: cannot open " << rootFileZG << std::endl;
        delete fWW;
        return;
    }

    TTree* tZG = (TTree*)fZG->Get("lheEvents");
    if (!applySelection(tZG)) {
        delete fWW;
        delete fZG;
        return;
    }

    plot1DCompare(cfg, fWW, fZG, outPdf, xMin, xMax, yMin, yMax, cut, selBranch, selMin, selMax);
    delete fWW;
    delete fZG;
}
