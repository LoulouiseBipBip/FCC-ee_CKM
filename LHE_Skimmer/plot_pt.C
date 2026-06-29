/*
  Backward-compatible wrapper. Prefer plot_lhe.C("quarkPt", ...).
*/
#include "plot_lhe.C"

void plot_pt(TString rootFile,
             TString compareFile = "",
             TString outPdf = "quark_pt.pdf",
             double xMin = 0.0,
             double xMax = 200.0) {
    plot_lhe("quarkPt", rootFile, compareFile, outPdf, xMin, xMax);
}
