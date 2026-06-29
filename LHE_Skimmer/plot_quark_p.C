/*
  Backward-compatible wrapper. Prefer plot_lhe.C("quarkP", ...).
*/
#include "plot_lhe.C"

void plot_quark_p(TString rootFile,
                  TString compareFile = "",
                  TString outPdf = "quark_p.pdf",
                  double xMin = 0.0,
                  double xMax = 200.0) {
    plot_lhe("quarkP", rootFile, compareFile, outPdf, xMin, xMax);
}
