/*
  Backward-compatible wrapper. Prefer plot_lhe.C("photonP_vs_quarkP", ...).
*/
#include "plot_lhe.C"

void plot_photon_p(TString filename,
                   TString outPdf = "photonP_vsZquarkP.pdf",
                   double photonPMin = 80.0,
                   double photonPMax = 120.0,
                   double quarkPMin = 0.0,
                   double quarkPMax = 130.0,
                   int nBins = 200) {
    plot_lhe("photonP_vs_quarkP", "", filename, outPdf,
             photonPMin, photonPMax, quarkPMin, quarkPMax, nBins);
}
