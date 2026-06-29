/*
  Plot photon |p| vs qq invariant mass m_{Zqq} from skimmer output.

  Usage:
    root -l -q 'plot_photon_p_massqq.C("path/to/unweighted_events.lhe_zgammaWITHOUTcuts.root")'
    root -l -q 'plot_photon_p_massqq.C("path/to/file.root", "photonP_vs_mZqq.pdf")'
    root -l -q 'plot_photon_p_massqq.C("file.root", "out.pdf", 80.0, 120.0, 40.0, 120.0, 200)'
*/

#include "TCanvas.h"
#include "TFile.h"
#include "TH2D.h"
#include "TPad.h"
#include "TTree.h"
#include "TString.h"

#include <iostream>

TH2D* draw2D(TTree* tree, const char* expr, const char* name,
             int nxb, double xlo, double xhi,
             int nyb, double ylo, double yhi,
             const char* title) {
    tree->Draw(Form("%s>>%s(%d,%g,%g,%d,%g,%g)", expr, name, nxb, xlo, xhi, nyb, ylo, yhi),
               "", "goff");
    TH2D* h = (TH2D*)gDirectory->Get(name);
    if (!h) return 0;
    h->SetDirectory(0);
    h->SetTitle(title);
    return h;
}

void plot_photon_p_massqq(TString filename = "/afs/cern.ch/user/l/lberiet/FCC-ee_CKM/MG5_aMC_v3_7_1/Zphoton_qqbar/Events/run_240GeV/unweighted_events.lhe_zgammaWITHOUTcuts.root",
                          TString outPdf = "photonP_vs_mZqq_240GeV.pdf",
                          double photonPMin = 80.0,
                          double photonPMax = 120.0,
                          double mZqqMin = 40.0,
                          double mZqqMax = 120.0,
                          int nBins = 200)
{
    TFile* f = TFile::Open(filename);
    if (!f || f->IsZombie()) {
        std::cerr << "ERROR: cannot open " << filename << std::endl;
        return;
    }

    TTree* tree = (TTree*)f->Get("lheEvents");
    if (!tree) {
        std::cerr << "ERROR: tree 'lheEvents' not found in " << filename << std::endl;
        delete f;
        return;
    }

    // ROOT Draw "y:x" puts x on the horizontal axis and y on the vertical axis.
    TH2D* h = draw2D(tree, "mZqq:photonP", "h2_photonP_vs_mZqq",
                     nBins, photonPMin, photonPMax, nBins, mZqqMin, mZqqMax,
                     "Z #rightarrow q#bar{q};photon |p| [GeV];m_{q#bar{q}} [GeV]");

    if (!h) {
        std::cerr << "ERROR: failed to fill 2D histogram (need photonP and mZqq branches)" << std::endl;
        delete f;
        return;
    }

    TCanvas* c = new TCanvas("c_photonP_vs_mZqq", "photon |p| vs m_{q#bar{q}}", 800, 600);
    gPad->SetRightMargin(0.15);
    h->Draw("COLZ");

    c->SaveAs(outPdf);
    std::cout << "Saved " << outPdf << std::endl;

    delete f;
}
