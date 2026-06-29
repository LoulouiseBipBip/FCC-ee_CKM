/*
  Plot photon pT vs Z-decay quark pT (LHE partons) from skimmer output.

  Usage:
    root -l -q 'plot_photon_pt.C("path/to/unweighted_events.lhe_zgammaWITHOUTcuts.root")'
    root -l -q 'plot_photon_pt.C("path/to/file.root", "photonPt_vsZquarkPt.pdf")'
    // To customize x axis (photonPt) range, supply xmin and xmax:
    root -l -q 'plot_photon_pt.C("file.root", "out.pdf", 0.0, 120.0)'
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

void plot_photon_pt(TString filename = "/afs/cern.ch/user/l/lberiet/FCC-ee_CKM/MG5_aMC_v3_7_1/Zphoton_qqbar/Events/run_240GeV/unweighted_events.lhe_zgammaWITHOUTcuts.root",
                    TString outPdf = "photonPt_vsZquarkPt_240GeV.pdf",
                    double photonPtMin = 20.0,
                    double photonPtMax = 120)
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

    const int nBins = 40;
    // y axis is quark pt, default range like photon pt unless you want to customize etc.
    const double quarkPtMin = 0.0;
    const double quarkPtMax = 120;

    // Leading / subleading Z-decay quark pT vs photon pT
    TH2D* hLead = draw2D(tree, "qJetPtLead:photonPt", "h2_photon_vs_qLead",
                         nBins, photonPtMin, photonPtMax, nBins, quarkPtMin, quarkPtMax,
                         "leading Z quark;photon p_{T} [GeV];leading quark p_{T} [GeV]");
    TH2D* hSub = draw2D(tree, "qJetPtSub:photonPt", "h2_photon_vs_qSub",
                        nBins, photonPtMin, photonPtMax, nBins, quarkPtMin, quarkPtMax,
                        "subleading Z quark;photon p_{T} [GeV];subleading quark p_{T} [GeV]");

    // Combined: both quarks from each event (2 entries per event)
    TH2D* hQ1 = draw2D(tree, "q1Pt:photonPt", "h2_photon_vs_q1",
                        nBins, photonPtMin, photonPtMax, nBins, quarkPtMin, quarkPtMax,
                        "Z decay quarks;photon p_{T} [GeV];quark p_{T} [GeV]");
    TH2D* hQ2 = draw2D(tree, "q2Pt:photonPt", "h2_photon_vs_q2",
                        nBins, photonPtMin, photonPtMax, nBins, quarkPtMin, quarkPtMax,
                        "Z decay quarks;photon p_{T} [GeV];quark p_{T} [GeV]");
    TH2D* hBoth = hQ1 ? (TH2D*)hQ1->Clone("h2_photon_vs_qBoth") : 0;
    if (hBoth && hQ2) hBoth->Add(hQ2);

    if (!hLead || !hSub || !hBoth) {
        std::cerr << "ERROR: failed to fill 2D histograms (need q1Pt/q2Pt/photonPt branches)" << std::endl;
        delete f;
        return;
    }

    TCanvas* c = new TCanvas("c_photon_vs_q", "photon p_{T} vs Z quark p_{T}", 1200, 500);
    c->Divide(3, 1);

    c->cd(1);
    gPad->SetRightMargin(0.15);
    hLead->Draw("COLZ");

    c->cd(2);
    gPad->SetRightMargin(0.15);
    hSub->Draw("COLZ");

    c->cd(3);
    gPad->SetRightMargin(0.15);
    hBoth->Draw("COLZ");

    c->SaveAs(outPdf);
    std::cout << "Saved " << outPdf << std::endl;

    delete f;
}
