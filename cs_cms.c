/*
  Plot MadGraph integrated cross section vs sqrt(s) for WW semileptonic runs.

  Reads the "Integrated weight (pb)" from each unweighted_events.lhe header
  under MG5_aMC_v3_7_1/WW_semileptonic/Events/run_<E>GeV/.

  Usage:
    root -l -q 'cs_cms.c()'
    root -l -q 'cs_cms.c("MG5_aMC_v3_7_1/WW_semileptonic/Events", "WW_xsec_vs_energy.pdf")'
*/

#include "TCanvas.h"
#include "TGraph.h"
#include "TLatex.h"
#include "TString.h"
#include "TSystem.h"
#include "TSystemDirectory.h"
#include "TList.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

struct EnergyPoint {
    double energy;
    double xsec;
};

bool parseEnergyFromRunName(const TString& name, double& energy) {
    // run_178GeV -> 178
    if (!name.BeginsWith("run_") || !name.EndsWith("GeV")) return false;
    TString num = name;
    num.ReplaceAll("run_", "");
    num.ReplaceAll("GeV", "");
    energy = num.Atof();
    return energy > 0;
}

bool parseCrossSection(const TString& lhePath, double& xsec) {
    std::ifstream in(lhePath.Data());
    if (!in.is_open()) return false;

    std::string line;
    while (std::getline(in, line)) {
        if (line.find("Integrated weight (pb)") != std::string::npos) {
            const auto colon = line.rfind(':');
            if (colon == std::string::npos) return false;
            std::istringstream iss(line.substr(colon + 1));
            iss >> xsec;
            return !iss.fail() && xsec > 0;
        }
        if (line.find("</header>") != std::string::npos) break;
    }
    return false;
}

TString findLheFile(const TString& runDir) {
    TString plain = runDir + "/unweighted_events.lhe";
    if (!gSystem->AccessPathName(plain.Data())) return plain;

    TString gz = runDir + "/unweighted_events.lhe.gz";
    if (!gSystem->AccessPathName(gz.Data())) return gz;

    return "";
}

void cs_cms(TString eventsDir = "MG5_aMC_v3_7_1/WW_semileptonic/Events",
            TString outPdf = "WW_semileptonic_xsec_vs_energy.pdf")
{
    if (gSystem->AccessPathName(eventsDir.Data())) {
        std::cerr << "ERROR: events directory not found: " << eventsDir << std::endl;
        return;
    }

    TSystemDirectory dir("Events", eventsDir);
    TList* entries = dir.GetListOfFiles();
    if (!entries) {
        std::cerr << "ERROR: cannot list " << eventsDir << std::endl;
        return;
    }

    std::vector<EnergyPoint> points;
    entries->Sort();

    TIter next(entries);
    while (TObject* obj = next()) {
        const TString name = obj->GetName();
        if (name == "." || name == "..") continue;

        double energy = 0;
        if (!parseEnergyFromRunName(name, energy)) continue;

        const TString runDir = eventsDir + "/" + name;
        const TString lhePath = findLheFile(runDir);
        if (lhePath == "") {
            std::cerr << "WARNING: no LHE file in " << runDir << std::endl;
            continue;
        }

        double xsec = 0;
        if (!parseCrossSection(lhePath, xsec)) {
            std::cerr << "WARNING: could not parse cross section from " << lhePath << std::endl;
            continue;
        }

        points.push_back({energy, xsec});
        std::cout << Form("  sqrt(s) = %5.0f GeV  ->  sigma = %.6f pb  (%s)",
                          energy, xsec, lhePath.Data()) << std::endl;
    }

    if (points.empty()) {
        std::cerr << "ERROR: no valid run points found in " << eventsDir << std::endl;
        return;
    }

    std::sort(points.begin(), points.end(),
              [](const EnergyPoint& a, const EnergyPoint& b) { return a.energy < b.energy; });

    TGraph* gr = new TGraph(points.size());
    gr->SetName("WW_xsec_vs_energy");
    for (size_t i = 0; i < points.size(); ++i) {
        gr->SetPoint(i, points[i].energy, points[i].xsec);
    }

    gr->SetTitle("WW semileptonic;#sqrt{s} [GeV];#sigma [pb]");
    gr->SetMarkerStyle(20);
    gr->SetMarkerSize(0.9);
    gr->SetMarkerColor(kBlue + 1);
    gr->SetLineColor(kBlue + 1);
    gr->SetLineWidth(2);

    TCanvas* c = new TCanvas("c_xsec", "WW cross section vs energy", 800, 600);
    c->SetGrid();
    gr->Draw("ALP");

    // Set x-axis limits
    double xmin = 155;
    double xmax = 290;
    gr->GetXaxis()->SetLimits(xmin, xmax);

    TLatex lat;
    lat.SetNDC();
    lat.SetTextSize(0.035);
    lat.DrawLatex(0.15, 0.92, "e^{+}e^{-} #rightarrow W^{+}W^{-} #rightarrow l#nu q#bar{q}");

    c->SaveAs(outPdf);
    std::cout << "Saved plot to " << outPdf << std::endl;
}
