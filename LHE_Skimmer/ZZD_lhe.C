/*
  Macro to translate MadGraph5_aMC@NLO LHE events into a .root file
  (tree + histograms). Supports:
    - WW semileptonic  (e+ e- > W+ W- > l nu + q qbar)
    - Z+photon hadronic (e+ e- > Z gamma, Z > q qbar)
    - legacy H->ZZ->4l (4e, 4mu, 2e2mu)

  Example syntax:
    root -l -q 'ZZD_lhe.C("MG5_aMC_v3_7_1/WW_semileptonic/Events/run_163GeV/unweighted_events.lhe", 10000, "WW")'
    root -l -q 'ZZD_lhe.C("MG5_aMC_v3_7_1/Zphoton_qqbar/Events/run_163GeV/unweighted_events.lhe", 10000, "Zgamma")'
*/

// C++
#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>

// ROOT
#include "Riostream.h"
#include "TTree.h"
#include "TFile.h"
#include "TLorentzVector.h"
#include "TH1.h"
#include "TString.h"

// HZZ4l angles (legacy 4l mode only)
#include "computeAngles.cc"

using namespace std;

const int LEPSIZE = 4;
const double PI = 3.1415926;
const double mZ_PDG = 91.188; // GeV

struct LHEParticle {
    int id;
    int status;
    TLorentzVector p4;
};

bool isChargedLepton(int id) {
    const int absId = abs(id);
    return absId == 11 || absId == 13 || absId == 15;
}

bool isNeutrino(int id) {
    const int absId = abs(id);
    return absId == 12 || absId == 14 || absId == 16;
}

bool isQuark(int id) {
    const int absId = abs(id);
    return absId >= 1 && absId <= 6;
}

bool isPhoton(int id) {
    return id == 22;
}

TString normalizeProcess(TString p) {
    p.ToLower();
    if (p == "ww" || p == "ww_mu" || p == "ww_e") return "ww";
    if (p == "zgamma" || p == "zg" || p == "zzgamma") return "zgamma";
    if (p == "zz4l") return "zz4l";
    return p;
}

TString detectProcess(const vector<LHEParticle>& fs) {
    int nLeptons = 0;
    int nNeutrinos = 0;
    int nQuarks = 0;
    int nPhotons = 0;
    int sumAbsLeptonId = 0;

    for (const auto& p : fs) {
        if (isChargedLepton(p.id)) {
            nLeptons++;
            sumAbsLeptonId += abs(p.id);
        } else if (isNeutrino(p.id)) {
            nNeutrinos++;
        } else if (isQuark(p.id)) {
            nQuarks++;
        } else if (isPhoton(p.id)) {
            nPhotons++;
        }
    }

    if (nLeptons == 4 && (sumAbsLeptonId == 44 || sumAbsLeptonId == 48 || sumAbsLeptonId == 52)) {
        return "zz4l";
    }
    if (nLeptons == 1 && nNeutrinos == 1 && nQuarks == 2) {
        return "ww";
    }
    if (nPhotons == 1 && nQuarks == 2) {
        return "zgamma";
    }
    return "unknown";
}

bool readMG5Event(ifstream& in, vector<LHEParticle>& particles, double& wt, int& idprup) {
    string line;
    while (getline(in, line)) {
        if (!in.good()) return false;
        if (line == "<event>") break;
    }
    if (!in.good()) return false;

    if (!getline(in, line) || !in.good()) return false;

    int nup;
    double scalup, aqedup, aqcdup;
    istringstream header(line);
    header >> nup >> idprup >> wt >> scalup >> aqedup >> aqcdup;
    if (nup <= 0) return false;

    particles.clear();
    particles.reserve(nup);

    for (int i = 0; i < nup; ++i) {
        if (!getline(in, line) || !in.good()) return false;
        LHEParticle part;
        int moth1, moth2, col1, col2;
        double mass, tau, spin;
        double px, py, pz, e;
        istringstream pline(line);
        pline >> part.id >> part.status >> moth1 >> moth2 >> col1 >> col2
              >> px >> py >> pz >> e >> mass >> tau >> spin;
        part.p4.SetPxPyPzE(px, py, pz, e);
        particles.push_back(part);
    }
    return true;
}

vector<LHEParticle> finalStateParticles(const vector<LHEParticle>& particles) {
    vector<LHEParticle> fs;
    for (const auto& p : particles) {
        if (p.status == 1) fs.push_back(p);
    }
    return fs;
}

void fillQuarkJetObservables(const LHEParticle& q1, const LHEParticle& q2,
                             double& q1Pt, double& q2Pt,
                             double& qJetPtLead, double& qJetPtSub,
                             double& q1P, double& q2P,
                             double& qJetPLead, double& qJetPSub,
                             double& qPairP,
                             double& q1Theta, double& q2Theta,
                             double& q1Eta, double& q2Eta,
                             double& qDeltaR, double& qOpenAngle,
                             TH1D* h_quarkPt, TH1D* h_quarkPtLead, TH1D* h_quarkPtSub,
                             TH1D* h_quarkP, TH1D* h_quarkPLead, TH1D* h_quarkPSub,
                             TH1D* h_quarkPairP,
                             TH1D* h_quarkTheta, TH1D* h_quarkCosTheta, TH1D* h_quarkEta,
                             TH1D* h_quarkDeltaR, TH1D* h_quarkOpenAngle,
                             double wt) {
    q1Pt = q1.p4.Pt();
    q2Pt = q2.p4.Pt();
    qJetPtLead = max(q1Pt, q2Pt);
    qJetPtSub = min(q1Pt, q2Pt);
    h_quarkPt->Fill(q1Pt, wt);
    h_quarkPt->Fill(q2Pt, wt);
    h_quarkPtLead->Fill(qJetPtLead, wt);
    h_quarkPtSub->Fill(qJetPtSub, wt);

    q1P = q1.p4.P();
    q2P = q2.p4.P();
    qJetPLead = max(q1P, q2P);
    qJetPSub = min(q1P, q2P);
    qPairP = (q1.p4 + q2.p4).P();
    h_quarkP->Fill(q1P, wt);
    h_quarkP->Fill(q2P, wt);
    h_quarkPLead->Fill(qJetPLead, wt);
    h_quarkPSub->Fill(qJetPSub, wt);
    h_quarkPairP->Fill(qPairP, wt);

    q1Theta = q1.p4.Theta();
    q2Theta = q2.p4.Theta();
    q1Eta = q1.p4.Eta();
    q2Eta = q2.p4.Eta();
    h_quarkTheta->Fill(q1Theta, wt);
    h_quarkTheta->Fill(q2Theta, wt);
    h_quarkCosTheta->Fill(q1.p4.CosTheta(), wt);
    h_quarkCosTheta->Fill(q2.p4.CosTheta(), wt);
    h_quarkEta->Fill(q1Eta, wt);
    h_quarkEta->Fill(q2Eta, wt);

    qDeltaR = q1.p4.DeltaR(q2.p4);
    h_quarkDeltaR->Fill(qDeltaR, wt);

    qOpenAngle = q1.p4.Vect().Angle(q2.p4.Vect());
    h_quarkOpenAngle->Fill(qOpenAngle, wt);
}

void ZZD_lhe(TString lheFile = "MG5_aMC_v3_7_1/WW_semileptonic/Events/run_163GeV/unweighted_events.lhe",
             long int Nevents = 10000,
             TString process = "auto",
             bool bApplyCuts = false,
             unsigned int kDebugLevel = 1) {
    ifstream in;
    in.open(lheFile.Data(), ios::in);
    if (!in.good()) {
        cerr << "ERROR: cannot open LHE file " << lheFile << endl;
        return;
    }

    TString processLabel = process;
    processLabel.ToLower();
    TString processMode = normalizeProcess(process);

    string outName(lheFile.Data());
    if (processLabel == "auto") {
        outName.append("_auto");
    } else {
        outName.append("_");
        outName.append(processLabel.Data());
    }
    outName.append(bApplyCuts ? "WITH" : "WITHOUT");
    outName.append("cuts.root");

    cout << " preparing root file: " << outName << endl;
    cout << " running for: " << Nevents << " events." << endl;
    cout << " requested process mode: " << process << endl;

    TFile* f = new TFile(outName.c_str(), "RECREATE");

    // Shared branches
    double wt = 1.0;
    int idprup = 0;
    int nFinalState = 0;

    // WW branches
    int lepId = 0, nuId = 0, q1Id = 0, q2Id = 0;
    double mWlep = 0.0, mWhad = 0.0, lepPt = 0.0, lepEta = 0.0, nuPt = 0.0, mqq = 0.0;

    // WW / Zgamma shared quark-jet branches (LHE parton pT and |p|)
    double q1Pt = 0.0, q2Pt = 0.0, qJetPtLead = 0.0, qJetPtSub = 0.0;
    double q1P = 0.0, q2P = 0.0, qJetPLead = 0.0, qJetPSub = 0.0, qPairP = 0.0;
    double q1Theta = 0.0, q2Theta = 0.0, q1Eta = 0.0, q2Eta = 0.0, qDeltaR = 0.0, qOpenAngle = 0.0;

    // Zgamma branches
    double mZqq = 0.0, photonE = 0.0, photonPt = 0.0, photonP = 0.0, photonEta = 0.0, photonAbsCosTheta = 0.0;

    // Legacy 4l branches
    int id1 = 0, id2 = 0, id3 = 0, id4 = 0;
    double pt3 = 0.0, pt4 = 0.0, pt5 = 0.0, pt6 = 0.0;
    double eta3 = 0.0, eta4 = 0.0, eta5 = 0.0, eta6 = 0.0;
    double phi3 = 0.0, phi4 = 0.0, phi5 = 0.0, phi6 = 0.0;
    double pto1 = 0.0, pto2 = 0.0, pto3 = 0.0, pto4 = 0.0;
    double mass4l = 0.0, pT4l = 0.0, eta4l = 0.0, rapidity4l = 0.0;
    float _costheta1 = 0.0, _costheta2 = 0.0, _costhetastar = 0.0, _phi = 0.0, _phi1 = 0.0;
    double costheta1 = 0.0, costheta2 = 0.0, costhetastar = 0.0, phi = 0.0, phi1 = 0.0;
    double massZ1 = 0.0, massZ2 = 0.0;

    TH1D* h_mWlep = new TH1D("mWlep", "mWlep", 100, 50.0, 200.0);
    TH1D* h_mWhad = new TH1D("mWhad", "mWhad", 100, 50.0, 200.0);
    TH1D* h_lepPt = new TH1D("lepPt", "lepPt", 100, 0.0, 200.0);
    TH1D* h_nuPt = new TH1D("nuPt", "nuPt", 100, 0.0, 200.0);
    TH1D* h_quarkPt = new TH1D("quarkPt", "quark p_{T} (LHE partons);p_{T} [GeV];entries", 50, 0.0, 200.0);
    TH1D* h_quarkPtLead = new TH1D("quarkPtLead", "leading quark p_{T};p_{T} [GeV];entries", 50, 0.0, 200.0);
    TH1D* h_quarkPtSub = new TH1D("quarkPtSub", "subleading quark p_{T};p_{T} [GeV];entries", 50, 0.0, 200.0);
    TH1D* h_quarkP = new TH1D("quarkP", "quark |p| (LHE partons);|p| [GeV];entries", 50, 0.0, 300.0);
    TH1D* h_quarkPLead = new TH1D("quarkPLead", "leading quark |p|;|p| [GeV];entries", 50, 0.0, 200.0);
    TH1D* h_quarkPSub = new TH1D("quarkPSub", "subleading quark |p|;|p| [GeV];entries", 50, 0.0, 200.0);
    TH1D* h_quarkPairP = new TH1D("quarkPairP", "qq pair |#vec{p}_{1}+#vec{p}_{2}|;|p| [GeV];entries", 50, 0.0, 200.0);
    TH1D* h_quarkTheta = new TH1D("quarkTheta", "quark #theta (LHE partons);#theta [rad];entries", 100, -4, 4);
    TH1D* h_quarkCosTheta = new TH1D("quarkCosTheta", "quark cos#theta (LHE partons);cos#theta;entries", 50, -1.0, 1.0);
    TH1D* h_quarkEta = new TH1D("quarkEta", "quark #eta (LHE partons);#eta;entries", 50, -5.0, 5.0);
    TH1D* h_quarkDeltaR = new TH1D("quarkDeltaR", "quark pair #Delta R;#Delta R;entries", 50, 0.0, 6.0);
    TH1D* h_quarkOpenAngle = new TH1D("quarkOpenAngle", "quark pair opening angle;#Delta#theta [rad];entries", 50, 0.0, PI);

    TH1D* h_mZqq = new TH1D("mZqq", "mZqq", 100, 40.0, 200.0);
    TH1D* h_photonE = new TH1D("photonE", "photonE", 100, 0.0, 200.0);
    TH1D* h_photonP = new TH1D("photonP", "photon |p|;|p| [GeV];entries", 50, 70.0, 130.0);
    TH1D* h_photonAbsCosTheta = new TH1D("photonAbsCosTheta", "photonAbsCosTheta", 50, 0.0, 1.0);

    TH1D* mass4l_hist_tchan = new TH1D("mass4l_tchan", "mass4l_tchan", 200, 0.0, 1000.0);
    TH1D* pT4l_hist_tchan = new TH1D("pT4l_tchan", "pT4l_tchan", 200, 0.0, 200.0);
    TH1D* massZ1_hist_tchan = new TH1D("massZ1_tchan", "massZ1_tchan", 100, 20.0, 120.0);
    TH1D* massZ2_hist_tchan = new TH1D("massZ2_tchan", "massZ2_tchan", 100, 0.0, 50.0);

    TTree* TT = new TTree("lheEvents", "MadGraph5 LHE events");
    TT->Branch("weight", &wt, "weight/D");
    TT->Branch("idprup", &idprup, "idprup/I");
    TT->Branch("nFinalState", &nFinalState, "nFinalState/I");

    TT->Branch("lepId", &lepId, "lepId/I");
    TT->Branch("nuId", &nuId, "nuId/I");
    TT->Branch("q1Id", &q1Id, "q1Id/I");
    TT->Branch("q2Id", &q2Id, "q2Id/I");
    TT->Branch("mWlep", &mWlep, "mWlep/D");
    TT->Branch("mWhad", &mWhad, "mWhad/D");
    TT->Branch("lepPt", &lepPt, "lepPt/D");
    TT->Branch("lepEta", &lepEta, "lepEta/D");
    TT->Branch("nuPt", &nuPt, "nuPt/D");
    TT->Branch("mqq", &mqq, "mqq/D");
    TT->Branch("q1Pt", &q1Pt, "q1Pt/D");
    TT->Branch("q2Pt", &q2Pt, "q2Pt/D");
    TT->Branch("qJetPtLead", &qJetPtLead, "qJetPtLead/D");
    TT->Branch("qJetPtSub", &qJetPtSub, "qJetPtSub/D");
    TT->Branch("q1P", &q1P, "q1P/D");
    TT->Branch("q2P", &q2P, "q2P/D");
    TT->Branch("qJetPLead", &qJetPLead, "qJetPLead/D");
    TT->Branch("qJetPSub", &qJetPSub, "qJetPSub/D");
    TT->Branch("qPairP", &qPairP, "qPairP/D");
    TT->Branch("q1Theta", &q1Theta, "q1Theta/D");
    TT->Branch("q2Theta", &q2Theta, "q2Theta/D");
    TT->Branch("q1Eta", &q1Eta, "q1Eta/D");
    TT->Branch("q2Eta", &q2Eta, "q2Eta/D");
    TT->Branch("qDeltaR", &qDeltaR, "qDeltaR/D");
    TT->Branch("qOpenAngle", &qOpenAngle, "qOpenAngle/D");

    TT->Branch("mZqq", &mZqq, "mZqq/D");
    TT->Branch("photonE", &photonE, "photonE/D");
    TT->Branch("photonPt", &photonPt, "photonPt/D");
    TT->Branch("photonP", &photonP, "photonP/D");
    TT->Branch("photonEta", &photonEta, "photonEta/D");
    TT->Branch("photonAbsCosTheta", &photonAbsCosTheta, "photonAbsCosTheta/D");

    TT->Branch("id3", &id1, "id3/I");
    TT->Branch("id4", &id2, "id4/I");
    TT->Branch("id5", &id3, "id5/I");
    TT->Branch("id6", &id4, "id6/I");
    TT->Branch("mass4l", &mass4l, "mass4l/D");
    TT->Branch("pT4l", &pT4l, "pT4l/D");
    TT->Branch("massZ1", &massZ1, "massZ1/D");
    TT->Branch("massZ2", &massZ2, "massZ2/D");
    TT->Branch("costheta1", &costheta1, "costheta1/D");
    TT->Branch("costheta2", &costheta2, "costheta2/D");
    TT->Branch("costhetastar", &costhetastar, "costhetastar/D");
    TT->Branch("phi", &phi, "phi/D");
    TT->Branch("phi1", &phi1, "phi1/D");

    long int nRead = 0;
    long int nAccepted = 0;
    long int nSkipped = 0;
    bool processLocked = (processMode != "auto");

    while (true) {
        if (Nevents > 0 && nRead >= Nevents) break;

        vector<LHEParticle> particles;
        if (!readMG5Event(in, particles, wt, idprup)) break;
        nRead++;

        vector<LHEParticle> fs = finalStateParticles(particles);
        nFinalState = fs.size();

        TString eventProcess = processLocked ? processMode : normalizeProcess(detectProcess(fs));
        std::cout << "Event " << nRead << ": process = " << eventProcess << std::endl;
        if (eventProcess == "unknown") {
            nSkipped++;
            continue;
        }

   

        bool pass = true;

        if (eventProcess == "ww") {
            LHEParticle lep, nu, q1, q2;
            bool foundLep = false, foundNu = false;
            int nQuarks = 0;
            for (const auto& p : fs) {
                if (!foundLep && isChargedLepton(p.id)) {
                    lep = p;
                    foundLep = true;
                } else if (!foundNu && isNeutrino(p.id)) {
                    nu = p;
                    foundNu = true;
                } else if (isQuark(p.id)) {
                    if (nQuarks == 0) q1 = p;
                    else q2 = p;
                    nQuarks++;
                }
            }
            if (!foundLep || !foundNu || nQuarks != 2) {
                nSkipped++;
                continue;
            }

            lepId = lep.id;
            nuId = nu.id;
            q1Id = q1.id;
            q2Id = q2.id;
            TLorentzVector pWlep = lep.p4 + nu.p4;
            TLorentzVector pWhad = q1.p4 + q2.p4;
            mWlep = pWlep.M();
            mWhad = pWhad.M();
            lepPt = lep.p4.Pt();
            lepEta = lep.p4.Eta();
            nuPt = nu.p4.Pt();
            mqq = pWhad.M();
            fillQuarkJetObservables(q1, q2, q1Pt, q2Pt, qJetPtLead, qJetPtSub,
                                    q1P, q2P, qJetPLead, qJetPSub, qPairP,
                                    q1Theta, q2Theta, q1Eta, q2Eta, qDeltaR, qOpenAngle,
                                    h_quarkPt, h_quarkPtLead, h_quarkPtSub,
                                    h_quarkP, h_quarkPLead, h_quarkPSub, h_quarkPairP,
                                    h_quarkTheta, h_quarkCosTheta, h_quarkEta,
                                    h_quarkDeltaR, h_quarkOpenAngle,
                                    wt);

            if (bApplyCuts) {
                pass = (lepPt > 10.0 && nuPt > 10.0 && mWlep > 50.0 && mWhad > 50.0);
            }

            if (pass) {
                h_mWlep->Fill(mWlep, wt);
                h_mWhad->Fill(mWhad, wt);
                h_lepPt->Fill(lepPt, wt);
                h_nuPt->Fill(nuPt, wt);
                TT->Fill();
                nAccepted++;
            } else {
                nSkipped++;
            }
        } else if (eventProcess == "zgamma") {
            LHEParticle photon, q1, q2;
            bool foundPhoton = false;
            int nQuarks = 0;
            for (const auto& p : fs) {
                if (!foundPhoton && isPhoton(p.id)) {
                    photon = p;
                    foundPhoton = true;
                } else if (isQuark(p.id)) {
                    if (nQuarks == 0) q1 = p;
                    else q2 = p;
                    nQuarks++;
                }
            }
            if (!foundPhoton || nQuarks != 2) {
                nSkipped++;
                continue;
            }

            q1Id = q1.id;
            q2Id = q2.id;
            TLorentzVector pZqq = q1.p4 + q2.p4;
            mZqq = pZqq.M();
            mqq = mZqq;
            photonE = photon.p4.E();
            photonPt = photon.p4.Pt();
            photonP = photon.p4.P();
            photonEta = photon.p4.Eta();
            photonAbsCosTheta = fabs(photon.p4.CosTheta());
            fillQuarkJetObservables(q1, q2, q1Pt, q2Pt, qJetPtLead, qJetPtSub,
                                    q1P, q2P, qJetPLead, qJetPSub, qPairP,
                                    q1Theta, q2Theta, q1Eta, q2Eta, qDeltaR, qOpenAngle,
                                    h_quarkPt, h_quarkPtLead, h_quarkPtSub,
                                    h_quarkP, h_quarkPLead, h_quarkPSub, h_quarkPairP,
                                    h_quarkTheta, h_quarkCosTheta, h_quarkEta,
                                    h_quarkDeltaR, h_quarkOpenAngle,
                                    wt);

            if (bApplyCuts) {
                pass = (mZqq > 40.0 && photonE > 5.0);
            }

            if (pass) {
                h_mZqq->Fill(mZqq, wt);
                h_photonE->Fill(photonE, wt);
                h_photonP->Fill(photonP, wt);
                h_photonAbsCosTheta->Fill(photonAbsCosTheta, wt);
                TT->Fill();
                nAccepted++;
            } else {
                nSkipped++;
            }
        } else if (eventProcess == "zz4l") {
            vector<LHEParticle> leptons;
            for (const auto& p : fs) {
                if (isChargedLepton(p.id)) leptons.push_back(p);
            }
            if (leptons.size() != 4) {
                nSkipped++;
                continue;
            }

            id1 = leptons[0].id;
            id2 = leptons[1].id;
            id3 = leptons[2].id;
            id4 = leptons[3].id;
            int sumAbsId = abs(id1) + abs(id2) + abs(id3) + abs(id4);
            if (sumAbsId != 44 && sumAbsId != 48 && sumAbsId != 52) {
                nSkipped++;
                continue;
            }

            TLorentzVector L11P4 = leptons[0].p4;
            TLorentzVector L12P4 = leptons[1].p4;
            TLorentzVector L21P4 = leptons[2].p4;
            TLorentzVector L22P4 = leptons[3].p4;

            TLorentzVector ZZ = L11P4 + L12P4 + L21P4 + L22P4;
            mass4l = ZZ.M();
            pT4l = ZZ.Pt();
            rapidity4l = ZZ.Rapidity();
            eta4l = ZZ.Eta();

            TLorentzVector tmpZ1 = L11P4 + L12P4;
            TLorentzVector tmpZ2 = L21P4 + L22P4;
            TLorentzVector Z1 = tmpZ1;
            TLorentzVector Z2 = tmpZ2;

            if (sumAbsId == 48) {
                mela::computeAngles(L11P4, id1, L12P4, id2, L21P4, id3, L22P4, id4,
                                    _costhetastar, _costheta1, _costheta2, _phi, _phi1);
                if (fabs(tmpZ1.M() - mZ_PDG) > fabs(tmpZ2.M() - mZ_PDG)) {
                    Z1 = tmpZ2;
                    Z2 = tmpZ1;
                    mela::computeAngles(L21P4, id3, L22P4, id4, L11P4, id1, L12P4, id2,
                                        _costhetastar, _costheta1, _costheta2, _phi, _phi1);
                }
            } else {
                TLorentzVector tmpZ3 = L11P4 + L22P4;
                TLorentzVector tmpZ4 = L21P4 + L12P4;
                double min_dM = min(min(fabs(tmpZ1.M() - mZ_PDG), fabs(tmpZ2.M() - mZ_PDG)),
                                    min(fabs(tmpZ3.M() - mZ_PDG), fabs(tmpZ4.M() - mZ_PDG)));
                mela::computeAngles(L11P4, id1, L12P4, id2, L21P4, id3, L22P4, id4,
                                    _costhetastar, _costheta1, _costheta2, _phi, _phi1);
                if (fabs(tmpZ2.M() - mZ_PDG) <= min_dM) {
                    Z1 = tmpZ2;
                    Z2 = tmpZ1;
                }
                if (fabs(tmpZ3.M() - mZ_PDG) <= min_dM) {
                    Z1 = tmpZ3;
                    Z2 = tmpZ4;
                }
                if (fabs(tmpZ4.M() - mZ_PDG) <= min_dM) {
                    Z1 = tmpZ4;
                    Z2 = tmpZ3;
                }
            }

            massZ1 = Z1.M();
            massZ2 = Z2.M();
            costheta1 = _costheta1;
            costheta2 = _costheta2;
            costhetastar = _costhetastar;
            phi = _phi;
            phi1 = _phi1;

            pt3 = L11P4.Pt(); eta3 = L11P4.Eta();
            pt4 = L12P4.Pt(); eta4 = L12P4.Eta();
            pt5 = L21P4.Pt(); eta5 = L21P4.Eta();
            pt6 = L22P4.Pt(); eta6 = L22P4.Eta();
            double ptArray[4] = {pt3, pt4, pt5, pt6};
            sort(ptArray, ptArray + LEPSIZE);
            pto1 = ptArray[3]; pto2 = ptArray[2]; pto3 = ptArray[1]; pto4 = ptArray[0];

            if (bApplyCuts) {
                pass = (pto1 > 20.0 && pto2 > 10.0 && pto3 > 5.0 && pto4 > 5.0
                        && fabs(eta3) < 2.5 && fabs(eta4) < 2.5
                        && fabs(eta5) < 2.5 && fabs(eta6) < 2.5
                        && massZ1 > 40.0);
            }

            if (pass) {
                mass4l_hist_tchan->Fill(mass4l, wt);
                pT4l_hist_tchan->Fill(pT4l, wt);
                massZ1_hist_tchan->Fill(massZ1, wt);
                massZ2_hist_tchan->Fill(massZ2, wt);
                TT->Fill();
                nAccepted++;
            } else {
                nSkipped++;
            }
        } else {
            nSkipped++;
        }

        if (kDebugLevel > 0 && (nRead % 1000) == 0) {
            printf("Analysed %ld events (%ld accepted, %ld skipped)\n", nRead, nAccepted, nSkipped);
        }
    }

    h_mWlep->Write();
    h_mWhad->Write();
    h_lepPt->Write();
    h_nuPt->Write();
    h_quarkPt->Write();
    h_quarkPtLead->Write();
    h_quarkPtSub->Write();
    h_quarkP->Write();
    h_quarkPLead->Write();
    h_quarkPSub->Write();
    h_quarkPairP->Write();
    h_quarkTheta->Write();
    h_quarkCosTheta->Write();
    h_quarkEta->Write();
    h_quarkDeltaR->Write();
    h_quarkOpenAngle->Write();
    h_mZqq->Write();
    h_photonE->Write();
    h_photonP->Write();
    h_photonAbsCosTheta->Write();
    mass4l_hist_tchan->Write();
    pT4l_hist_tchan->Write();
    massZ1_hist_tchan->Write();
    massZ2_hist_tchan->Write();
    TT->Write();

    printf("Read %ld events\n", nRead);
    printf("Accepted %ld events\n", nAccepted);
    printf("Skipped %ld events\n", nSkipped);

    in.close();
    f->Write();
    f->Close();
    delete f;
}
