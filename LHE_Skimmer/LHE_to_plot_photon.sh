#!/bin/bash
# Skim two LHE samples and produce a quark-pT comparison plot.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CMS_ENERGY=110
PTA=20.0
NEVENTS=10000
photonPMin=10
photonPMax=20
mZqqMin=80
mZqqMax=100
nBins=200
LHE_Zgamma="${PROJECT_ROOT}/MG5_aMC_v3_7_1/Zphoton_qqbar_mefixed/Events/run_${CMS_ENERGY}GeV/unweighted_events.lhe"


ROOT_Zgamma="${LHE_Zgamma}_zgammaWITHOUTcuts.root"
OUT_PDF="${PROJECT_ROOT}/plots_quark_vs_photon_p/${CMS_ENERGY}GeV/Z_photon_p_massqq.pdf"


for f in "${LHE_Zgamma}"; do
    if [[ ! -f "${f}" ]]; then
        echo "ERROR: LHE file not found: ${f}" >&2
        exit 1
    fi
done

mkdir -p "$(dirname "${OUT_PDF}")"
cd "${SCRIPT_DIR}"

need_skim_zgamma=1

if [[ -f "${ROOT_Zgamma}" ]]; then
    echo "Found existing ROOT file for Z+gamma: ${ROOT_Zgamma}, skipping skimming."
    need_skim_zgamma=0
fi

if [[ $need_skim_zgamma -eq 1 ]]; then
    echo "Skimming Z+gamma ..."
    root -l -b -q "ZZD_lhe.C(\"${LHE_Zgamma}\", ${NEVENTS}, \"Zgamma\")"
fi

echo "Plotting ..."
root -l -b -q "plot_photon_p_massqq.C(\"${ROOT_Zgamma}\", \"${OUT_PDF}\", ${photonPMin}, ${photonPMax}, ${mZqqMin}, ${mZqqMax}, ${nBins})"

echo "Done: ${OUT_PDF}"
