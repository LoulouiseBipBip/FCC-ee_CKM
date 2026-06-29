#!/bin/bash
# Skim LHE samples and plot a chosen observable from the skimmer ROOT files.
#
# Usage:
#   ./LHE_to_plot.sh [variable] [xMin] [xMax] [yMin] [yMax] [selVariable] [selMin] [selMax]
#
# Omit xMin/xMax/yMin/yMax (or leave unset) to auto-range each axis from the
# plotted distribution with a 5% margin. Partial overrides are supported.
#
# Variables:
#   quarkPairP         WW vs Z+gamma |p1+p2| (default)
#   quarkPt            WW vs Z+gamma quark pT
#   quarkP             WW vs Z+gamma quark |p|
#   mZqq               WW (mWhad) vs Z+gamma m_{q#bar{q}}
#   quarkTheta         WW vs Z+gamma parton #theta (per quark)
#   quarkEta           WW vs Z+gamma parton #eta (per quark)
#   quarkDeltaR        WW vs Z+gamma #Delta R between the two partons
#   quarkOpenAngle     WW vs Z+gamma opening angle between the two partons
#   photonP            Z+gamma photon |p| (1D)
#   photonP_vs_quarkP  Z+gamma 2D: photon |p| vs quark |p|
#
# Aliases: pt, p, qpairp/quarkpairp, mzqq, theta/quarktheta, eta/quarketa,
#          deltar/quarkdeltar, openangle/quarkopenangle, photonp, photon_vs_quarkp
#
# Optional event selection (args 6-8): keep only Z+gamma events where selVariable
# is within [selMin, selMax]. WW sample is always unfiltered in comparison plots.
# Any lheEvents branch name works, e.g. qPairP, photonP, photonEta, mZqq
#   qPairP, q1P, q2P, q1Pt, q1Theta, q1Eta, qDeltaR, qOpenAngle, photonP, photonEta, mZqq, mWhad, photonAbsCosTheta
#
# Selection can also be set via environment variables (handy for 1D plots):
#   SEL_VAR=qPairP SEL_MIN=50 SEL_MAX=120 ./LHE_to_plot.sh quarkP 0 140
#
# Examples:
#   ./LHE_to_plot.sh quarkP 0 140
#   ./LHE_to_plot.sh quarkPt 0 100
#   ./LHE_to_plot.sh photonP 0 120
#   ./LHE_to_plot.sh photonP_vs_quarkP 10 20 0 130
#   ./LHE_to_plot.sh quarkP 0 140 0 130 qPairP 50 120
#   SEL_VAR=photonAbsCosTheta SEL_MIN=0.9 SEL_MAX=1.0 ./LHE_to_plot.sh photonP 0 120

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

AUTO=-999

VARIABLE="${1:-photonp}"
XMIN="${2:-70}"
XMAX="${3:-130}"
YMIN="${4:-$AUTO}"
YMAX="${5:-$AUTO}"
SEL_VAR="${SEL_VAR:-${6:-}}"
SEL_MIN="${SEL_MIN:-${7:--1}}"
SEL_MAX="${SEL_MAX:-${8:--1}}"

CMS_ENERGY_Z=240
CMS_ENERGY_WW=240
NEVENTS=10000

LHE_Zgamma="${PROJECT_ROOT}/MG5_aMC_v3_7_1/Zphoton_qqbar_mefixed/Events/run_${CMS_ENERGY_Z}GeV/unweighted_events.lhe"
LHE_WW="${PROJECT_ROOT}/MG5_aMC_v3_7_1/WW_mu_semileptonic/Events/run_${CMS_ENERGY_WW}GeV/unweighted_events.lhe"

ROOT_Zgamma="${LHE_Zgamma}_zgammaWITHOUTcuts.root"
ROOT_WW="${LHE_WW}_ww_muWITHOUTcuts.root"
OUT_PDF="${PROJECT_ROOT}/plots_final/Zgamma${CMS_ENERGY_Z}GeV_WW${CMS_ENERGY_WW}GeV/${SEL_VAR}_${SEL_MIN}_${SEL_MAX}/${VARIABLE}_nolog.pdf"

# Accept any variable (let ROOT handle variable errors)
VARIABLE_LC="$(echo "${VARIABLE}" | tr '[:upper:]' '[:lower:]')"
NEEDS_WW=1
case "${VARIABLE_LC}" in
    photonp|photonp_vs_quarkp|photon_vs_quarkp)
        NEEDS_WW=0
        ;;
    *)
        NEEDS_WW=1
        ;;
esac

if [[ ${NEEDS_WW} -eq 1 ]]; then
    for f in "${LHE_Zgamma}" "${LHE_WW}"; do
        if [[ ! -f "${f}" ]]; then
            echo "ERROR: LHE file not found: ${f}" >&2
            exit 1
        fi
    done
else
    if [[ ! -f "${LHE_Zgamma}" ]]; then
        echo "ERROR: LHE file not found: ${LHE_Zgamma}" >&2
        exit 1
    fi
fi

mkdir -p "$(dirname "${OUT_PDF}")"
cd "${SCRIPT_DIR}"

need_skim_zgamma=1
need_skim_ww=0
if [[ ${NEEDS_WW} -eq 1 ]]; then
    need_skim_ww=1
fi

if [[ -f "${ROOT_Zgamma}" ]]; then
    echo "Found existing ROOT file for Z+gamma: ${ROOT_Zgamma}, skipping skimming."
    need_skim_zgamma=0
fi

if [[ ${NEEDS_WW} -eq 1 && -f "${ROOT_WW}" ]]; then
    echo "Found existing ROOT file for WW: ${ROOT_WW}, skipping skimming."
    need_skim_ww=0
fi

if [[ $need_skim_zgamma -eq 1 ]]; then
    echo "Skimming Z+gamma ..."
    root -l -b -q "ZZD_lhe.C(\"${LHE_Zgamma}\", ${NEVENTS}, \"Zgamma\")"
fi

if [[ $need_skim_ww -eq 1 ]]; then
    echo "Skimming WW ..."
    root -l -b -q "ZZD_lhe.C(\"${LHE_WW}\", ${NEVENTS}, \"WW_mu\")"
fi

echo "Plotting ${VARIABLE} ..."
if [[ "${XMIN}" == "${AUTO}" || "${XMAX}" == "${AUTO}" || "${YMIN}" == "${AUTO}" || "${YMAX}" == "${AUTO}" ]]; then
    echo "Axis auto-range enabled for unset limits (5% margin)."
fi
if [[ -n "${SEL_VAR}" ]]; then
    echo "Event selection: ${SEL_VAR} in [${SEL_MIN}, ${SEL_MAX}]"
fi
root -l -b -q "plot_lhe.C(\"${VARIABLE}\", \"${ROOT_WW}\", \"${ROOT_Zgamma}\", \"${OUT_PDF}\", ${XMIN}, ${XMAX}, ${YMIN}, ${YMAX}, 200, \"${SEL_VAR}\", ${SEL_MIN}, ${SEL_MAX})"

echo "Done: ${OUT_PDF}"
