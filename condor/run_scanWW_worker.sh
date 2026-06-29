#!/bin/bash
# HTCondor worker: generate one energy point from the WW scan.
#
# Called by Condor with:
#   run_scanWW_worker.sh <sqrt(s) in GeV>

set -euo pipefail

ENERGY="${1:-}"
if [[ -z "${ENERGY}" ]]; then
    echo "ERROR: missing energy argument (GeV)" >&2
    exit 1
fi

PROJECT_ROOT="/afs/cern.ch/user/l/lberiet/FCC-ee_CKM"
# shellcheck source=scanWW_config.sh
source "${PROJECT_ROOT}/condor/scanWW_config.sh"

JOB_TAG="${CLUSTER_ID:-local}.${PROC_ID:-0}"
RUNNAME="run_${ENERGY}GeV"
LHE_AFS="${PROC_DIR}/Events/${RUNNAME}/unweighted_events.lhe.gz"
LOCAL_PROC="${TMPDIR:-/tmp}/mg5_${JOB_TAG}_$$"

cleanup() {
    rm -rf "${LOCAL_PROC}"
}
trap cleanup EXIT

echo "======================================================"
echo " Condor worker ${JOB_TAG}"
echo " sqrt(s) = ${ENERGY} GeV"
echo " AFS process: ${PROC_DIR}"
echo " Local copy:  ${LOCAL_PROC}"
echo " Started: $(date -Is)"
echo " Host:    $(hostname)"
echo "======================================================"

if [[ ! -d "${PROC_DIR}" ]]; then
    echo "ERROR: process directory not found: ${PROC_DIR}" >&2
    exit 1
fi

if [[ -f "${LHE_AFS}" ]]; then
    echo "LHE already exists, skipping: ${LHE_AFS}"
    exit 0
fi

if [[ ! -f "${MG5_BIN}" ]]; then
    echo "ERROR: mg5_aMC not found: ${MG5_BIN}" >&2
    exit 1
fi

echo "Copying process directory to local disk ..."
T0=$(date +%s)
rsync -a \
    --exclude 'Events/run_*' \
    "${PROC_DIR}/" "${LOCAL_PROC}/"
echo "Local copy ready in $(( $(date +%s) - T0 )) s ($(du -sh "${LOCAL_PROC}" | cut -f1))"

EBEAM=$(echo "scale=4; ${ENERGY}/2" | bc)
TMP_SCRIPT="$(mktemp "${TMPDIR:-/tmp}/mg5_launch_${ENERGY}GeV_XXXXXX.mg5")"

cat > "${TMP_SCRIPT}" <<MGEOF
launch ${LOCAL_PROC} --name ${RUNNAME}
set run_card lpp1 0
set run_card lpp2 0
set run_card ebeam1 ${EBEAM}
set run_card ebeam2 ${EBEAM}
set run_card nevents ${NEVENTS}
set run_card cut_decays False
MGEOF

echo "Launch script: ${TMP_SCRIPT}"
echo "ebeam = ${EBEAM} GeV, nevents = ${NEVENTS}"
echo "Starting mg5_aMC at $(date -Is)"

set +e
python3 "${MG5_BIN}" "${TMP_SCRIPT}" 2>&1
MG5_RC=$?
set -e
rm -f "${TMP_SCRIPT}"

if [[ ${MG5_RC} -ne 0 ]]; then
    echo "ERROR: mg5_aMC failed with exit code ${MG5_RC}" >&2
    exit "${MG5_RC}"
fi

LHE_LOCAL="${LOCAL_PROC}/Events/${RUNNAME}/unweighted_events.lhe.gz"
if [[ ! -f "${LHE_LOCAL}" ]]; then
    echo "ERROR: expected LHE not found locally: ${LHE_LOCAL}" >&2
    exit 1
fi

echo "Copying events back to AFS ..."
mkdir -p "${PROC_DIR}/Events/${RUNNAME}"
rsync -a "${LOCAL_PROC}/Events/${RUNNAME}/" "${PROC_DIR}/Events/${RUNNAME}/"

if [[ -f "${LHE_AFS}" ]]; then
    echo "Done: ${LHE_AFS} ($(du -sh "${LHE_AFS}" | cut -f1))"
    exit 0
fi

echo "ERROR: LHE not found on AFS after rsync: ${LHE_AFS}" >&2
exit 1
