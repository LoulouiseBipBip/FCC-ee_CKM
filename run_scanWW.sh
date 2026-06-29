#!/bin/bash
# run_scanWW.sh — generate e+e- -> WW (semileptonic) at multiple cms energies
#
# Usage:
#   ./run_scanWW.sh
#
# Each run outputs events to:
#   MG5_aMC_v3_7_1/WW_mu_semileptonic/Events/run_<E>GeV/unweighted_events.lhe.gz
#
# To add more energies, edit ENERGIES below.

set -euo pipefail

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
MG5_DIR="/afs/cern.ch/user/l/lberiet/FCC-ee_CKM/MG5_aMC_v3_7_1"
PROC_DIR="${MG5_DIR}/WW_mu_semileptonic_MINUS_TEST/"
MG5_BIN="${MG5_DIR}/bin/mg5_aMC"

NEVENTS=10000          # number of unweighted events per energy point

ENERGIES=(163)

# --------------------------------------------------------------------------
# Sanity checks
# --------------------------------------------------------------------------
if [ ! -d "${PROC_DIR}" ]; then
    echo "ERROR: Process directory not found: ${PROC_DIR}"
    echo "Run proc_card_WW.dat with mg5_aMC first."
    exit 1
fi

if [ ! -f "${PROC_DIR}/SubProcesses/proc_characteristics" ]; then
    echo "ERROR: proc_characteristics missing — process directory incomplete."
    exit 1
fi

# --------------------------------------------------------------------------
# Loop over energies
# --------------------------------------------------------------------------
for E in "${ENERGIES[@]}"; do
    EBEAM=$(echo "scale=4; ${E}/2" | bc)
    RUNNAME="run_${E}GeV"

    echo "======================================================"
    echo " Generating events: sqrt(s) = ${E} GeV  (ebeam = ${EBEAM} GeV)"
    echo " Run name: ${RUNNAME}"
    echo "======================================================"

    # Skip if already done
    LHE="${PROC_DIR}/Events/${RUNNAME}/unweighted_events.lhe.gz"
    if [ -f "${LHE}" ]; then
        echo "  -> LHE file already exists, skipping."
        continue
    fi

    # Write a temporary mg5 launch script for this energy point
    TMP_SCRIPT=$(mktemp /tmp/mg5_launch_XXXXXX.mg5)
    cat > "${TMP_SCRIPT}" <<MGEOF
launch ${PROC_DIR} --name ${RUNNAME}
set run_card lpp1 0
set run_card lpp2 0
set run_card ebeam1 ${EBEAM}
set run_card ebeam2 ${EBEAM}
set run_card nevents ${NEVENTS}
set run_card cut_decays False
MGEOF

    python3 "${MG5_BIN}" "${TMP_SCRIPT}" 2>&1
    rm -f "${TMP_SCRIPT}"

    if [ -f "${LHE}" ]; then
        SIZE=$(du -sh "${LHE}" | cut -f1)
        echo "  -> Done. LHE file: ${LHE} (${SIZE})"
    else
        echo "  -> WARNING: LHE file not found after run"
    fi

    echo ""
done

echo "======================================================"
echo " Scan complete. Results summary:"
echo "======================================================"
for E in "${ENERGIES[@]}"; do
    LHE="${PROC_DIR}/Events/run_${E}GeV/unweighted_events.lhe.gz"
    if [ -f "${LHE}" ]; then
        echo "  sqrt(s) = ${E} GeV -> ${LHE}"
    else
        echo "  sqrt(s) = ${E} GeV -> NOT GENERATED"
    fi
done
