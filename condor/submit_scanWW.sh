#!/bin/bash
# Generate the energy list and submit the WW scan to HTCondor.
#
# Usage:
#   ./condor/submit_scanWW.sh              # submit all missing energies
#   ./condor/submit_scanWW.sh --prepare    # compile process dir once, then submit
#   ./condor/submit_scanWW.sh --dry-run    # show what would be submitted
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scanWW_config.sh
source "${SCRIPT_DIR}/scanWW_config.sh"

PREPARE=0
DRY_RUN=0
for arg in "$@"; do
    case "${arg}" in
        --prepare) PREPARE=1 ;;
        --dry-run) DRY_RUN=1 ;;
        -h|--help)
            sed -n '2,8p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown option: ${arg}" >&2
            exit 1
            ;;
    esac
done

mkdir -p "${LOG_DIR}"
write_scanWW_pending_energy_list
chmod +x "${SCRIPT_DIR}/run_scanWW_worker.sh"

TOTAL=$(wc -l < "${ENERGY_LIST}")
PENDING=$(wc -l < "${PENDING_ENERGY_LIST}")
echo "Energy points in scan: ${TOTAL}"
echo "Full list:   ${ENERGY_LIST}"
echo "Submit list: ${PENDING_ENERGY_LIST} (${PENDING} jobs)"
echo ""

DONE=$((TOTAL - PENDING))
echo "Already complete: ${DONE}/${TOTAL}"
echo "Still missing:    ${PENDING}/${TOTAL}"
echo ""

if [[ ${PREPARE} -eq 1 ]]; then
    FIRST_MISSING=""
    while read -r E; do
        LHE="${PROC_DIR}/Events/run_${E}GeV/unweighted_events.lhe.gz"
        if [[ ! -f "${LHE}" ]]; then
            FIRST_MISSING="${E}"
            break
        fi
    done < "${ENERGY_LIST}"

    if [[ -n "${FIRST_MISSING}" ]]; then
        echo "Preparing process directory (compile) at sqrt(s)=${FIRST_MISSING} GeV ..."
        "${SCRIPT_DIR}/run_scanWW_worker.sh" "${FIRST_MISSING}"
        echo ""
    else
        echo "All energies already generated; nothing to prepare."
        echo ""
    fi
fi

if [[ ${PENDING} -eq 0 ]]; then
    echo "Nothing to submit — all LHE files exist."
    exit 0
fi

if [[ ${DRY_RUN} -eq 1 ]]; then
    echo "Dry run — would submit ${PENDING} jobs for:"
    cat "${PENDING_ENERGY_LIST}" | sed 's/^/  sqrt(s) = /; s/$/ GeV/'
    exit 0
fi

echo "Submitting to HTCondor ..."
if ! timeout 30 condor_q -totals "${USER}" >/dev/null 2>&1; then
    echo ""
    echo "ERROR: HTCondor is not responding (condor_q timed out after 30 s)."
    echo "This is a CERN batch connectivity issue, not a problem in run_scanWW.sh."
    echo ""
    echo "condor_submit often hangs at 'Submitting job(s)' while storing Kerberos"
    echo "credentials with the schedd (CredD / SECMAN). Typical causes:"
    echo "  - temporary lxbatch / HTCondor outage"
    echo "  - expired or missing Kerberos ticket (check: klist)"
    echo ""
    echo "Try:"
    echo "  kinit                          # refresh Kerberos ticket"
    echo "  condor_q -totals               # should return quickly, not hang"
    echo "  ./condor/submit_scanWW.sh      # retry once condor_q works"
    echo ""
    echo "Meanwhile, run locally:  ./run_scanWW.sh <energy>"
    exit 1
fi

if ! timeout 60 condor_submit "${SCRIPT_DIR}/submit_scanWW.sub"; then
    echo ""
    echo "ERROR: condor_submit failed or timed out after 60 s."
    echo "If it hung at 'Submitting job(s)', HTCondor credential storage is stuck."
    echo "Check: klist ; condor_q -totals"
    exit 1
fi

echo ""
echo "Jobs run in parallel: each worker copies the MG5 process dir (~73M)"
echo "to local disk, generates events, then rsyncs the LHE back to AFS."
echo ""
echo "Monitor with:"
echo "  ${SCRIPT_DIR}/monitor_scanWW.sh"
echo "  ${SCRIPT_DIR}/show_logs.sh --tail"
