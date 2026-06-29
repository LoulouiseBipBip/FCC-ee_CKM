#!/bin/bash
# Dashboard for the WW scan Condor submission and LHE output.
#
# Usage:
#   ./condor/monitor_scanWW.sh
#   ./condor/monitor_scanWW.sh --watch     # refresh every 30 s
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scanWW_config.sh
source "${SCRIPT_DIR}/scanWW_config.sh"

WATCH=0
REFRESH=30
for arg in "$@"; do
    case "${arg}" in
        --watch) WATCH=1 ;;
        -h|--help)
            sed -n '2,6p' "$0"
            exit 0
            ;;
    esac
done

print_status() {
    clear 2>/dev/null || true
    echo "============================================================"
    echo " WW scan monitor — $(date)"
    echo " Process: ${PROC_DIR}"
    echo "============================================================"
    echo ""

    echo "--- HTCondor queue (your jobs matching scanWW) ---"
    if condor_q -nobatch "${USER}" 2>/dev/null | grep -E 'scanWW|ID +OWNER' || true | head -40; then
        :
    else
        echo "(no matching jobs in queue, or condor_q unavailable)"
    fi
    echo ""

    echo "--- Recent Condor history (last 10 scanWW jobs) ---"
    condor_history "${USER}" -limit 10 -af ClusterId ProcId JobStatus ExitCode Cmd 2>/dev/null \
        | grep scanWW_worker || echo "(none)"
    echo ""

    if [[ ! -f "${ENERGY_LIST}" ]]; then
        write_scanWW_energy_list
    fi

    TOTAL=$(wc -l < "${ENERGY_LIST}")
    DONE=0
    MISSING=0

    echo "--- LHE generation progress ---"
    printf "%-8s %-10s %s\n" "sqrt(s)" "status" "path"
    while read -r E; do
        RUNNAME="run_${E}GeV"
        LHE="${PROC_DIR}/Events/${RUNNAME}/unweighted_events.lhe.gz"
        if [[ -f "${LHE}" ]]; then
            SIZE=$(du -sh "${LHE}" | cut -f1)
            printf "%-8s %-10s %s (%s)\n" "${E}" "DONE" "${LHE}" "${SIZE}"
            ((DONE++)) || true
        else
            printf "%-8s %-10s %s\n" "${E}" "MISSING" "${LHE}"
            ((MISSING++)) || true
        fi
    done < "${ENERGY_LIST}"

    echo ""
    echo "Summary: ${DONE}/${TOTAL} complete, ${MISSING} missing"
    echo ""
    echo "Logs: ${LOG_DIR}/scanWW_<ClusterId>.<ProcId>.{out,err,log}"
    echo "Help: ${SCRIPT_DIR}/show_logs.sh --failed"
}

if [[ ${WATCH} -eq 1 ]]; then
    while true; do
        print_status
        sleep "${REFRESH}"
    done
else
    print_status
fi
