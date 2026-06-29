#!/bin/bash
# Inspect Condor logs for the WW scan.
#
# Usage:
#   ./condor/show_logs.sh                     # list recent log files
#   ./condor/show_logs.sh --tail              # tail all *.err files
#   ./condor/show_logs.sh --failed            # show stderr of non-zero exits
#   ./condor/show_logs.sh 12345.6             # show logs for ClusterId.ProcId
#   ./condor/show_logs.sh 12345.6 --follow    # tail -f that job's .out
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scanWW_config.sh
source "${SCRIPT_DIR}/scanWW_config.sh"

JOB=""
MODE="list"
FOLLOW=0

for arg in "$@"; do
    case "${arg}" in
        --tail) MODE="tail" ;;
        --failed) MODE="failed" ;;
        --follow|-f) FOLLOW=1 ;;
        -h|--help)
            sed -n '2,9p' "$0"
            exit 0
            ;;
        *)
            if [[ -z "${JOB}" ]]; then
                JOB="${arg}"
            fi
            ;;
    esac
done

mkdir -p "${LOG_DIR}"

show_job_logs() {
    local id="$1"
    local out="${LOG_DIR}/scanWW_${id}.out"
    local err="${LOG_DIR}/scanWW_${id}.err"
    local log="${LOG_DIR}/scanWW_${id}.log"

    echo "========== ${id} =========="
    for f in "${out}" "${err}" "${log}"; do
        if [[ -f "${f}" ]]; then
            echo "--- ${f} ---"
            if [[ ${FOLLOW} -eq 1 && "${f}" == "${out}" ]]; then
                tail -f "${f}"
                return
            fi
            tail -n 40 "${f}"
            echo ""
        else
            echo "(missing ${f})"
        fi
    done
}

case "${MODE}" in
    list)
        if [[ -n "${JOB}" ]]; then
            show_job_logs "${JOB}"
            exit 0
        fi
        echo "Recent scanWW logs in ${LOG_DIR}:"
        ls -lt "${LOG_DIR}"/scanWW_* 2>/dev/null | head -30 || echo "(no logs yet)"
        ;;
    tail)
        shopt -s nullglob
        files=("${LOG_DIR}"/scanWW_*.err)
        if [[ ${#files[@]} -eq 0 ]]; then
            echo "No .err logs found in ${LOG_DIR}"
            exit 0
        fi
        tail -n 20 "${files[@]}"
        ;;
    failed)
        shopt -s nullglob
        found=0
        for err in "${LOG_DIR}"/scanWW_*.err; do
            base=$(basename "${err}" .err)
            id=${base#scanWW_}
            out="${LOG_DIR}/${base}.out"
            # Heuristic: mg5 failure or non-empty err with ERROR
            if grep -qiE 'ERROR|fail to reach target|NoDiagramException|Traceback' "${err}" 2>/dev/null; then
                found=1
                show_job_logs "${id}"
            elif [[ -s "${err}" ]] && ! grep -q 'LHE already exists' "${out}" 2>/dev/null; then
                found=1
                show_job_logs "${id}"
            fi
        done
        if [[ ${found} -eq 0 ]]; then
            echo "No obvious failures in ${LOG_DIR}"
        fi
        ;;
esac
