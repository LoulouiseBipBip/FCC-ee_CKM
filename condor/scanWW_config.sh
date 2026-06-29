# Shared configuration for run_scanWW.sh and Condor submission.
# shellcheck shell=bash

PROJECT_DIR="/afs/cern.ch/user/l/lberiet/FCC-ee_CKM"
MG5_DIR="${PROJECT_DIR}/MG5_aMC_v3_7_1"
PROC_DIR="${MG5_DIR}/WW_mu_semileptonic"
MG5_BIN="${MG5_DIR}/bin/mg5_aMC"

NEVENTS=10000

CONDOR_DIR="${PROJECT_DIR}/condor"
LOG_DIR="${CONDOR_DIR}/logs"
ENERGY_LIST="${CONDOR_DIR}/energies.txt"
PENDING_ENERGY_LIST="${CONDOR_DIR}/energies_pending.txt"

# Return the energy scan list (one energy per line, deduplicated, sorted numerically).
scanWW_energies() {
    local -a energies=(157 160 162 163 165 167)
    local e
    for e in $(seq 167 5 295); do
        energies+=("${e}")
    done

    printf '%s\n' "${energies[@]}" | awk '!seen[$0]++' | sort -n
}

write_scanWW_energy_list() {
    mkdir -p "${CONDOR_DIR}"
    scanWW_energies > "${ENERGY_LIST}"
}

write_scanWW_pending_energy_list() {
    write_scanWW_energy_list
    : > "${PENDING_ENERGY_LIST}"
    while read -r E; do
        LHE="${PROC_DIR}/Events/run_${E}GeV/unweighted_events.lhe.gz"
        [[ -f "${LHE}" ]] || echo "${E}" >> "${PENDING_ENERGY_LIST}"
    done < "${ENERGY_LIST}"
}
