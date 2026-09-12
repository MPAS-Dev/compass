#!/bin/bash
#
# run_compass_integration.sh
#
# Automates a baseline + testing compass "full_integration" landice suite
# comparison on Perlmutter (pm-cpu).
#
# ALL configuration (paths, refs, remotes, account, qos, etc.) is set by
# editing the variables in the CONFIG section below. The only supported
# command-line arguments are --force-baseline / --force-testing / --force,
# used to force re-cloning/re-building a stage that was already set up by a
# previous run of this script.
#
# Workflow (one command, no manual steps):
#   1. BASELINE prep (synchronous, on the login node):
#        - clone compass from COMPASS_REMOTE_BASELINE @ COMPASS_REF_BASELINE
#          (skipped if already present, unless --force/--force-baseline)
#        - ./deploy.py --with-albany ...     (skipped if load script exists)
#        - source the generated load script
#        - clone MALI-Dev from MALI_REMOTE_BASELINE @ MALI_REF_BASELINE
#        - compile MALI (mpas-albany-landice)
#        - `compass suite -c landice -t full_integration -w <suite-baseline> -s`
#   2. Submit the compass-generated suite job script for BASELINE via sbatch.
#   3. TESTING prep (synchronous, on the login node):
#        - reuse the BASELINE compass checkout if COMPASS_REMOTE_TESTING/
#          COMPASS_REF_TESTING are left blank; otherwise clone/deploy a
#          separate compass copy for testing
#        - clone/build a *separate* MALI-Dev copy from MALI_REMOTE_TESTING @
#          MALI_REF_TESTING
#        - `compass suite -c landice -t full_integration -b <suite-baseline> \
#              -w <suite-testing> -s`
#   4. Submit the compass-generated suite job script for TESTING via sbatch
#      with `--dependency=afterok:<baseline_run_jobid>` so it automatically
#      waits for the baseline run to finish successfully before starting --
#      no manual waiting required.
#
# Hardcoded directory layout under ROOT_WORK_DIR:
#   <ROOT_WORK_DIR>/compass-baseline/   compass checkout used for baseline
#   <ROOT_WORK_DIR>/compass-testing/    compass checkout used for testing (only
#                                       created if a separate compass ref/remote
#                                       is requested for testing)
#   <ROOT_WORK_DIR>/MALI-baseline/      MALI-Dev checkout/build for baseline
#   <ROOT_WORK_DIR>/MALI-testing/       MALI-Dev checkout/build for testing
#   <ROOT_WORK_DIR>/suite-baseline/     compass suite -w for baseline
#   <ROOT_WORK_DIR>/suite-testing/      compass suite -w for testing
#
set -euo pipefail

# =============================================================================
# CONFIG -- edit these values for your run. Nothing here is a CLI argument.
# =============================================================================

# Single root location; everything else lives in a hardcoded structure below it.
ROOT_WORK_DIR="/pscratch/sd/h/hoffman2/COMPASS/run"

# SLURM settings for the compass suite *run* job (the actual integration test
# execution). The suite itself typically completes in ~12 minutes.
ACCOUNT="m4274"
QOS="debug"
RUN_WALLTIME="00:30:00"

# compass deploy.py / build settings (shared by baseline and testing).
MACHINE="pm-cpu"
COMPILER="gnu"
MPI="mpich"

# compass suite selection.
SUITE_CORE="landice"
SUITE_NAME="full_integration"

# --- compass source ----------------------------------------------------------
COMPASS_REMOTE_BASELINE="git@github.com:MPAS-Dev/compass.git"
COMPASS_REF_BASELINE="main"

# If set, use this already-cloned (and optionally already-deployed) compass
# directory for baseline instead of managing a clone under ROOT_WORK_DIR.
# git clone/checkout is never performed on this directory (your working tree
# is left untouched); COMPASS_REMOTE_BASELINE/COMPASS_REF_BASELINE above are
# ignored when this is set. deploy.py is still run here if no matching
# load_compass_<machine>_<compiler>_<mpi>.sh is found (i.e. env not yet
# deployed), otherwise it is skipped.
EXISTING_COMPASS_DIR_BASELINE=""

# Leave both of these blank to reuse the baseline compass checkout for testing
# (recommended when you are only testing a MALI change). Set both to test a
# different compass remote/ref as well (a separate compass-testing/ checkout
# will be created).
COMPASS_REMOTE_TESTING=""
COMPASS_REF_TESTING=""

# Same as EXISTING_COMPASS_DIR_BASELINE, but for testing. Only relevant when a
# separate testing compass checkout is in use (i.e. COMPASS_REMOTE_TESTING or
# COMPASS_REF_TESTING is set above); ignored when testing reuses the baseline
# compass checkout.
EXISTING_COMPASS_DIR_TESTING=""

# --- MALI-Dev source ----------------------------------------------------------
MALI_REMOTE_BASELINE="git@github.com:MALI-Dev/E3SM.git"
MALI_REF_BASELINE="develop"

MALI_REMOTE_TESTING="git@github.com:MALI-Dev/E3SM.git"
MALI_REF_TESTING="matthewhoffman/mali/spatial-damage-threshold"

# =============================================================================
# End of CONFIG. You should not need to edit anything below this line.
# =============================================================================

FORCE_BASELINE=false
FORCE_TESTING=false

usage() {
    cat <<EOF
Usage: $0 [--force-baseline] [--force-testing] [--force]

All other configuration (paths, refs, remotes, account, qos, etc.) is set by
editing the CONFIG section at the top of this script.

  --force-baseline   redo baseline clone/deploy/build/setup even if present
  --force-testing    redo testing clone/deploy/build/setup even if present
  --force            shorthand for --force-baseline --force-testing
  -h, --help         show this help
EOF
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force-baseline) FORCE_BASELINE=true; shift ;;
        --force-testing) FORCE_TESTING=true; shift ;;
        --force) FORCE_BASELINE=true; FORCE_TESTING=true; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

SAME_COMPASS=true
if [[ -n "${COMPASS_REMOTE_TESTING}" || -n "${COMPASS_REF_TESTING}" ]]; then
    SAME_COMPASS=false
fi

mkdir -p "${ROOT_WORK_DIR}"
ROOT_WORK_DIR="$(cd "${ROOT_WORK_DIR}" && pwd)"

# Hardcoded organizational structure under ROOT_WORK_DIR (unless an
# EXISTING_COMPASS_DIR_* override is given above).
if [[ -n "${EXISTING_COMPASS_DIR_BASELINE}" ]]; then
    BASELINE_COMPASS_DIR="${EXISTING_COMPASS_DIR_BASELINE}"
else
    BASELINE_COMPASS_DIR="${ROOT_WORK_DIR}/compass-baseline"
fi
if [[ -n "${EXISTING_COMPASS_DIR_TESTING}" ]]; then
    TESTING_COMPASS_DIR="${EXISTING_COMPASS_DIR_TESTING}"
else
    TESTING_COMPASS_DIR="${ROOT_WORK_DIR}/compass-testing"
fi
BASELINE_MALI_DIR="${ROOT_WORK_DIR}/MALI-baseline"
TESTING_MALI_DIR="${ROOT_WORK_DIR}/MALI-testing"
BASELINE_WORK_DIR="${ROOT_WORK_DIR}/suite-baseline"
TESTING_WORK_DIR="${ROOT_WORK_DIR}/suite-testing"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# clone_or_checkout <dir> <remote_url> <ref> <force>
clone_or_checkout() {
    local dir="$1" remote_url="$2" ref="$3" force="$4"

    if [[ -d "${dir}/.git" && "${force}" == "false" ]]; then
        log "Repo already present at ${dir}, skipping clone (use --force to redo)."
        return 0
    fi

    if [[ -d "${dir}" ]]; then
        log "Removing existing ${dir} (force redo)."
        rm -rf "${dir}"
    fi

    log "Cloning ${remote_url} into ${dir}"
    git clone "${remote_url}" "${dir}"

    (
        cd "${dir}"
        local current_branch
        current_branch=$(git rev-parse --abbrev-ref HEAD)
        if [[ -n "${ref}" && "${ref}" != "${current_branch}" ]]; then
            log "Checking out ${ref} in ${dir}"
            git fetch origin
            git checkout "${ref}"
        fi
    )
}

# deploy_compass_env <compass_dir> <force> -> echoes path to load script on success
deploy_compass_env() {
    local compass_dir="$1" force="$2"

    if [[ ! -f "${compass_dir}/deploy.py" ]]; then
        log "ERROR: ${compass_dir} does not look like a compass checkout (no deploy.py found)"
        exit 1
    fi

    local existing
    existing=$(find "${compass_dir}" -maxdepth 1 -name "load_compass_${MACHINE}_${COMPILER}_${MPI}.sh" 2>/dev/null | head -n1 || true)

    if [[ -n "${existing}" && "${force}" == "false" ]]; then
        log "Compass env already deployed in ${compass_dir}, reusing ${existing}."
        echo "${existing}"
        return 0
    fi

    log "Running deploy.py in ${compass_dir} (machine=${MACHINE} compiler=${COMPILER} mpi=${MPI})"
    (
        cd "${compass_dir}"
        ./deploy.py --with-albany --compiler "${COMPILER}" --mpi "${MPI}" --machine "${MACHINE}"
    )

    existing=$(find "${compass_dir}" -maxdepth 1 -name "load_compass_${MACHINE}_${COMPILER}_${MPI}.sh" 2>/dev/null | head -n1 || true)
    if [[ -z "${existing}" ]]; then
        log "ERROR: could not find generated load_compass_*.sh in ${compass_dir}"
        exit 1
    fi
    echo "${existing}"
}

# compile_mali <mali_dir> <force>
compile_mali() {
    local mali_dir="$1" force="$2"
    local landice_dir="${mali_dir}/components/mpas-albany-landice"
    local binary="${landice_dir}/landice_model"

    if [[ -f "${binary}" && "${force}" == "false" ]]; then
        log "MALI already built at ${binary}, skipping (use --force to redo)."
        return 0
    fi

    log "Compiling MALI in ${landice_dir}"
    (
        cd "${landice_dir}"
        make -j 4 gnu-cray ALBANY=true DEBUG=true
    )
}

# setup_suite <compass_dir> <work_dir> <baseline_dir_or_empty>
# Returns (echoes) the path to the compass-generated suite job script.
setup_suite() {
    local compass_dir="$1" work_dir="$2" baseline_dir="$3"
    mkdir -p "${work_dir}"

    log "Setting up ${SUITE_CORE}/${SUITE_NAME} suite in ${work_dir}"
    (
        cd "${compass_dir}"
        if [[ -n "${baseline_dir}" ]]; then
            compass suite -c "${SUITE_CORE}" -t "${SUITE_NAME}" -b "${baseline_dir}" -w "${work_dir}" -s
        else
            compass suite -c "${SUITE_CORE}" -t "${SUITE_NAME}" -w "${work_dir}" -s
        fi
    )

    local job_script
    job_script=$(find "${work_dir}" -maxdepth 1 -name "job_script.*.sh" 2>/dev/null | head -n1 || true)
    if [[ -z "${job_script}" ]]; then
        log "ERROR: could not find compass-generated job_script.*.sh in ${work_dir}"
        exit 1
    fi
    echo "${job_script}"
}

# patch_and_submit <job_script> <dependency_jobid_or_empty> -> echoes new jobid
patch_and_submit() {
    local job_script="$1" dep_jobid="$2"

    # Patch account / qos / walltime directives defensively, whatever compass put there.
    sed -i -E \
        -e "s/^#SBATCH[[:space:]]+(-A|--account=?)[[:space:]]*.*/#SBATCH -A ${ACCOUNT}/" \
        -e "s/^#SBATCH[[:space:]]+(--qos=?)[[:space:]]*.*/#SBATCH --qos=${QOS}/" \
        -e "s/^#SBATCH[[:space:]]+(-t|--time=?)[[:space:]]*.*/#SBATCH --time=${RUN_WALLTIME}/" \
        "${job_script}"

    # Add directives if they weren't present at all.
    grep -q '^#SBATCH -A' "${job_script}" || sed -i "1a #SBATCH -A ${ACCOUNT}" "${job_script}"
    grep -q '^#SBATCH --qos' "${job_script}" || sed -i "1a #SBATCH --qos=${QOS}" "${job_script}"
    grep -q '^#SBATCH --time' "${job_script}" || sed -i "1a #SBATCH --time=${RUN_WALLTIME}" "${job_script}"

    local dep_args=()
    if [[ -n "${dep_jobid}" ]]; then
        dep_args=(--dependency="afterok:${dep_jobid}")
        log "Submitting ${job_script} with dependency afterok:${dep_jobid}"
    else
        log "Submitting ${job_script}"
    fi

    local jobid
    jobid=$(sbatch --parsable "${dep_args[@]}" "${job_script}")
    echo "${jobid}"
}

# ---------------------------------------------------------------------------
# BASELINE
# ---------------------------------------------------------------------------
log "=== BASELINE: compass checkout ==="
if [[ -n "${EXISTING_COMPASS_DIR_BASELINE}" ]]; then
    log "Using existing compass checkout at ${BASELINE_COMPASS_DIR} (no clone/checkout performed)"
else
    clone_or_checkout "${BASELINE_COMPASS_DIR}" "${COMPASS_REMOTE_BASELINE}" \
        "${COMPASS_REF_BASELINE}" "${FORCE_BASELINE}"
fi

log "=== BASELINE: deploy compass env ==="
if [[ -n "${EXISTING_COMPASS_DIR_BASELINE}" ]]; then
    BASELINE_LOAD_SCRIPT=$(deploy_compass_env "${BASELINE_COMPASS_DIR}" "false")
else
    BASELINE_LOAD_SCRIPT=$(deploy_compass_env "${BASELINE_COMPASS_DIR}" "${FORCE_BASELINE}")
fi
log "Sourcing ${BASELINE_LOAD_SCRIPT}"
# shellcheck disable=SC1090
source "${BASELINE_LOAD_SCRIPT}"

log "=== BASELINE: MALI-Dev checkout ==="
clone_or_checkout "${BASELINE_MALI_DIR}" "${MALI_REMOTE_BASELINE}" \
    "${MALI_REF_BASELINE}" "${FORCE_BASELINE}"

log "=== BASELINE: compile MALI ==="
compile_mali "${BASELINE_MALI_DIR}" "${FORCE_BASELINE}"

log "=== BASELINE: compass suite setup ==="
BASELINE_JOB_SCRIPT=$(setup_suite "${BASELINE_COMPASS_DIR}" "${BASELINE_WORK_DIR}" "")

log "=== BASELINE: submit suite run job ==="
BASELINE_RUN_JOBID=$(patch_and_submit "${BASELINE_JOB_SCRIPT}" "")
echo "${BASELINE_RUN_JOBID}" > "${ROOT_WORK_DIR}/.baseline_run_jobid"
log "Baseline suite run job submitted: ${BASELINE_RUN_JOBID}"

# ---------------------------------------------------------------------------
# TESTING
# ---------------------------------------------------------------------------
if [[ "${SAME_COMPASS}" == "true" ]]; then
    TESTING_COMPASS_DIR="${BASELINE_COMPASS_DIR}"
    TESTING_LOAD_SCRIPT="${BASELINE_LOAD_SCRIPT}"
    log "=== TESTING: reusing baseline compass checkout (${TESTING_COMPASS_DIR}) ==="
else
    log "=== TESTING: separate compass checkout (remote=${COMPASS_REMOTE_TESTING} ref=${COMPASS_REF_TESTING}) ==="
    if [[ -n "${EXISTING_COMPASS_DIR_TESTING}" ]]; then
        log "Using existing compass checkout at ${TESTING_COMPASS_DIR} (no clone/checkout performed)"
    else
        clone_or_checkout "${TESTING_COMPASS_DIR}" "${COMPASS_REMOTE_TESTING}" \
            "${COMPASS_REF_TESTING}" "${FORCE_TESTING}"
    fi

    log "=== TESTING: deploy compass env ==="
    if [[ -n "${EXISTING_COMPASS_DIR_TESTING}" ]]; then
        TESTING_LOAD_SCRIPT=$(deploy_compass_env "${TESTING_COMPASS_DIR}" "false")
    else
        TESTING_LOAD_SCRIPT=$(deploy_compass_env "${TESTING_COMPASS_DIR}" "${FORCE_TESTING}")
    fi
    log "Sourcing ${TESTING_LOAD_SCRIPT}"
    # shellcheck disable=SC1090
    source "${TESTING_LOAD_SCRIPT}"
fi

log "=== TESTING: MALI-Dev checkout ==="
clone_or_checkout "${TESTING_MALI_DIR}" "${MALI_REMOTE_TESTING}" \
    "${MALI_REF_TESTING}" "${FORCE_TESTING}"

log "=== TESTING: compile MALI ==="
compile_mali "${TESTING_MALI_DIR}" "${FORCE_TESTING}"

log "=== TESTING: compass suite setup (baseline=${BASELINE_WORK_DIR}) ==="
TESTING_JOB_SCRIPT=$(setup_suite "${TESTING_COMPASS_DIR}" "${TESTING_WORK_DIR}" "${BASELINE_WORK_DIR}")

log "=== TESTING: submit suite run job (depends on baseline run ${BASELINE_RUN_JOBID}) ==="
TESTING_RUN_JOBID=$(patch_and_submit "${TESTING_JOB_SCRIPT}" "${BASELINE_RUN_JOBID}")
echo "${TESTING_RUN_JOBID}" > "${ROOT_WORK_DIR}/.testing_run_jobid"

log "=== DONE ==="
log "Baseline suite run job: ${BASELINE_RUN_JOBID}  (work dir: ${BASELINE_WORK_DIR})"
log "Testing  suite run job: ${TESTING_RUN_JOBID}  (work dir: ${TESTING_WORK_DIR}, depends on ${BASELINE_RUN_JOBID})"
log "Monitor with: squeue -u \$USER"
