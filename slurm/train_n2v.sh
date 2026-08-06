#!/bin/bash
#SBATCH --account=dlthings
#SBATCH --job-name=n2v_train
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="infiniband&gpuram32gb"
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Train Noise2Void on the cluster.
#
#   sbatch slurm/train_n2v.sh --noisy-dir /path/to/noisy --out-dir /path/to/experiment
#
# Prerequisite: run ./install.sh once on a LOGIN node. Compute nodes have no outbound
# network, so the pixi environment must already be solved before the job starts.
#
# --account/--partition/--constraint above are copied from the working setup in
# napari_sam. Adjust them if your allocation differs.

set -euo pipefail

function usage() {
    echo "Usage: sbatch $0 --noisy-dir <dir> --out-dir <dir> [options]"
    echo "  -n, --noisy-dir    Directory of noisy TIFFs (required)"
    echo "  -o, --out-dir      Output dir for checkpoints/ and logs/ (required)"
    echo "      --epochs       Training epochs (default: 50)"
    echo "      --batch-size   Batch size (default: 32)"
    echo "      --patch-size   Crop size, multiple of 8 (default: 128)"
    echo "      --lr           Learning rate (default: 3e-4)"
    echo "      --mask-ratio   Fraction of blind-spot pixels (default: 0.02)"
    echo "  -h, --help         Show this help"
}

function display_memory_usage() {
    set +eu
    echo -n "[INFO] [$(date -Iseconds)] [$$] Max memory usage in bytes: "
    cat /sys/fs/cgroup/memory/slurm/uid_$(id -u)/job_${SLURM_JOB_ID}/memory.max_usage_in_bytes
    echo
}
trap display_memory_usage EXIT

START=$(date +%s)
STARTDATE=$(date -Iseconds)
echo "[INFO] [$STARTDATE] [$$] Starting SLURM job ${SLURM_JOB_ID:-local}"
echo "[INFO] [$STARTDATE] [$$] Running on $(hostname -s)"
echo "[INFO] [$STARTDATE] [$$] Working directory: $(pwd)"

### ---------------------------------------------------------------- arguments
NOISY_DIR=""
OUT_DIR=""
EPOCHS=50
BATCH_SIZE=32
PATCH_SIZE=128
LR=3e-4
MASK_RATIO=0.02

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--noisy-dir)  NOISY_DIR="$2"; shift 2 ;;
        -o|--out-dir)    OUT_DIR="$2";   shift 2 ;;
        --epochs)        EPOCHS="$2";    shift 2 ;;
        --batch-size)    BATCH_SIZE="$2"; shift 2 ;;
        --patch-size)    PATCH_SIZE="$2"; shift 2 ;;
        --lr)            LR="$2";        shift 2 ;;
        --mask-ratio)    MASK_RATIO="$2"; shift 2 ;;
        -h|--help)       usage; exit 0 ;;
        --)              shift; break ;;
        *)               echo "[ERROR] Unknown argument: $1"; usage; exit 2 ;;
    esac
done

if [[ -z "$NOISY_DIR" || -z "$OUT_DIR" ]]; then
    echo "[ERROR] --noisy-dir and --out-dir are required"
    usage
    exit 2
fi

if [[ ! -d "$NOISY_DIR" ]]; then
    echo "[ERROR] Input directory does not exist: $NOISY_DIR"
    exit 2
fi

### ---------------------------------------------------------------- environment
module purge

# Match thread counts to the CPU allocation; the default would oversubscribe the node.
THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"
echo "[INFO] [$STARTDATE] [$$] Using $THREADS CPU threads"

WD="$(realpath .)"
export PATH=$PATH:"$WD/infrastructure/apps/pixi/bin"
export PIXI_CACHE_DIR="$WD/infrastructure/apps/pixi/.pixi_cache"
export TMPDIR="$WD/infrastructure/.tmp_$USER"
mkdir -p "$TMPDIR" logs

if [[ ! -x "$WD/infrastructure/apps/pixi/bin/pixi" ]]; then
    echo "[ERROR] pixi not found. Run ./install.sh on a login node first."
    exit 1
fi

# --frozen: fail loudly instead of trying to re-solve (which needs network we do not have).
PIXI="$WD/infrastructure/apps/pixi/bin/pixi"
echo "[INFO] GPU visible to the job:"
$PIXI run --frozen python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU - falling back to CPU')"

### ---------------------------------------------------------------- run
$PIXI run --frozen python -m em_denoise.n2v.train_n2v train \
    "$NOISY_DIR" \
    "$OUT_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --patch-size "$PATCH_SIZE" \
    --lr "$LR" \
    --mask-ratio "$MASK_RATIO" \
    --num-workers "$THREADS"

END=$(date +%s)
ENDDATE=$(date -Iseconds)
echo "[INFO] [$ENDDATE] [$$] Execution time (seconds): $(( END - START ))"
echo "[INFO] [$ENDDATE] [$$] Checkpoints in: $OUT_DIR/checkpoints"
