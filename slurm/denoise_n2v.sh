#!/bin/bash
#SBATCH --account=dlthings
#SBATCH --job-name=n2v_denoise
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="infiniband&gpuram32gb"
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Apply a trained Noise2Void checkpoint to a directory of TIFFs.
#
#   sbatch slurm/denoise_n2v.sh -i /path/in -o /path/out -c /path/checkpoints/best_model.pt

set -euo pipefail

function usage() {
    echo "Usage: sbatch $0 -i <input_dir> -o <output_dir> -c <checkpoint> [--tile-size N]"
}

INPUT_DIR=""
OUTPUT_DIR=""
CHECKPOINT=""
TILE_SIZE=512

while [[ $# -gt 0 ]]; do
    case "$1" in
        -i|--input)      INPUT_DIR="$2";  shift 2 ;;
        -o|--output)     OUTPUT_DIR="$2"; shift 2 ;;
        -c|--checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --tile-size)     TILE_SIZE="$2";  shift 2 ;;
        -h|--help)       usage; exit 0 ;;
        *)               echo "[ERROR] Unknown argument: $1"; usage; exit 2 ;;
    esac
done

if [[ -z "$INPUT_DIR" || -z "$OUTPUT_DIR" || -z "$CHECKPOINT" ]]; then
    echo "[ERROR] --input, --output and --checkpoint are required"
    usage
    exit 2
fi

module purge
THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="$THREADS"

WD="$(realpath .)"
export PATH=$PATH:"$WD/infrastructure/apps/pixi/bin"
export PIXI_CACHE_DIR="$WD/infrastructure/apps/pixi/.pixi_cache"
export TMPDIR="$WD/infrastructure/.tmp_$USER"
mkdir -p "$TMPDIR" logs

"$WD/infrastructure/apps/pixi/bin/pixi" run --frozen python -m em_denoise.n2v.train_n2v denoise \
    "$INPUT_DIR" \
    "$OUTPUT_DIR" \
    "$CHECKPOINT" \
    --tile-size "$TILE_SIZE"

echo "[INFO] Denoised images written to: $OUTPUT_DIR"
