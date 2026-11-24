#!/bin/bash
#SBATCH --account=fri-contom
#SBATCH --job-name=em_n2n_train
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --constraint=infiniband
#SBATCH --output=logs/n2n-%j.out
#SBATCH --error=logs/n2n-%j.err

set -euo pipefail

START=$(date +%s)
STARTDATE=$(date -Iseconds)
echo "[INFO] [$STARTDATE] Job $SLURM_JOB_ID on $(hostname -s)"
echo "[INFO] [$STARTDATE] Working directory: $(pwd)"

# 1. Disable BLAS/OpenMP internal oversubscription
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLOSC_NTHREADS=1

# Redirect Triton / TorchInductor caches off read-only $HOME onto local node scratch
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/triton_cache_${SLURM_JOB_ID:-$$}"
export TORCHINDUCTOR_CACHE_DIR="${TMPDIR:-/tmp}/inductor_cache_${SLURM_JOB_ID:-$$}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

# 2. Repo & environment on /tachyon (adjust REPO_DIR if cloned elsewhere)
REPO_DIR="/tachyon/groups/scratch/gmicro_sem/gfriedri/tgan/em_denoise"
cd "$REPO_DIR"
mkdir -p logs

# Expose .venv directly so compute nodes never touch read-only ~/.cache/uv
export PATH="$REPO_DIR/.venv/bin:$PATH"

CONFIG="$REPO_DIR/src/em_denoise_refactor/config_n2n_slurm.yaml"

# 3. Verify GPU visibility
python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

# 4. Run N2N training + evaluation
python -m em_denoise_refactor.execution.train_and_infer --config "$CONFIG"
EXITCODE=$?

END=$(date +%s)
ENDDATE=$(date -Iseconds)
echo "[INFO] [$ENDDATE] Job $SLURM_JOB_ID finished with exit code $EXITCODE"
echo "[INFO] [$ENDDATE] Execution time: $(( END - START )) seconds"
exit $EXITCODE
