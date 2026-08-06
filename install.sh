#!/usr/bin/env bash
# Install pixi into the repo-local infrastructure directory.
#
# Run this ONCE on a login node (compute nodes have no outbound network). Same pattern as
# napari_sam / vlad_segmentation, so pixi never lands in the quota-limited home directory.
set -euo pipefail

export PIXI_HOME="$(pwd)/infrastructure/apps/pixi"
export PIXI_NO_PATH_UPDATE=1
export TMP_DIR="$(pwd)/infrastructure/.tmp_$USER"
mkdir -p "$TMP_DIR"

curl -fsSL https://pixi.sh/install.sh | bash

# Solve and materialize the environment now, while the network is still reachable.
export PATH="$PATH:$PIXI_HOME/bin"
export PIXI_CACHE_DIR="$PIXI_HOME/.pixi_cache"
pixi install

echo "Done. Submit jobs with: sbatch slurm/train_n2v.sh --noisy-dir ... --out-dir ..."
