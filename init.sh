#!/usr/bin/env bash
# Initialize pixi paths, cache and temp dirs. Source this before calling pixi by hand:
#   source init.sh && pixi run python -m em_denoise.n2v.train_n2v --help
WD="$(realpath .)"
export PATH=$PATH:"$WD/infrastructure/apps/pixi/bin"
export PIXI_CACHE_DIR="$WD/infrastructure/apps/pixi/.pixi_cache"
export TMPDIR="$WD/infrastructure/.tmp_$USER"
mkdir -p "$TMPDIR"
