#!/bin/bash
# VGG16 interconnect sweep: 7 configs x 1 epoch, 2 batches
set -e
export PATH=$HOME/.local/go/bin:$HOME/go/bin:$PATH

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="$REPO_ROOT/experiments/results/vgg16"
SAMPLE_DIR="$REPO_ROOT/amd/samples/vgg16"

mkdir -p "$RESULTS_DIR"
cd "$SAMPLE_DIR"

declare -a CONFIGS=(
    "pcie_gen4_x16    32000000000   50"
    "pcie_gen5_x16    64000000000   40"
    "pcie_gen6_x16   128000000000   30"
    "elec_high_bw    256000000000   20"
    "optical_low      512000000000   5"
    "optical_mid     1024000000000   3"
    "optical_high    2048000000000   2"
)

for config_spec in "${CONFIGS[@]}"; do
    read -r name bw latency <<< "$config_spec"
    out_dir="$RESULTS_DIR/$name"
    mkdir -p "$out_dir"

    if [ -f "$out_dir/done.marker" ]; then
        echo "[SKIP] $name"
        continue
    fi

    echo "[RUN]  $name (BW=$bw, Lat=$latency)"
    rm -f akita_sim_*.sqlite3

    ./vgg16 \
        -timing \
        -gpus=1,2 \
        -num-accel=2 \
        -accel-config=accel_config.json \
        -accel-interconnect-bw="$bw" \
        -accel-interconnect-latency="$latency" \
        --report-all \
        -epoch=1 \
        -max-batch-per-epoch=2 \
        -batch-size=8 \
        > "$out_dir/stdout.log" 2>&1

    db_file=$(ls -1t akita_sim_*.sqlite3 2>/dev/null | head -1)
    if [ -n "$db_file" ]; then
        mv "$db_file" "$out_dir/metrics.sqlite3"
    fi

    touch "$out_dir/done.marker"
    echo "[DONE] $name"
done

echo "VGG16 sweep complete! Results in $RESULTS_DIR"
