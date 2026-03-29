#!/bin/bash
# Interconnect Comparison Experiments: Optical vs Electrical
# Sweeps bandwidth and latency across all DNN benchmarks with accelerator offloading
#
# Interconnect configurations:
#   Electrical (PCIe Gen4/5):  BW = 64-256 GB/s, Latency = 20-50 cycles
#   Optical (photonic):        BW = 512-2048 GB/s, Latency = 2-5 cycles
#
# Usage: bash experiments/interconnect_comparison.sh
# Results go into experiments/results/<benchmark>/<config>/

set -e

export PATH=$HOME/.local/go/bin:$HOME/go/bin:$PATH

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="$REPO_ROOT/experiments/results"
SAMPLES_DIR="$REPO_ROOT/amd/samples"

mkdir -p "$RESULTS_DIR"

# --- Interconnect configurations ---
# Format: "name bw_bytes_per_sec latency_cycles"
CONFIGS=(
    # Electrical (PCIe-class)
    "pcie_gen4_x16    32000000000   50"   # 32 GB/s, 50 cycles (~PCIe Gen4 x16)
    "pcie_gen5_x16    64000000000   40"   # 64 GB/s, 40 cycles (~PCIe Gen5 x16)
    "pcie_gen6_x16   128000000000   30"   # 128 GB/s, 30 cycles (~PCIe Gen6 x16)
    "elec_high_bw    256000000000   20"   # 256 GB/s, 20 cycles (high-end electrical)

    # Optical (photonic interconnect)
    "optical_low      512000000000   5"   # 512 GB/s, 5 cycles (conservative optical)
    "optical_mid     1024000000000   3"   # 1 TB/s, 3 cycles (mid-range optical)
    "optical_high    2048000000000   2"   # 2 TB/s, 2 cycles (aggressive optical)
)

# --- Benchmarks ---
# Format: "name dir_name gpu_flags extra_flags"
BENCHMARKS=(
    "minerva  minerva  1,2  -epoch=1"
    "lenet    lenet    1,2  -epoch=1"
    "vgg16    vgg16    1,2  -epoch=1 -max-batch-per-epoch=2 -batch-size=8"
)

run_experiment() {
    local bench_name="$1"
    local bench_dir="$2"
    local gpu_flags="$3"
    local extra_flags="$4"
    local config_name="$5"
    local bw="$6"
    local latency="$7"

    local out_dir="$RESULTS_DIR/${bench_name}/${config_name}"
    mkdir -p "$out_dir"

    # Skip if already completed
    if [ -f "$out_dir/done.marker" ]; then
        echo "  [SKIP] $bench_name / $config_name (already done)"
        return 0
    fi

    local sample_dir="$SAMPLES_DIR/$bench_dir"
    cd "$sample_dir"

    # Build if needed
    if [ ! -f "$bench_dir" ] && [ ! -f "$bench_name" ]; then
        go build 2>&1
    fi

    local binary
    binary=$(ls -1 "$bench_dir" "$bench_name" 2>/dev/null | head -1)
    if [ -z "$binary" ]; then
        # binary name matches directory name
        binary="./$bench_dir"
    else
        binary="./$binary"
    fi

    echo "  [RUN]  $bench_name / $config_name (BW=${bw}, Lat=${latency})"

    # Run simulation
    $binary \
        -timing \
        -gpus="$gpu_flags" \
        -num-accel=2 \
        -accel-config=accel_config.json \
        -accel-interconnect-bw="$bw" \
        -accel-interconnect-latency="$latency" \
        --report-all \
        $extra_flags \
        > "$out_dir/stdout.log" 2>&1

    # Move the SQLite results file
    local db_file
    db_file=$(ls -1t akita_sim_*.sqlite3 2>/dev/null | head -1)
    if [ -n "$db_file" ]; then
        mv "$db_file" "$out_dir/metrics.sqlite3"
    fi

    touch "$out_dir/done.marker"
    echo "  [DONE] $bench_name / $config_name"
}

echo "========================================"
echo "Interconnect Comparison Experiments"
echo "========================================"
echo ""
echo "Benchmarks: ${#BENCHMARKS[@]}"
echo "Configs:    ${#CONFIGS[@]}"
echo "Total runs: $(( ${#BENCHMARKS[@]} * ${#CONFIGS[@]} ))"
echo ""

for bench_spec in "${BENCHMARKS[@]}"; do
    read -r bench_name bench_dir gpu_flags extra_flags <<< "$bench_spec"
    echo "--- Benchmark: $bench_name ---"

    for config_spec in "${CONFIGS[@]}"; do
        read -r config_name bw latency <<< "$config_spec"
        run_experiment "$bench_name" "$bench_dir" "$gpu_flags" "$extra_flags" \
                       "$config_name" "$bw" "$latency"
    done

    echo ""
done

echo "========================================"
echo "All experiments complete!"
echo "Results in: $RESULTS_DIR"
echo "========================================"
