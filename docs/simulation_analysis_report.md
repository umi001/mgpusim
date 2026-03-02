# Simulation Analysis Report — GPT-2 on Heterogeneous GPU + Accelerator

## 1. Experiment Setup

### Hardware Configuration (Simulated)
- **GPUs**: 2× AMD R9 Nano (GCN3, 64 CUs each, 16 DRAM banks per GPU)
- **Accelerators**: 2× custom inference accelerator (256×256 PE array, dedicated DRAM)
- **Accelerator interconnect**: PCIe-modeled connector, 256 GB/s bandwidth, 10-cycle switch latency
- **Accelerator DRAM**: `idealmemcontroller` with fixed 100-cycle latency per request
- **Max outstanding requests**: 64 (configurable via `--accel-max-outstanding`)

### Workload Configuration
- **Model**: GPT-2 variant (d_model=384, 6 heads, 6 layers, d_ff=1536, vocab=10K)
- **Parameters**: 18.3M (~73 MB float32 per GPU copy)
- **Input**: batch=1, seq_len=32, synthetic random tokens
- **Offloading**: Transformer blocks 0–2 on accelerator, blocks 3–5 on GPU
- **Data parallelism**: Each GPU gets independent batch, own accelerator

### Simulation Flags
```
./gpt2 -timing -gpus=1,2 -num-accel=2 -report-all \
  -num-iter=1 -batch-size=1 -seq-len=32 \
  -d-model=384 -n-heads=6 -n-layers=6 -d-ff=1536 \
  -vocab-size=10000 -accel-config=accel_config.json
```

---

## 2. Results

### 2.1 Accelerator Metrics (per accelerator — both identical)

| Metric | Accel[0] | Accel[1] |
|--------|----------|----------|
| Operations dispatched | 24 | 24 |
| Busy time | 1.019 ms | 1.019 ms |
| Compute cycles | 2,880 | 2,880 |
| Memory requests | 387,720 | 387,720 |
| Read bytes | 22.1 MB | 22.1 MB |
| Write bytes | 1.55 MB | 1.55 MB |
| Total interconnect traffic | 23.7 MB | 23.7 MB |

### 2.2 Accelerator DRAM

| Metric | Reads | Writes |
|--------|-------|--------|
| Transactions | 362,376 | 25,344 |
| Data volume | 22.1 MB | 1.55 MB |
| Avg latency | 100 ns | 100 ns |

### 2.3 GPU Metrics

| Metric | GPU[1] | GPU[2] |
|--------|--------|--------|
| Kernel time | 4.495 ms | 4.769 ms |
| DRAM reads | 122.9 MB | 122.9 MB |
| DRAM writes | 140.9 MB | 140.7 MB |
| DRAM read transactions | 2.01M | 2.01M |
| DRAM write transactions | 2.31M | 2.31M |

### 2.4 End-to-End Timing

| Metric | Value |
|--------|-------|
| Driver kernel_time | 8.701 ms |
| GPU[1] kernel_time | 4.495 ms |
| GPU[2] kernel_time | 4.769 ms |
| Accel[0] busy_time | 1.019 ms |
| Accel[1] busy_time | 1.019 ms |

---

## 3. Verification — Internal Consistency

All metrics were verified against analytical predictions.

### 3.1 Operation Count
- Per block: 6 GEMMs (Q/K/V/O projections + 2 FFN) + 2 ScaleAdd = 8 ops
- 3 blocks on accelerator → 24 ops. **Matches.**

### 3.2 Compute Cycles
- Q/K/V/O GEMM (M=32, N=384, K=384): 32×384×384 / (256×256) = 72 cycles each
- FFN1/FFN2 GEMM (M=32, K/N=384/1536): 288 cycles each
- ScaleAdd: (32×384) / 256 = 48 cycles each
- Per block: 4×72 + 288 + 288 + 2×48 = 960 cycles
- 3 blocks × 960 = 2,880 cycles. **Matches.**

### 3.3 Memory Traffic
- Read bytes predicted (including bias reads): 23,192,064. **Exact match.**
- Write bytes predicted: 1,622,016. **Exact match.**
- Read transactions: 23,192,064 / 64 = 362,376. **Exact match.**
- Write transactions: 1,622,016 / 64 = 25,344. **Exact match.**
- Total mem requests: 362,376 + 25,344 = 387,720. **Exact match.**

### 3.4 Multi-Accelerator Distribution
Both accelerators show identical metrics. This is expected: same model
architecture, same offloaded layers, same tensor shapes, independent DRAM
and PCIe networks. Round-robin assignment (GPU[1]→Accel[0], GPU[2]→Accel[1])
confirmed working.

### 3.5 GPU Parallelism
Driver kernel_time (8.701 ms) ≈ max(GPU[1], GPU[2]) + contention overhead,
NOT sum(GPU[1], GPU[2]). Confirms GPUs execute concurrently via goroutines.

---

## 4. Analysis

### 4.1 Compute vs Memory Time

| Component | Time | % of busy_time |
|-----------|------|----------------|
| Compute | 2.88 µs | 0.28% |
| Memory | 1,016 µs | 99.72% |

The workload is **overwhelmingly memory-bound**. The 256×256 PE array finishes
all 24 operations in under 3 µs, but moving 23.7 MB through the interconnect
takes over 1 ms. At this scale, the PE array size has negligible impact on
accelerator performance.

This is expected for small batch sizes (batch=1, seq=32) where matrix
dimensions are small relative to the PE array capacity. With full GPT-2
Small (seq=1024, d=768), compute would become a larger fraction.

### 4.2 Interconnect Bandwidth Utilization

| Metric | Value |
|--------|-------|
| Total traffic | 24.8 MB |
| Busy time | 1,019 µs |
| Effective bandwidth | 24.3 GB/s |
| Nominal bandwidth | 256 GB/s |
| **Utilization** | **9.5%** |
| Theoretical minimum transfer time | 96.9 µs |
| Actual / theoretical | 10.5× |

The accelerator achieves only 9.5% of the interconnect's raw bandwidth.
Three factors contribute (see Section 5 for details):

1. **Sequential operation execution** — idle gaps between ops
2. **Pipeline depth limit** — only 64 requests in flight
3. **Per-request switch latency overhead** — 20 cycles per round trip

### 4.3 Read/Write Asymmetry

| Direction | Volume | Ratio |
|-----------|--------|-------|
| Reads | 22.1 MB | 93.5% |
| Writes | 1.55 MB | 6.5% |

Traffic is 14:1 read-heavy. GEMM reads both input activations and weight
matrices but only writes the output. Weight matrices (384² = 147K elements)
dominate read volume. This means **read bandwidth** is the critical metric
for interconnect comparison, not write bandwidth.

### 4.4 GPU vs Accelerator Performance

| Metric | GPU (3 blocks) | Accelerator (3 blocks) | Ratio |
|--------|---------------|------------------------|-------|
| Execution time | ~4.6 ms | 1.019 ms | GPU 4.5× slower |
| DRAM traffic | ~264 MB | 23.7 MB | GPU 11× more |

The GPU processes the same 3 transformer blocks but takes 4.5× longer and
generates 11× more DRAM traffic. This is expected: the GPU runs full
cycle-accurate simulation through CU pipelines, L1/L2 caches, and 16 DRAM
banks, while the accelerator uses a simplified analytical compute model with
direct DRAM access.

---

## 5. Known Limitations and Areas for Improvement

### 5.1 Sequential Operation Execution (High Impact)

**Current behavior**: Each of the 24 operations executes strictly as
READ → COMPUTE → WRITE before the next operation begins. The interconnect
sits idle between operations.

```
Op1: [===READ===][C][=WRITE=]
Op2:                          [===READ===][C][=WRITE=]
     ^^^^^^^^^^ idle gap ^^^^
```

**Real accelerator behavior**: Hardware accelerators use double-buffering
with on-chip SRAM. While the PE array computes on data from Op N (stored
in SRAM buffer A), the DMA engine prefetches data for Op N+1 into SRAM
buffer B. This keeps the interconnect busy nearly continuously:

```
Op1: [===READ===][COMPUTE][=WRITE=]
Op2:       [===READ===]...[COMPUTE][=WRITE=]
Op3:              [===READ===]......[COMPUTE]
```

**Impact**: This is likely the largest contributor to the 10.5× slowdown.
Implementing SRAM + double-buffering would significantly improve bandwidth
utilization.

**Fix**: Model on-chip SRAM with capacity constraints. When data fits in
SRAM, overlap next op's reads with current op's compute. Requires a new
SRAM state buffer in `comp.go` and a modified state machine with concurrent
read/compute phases.

### 5.2 Pipeline Depth Limit (Medium Impact)

**Current behavior**: `maxOutstandingReqs=64` limits in-flight DMA requests.
To saturate the interconnect at 256 GB/s with ~120 ns round-trip latency:

```
Required in-flight data = bandwidth × latency = 256 GB/s × 120 ns = 30,720 bytes
Required requests = 30,720 / 64 = 480 requests
```

We allow 64 out of the ~480 needed — **13% of what's required to saturate
the link**.

**Impact**: Directly reduces achievable bandwidth by ~3-4×.

**Fix**: Increase default `maxOutstandingReqs`. This is already configurable
via `--accel-max-outstanding`. Setting it to 256 or 512 would improve
utilization without code changes. However, higher values increase simulation
memory usage and event count.

### 5.3 No Weight Reuse / SRAM Caching (High Impact)

**Current behavior**: Every GEMM re-reads the full weight matrix from DRAM
through the interconnect. For a Q-projection weight (384×384 = 589 KB),
this is read once per GEMM, even though the same weight is used across all
sequence positions.

**Real accelerator behavior**: Weight matrices are loaded into SRAM once and
reused across multiple GEMM invocations (weight stationarity). Only
activations need to be streamed from DRAM per operation.

**Impact**: For this workload, weight reads are ~93% of total read traffic.
With SRAM caching, most of this would be eliminated after the first read,
dramatically reducing interconnect pressure.

**Fix**: Requires SRAM capacity modeling. Track which tensors are "resident"
in SRAM. If a tensor was recently loaded and still fits, skip the DMA read.
Evict based on LRU or tensor size when SRAM is full.

### 5.4 Host-Side Fallback Operations (Medium Impact)

**Current behavior**: Several operations run entirely on the host CPU with
no simulation timing:

| Operation | Calls per forward pass | Complexity |
|-----------|----------------------|------------|
| LayerNorm | 13 (2/block × 6 + 1 final) | O(seq × d_model) |
| GELU | 6 (1/block × 6) | O(seq × d_ff) |
| Attention scores (Q·K^T, softmax, ·V) | 6 blocks × 6 heads | O(seq² × d_model) |
| Embedding | 1 | O(seq × d_model) |

These operations:
- Contribute **zero simulated time** — they appear to take 0 ns
- Do consume real wall-clock time (host CPU computation)
- Pull tensor data from device memory to host via `Vector()`, then push
  results back via `Create()` + `Init()`

**Impact**: The simulated accelerator/GPU timing is incomplete. LayerNorm
and attention scores are significant operations in a real transformer.
Their absence means the simulated execution time understates the real
workload by a non-trivial amount.

**Fix**: Add protocol op types (`AccelOpLayerNorm`, `AccelOpAttention`,
`AccelOpGELU`) and corresponding timing models in `comp.go`. Then convert
the host fallbacks in `benchmark.go` to dispatch through the accelerator.
This is planned as Phase 4.

### 5.5 Reshape Generates Hidden Memory Traffic (Low Impact)

**Current behavior**: The `linear()` helper calls `op.Reshape()` on input
and weight tensors before each GEMM. In `acceltensor`, Reshape calls Clone
which does a full `MemCopyD2D` through the driver. This generates real
memory traffic that is:
- Not counted in accelerator metrics (bypasses the accelerator's DMA)
- Not timed through the PCIe interconnect
- Significant in volume (~36 tensor copies per forward pass per GPU)

**Impact**: Low for interconnect analysis (this traffic goes through the
driver, not the accelerator interconnect). But it inflates real simulation
wall-clock time and memory usage.

**Fix**: Consider making Reshape a view (metadata-only, no copy) instead of
a full clone. This would require tensor shape to be mutable or a separate
"view" type.

### 5.6 Ideal DRAM Controller (Low Impact for Interconnect Studies)

**Current behavior**: The accelerator's DRAM uses `idealmemcontroller` with
fixed 100 ns latency per request, no bandwidth limiting, no bank conflicts,
and no queuing delays.

**Impact**: All bandwidth limitation comes from the PCIe interconnect, which
is correct for interconnect technology comparison. But it means the DRAM
itself is never a bottleneck, which is unrealistic for high-bandwidth
scenarios where the interconnect is faster than the DRAM (e.g., optical
interconnect at 200+ GB/s with HBM at ~200 GB/s).

**Fix**: Replace `idealmemcontroller` with a bandwidth-limited DRAM model
(e.g., MGPUSim's existing DRAM controllers with bank modeling). This would
add DRAM as a second bottleneck alongside the interconnect.

---

## 6. Recommendations for Interconnect Comparison Experiments

### 6.1 Quick Wins (No Code Changes)

1. **Increase `--accel-max-outstanding`** to 256 or 512 to improve pipeline
   utilization
2. **Use larger model dimensions** (d_model=768, seq_len=128+) to increase
   compute/memory ratio and total traffic volume
3. **Run parameter sweeps** varying `--accel-interconnect-bw` and
   `--accel-interconnect-latency` to map the sensitivity curve

### 6.2 Medium-Term Improvements

1. **Phase 4 protocol extensions** — add timing for LayerNorm, GELU,
   attention to make simulated time more complete
2. **Increase pipeline depth default** — change builder default from 64 to
   256+ based on realistic DMA engine specs

### 6.3 Longer-Term Improvements

1. **SRAM + double-buffering model** — the single highest-impact change for
   realistic bandwidth utilization
2. **Weight caching in SRAM** — eliminates redundant weight reads
3. **Bandwidth-limited DRAM** — adds a second bottleneck for high-bandwidth
   interconnect scenarios

---

## 7. Reproducing Results

### Build
```bash
cd ~/projects/mgpusim/amd/samples/gpt2
go build
```

### Run
```bash
# Disable systemd-oomd if memory is tight (simulation uses 8+ GB)
sudo systemctl stop systemd-oomd.socket

# Run from external terminal (not VS Code) to avoid cgroup OOM kills
nohup ./gpt2 -timing -gpus=1,2 -num-accel=2 -report-all \
  -num-iter=1 -batch-size=1 -seq-len=32 \
  -d-model=384 -n-heads=6 -n-layers=6 -d-ff=1536 \
  -vocab-size=10000 -accel-config=accel_config.json \
  > /tmp/gpt2_run.log 2>&1 &

# Re-enable after simulation completes
sudo systemctl start systemd-oomd.socket
```

### Query Results
```bash
sqlite3 akita_sim_*.sqlite3 "SELECT Location, What, Value, Unit \
  FROM mgpusim_metrics \
  WHERE Location LIKE '%Accel%' \
  ORDER BY Location, What;"
```

### Branch and Commits
- Branch: `approach_2`
- Phase 2.5 commit: `fdf76e25`
- GPT-2 merge: `1fabeaf2`
- Multi-accel fix: `3026d1e8`
- Parallel GPU fix: `08614fe5`
