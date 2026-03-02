# Heterogeneous Accelerator Implementation & GPT-2 Benchmark Analysis

## Part 1: Accelerator Implementation

### 1. Combined Memory Hierarchy

The diagram below shows the full memory hierarchy for both GPU and accelerator
as implemented in our codebase. Both devices share the same `globalStorage`
backing store.

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                              System PCIe (commands only)                              │
│                                                                                      │
│    Driver ←──→ RootComplex ←──→ ┌──────────────────┐  ┌─────────────────────┐        │
│    (gpuPort,     (PCIe 4.0)     │ GPU[i] Switch    │  │ Accel[j] Switch     │        │
│     accelPort,                  │  - CP port       │  │  - ToDriver port    │        │
│     mmuPort)                    │  - RDMA ports    │  │    (commands only)   │        │
│                                 │  - PMC port      │  │                     │        │
│                                 │  - L2TLB ports   │  │                     │        │
│                                 └──────────────────┘  └─────────────────────┘        │
└──────────────────────────────────────────────────────────────────────────────────────┘

┌─ GPU Side (per GPU) ─────────────────────┐    ┌─ Accelerator Side (per Accel) ──────┐
│                                          │    │                                      │
│  ┌──────────────────────────────────┐    │    │  ┌──────────────────────────────┐    │
│  │  Compute Units (4-6 per SA)     │    │    │  │  AccelUnit                   │    │
│  │  - Wavefront execution          │    │    │  │  - PE Array 256×256          │    │
│  │  - 64 threads per wavefront     │    │    │  │  - 32MB SRAM (not modeled)   │    │
│  │  - Per-CU register files        │    │    │  │  - Analytical timing         │    │
│  └──────┬────────┬────────┬────────┘    │    │  │  - 3-phase state machine     │    │
│         │        │        │             │    │  │    READ→COMPUTE→WRITE        │    │
│   Vector│  Scalar│  Instr │             │    │  └──────────────┬───────────────┘    │
│         ▼        ▼        ▼             │    │                 │ ToMem              │
│  ┌──────────┐ ┌──────┐ ┌──────┐         │    │                 │ (64B requests)     │
│  │ L1V ROB  │ │L1S   │ │L1I   │         │    │                 ▼                    │
│  │ AddrTrans│ │ROB   │ │Cache │         │    │  ┌──────────────────────────────┐    │
│  │ L1V TLB  │ │AddrT │ │AddrT │         │    │  │  Internal PCIe Connector     │    │
│  │ L1V Cache│ │L1S   │ │L1I   │         │    │  │  - Bandwidth: configurable   │    │
│  │ (per CU) │ │TLB   │ │TLB   │         │    │  │    default 256 GB/s          │    │
│  └────┬─────┘ │Cache │ │      │         │    │  │  - Switch latency: config    │    │
│       │       └──┬───┘ └──┬───┘         │    │  │    default 10 cycles         │    │
│       │          │        │             │    │  │  - Flit-based serialization   │    │
│       └──────────┴────────┘             │    │  │  - maxOutstandingReqs: 64     │    │
│                  │                      │    │  └──────────────┬───────────────┘    │
│         L1ToL2Conn (DirectConnection)   │    │                 │                    │
│                  │                      │    │                 │                    │
│                  ▼                      │    │                 ▼                    │
│  ┌──────────────────────────────────┐   │    │  ┌──────────────────────────────┐    │
│  │  L2 Cache × 16                  │   │    │  │  DRAM Controller             │    │
│  │  - Interleaved (128B granularity)│   │    │  │  - Ideal (100 cycles flat)   │    │
│  │  - 16 cache slices              │   │    │  │  - Single controller         │    │
│  └──────────────┬───────────────────┘   │    │  │  - No bank conflicts         │    │
│                 │                       │    │  └──────────────┬───────────────┘    │
│    L2ToDRAMConn (DirectConnection)      │    │                 │                    │
│                 │                       │    │                 │                    │
│                 ▼                       │    │                 │                    │
│  ┌──────────────────────────────────┐   │    │                 │                    │
│  │  DRAM Controllers × 16          │   │    │                 │                    │
│  │  - Ideal (100 cycles flat)      │   │    │                 │                    │
│  │  - 16 parallel memory banks     │   │    │                 │                    │
│  └──────────────┬───────────────────┘   │    │                 │                    │
│                 │                       │    │                 │                    │
└─────────────────┼───────────────────────┘    └─────────────────┼────────────────────┘
                  │                                              │
                  ▼                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                     │
│                          globalStorage (unified backing store)                       │
│                                                                                     │
│  Address space partitioning:                                                        │
│    GPU[0]: [0, 4GB)                                                                 │
│    GPU[1]: [4GB, 8GB)                                                               │
│    Accel[0]: [(numGPUs+1)*4GB, (numGPUs+2)*4GB)                                     │
│                                                                                     │
│  Both GPU DRAM controllers and Accel DRAM controller read/write to the SAME         │
│  backing store. Tensors allocated by one device are directly accessible by the      │
│  other via shared virtual addresses.                                                │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘

TLB Hierarchy (GPU only):
  L1V TLB (per CU) ──→ L2 TLB (per GPU) ──→ MMU (system-level)
  L1S TLB (per SA)  ─┘
  L1I TLB (per SA)  ─┘
  (Accelerator has no TLB — uses physical addresses directly)
```

**Key implementation files:**
- GPU hierarchy: `amd/samples/runner/timingconfig/r9nano.go` (L1→L2→DRAM wiring)
- GPU shader array: `amd/samples/runner/timingconfig/shaderarray.go` (CU + L1 caches + TLBs)
- Accelerator hierarchy: `amd/samples/runner/timingconfig/accelbuilder/builder.go` (AccelUnit→PCIe→DRAM)
- System-level PCIe: `amd/samples/runner/timingconfig/builder.go` (driver↔GPU↔accel routing)
- Shared storage: `amd/samples/runner/timingconfig/builder.go:116-117` (single `mem.Storage` instance)

**What flows where:**
- System PCIe carries only **commands and responses** (LaunchKernelCommand, AccelInferenceReq/Rsp) — not tensor data.
- GPU tensor data flows through the GPU's own L1→L2→DRAM path within the GPU domain.
- Accelerator tensor data flows through the internal PCIe→DRAM path within the accelerator domain.
- No cross-device data transfer messages exist. The `globalStorage` shared backing store is the implicit data channel.

### 2. Implementation Trade-offs

#### 2.1 Host Fallback Pattern: "Functional on CPU, Timing on Accelerator"

**What we do:** In `acceltensor/operator.go`, each compute operation follows this pattern:
1. `t.Vector()` — copies tensor from device to host (D2H)
2. CPU computes the result (Go code)
3. `op.Init(out, data)` — copies result from host to device (H2D)
4. `driver.AccelInference(...)` — dispatches `AccelInferenceReq` for timing

**Example** (`acceltensor/operator.go`, Softmax):
```go
inData := t.Vector()           // D2H copy
// ... CPU softmax computation ...
out := o.Create(size)
o.Init(out, outData)           // H2D copy
o.driver.AccelInference(...)   // timing dispatch
```

**Trade-off:**
- Simulation wall-clock time is bloated by two full tensor copies per operation. For a 768×768 weight matrix, that's ~2.4MB copied twice through the host, per operation.
- The reported timing metrics (busy_time, total_compute_cycles) correctly reflect only the accelerator's modeled cycles — the host copies are invisible to the timing model.
- **Is it sufficient?** Yes for our use cases. The reported timing is what we analyze. Simulation speed is a convenience issue, not an accuracy issue.
- **What we lose:** Can't study scenarios where accelerator functional correctness matters (e.g., quantization effects on output accuracy, numerical divergence between FP16 and FP32 paths).

#### 2.2 Attention Scores Computed on CPU (GPT-2-specific)

**What we do:** In `gpt2/benchmark.go`, the multi-head attention computation is split:
- **Gemm-dispatched** (timed): Q, K, V, O linear projections — 4 Gemm calls per block
- **CPU fallback** (untimed): Q@K^T dot products, softmax over scores, scores@V multiplication

The attention score computation (`attentionHead()`, line 522) runs entirely on the host with Go loops.

**Trade-off:**
- For GPT-2 Small with seq_len=128: attention scores are O(batch × heads × seq^2 × head_dim) = O(1 × 12 × 128^2 × 64) ≈ 12.6M FLOPs per block. The Q/K/V projections are O(batch × seq × d_model^2) = O(1 × 128 × 768^2) ≈ 75.5M FLOPs per projection × 4 = 302M FLOPs per block. Attention scores are ~4% of total block FLOPs — **acceptable to omit**.
- At seq_len=1024: attention becomes O(12 × 1024^2 × 64) ≈ 805M FLOPs per block vs projections O(1 × 1024 × 768^2) × 4 = 2.42B. Attention is ~25% — starts to matter.
- At seq_len=2048+: attention dominates quadratically and the CPU fallback creates a significant blind spot.
- **Is it sufficient?** For our default configuration (seq=128), yes. For long-sequence studies, we would need to dispatch attention scores to the accelerator as well.

#### 2.3 LayerNorm, GELU, Embedding as CPU Fallback

**What we do:**
- **LayerNorm** (`gpt2/benchmark.go:393`): Full CPU implementation with mean/variance/normalize. 25 calls per forward pass (2 per block + 1 final).
- **GELU** (`gpt2/benchmark.go:608`): Tanh approximation on CPU. 12 calls per forward pass.
- **Embedding** (`gpt2/benchmark.go:353`): Lookup table + position add on CPU. 1 call.

None of these generate `AccelInferenceReq`. No timing contribution.

**Trade-off:**
- LayerNorm per call: O(batch × seq × d_model) = O(128 × 768) ≈ 98K FLOPs × 25 = 2.5M total.
- GELU per call: O(batch × seq × d_ff) = O(128 × 3072) ≈ 393K FLOPs × 12 = 4.7M total.
- Embedding: O(batch × seq × d_model) = O(128 × 768) ≈ 98K FLOPs.
- Total untimed: ~7.3M FLOPs vs timed GEMM: ~3.76B FLOPs. That's **<0.2% of total FLOPs**.
- **Is it sufficient?** Yes. These ops are negligible in compute. On real hardware they are memory-bound and would add modest latency, but for interconnect comparison studies the relative results are unaffected.

#### 2.4 Shared globalStorage — No Explicit Data Transfers

**What we do:** Both GPU DRAM controllers (16 banks) and the accelerator DRAM controller share the same `mem.Storage` instance (`globalStorage`). Tensors allocated via `driver.AllocateMemory()` are placed in this unified store. Address space partitioned per device (GPU[0] at offset 0, GPU[1] at 4GB, etc.).

The `DeviceTensor` interface (`tensor/tensor.go:37-40`) exposes `Ptr()` which returns a `driver.Ptr` in this shared space. The `ptrOf()` helper in `acceltensor/operator.go:73-79` extracts pointers from either `gputensor.Tensor` or `acceltensor.Tensor` via this interface — enabling cross-operator tensor passing with zero copies.

**Trade-off:**
- **Accurate for MI300A-style unified memory.** No explicit DMA between GPU and accelerator is needed, matching the real hardware behavior where both share HBM.
- **Not accurate for discrete devices.** A discrete GPU + discrete accelerator on separate memory buses would require explicit PCIe DMA transfers at device boundaries. We don't model this.
- **Missing:** No cache coherence overhead between GPU and accelerator. No Infinity Fabric delays. No bandwidth contention from concurrent GPU + accelerator DRAM access (they have separate controllers).
- **Is it sufficient?** For our target use case (MI300A-like unified memory, interconnect technology comparison), yes. The shared memory model is architecturally correct.

#### 2.5 No SRAM Capacity Checks or Tiling

**What we do:** The accelerator has a `sramSizeBytes` field (32MB default) but it is never checked. All tensors are assumed to fit in SRAM. The READ phase loads the full tensor, COMPUTE runs on it, WRITE stores the full result.

**Problem for GPT-2:**
- Output projection weight: [768 × 50257] × 4 bytes = **147.6 MB** — does NOT fit in 32MB SRAM.
- FFN W1: [768 × 3072] × 4 = 9.4 MB — fits.
- FFN W2: [3072 × 768] × 4 = 9.4 MB — fits.
- Q/K/V/O projections: [768 × 768] × 4 = 2.4 MB each — fits.

**Trade-off:**
- For tensors exceeding SRAM, real hardware would tile: chunk the tensor, DMA each tile, compute, write back. This adds `num_tiles` factor to latency (partially mitigated by double-buffering).
- Compute cycles are underestimated for large tensors because we don't model the tiling overhead (pipeline fills/drains per tile).
- **However**, the full-traffic Phase 2.5 memory DMA is correct regardless — we read/write the full tensor data through the interconnect. So memory bandwidth effects are properly captured.
- **Is it sufficient?** For interconnect comparison (our primary use case), yes — memory traffic drives the comparison. For absolute accelerator latency estimation, the compute cycles are optimistic for large tensors.

#### 2.6 Sequential Execution Phases (No Compute-Memory Overlap)

**What we do:** Each operation executes as READ → COMPUTE → WRITE sequentially. The accelerator reads all input tensors from DRAM, then counts down compute cycles, then writes the output.

**Trade-off:**
- Real accelerators use double-buffering: while computing on tile N from SRAM buffer A, DMA fetches tile N+1 into buffer B. This can hide ~50% of memory latency for memory-bound ops.
- Our model overestimates total operation latency, particularly for memory-bound ops (ScaleAdd, Adam, RMSProp) where read/write time dominates.
- **Is it sufficient?** Absolute timing is pessimistic. But for comparing two interconnect configurations (optical vs electrical), the relative difference is preserved because both configurations experience the same sequential phase penalty.

#### 2.7 Analytical Compute Timing Models

**What we do** (`accelerator/comp.go`, `estimateComputeCycles()`):

| Op Category | Formula | Rationale |
|-------------|---------|-----------|
| GEMM | `M × N × K / (peRows × peCols)` | Full 2D array utilization, 1 MAC/PE/cycle |
| Element-wise (ReLU, etc.) | `elements / (peRows × peCols)` | Array as ALU farm |
| ScaleAdd | `3 × elements / peCols` | 2 reads + 1 write, 1D feed rate |
| Softmax | `3 × elements / peCols` | 3-pass (exp, sum, divide) |
| Adam | `7 × elements / peCols` | 4 reads + 3 writes |
| RMSProp | `5 × elements / peCols` | 3 reads + 2 writes |
| CrossEntropy | `2 × elements / peCols` | 1 read + 1 write |
| Reduction | `elements / peCols` | Single pass |

Memory-bound ops use `peCols` (256) as divisor, not `peRows × peCols` (65,536), because data feeds along one dimension of the systolic array — the 1D feed rate is the bottleneck, not the 2D array capacity.

**Trade-off:**
- No pipeline fill/drain latency for GEMM. A [128×768] × [768×768] GEMM starts producing output immediately instead of requiring 768 cycles to fill the pipeline. Error: ~768 cycles out of ~1,152 total = potentially significant for small tensors, negligible for large ones.
- No tiling overhead in the formula. Large tensors that exceed SRAM would need multiple passes.
- **Is it sufficient?** For system-level questions ("is this workload compute-bound or memory-bound?", "which interconnect technology reduces accelerator busy time?"), yes. Not for microarchitecture studies or absolute performance prediction.

#### 2.8 Dedicated DRAM Controller per Accelerator

**What we do:** Each accelerator gets its own `idealmemcontroller` in `accelbuilder/builder.go`. This controller is separate from the GPU's 16 DRAM banks. The interconnect between accelerator and its DRAM is the internal PCIe connector (parameterizable). The GPU's memory path (L1→L2→DRAM) is entirely separate.

**Trade-off:**
- Cannot study scenarios where GPU and accelerator compete for shared DRAM bandwidth. On real MI300A, both devices access the same HBM and can create contention.
- The accelerator's DRAM controller is ideal (100 cycles flat, no banking, no scheduling). All bandwidth limitation comes from the PCIe interconnect, which is the component we're studying.
- **Is it sufficient?** For interconnect technology comparison, yes — the interconnect is the variable under study, and having an ideal DRAM controller isolates the interconnect effect. For studying memory architecture trade-offs (shared vs dedicated HBM), we would need to model a shared controller with contention.

#### 2.9 Full-Traffic Phase 2.5 Memory Model

**What we do:** Every 64 bytes of tensor data generates a real `mem.ReadReq` or `mem.WriteReq` that flows through the internal PCIe connector. The accelerator issues requests progressively with backpressure control (`maxOutstandingReqs` = 64 default).

For a GEMM reading input [128×768] + weights [768×768]:
- Input: 128 × 768 × 4 = 393,216 bytes → 6,144 read requests
- Weights: 768 × 768 × 4 = 2,359,296 bytes → 36,864 read requests
- Output: 128 × 768 × 4 = 393,216 bytes → 6,144 write requests
- Total: 49,152 memory transactions through the interconnect

Memory bandwidth emerges from PCIe flit serialization — not from a formula. Changing `--accel-interconnect-bw` from 32 GB/s to 256 GB/s naturally reduces memory phase duration.

**Key implementation details:**
- `tensorTransfer` struct tracks per-tensor DMA progress (bytes sent, responses expected/received)
- `processReadPhase()` issues requests in a loop bounded by `maxOutstandingReqs` and port backpressure
- `processMemRsp()` drains all pending responses per tick
- Transition to next phase only when all responses received

**Trade-off:**
- Simulation speed decreases significantly vs analytical model (thousands of events per operation vs one).
- Real-world accuracy improves: bandwidth saturation, queuing delays, and contention effects emerge naturally.
- **Is it sufficient?** Yes — this is specifically designed for interconnect technology comparison, which is our primary research question.

### 3. Is the GPU + Accelerator System Properly Modeled for Our Use Cases?

| Use Case | Assessment | Why |
|----------|------------|-----|
| **Interconnect technology comparison** (optical vs electrical) | **Sufficient** | Full-traffic DMA through parameterizable PCIe captures BW/latency effects. This is the primary use case. |
| **Layer offloading exploration** (which blocks on which device) | **Mostly sufficient** | GEMM-dominated ops are accurately timed. CPU-fallback ops (LayerNorm, GELU, attention scores) create modest blind spots (<5% of FLOPs for GPT-2 Small seq=128). |
| **Absolute accelerator latency prediction** | **Optimistic** | Missing: SRAM tiling for large tensors, compute-memory overlap (double-buffering), pipeline fill/drain. Reported latency is a lower bound. |
| **End-to-end LLM workload characterization** | **Partially sufficient** | Forward pass structure is correct. Absolute timing is optimistic. Relative comparisons between configs remain valid. |
| **Multi-GPU + accelerator scaling** | **Sufficient** | Round-robin accelerator distribution, independent contexts per GPU, data parallelism all work correctly. |
| **GPU-accelerator memory contention** | **Not modeled** | Separate DRAM controllers. Cannot study shared memory bandwidth scenarios. |
| **Quantization / mixed precision** | **Not modeled** | All ops FP32. Protocol has TODO comments for data type fields. |

### 4. Configuration and Metrics

#### JSON Config Files

**VGG16** (`amd/benchmarks/dnn/training_benchmarks/vgg16/accel_config.json`):
```json
{"accel_layers": [32, 34]}
```
Layer IDs are 0-based indices of compute layers (Conv2D and FC). ReLU/pooling inherit operator from the preceding compute layer.

**GPT-2** (passed via `--accel-config` flag):
```json
{"accel_blocks": [0, 1, 2, 3, 4]}
```
Block IDs are 0-based transformer block indices (0 to NLayers-1). All ops within a block (projections, FFN) use the assigned operator. LayerNorm/GELU/attention scores always use CPU fallback regardless of assignment.

If the config file is missing or omitted, all layers/blocks run on GPU — full backward compatibility.

#### CLI Flags

**Accelerator flags** (`amd/samples/runner/flag.go`):
```
--num-accel=N                              # Number of accelerators (default 0)
--accel-interconnect-bw=<bytes/sec>        # Accel↔DRAM bandwidth (default 256000000000 = 256 GB/s)
--accel-interconnect-latency=<cycles>      # Switch latency (default 10)
--accel-max-outstanding=<count>            # DMA queue depth (default 64)
```

**GPT-2 flags** (`amd/samples/gpt2/main.go`):
```
--num-iter=N       # Forward pass iterations (default 1)
--batch-size=N     # Sequences per batch (default 1)
--seq-len=N        # Tokens per sequence (default 128)
--d-model=N        # Hidden size (default 768)
--n-heads=N        # Attention heads (default 12)
--n-layers=N       # Transformer blocks (default 12)
--d-ff=N           # FFN inner dimension (default 3072)
--vocab-size=N     # Vocabulary size (default 50257)
--accel-config=PATH  # Accelerator config JSON
```

#### SQLite Metrics

Output in `akita_sim_*.sqlite3`, table `mgpusim_metrics`:

| Metric | Unit | Description |
|--------|------|-------------|
| `busy_time` | seconds | Total time accelerator was active |
| `op_count` | count | Number of operations executed |
| `total_compute_cycles` | cycles | Sum of analytical compute cycles |
| `total_read_bytes` | bytes | Data read from DRAM |
| `total_write_bytes` | bytes | Data written to DRAM |
| `total_mem_reqs` | count | Actual memory transactions (Phase 2.5) |

#### Example Commands

```bash
# GPU-only GPT-2 (no accelerator)
cd amd/samples/gpt2 && go build
./gpt2 -timing -gpus=1 -report-all

# GPU + 1 accelerator, blocks 0-5 offloaded
./gpt2 -timing -gpus=1 -num-accel=1 \
  -accel-config=accel_config.json -report-all

# Electrical interconnect (PCIe 4.0 x16)
./gpt2 -timing -gpus=1 -num-accel=1 \
  -accel-interconnect-bw=32000000000 \
  -accel-interconnect-latency=140 \
  -accel-config=accel_config.json -report-all

# Optical interconnect (photonic link)
./gpt2 -timing -gpus=1 -num-accel=1 \
  -accel-interconnect-bw=200000000000 \
  -accel-interconnect-latency=10 \
  -accel-config=accel_config.json -report-all

# Multi-GPU + multi-accelerator
./gpt2 -timing -gpus=1,2 -num-accel=2 \
  -accel-config=accel_config.json -report-all
```

---

## Part 2: GPT-2 Benchmark

### 1. Why Standalone (Not Using Existing Trainer/Layer Infrastructure)

The GPT-2 benchmark (`amd/benchmarks/dnn/training_benchmarks/gpt2/benchmark.go`, 658 lines) is implemented as a standalone benchmark, not using the existing `layers.Layer` or `training.Trainer` infrastructure. Three reasons:

1. **Weight tensor count mismatch.** Each transformer block has 16 weight tensors (ln1Gamma, ln1Beta, wq, wk, wv, wo, bq, bk, bv, bo, ln2Gamma, ln2Beta, ffnW1, ffnB1, ffnW2, ffnB2). The existing `Layer` interface exposes `Parameters() tensor.Tensor` (singular) and `Gradients() tensor.Tensor` (singular). A transformer block doesn't fit this contract.

2. **Forward-only inference.** The existing `Trainer` assumes training (forward + backward + update). GPT-2 runs forward-only. Using the trainer would require implementing unused backward pass methods.

3. **Attention mechanism.** Multi-head self-attention (Q@K^T scaling, softmax, score@V) has no analog in the existing Conv2D/FC/ReLU/Pooling layer library. It would require a new Layer type with different semantics (multi-input, head splitting/merging).

### 2. Architecture — Exact Structure

**Model configuration** (GPT-2 Small defaults):

| Parameter | Value | Source |
|-----------|-------|--------|
| d_model | 768 | Hidden size |
| n_heads | 12 | Attention heads |
| n_layers | 12 | Transformer blocks |
| d_ff | 3072 | FFN inner dim (4 × d_model) |
| vocab_size | 50257 | BPE vocabulary |
| seq_len | 1024 (default in struct, 128 in CLI) | Sequence length |
| head_dim | 64 | d_model / n_heads |

**Parameter allocation** (`allocateParameters()`, line 206):

| Component | Shape | Elements | Bytes (FP32) |
|-----------|-------|----------|-------------|
| Token embedding | [50257 × 768] | 38,597,376 | 147.3 MB |
| Position embedding | [1024 × 768] | 786,432 | 3.0 MB |
| Per block (×12): | | | |
| - ln1Gamma, ln1Beta | [768] × 2 | 1,536 | 6 KB |
| - wq, wk, wv, wo | [768 × 768] × 4 | 2,359,296 | 9.0 MB |
| - bq, bk, bv, bo | [768] × 4 | 3,072 | 12 KB |
| - ln2Gamma, ln2Beta | [768] × 2 | 1,536 | 6 KB |
| - ffnW1 | [768 × 3072] | 2,359,296 | 9.0 MB |
| - ffnB1 | [3072] | 3,072 | 12 KB |
| - ffnW2 | [3072 × 768] | 2,359,296 | 9.0 MB |
| - ffnB2 | [768] | 768 | 3 KB |
| Final LN gamma, beta | [768] × 2 | 1,536 | 6 KB |
| Output projection | [768 × 50257] | 38,597,376 | 147.3 MB |
| **Total** | | **~163,037,184** | **~622 MB** |

**Weight initialization** (`randomizeParameters()`, line 244):
- Weight matrices: Xavier initialization (scale = 1/sqrt(fan_in))
- LN gamma: initialized to 1.0
- LN beta and biases: initialized to 0.0

**Forward pass call chain:**

```
Run() → for each iteration:
  forward(inst, tokens)
    ├── embed(inst, tokens)                          # CPU: token + position lookup
    ├── for i = 0..11:
    │     transformerBlock(inst, i, x, batchSeq)
    │       ├── Clone(x) → residual
    │       ├── layerNormReplace(op, x, ln1G, ln1B)  # CPU: mean/var/normalize
    │       ├── multiHeadAttention(inst, op, normed, blk, batchSeq)
    │       │     ├── linear(op, x, wq, bq, ...)     # GEMM (GPU/Accel)
    │       │     ├── linear(op, x, wk, bk, ...)     # GEMM (GPU/Accel)
    │       │     ├── linear(op, x, wv, bv, ...)     # GEMM (GPU/Accel)
    │       │     ├── attentionHead() × NHeads × Batch  # CPU: Q@K^T, softmax, @V
    │       │     └── linear(op, attnOut, wo, bo, ...)   # GEMM (GPU/Accel)
    │       ├── ScaleAdd(1, 1, residual, attnOut)    # GPU/Accel
    │       ├── Clone(x) → residual2
    │       ├── layerNormReplace(op, x, ln2G, ln2B)  # CPU: mean/var/normalize
    │       ├── ffn(op, normed2, blk)
    │       │     ├── linear(op, x, ffnW1, ffnB1, ...)  # GEMM (GPU/Accel)
    │       │     ├── geluReplace(op, h)                 # CPU: tanh approximation
    │       │     └── linear(op, h, ffnW2, ffnB2, ...)   # GEMM (GPU/Accel)
    │       └── ScaleAdd(1, 1, residual2, ffnOut)    # GPU/Accel
    ├── layerNormReplace(gpuOp, x, finalLNG, finalLNB)   # CPU
    └── linear(gpuOp, x, outputW, nil, ...)              # GEMM (GPU)
```

### 3. GPU/Accelerator vs CPU Dispatch

**Gemm-dispatched (timing-accurate):**

| Operation | Count per Forward Pass | Tensor Dimensions (seq=128) |
|-----------|----------------------|----------------------------|
| Q projection | 12 (1 per block) | [128, 768] × [768, 768] |
| K projection | 12 | [128, 768] × [768, 768] |
| V projection | 12 | [128, 768] × [768, 768] |
| O projection | 12 | [128, 768] × [768, 768] |
| FFN W1 | 12 | [128, 768] × [768, 3072] |
| FFN W2 | 12 | [128, 3072] × [3072, 768] |
| Output projection | 1 | [128, 768] × [768, 50257] |
| **Total Gemm calls** | **73** | |

Also dispatched via `ScaleAdd`: 24 calls (2 per block, for residual connections).

**CPU fallback (no timing):**

| Operation | Count | FLOPs per Call (seq=128) |
|-----------|-------|--------------------------|
| Embedding lookup | 1 | ~98K |
| LayerNorm | 25 (2/block + 1 final) | ~98K |
| GELU activation | 12 (1/block) | ~393K |
| Attention scores (Q@K^T, softmax, @V) | 12 blocks × 12 heads | ~1.05M per head |

**FLOP budget summary (seq=128, batch=1):**

- Timed Gemm FLOPs: 73 Gemm calls, dominated by the 48 Q/K/V/O projections at 75.5M each + 24 FFN projections at 302M each ≈ **10.9B FLOPs**
- Untimed CPU FLOPs: ~7.3M (LayerNorm + GELU) + ~151M (attention scores) ≈ **158M FLOPs**
- **Ratio: ~98.6% of FLOPs are timed**

### 4. Realism Assessment

#### What's Faithful to Real GPT-2

1. **Correct parameter shapes and counts.** Token/position embeddings, per-block Q/K/V/O weight matrices, FFN W1/W2, LayerNorm parameters — all match the real GPT-2 architecture. Total count (~163M) matches published GPT-2 Small.

2. **Pre-norm residual connections.** The benchmark uses the GPT-2 ordering: LayerNorm before attention/FFN, then add residual (not post-norm like original Transformer). Implemented correctly in `transformerBlock()`.

3. **GELU activation.** Uses the exact tanh approximation formula from the GPT-2 paper:
   ```
   GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
   ```

4. **Multi-head attention structure.** Correct head splitting (d_model / n_heads = 64), independent per-head attention computation, correct index arithmetic via `mhaIndex()`.

5. **Numerically stable softmax.** Max-subtraction before exponentiation prevents overflow — correct implementation.

6. **Configurable model dimensions.** Can set d_model, n_heads, n_layers, d_ff, vocab_size, seq_len, batch_size via CLI flags to match GPT-2 Small/Medium/Large/XL.

7. **Xavier weight initialization.** Scale = 1/sqrt(fan_in), matching standard transformer initialization.

#### What's Simplified or Missing

1. **No causal (autoregressive) masking.** Real GPT-2 masks future tokens in attention — position i can only attend to positions [0, i]. Our benchmark computes full bidirectional attention. This means attention score computation is identical in FLOPs but semantically different (model would produce different outputs).

2. **No KV cache.** In real autoregressive inference, K and V are cached from previous tokens. Our benchmark recomputes K, V for all positions every time. This affects the realistic latency profile but not the FLOP count for a single forward pass.

3. **Attention scores on CPU.** Q@K^T, softmax, and scores@V are computed on the host (see Section 2.2 above). Only the linear projections (Q/K/V/O) go through the accelerator.

4. **Forward-only (no training).** No backward pass, no gradient computation, no optimizer step. The benchmark is inference-only. The `Verify()` method panics with "not implemented".

5. **Synthetic random tokens.** Input is `rand.Intn(vocabSize)` — no tokenizer, no real text. This is fine because the compute pattern is identical regardless of token values.

6. **No dropout.** GPT-2 uses dropout in attention and FFN during training. Since we're inference-only, this is not a limitation.

7. **Flat (1D) tensor storage.** Tensors are stored as 1D vectors and reshaped for Gemm via `op.Reshape()`. The Q/K/V head splitting is done in host-side index arithmetic (`mhaIndex()`), not via tensor reshape operations. This means no 4D tensor operations flow through the accelerator.

8. **No weight tying.** Real GPT-2 ties token embedding and output projection weights. Our benchmark allocates them separately (~147MB each), doubling this portion of memory.

9. **Batch size = 1 default.** While configurable, default batch=1 means most tensors are small relative to the PE array capacity. This makes the workload more memory-bound than a batched deployment would be.

### 5. Memory Footprint

**Per GPU instance:**
- Parameters: ~163M × 4 bytes = **~622 MB**
  - Largest single tensor: output projection [768 × 50257] = 147.3 MB
  - Token embedding: 147.3 MB
  - 12 blocks × ~27.2 MB per block = 326.4 MB
- Activations (seq=128, batch=1):
  - Intermediate tensors: [128, 768] = 393 KB each (reused/freed)
  - FFN intermediate: [128, 3072] = 1.5 MB (temporary)
  - Attention scores: 12 heads × 128 × 128 × 8 bytes = 1.5 MB (host-side, not device)
- **Total per GPU: ~622 MB** (dominated by parameters)

**Multi-GPU:** Each GPU gets an independent copy of all parameters via `InitWithExistingPID`. 2 GPUs = ~1.24 GB total.

### 6. Multi-GPU and Accelerator Support

**Data parallelism** (`createGPUInstance()`, line 131):
- Each GPU gets its own `Context`, `GPUOperator`, and full parameter copy.
- Contexts share the same PID via `InitWithExistingPID()`.
- Each GPU runs an independent forward pass on the same batch (no gradient averaging since inference-only).

**Accelerator assignment** (`createGPUInstance()`, line 146):
- Round-robin distribution: GPU instance i gets accelerator `i % numAccelerators`.
- Per-block operator: `blockOps[i]` is either `gpuOp` or `accelOp` based on config.
- Fix in commit `3026d1e`: originally hardcoded to accelerator 0, now properly distributes.

**Tested configurations** (from commit message):
- Single GPU timing mode
- Multi-GPU (2 GPUs) timing mode
- Single GPU + 1 accelerator (block 0 offloaded)
- Multi-GPU + multi-accelerator (both GPUs dispatch to shared accelerator)

### 7. Key Files

| File | Purpose |
|------|---------|
| `amd/benchmarks/dnn/training_benchmarks/gpt2/benchmark.go` | GPT-2 benchmark (658 lines) |
| `amd/samples/gpt2/main.go` | Sample runner entry point |
| `amd/protocol/accelprotocol.go` | AccelInferenceReq/Rsp, 15 op types |
| `amd/timing/accelerator/comp.go` | Accelerator component, 3-phase state machine, timing models |
| `amd/timing/accelerator/builder.go` | Accelerator component builder |
| `amd/samples/runner/timingconfig/accelbuilder/builder.go` | Platform-level accelerator builder (PCIe wiring) |
| `amd/samples/runner/timingconfig/builder.go` | System-level builder (GPU + accel + driver) |
| `amd/samples/runner/timingconfig/r9nano.go` | GPU memory hierarchy (L1→L2→DRAM) |
| `amd/samples/runner/timingconfig/shaderarray.go` | Shader array internals (CUs, L1 caches, TLBs) |
| `amd/benchmarks/dnn/acceltensor/operator.go` | Accelerator tensor operator (dispatch + CPU fallback) |
| `amd/benchmarks/dnn/tensor/tensor.go` | Tensor and DeviceTensor interfaces |
| `amd/driver/api.go` | SelectAccelerator(), AccelInference() APIs |
| `amd/samples/runner/flag.go` | CLI flags for accelerator config |
| `amd/samples/runner/report.go` | Metrics collection and SQLite output |
