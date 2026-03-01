# Approach 2: Heterogeneous GPU + Accelerator Simulation Design Document

## 1. Motivation and Problem Statement

Modern AI hardware increasingly adopts heterogeneous architectures that
combine general-purpose GPU compute units with fixed-function inference
accelerators on the same die or package. AMD's MI300A (GPU + CPU on the same
package), Google's TPU-integrated systems, and Intel's Gaudi all exemplify
this trend. Yet cycle-accurate simulation infrastructure for studying these
heterogeneous configurations is largely absent — existing GPU simulators
model only the GPU pipeline, and existing accelerator simulators model only
the accelerator.

**The core research question:** How does a heterogeneous GPU + accelerator
system behave when running a real LLM workload, where some layers execute on
the GPU's programmable compute units (with full memory hierarchy simulation)
and other layers execute on a fixed-function accelerator (with analytical
timing)?

MGPUSim is a cycle-accurate AMD GCN3 GPU simulator built on the Akita
discrete-event simulation framework. It models the full GPU pipeline:
wavefront scheduling, SIMD execution, register files, L1/L2 caches, DRAM
controllers, TLBs, RDMA for multi-GPU, and page migration. Our extension
adds a co-simulated inference accelerator alongside the existing GPU, sharing
the same memory address space and simulation engine.

## 2. Design Goals

1. **Minimal invasion of existing codebase.** The GPU simulation must remain
   unmodified. All accelerator logic is additive — new packages, new
   protocol messages, new driver APIs. No existing benchmark needs
   modification to continue working on GPU-only configurations.

2. **Shared memory space.** The accelerator and GPU operate on the same
   `globalStorage` backing store. A tensor allocated by one device can be
   read by the other without explicit DMA transfers. This models the
   shared-memory architecture of packages like MI300A.

3. **Plug-in operator model.** Benchmarks select an *operator*
   (`tensor.Operator` interface) per layer/block. The same benchmark code
   works with `gputensor.GPUOperator` (GPU kernels) or
   `acceltensor.Operator` (accelerator dispatch) with no source changes
   beyond operator construction.

4. **Timing fidelity proportional to importance.** GEMM (which dominates
   transformer compute) gets the most detailed timing model. Memory-bound
   element-wise ops get bandwidth-aware analytical estimates. Data layout
   transforms (Im2Col, Transpose) get CPU fallback with no timing. This
   is deliberate: we invest modeling effort where it affects the results
   the most.

5. **Configuration-driven heterogeneity.** A JSON config file maps
   layers/blocks to devices at runtime. No recompilation needed to explore
   different offloading strategies.

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Benchmark                               │
│  (e.g., VGG16, Minerva, GPT-2)                              │
│                                                             │
│  Layer 1 → gpuOp.Gemm(...)     Layer 2 → accelOp.Gemm(...) │
│            ↓                              ↓                 │
│  gputensor.GPUOperator          acceltensor.Operator        │
│  (launches HIP kernels)         (AccelInferenceReq)         │
└──────────┬──────────────────────────────────┬───────────────┘
           │                                  │
           ▼                                  ▼
┌──────────────────┐              ┌───────────────────────┐
│   GPU Driver     │              │    Driver (accelPort)  │
│                  │              │                        │
│  LaunchKernel()  │              │  AccelInference()      │
│  MemCopy()       │              │  EnqueueAccelInference│
└──────┬───────────┘              └───────────┬───────────┘
       │                                      │
       ▼                                      ▼
┌──────────────┐                  ┌─────────────────────────┐
│  GPU CUs     │                  │  Accelerator Comp       │
│  L1/L2/DRAM  │                  │  (Akita TickingComponent)│
│  TLB/RDMA    │                  │                         │
│  Full timing │                  │  estimateCycles()       │
│  pipeline    │                  │  cycle countdown        │
└──────────────┘                  └─────────────────────────┘
       │                                      │
       ▼                                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    globalStorage (shared)                     │
│             (single mem.Storage backing both devices)         │
└─────────────────────────────────────────────────────────────┘
```

## 4. Key Design Decisions and Rationale

### 4.1 Why a Separate `acceltensor.Operator` Instead of Transparent Routing

**Decision:** The benchmark explicitly chooses which operator to use per
layer. The driver does NOT intercept operations and transparently route them.

**Rationale:** In real heterogeneous hardware, the compiler or runtime makes
explicit placement decisions — CUDA/HIP graph partitioning, ONNX Runtime
execution providers, TensorRT layer assignment. There is no magic layer that
"just knows" where to run each op. By making the benchmark (acting as the
application/framework) explicitly select the operator, we model this
real-world decision point. This also gives researchers full control over
placement strategies for experimentation.

**Alternative considered:** A middleware layer in the driver that
intercepts all tensor operations and routes based on op type and tensor
size. Rejected because: (a) it conflates the simulation infrastructure with
the scheduling policy being studied, and (b) it adds complexity to the
driver's already intricate command processing pipeline.

### 4.2 Why the `tensor.Operator` Interface (Not Concrete Types)

**Decision:** All benchmarks, layers, the trainer, and the optimizer use the
`tensor.Operator` interface and `tensor.DeviceTensor` interface rather than
concrete `*gputensor.GPUOperator` or `*gputensor.Tensor` types.

**Rationale:** This was the enabling abstraction for the entire approach.
Before our changes, the `DataParallelismMultiGPUTrainer` held
`[]*gputensor.GPUOperator` and cast gradients to `*gputensor.Tensor`. This
made it impossible to mix GPU and accelerator operators in the same training
loop. By generalizing to interfaces:

- The trainer can hold a mix of GPU and accelerator operators in its
  `TensorOperators` slice.
- A single forward pass can use `gpuOp` for embedding layers and `accelOp`
  for attention layers, with tensors passing between them via the shared
  `DeviceTensor` interface.
- The `ptrOf()` helper in acceltensor extracts the device pointer from
  *any* tensor implementing `DeviceTensor`, enabling cross-operator tensor
  compatibility.

This was implemented across 7 commits, starting with the `DeviceTensor`
interface addition (commit `7c3a767a`) and propagating through the trainer
(`272bbe1c`), all benchmarks (`11637817`, `c8b8c6d8`), and the operator
itself (`c75cd964`).

### 4.3 Why Shared `globalStorage` (No Explicit Transfers)

**Decision:** Both GPU and accelerator tensors are backed by the same
`mem.Storage` instance. `acceltensor.Tensor` and `gputensor.Tensor` can
reference the same physical addresses.

**Rationale:** This models a unified memory architecture (like MI300A's
unified HBM). In such systems, the GPU and accelerator share the same
physical memory, and data movement between them is implicit — there's no
explicit "copy from GPU to accelerator" step. A tensor allocated for the
GPU's FC layer can be directly passed to the accelerator's attention layer.

**Tradeoff acknowledged:** This means we don't model any cache coherence
overhead or bandwidth contention between the GPU and accelerator accessing
the same data. In a real MI300A, shared HBM access involves the Infinity
Fabric interconnect, and concurrent access patterns would create contention.
This is a Phase 2 improvement: wiring the accelerator's `ToMem` port into
the memory hierarchy would capture these effects.

### 4.4 Why "Functional on CPU, Timing on Accelerator"

**Decision:** Each acceltensor operation computes its result on the host CPU
(using `Vector()`/`Init()` or `CPUOperator`), writes the result to device
memory, then dispatches an `AccelInferenceReq` for timing estimation. The
accelerator component counts down cycles but never touches actual data.

**Rationale:** This is the same functional/timing split used throughout
MGPUSim. The GPU's emulation mode computes results functionally; timing mode
adds pipeline delays. Our accelerator works the same way — correctness
comes from the CPU computation, and the `AccelInferenceReq` round-trip
models the time the operation *would* take on hardware.

This approach has two key benefits:
1. **Correctness is trivially verified.** The CPU computation matches the
   reference `CPUOperator` exactly. We don't need to implement a
   functionally correct accelerator datapath — just timing.
2. **The accelerator component stays simple.** It's a cycle counter with
   analytical models, not a full RTL simulation. This is appropriate for
   studying system-level effects (when is the accelerator idle? what's the
   GPU-accelerator overlap?) rather than microarchitectural details.

**What this means for data movement:** Currently, every op does a full
tensor round-trip through the host CPU (`Vector()` copies device→host,
`Init()` copies host→device). This is functionally correct but means the
real host-side wall-clock time is dominated by these copies. In a real
system, the accelerator would compute in-place on device memory. The
simulated time (reported in metrics) correctly reflects only the accelerator
cycle count — the host-side copies are "simulation overhead" that doesn't
affect the reported timing.

### 4.5 Protocol Message Design

**Decision:** A single `AccelInferenceReq` message type carries all
operation types, with an `AccelOpType` enum and an `AccelOpParams` struct.

**Rationale:** In the Akita simulation framework, messages flow between
ports. Having one message type for all accelerator ops keeps the protocol
simple and the driver's send/receive logic uniform. The `AccelOpType` enum
tells the accelerator *what* to model, and `AccelOpParams` carries
operation-specific parameters (matrix dimensions for GEMM, scalar
coefficients for ScaleAdd, etc.).

**The `AccelOpParams` struct intentionally has "extra" fields.** For GEMM,
`M/N/K/Alpha/Beta/TransposeA/TransposeB` are used. For ScaleAdd, only
`Alpha/Beta` are used. For ReLU, none are used (the timing model only needs
`InputSize`). This is deliberate — a flat struct with unused fields is
simpler than a polymorphic parameter hierarchy, and the struct is only
instantiated per-operation (not per-element), so the wasted bytes are
negligible.

**The `InputSize/OutputSize/WeightsSize [4]uint32` fields** encode tensor
shapes in a uniform `[N, C, H, W]` format with unused dimensions set to 1.
For a 2D matrix `[128, 768]`, this becomes `[128, 768, 1, 1]`. The timing
model computes total elements by multiplying all four dimensions, so the
padding with 1s is mathematically neutral.

### 4.6 Accelerator Timing Model Design

**Decision:** The timing model is analytical (formula-based), not
trace-driven or cycle-accurate. Different op categories get different
formulas.

**Rationale and per-op reasoning:**

#### GEMM (compute-bound)
```
cycles = M × N × K / (peArrayRows × peArrayCols)
```
GEMM is the canonical compute-bound workload for a systolic array.  Each
multiply-accumulate (MAC) operation occupies one PE for one cycle. A 256×256
array performs 65,536 MACs/cycle. The total work is M×N×K MACs (one per
output element, each requiring K multiply-adds). This gives the
theoretical peak throughput — real hardware would add tiling overhead,
pipeline fill/drain latency, and memory stalls, but this is a reasonable
first-order model.

#### Element-wise ops — ReLU, ElementWiseMul (memory-bound)
```
cycles = totalElements / (peArrayRows × peArrayCols)
```
Element-wise ops perform O(1) arithmetic per element. On a systolic array,
these would be executed using the PEs as simple ALUs, with throughput
limited by how fast data can be fed. The current model divides by the full
array size, which assumes the array can process one element per PE per
cycle. This is optimistic (real hardware would be limited by SRAM read
bandwidth, not PE count), but provides a reasonable baseline that Phase 2's
roofline model will refine.

#### ScaleAdd — `alpha*A + beta*B` (memory-bound, 2 inputs)
```
cycles = 3 × totalElements / peArrayCols
```
ScaleAdd reads two input tensors and writes one output = 3 memory passes.
We divide by `peArrayCols` (not the full array) because this operation
doesn't benefit from 2D array parallelism — it's a simple vector operation
that pipelines through one dimension. The factor of 3 accounts for the
three memory streams. This is more realistic than the element-wise model
for operations with multiple tensor operands.

#### Softmax (multi-pass, memory-bound)
```
cycles = 3 × totalElements / peArrayCols
```
Softmax requires three passes over the data: (1) compute exp of each
element, (2) sum across the row, (3) divide by the sum. Each pass is
memory-bound and pipelines through the vector lanes (peArrayCols). The
factor of 3 is exact for the number of passes — real implementations may
fuse passes or use online algorithms, but three-pass is the standard
approach.

#### Reduction / Sum (memory-bound + reduction tree)
```
cycles = totalElements / peArrayCols
```
A sum reduction reads all input elements once. The reduction tree adds
log(N) accumulation steps, but these are dominated by the initial read
pass for any reasonably sized tensor. We use a single pass as the
estimate.

#### Adam optimizer (memory-intensive, 7 tensor streams)
```
cycles = 7 × totalElements / peArrayCols
```
Adam reads 4 tensors per element (params, gradients, vHistory, sHistory)
and writes 3 (params, vHistory, sHistory). Each stream must pass through
the memory interface. The compute is trivial (a few multiplies and adds per
element). The factor of 7 directly reflects the number of memory streams.
This is a key insight: optimizer steps are often the most memory-bandwidth-
intensive operations in training, and this model captures that.

#### RMSProp optimizer (5 tensor streams)
```
cycles = 5 × totalElements / peArrayCols
```
Same reasoning as Adam but with fewer tensor operands: reads 3
(params, gradients, sHistory), writes 2 (params, sHistory).

#### CrossEntropy and derivatives (lightweight)
```
cycles = 2 × totalElements / peArrayCols
```
Cross-entropy loss reads the prediction tensor and produces either a scalar
(loss) or a same-sized tensor (derivative). The 2× factor accounts for one
read and one write pass. These operations touch relatively few elements
compared to GEMM or optimizer steps (only batch_size × num_classes), so
their contribution to total accelerator time is small.

#### Why `peArrayCols` for memory-bound ops, not `peArrayRows × peArrayCols`?

For compute-bound GEMM, the full 2D array is utilized — each PE performs
useful work simultaneously. For memory-bound ops, the bottleneck is feeding
data to the array. A systolic array typically loads data along one dimension
(rows or columns) from SRAM, and each column (or row) processes elements
independently. The throughput is limited by this 1D feed rate, not the full
2D array utilization. Using `peArrayCols` (= 256 for our default
configuration) as the divisor for memory-bound ops is therefore more
realistic than using `peArrayRows × peArrayCols` (= 65,536).

### 4.7 CPU Fallback Ops: Im2Col, Transpose, Rotate180, Dilate, Pooling

**Decision:** These operations are implemented by delegating to
`tensor.CPUOperator` with no accelerator dispatch (no timing).

**Rationale:**
- **Im2Col** is a data layout transformation, not a computation. On real
  hardware, it's often fused into the GEMM (implicit im2col) rather than
  materialized as a separate tensor. Modeling it as an accelerator op would
  overcount the actual hardware cost.
- **Transpose** reshuffles data in memory. On an accelerator, this would
  be handled by the DMA engine or memory controller, not the systolic
  array. Until Phase 2 wires memory transactions, there's no infrastructure
  to model this correctly.
- **Rotate180, Dilate** are used only in Conv2D backward passes and are
  simple data rearrangements.
- **Pooling** is comparatively cheap (no multiply-accumulates) and is often
  executed on a dedicated pooling unit, not the systolic array. CPU fallback
  is functionally correct and contributes negligible simulated time.

These ops are **not needed for GPT-2** (the target workload). They exist to
support Conv2D-based benchmarks (VGG16, LeNet) that serve as validation
stepping-stones.

### 4.8 Configuration-Driven Layer Assignment

**Decision:** A JSON configuration file (`accel_config.json`) specifies
which layers or transformer blocks run on the accelerator vs. the GPU.

**Rationale:** This enables rapid experimentation with offloading strategies
without recompilation. For VGG16, the config maps layer IDs (0-based Conv2D
and FC layer indices) to the accelerator. For GPT-2, it maps transformer
block indices. Example:

```json
{"accel_layers": [3, 4, 5]}          // VGG16: FC layers on accelerator
{"accel_blocks": [0, 1, 2, 3, 4]}    // GPT-2: first 5 blocks on accelerator
```

The benchmark reads this config at startup and constructs the appropriate
operator (`gputensor.GPUOperator` or `acceltensor.Operator`) per layer.
If the config file is missing, all layers run on GPU — maintaining full
backward compatibility.

## 5. Implementation Walkthrough

### 5.1 Infrastructure Layer (Commit `90eaf4c6`)

The initial commit introduced the complete infrastructure in 13 new files
(1,424 lines added, 1 line modified):

| Component | File | Purpose |
|-----------|------|---------|
| **Protocol** | `protocol/accelprotocol.go` | `AccelInferenceReq/Rsp` messages, `AccelOpType` enum, `AccelOpParams` |
| **Accelerator** | `timing/accelerator/comp.go` | Akita `TickingComponent` with `Tick()` loop, timing model, metrics |
| **Accelerator** | `timing/accelerator/builder.go` | Builder pattern for configuring PE array size, SRAM, bandwidth |
| **Operator** | `acceltensor/operator.go` | `tensor.Operator` impl with `Gemm` dispatch + host fallbacks |
| **Operator** | `acceltensor/tensor.go` | Tensor struct with device pointer, `Vector()`, `Ptr()` |
| **Driver** | `driver/api.go` | `SelectAccelerator()`, `AccelInference()` synchronous API |
| **Driver** | `driver/driver.go` | `RegisterAccelerator()`, `sendToAccelerators()`, response handling |
| **Driver** | `driver/command.go` | `AccelInferenceCommand` command type |
| **Platform** | `accelbuilder/builder.go` | Platform-level builder, wires accelerator to driver via PCIe |
| **Platform** | `timingconfig/builder.go` | Extended to support `--num-accel` flag |
| **Runner** | `runner/flag.go` | Added `--num-accel` flag |
| **Runner** | `runner/report.go` | Accelerator metric collection (busy_time, op_count, total_compute_cycles) |

### 5.2 Interface Generalization (Commits `7c3a767a` through `11637817`)

Three commits transformed the existing codebase from concrete GPU types to
interfaces:

1. **`DeviceTensor` interface** (`7c3a767a`): Added `tensor.DeviceTensor`
   extending `tensor.Tensor` with `Ptr() driver.Ptr`. Both `gputensor.Tensor`
   and `acceltensor.Tensor` already satisfied this interface — zero
   changes to existing types.

2. **Trainer generalization** (`272bbe1c`): Changed
   `DataParallelismMultiGPUTrainer.TensorOperators` from
   `[]*gputensor.GPUOperator` to `[]tensor.Operator`. Changed gradient
   type assertions from `*gputensor.Tensor` to `tensor.DeviceTensor`.
   Single-file, 3-line change with massive architectural impact.

3. **Benchmark generalization** (`11637817`): Updated LeNet, Minerva, and
   XOR benchmarks to use `tensor.Operator` and `tensor.DeviceTensor`
   instead of concrete GPU types. These benchmarks now work with any
   operator implementation.

### 5.3 VGG16 Mixed Mode (Commit `c8b8c6d8`)

The VGG16 benchmark was extended with per-layer operator selection:
- Reads `accel_config.json` at startup
- Maps Conv2D/FC layer IDs to a set of "accelerator layers"
- ReLU and pooling layers inherit the operator of the preceding layer
  (they are ancillary to the main compute layer)
- Constructs either `gputensor.GPUOperator` or `acceltensor.Operator`
  per layer group

### 5.4 Minerva End-to-End Proof (Commit `c75cd964`)

The Minerva benchmark (a simple 3-layer MLP) was the first end-to-end
proof of concept:
- Selected FC layers run on the accelerator
- The trainer (using the generalized `tensor.Operator` interface) handles
  mixed operators seamlessly
- Accelerator metrics (busy_time, op_count, total_compute_cycles) appear
  correctly in the SQLite output database

### 5.5 Phase 1: Complete Accelerator Op Dispatch (Current Work)

Converted all remaining host-fallback operations in `acceltensor/operator.go`
to dispatch `AccelInferenceReq` for timing:

**11 ops now timing-accurate:** ScaleAdd, Softmax, ReluForward,
ReluBackward, ElementWiseMul, Sum, Adam, RMSProp, CrossEntropy,
CrossEntropyDerivative, SoftmaxCrossEntropyDerivative.

**8 ops implemented as CPU fallback** (previously panicked): Im2Col,
Transpose, Rotate180, Dilate, MaxPoolingForward/Backward,
AvgPoolingForward/Backward.

**7 new protocol constants added:** AccelOpScaleAdd, AccelOpReduction,
AccelOpAdam, AccelOpRMSProp, AccelOpCrossEntropy, AccelOpCrossEntropyDeriv,
AccelOpSoftmaxCrossEntropyDeriv.

**6 new timing estimate functions added** with memory-bandwidth-aware
analytical models.

Additionally, the `Softmax` implementation was improved with numerical
stability (max-subtraction before exponentiation), which prevents overflow
for large logit values that occur in real transformer attention patterns.

### 5.6 Phase 2: Memory Hierarchy Wiring and Roofline Model

Phase 2 addresses the most significant gap in the timing model: the
accelerator's memory subsystem. Before Phase 2, all accelerator timing was
purely compute-based — the `ToMem` port was disconnected, and the timing
model only counted arithmetic cycles.

#### The Problem

Real accelerators are frequently **memory-bound**, not compute-bound. A
256×256 systolic array running at 1 GHz can perform 65,536 MACs/cycle —
far more than the memory system can feed for most operations:

- A ScaleAdd (alpha*A + beta*B) reads 2 tensors and writes 1 tensor. For a
  768-element vector, that's 3 × 768 × 4 = 9,216 bytes of memory traffic.
  At 256 bytes/cycle, that's 36 memory cycles vs. 3 compute cycles.
  **12× memory-bound.**

- A GEMM [128×768] × [768×768] requires 128×768×768 ≈ 75.5M MACs.
  Compute cycles = 75.5M / 65,536 = 1,152 cycles.
  Memory reads = (128×768 + 768×768) × 4 ≈ 2.75 MB; at 256 bytes/cycle
  = 10,750 cycles. **9.3× memory-bound** (before data reuse from SRAM).

Without memory modeling, the timing reports are unrealistically optimistic
for memory-bound operations.

#### Design: DMA-Based Memory Model

Real fixed-function accelerators (Google TPU, Groq TSP) use **DMA engines**
for bulk data transfers between off-chip DRAM and on-chip SRAM. The data
flow for each operation is:

```
1. Driver sends command → "compute GEMM on tensor at address X"
2. Accelerator DMA engine reads tensor from DRAM → on-chip SRAM (READ)
3. Systolic array reads from SRAM, computes, writes result to SRAM
4. DMA engine writes result from SRAM → DRAM (WRITE)
5. Accelerator sends response → "done"
```

This is fundamentally different from how GPUs access memory — GPUs issue
individual cache-line (64-byte) requests from each CU wavefront through
L1/L2/DRAM. Accelerators perform bulk DMA transfers of entire tiles.

We model this in three parts:

1. **DRAM infrastructure**: Wire the accelerator's `ToMem` port to an
   `idealmemcontroller` (same component used by GPUs) via a
   `directconnection`. This puts the accelerator in the Akita simulation
   framework's memory event system with a 100-cycle DRAM latency.

2. **Analytical bandwidth model**: For each operation, compute the total
   memory traffic (read bytes + write bytes) based on the op type and tensor
   dimensions. Calculate `mem_cycles = total_bytes / memBandwidthBW` where
   `memBandwidthBW` = 256 bytes/cycle (configurable).

3. **Roofline execution**: `total_cycles = max(compute_cycles, mem_cycles)`.
   The accelerator issues one `mem.ReadReq` per input tensor through `ToMem`
   for DRAM latency modeling, then counts down the roofline cycles, then
   issues a `mem.WriteReq` for the output. This gives a three-phase
   execution: READ → COMPUTE → WRITE.

The roofline model (`latency = max(compute, memory)`) is the standard
analytical model used in architecture research (Williams et al., 2009).
It captures the fundamental bottleneck — whether the operation is limited
by arithmetic throughput or memory bandwidth — without requiring detailed
microarchitectural simulation.

#### Memory Traffic Per Operation Type

| Operation | Read bytes | Write bytes | Bottleneck |
|-----------|-----------|-------------|------------|
| GEMM (M,N,K) | (M×K + K×N) × 4 | M×N × 4 | Usually compute for large K |
| Conv2D | (batch×inC×H×W + outC×inC×kH×kW) × 4 | batch×outC×oH×oW × 4 | Compute |
| ReLU/ElementWise | N × 4 | N × 4 | Memory |
| ScaleAdd | 2×N × 4 | N × 4 | Memory |
| Softmax | N × 4 | N × 4 | Memory |
| Reduction/Sum | N × 4 | 4 (scalar) | Memory |
| Adam | 4×N × 4 | 3×N × 4 | Memory |
| RMSProp | 3×N × 4 | 2×N × 4 | Memory |
| CrossEntropy | 2×N × 4 | N × 4 | Memory |

N = total elements from tensor dimensions.

#### What Phase 2 Does NOT Model (Future Work)

- **SRAM capacity and tiling**: When total data exceeds the 32MB on-chip
  SRAM, the operation must be tiled (broken into chunks that fit). Each tile
  goes through a separate DMA-read → compute → DMA-write cycle. Tiling
  increases total latency by `num_tiles` factor, partially mitigated by
  double-buffering (prefetching tile N+1 while computing tile N).

- **Double-buffering / compute-memory overlap**: Real accelerators pipeline
  DMA transfers with computation. While computing on data in SRAM buffer A,
  the DMA engine fills SRAM buffer B with the next tile. This overlap is
  not modeled — we use a sequential READ → COMPUTE → WRITE pipeline.

- **Per-cache-line memory transactions**: We issue one probe `ReadReq` per
  tensor (64 bytes) for the DRAM latency event, not per cache line. The
  bulk bandwidth is modeled analytically.

- **DRAM bank conflicts and row buffer effects**: The `idealmemcontroller`
  is ideal — every access takes exactly 100 cycles. A realistic DRAM
  controller would model bank conflicts, row buffer hits/misses, and
  scheduling policies.

### 5.7 Phase 2.5: Full-Traffic Memory Simulation for Interconnect Studies

Phase 2.5 replaces the analytical bandwidth model with full-traffic
simulation, enabling the study of different interconnect technologies
(optical vs electrical) between the accelerator and its DRAM.

#### Motivation

The Phase 2 analytical model (`mem_cycles = total_bytes / bandwidth`)
computed bandwidth inside the accelerator component as a constant. Changing
the interconnect from electrical to optical would have zero effect on the
simulation — no traffic actually flowed through the interconnect. For
research comparing interconnect technologies, real memory requests must flow
through a parameterizable interconnect component so that bandwidth and
latency effects emerge naturally from the simulation.

#### Architecture Change

```
BEFORE (Phase 2 — analytical):
  AccelComp.ToMem → directconnection (0 latency) → idealmemcontroller
  Bandwidth: hardcoded formula in comp.go
  Traffic: 1 probe (64B) per tensor

AFTER (Phase 2.5 — full traffic):
  AccelComp.ToMem → PCIe network (configurable BW + latency) → idealmemcontroller
  Bandwidth: emerges from PCIe flit serialization
  Traffic: ceil(tensorBytes/64) requests per tensor
```

#### Key Design Decisions

1. **PCIe connector as interconnect model**: The Akita PCIe connector
   provides flit-based bandwidth serialization and configurable switch
   latency. For optical vs electrical comparison:
   - Electrical: `--accel-interconnect-bw=32000000000 --accel-interconnect-latency=140`
   - Optical: `--accel-interconnect-bw=200000000000 --accel-interconnect-latency=10`

2. **Full traffic volume**: Every 64 bytes of tensor data generates a real
   `mem.ReadReq` or `mem.WriteReq`. A GEMM reading 100KB generates ~1,600
   requests that flow through the interconnect, experiencing realistic
   bandwidth contention and latency.

3. **Progressive issuing with backpressure**: The accelerator limits
   in-flight requests via `maxOutstandingReqs` (default 64). When the
   interconnect's buffers are full, the accelerator stalls until responses
   arrive — natural flow control.

4. **Sequential phases preserved**: READ → COMPUTE → WRITE. Memory time
   now emerges from simulation rather than being computed analytically.
   Total operation time = read_time (emergent) + compute_time (analytical)
   + write_time (emergent).

5. **Internal PCIe network**: The accelerator-to-DRAM interconnect is a
   separate PCIe network created inside `accelbuilder`, distinct from the
   system-level PCIe that connects GPUs and the driver. No port conflicts.

#### What Phase 2.5 Enables

- **Interconnect technology comparison**: Swap bandwidth and latency
  parameters to model electrical, optical, or any custom interconnect
- **Bandwidth saturation analysis**: Real traffic reveals when the
  interconnect becomes the bottleneck vs when compute dominates
- **Contention modeling**: Multiple requests queue at PCIe endpoints,
  revealing queuing effects absent in the analytical model
- **Accurate per-operation timing**: Memory-bound ops naturally take longer
  when interconnect bandwidth is lower

#### Configuration

CLI flags:
- `--accel-interconnect-bw=<bytes/sec>` (default: 256 GB/s)
- `--accel-interconnect-latency=<cycles>` (default: 10)
- `--accel-max-outstanding=<count>` (default: 64)

## 6. What the Simulation Currently Captures

### Accurately modeled:
- **GPU compute timing**: Full cycle-accurate simulation of GCN3 wavefront
  execution, pipeline stages, register file contention
- **GPU memory hierarchy**: L1/L2 cache hit/miss, DRAM controller queuing,
  TLB translation, page migration
- **Multi-GPU communication**: RDMA transfers, PCIe bandwidth
- **Accelerator compute timing**: Analytical models for all DNN operations,
  proportional to operation size and type
- **Accelerator utilization**: Busy time, operation count, total compute
  cycles reported to SQLite
- **Heterogeneous scheduling**: Which layers run where, configurable at
  runtime

### Modeled after Phase 2.5:
- **Full-traffic memory simulation**: Every 64 bytes of tensor data
  generates a real `mem.ReadReq`/`mem.WriteReq` flowing through the
  interconnect. Memory bandwidth and latency emerge from the simulation.
- **Parameterizable interconnect**: The accelerator-to-DRAM link uses a
  PCIe connector with configurable bandwidth and switch latency. Swap
  parameters to model electrical vs optical interconnects.
- **Bandwidth contention**: Requests queue at PCIe endpoints when bandwidth
  is saturated. Memory-bound ops naturally take longer with lower
  interconnect bandwidth.
- **DMA backpressure**: The accelerator limits in-flight requests via
  `maxOutstandingReqs`, providing natural flow control.
- **Memory traffic metrics**: Total read bytes, write bytes, and total
  memory transactions reported per accelerator in SQLite.

### Not yet modeled (future work):
- **SRAM capacity constraints and tiling**: The timing model doesn't check
  whether tensors fit in the 32MB on-chip SRAM. Large tensors would need
  tiling with DRAM spill, adding latency.
- **Double-buffering / compute-memory overlap**: Real accelerators pipeline
  DMA transfers with computation on the previous tile. We use sequential
  READ → COMPUTE → WRITE phases.
- **GPU-accelerator shared memory contention**: The accelerator uses a
  dedicated DRAM controller — no shared L2 or bandwidth contention with
  GPUs. (Shared DRAM is a future knob for studying memory architecture.)
- **Pipeline and startup overhead**: Each operation starts immediately with
  no pipeline fill latency.
- **DRAM bank conflicts**: The idealmemcontroller is ideal — no row buffer
  effects, scheduling, or contention.

## 7. Directory Structure

```
amd/
├── benchmarks/dnn/
│   ├── acceltensor/
│   │   ├── operator.go          # tensor.Operator for accelerator dispatch
│   │   └── tensor.go            # Tensor with device pointer
│   ├── gputensor/
│   │   └── operator.go          # tensor.Operator for GPU kernel launch
│   ├── tensor/
│   │   ├── tensor.go            # Tensor, DeviceTensor interfaces
│   │   └── operator.go          # Operator interface, CPUOperator
│   └── training_benchmarks/
│       ├── vgg16/               # VGG16 with mixed GPU+accel support
│       ├── minerva/             # Minerva with accel support
│       ├── lenet/               # LeNet (generalized to interfaces)
│       └── xor/                 # XOR (generalized to interfaces)
├── driver/
│   ├── api.go                   # AccelInference(), SelectAccelerator()
│   ├── driver.go                # RegisterAccelerator(), message routing
│   └── command.go               # AccelInferenceCommand
├── protocol/
│   └── accelprotocol.go         # AccelInferenceReq/Rsp, op types
├── timing/accelerator/
│   ├── comp.go                  # Accelerator Akita component + timing
│   └── builder.go               # Builder for accelerator component
└── samples/runner/
    ├── timingconfig/
    │   ├── accelbuilder/        # Platform-level accelerator builder
    │   └── builder.go           # Extended for --num-accel
    ├── flag.go                  # --num-accel flag
    └── report.go                # Accelerator metric reporting
```

## 8. How to Run

```bash
# GPU-only (existing behavior, unchanged)
cd amd/samples/minerva && go build && ./minerva -timing -gpus=1 -report-all

# GPU + 1 accelerator (all layers on GPU, acc metrics reported but idle)
./minerva -timing -gpus=1 -num-accel=1 -report-all

# GPU + 1 accelerator with specific layers offloaded
# (edit accel_config.json to specify which layers)
./minerva -timing -gpus=1 -num-accel=1 -report-all -accel-config=accel_config.json

# Multi-GPU + accelerator
./minerva -timing -gpus=1,2 -num-accel=1 -report-all
```

Metrics are written to `akita_sim_*.sqlite3`, table `mgpusim_metrics`.
Accelerator-specific metrics: `busy_time`, `op_count`,
`total_compute_cycles`.

## 9. Relationship to Paper

This work can support a paper with the following angles:

1. **Infrastructure contribution**: A reusable framework for simulating
   heterogeneous GPU + accelerator systems in a cycle-accurate GPU simulator,
   with minimal changes to the existing codebase (~1,400 lines for
   infrastructure, ~200 lines to generalize interfaces).

2. **Design space exploration**: Using the JSON config to sweep different
   layer-to-device assignments and measure the resulting accelerator
   utilization, GPU idle time, and end-to-end simulated latency.

3. **Timing model validation** (Phase 2): Once the accelerator's memory
   port is wired to DRAM, comparing the analytical timing estimates against
   the transaction-level memory simulation to quantify the error of pure
   analytical models.

4. **LLM workload characterization**: Using the GPT-2 benchmark to profile
   which transformer operations dominate accelerator time, where the
   GPU-accelerator handoff creates bubbles, and how data parallelism
   interacts with heterogeneous placement.
