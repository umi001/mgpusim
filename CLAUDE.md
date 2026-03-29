# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build, Test, and Lint Commands

```bash
# Build all packages (~16s clean, ~70s first run with deps)
go build ./...

# Run unit tests (~17s) - uses Ginkgo framework
ginkgo -r --skip-package=nvidia

# Run specific package tests
go test ./amd/emu/... -v

# Lint AMD code
golangci-lint run ./amd/... --timeout=10m

# Run acceptance tests (~3.5 min for single GPU)
cd amd/tests/acceptance && go build && ./acceptance -num-gpu=1

# Run sample simulation
cd amd/samples/fir && go build && ./fir -timing --report-all -length=64 -verify
```

## Architecture Overview

MGPUSim is a cycle-accurate GPU simulator modeling AMD GCN3 instruction set architecture. It uses the Akita discrete-event simulation framework (`github.com/sarchlab/akita/v4`).

### Two Simulation Modes

1. **Emulation Mode** (`amd/emu/`): Fast functional simulation
   - `ComputeUnit` executes wavefronts instruction-by-instruction
   - `ALUImpl` implements all GCN3 instruction semantics (VALU, SALU, LDS, FLAT, etc.)
   - Used for correctness verification

2. **Timing Mode** (`amd/timing/`): Cycle-accurate performance simulation
   - `cu/` - Compute Unit with pipeline stages, scheduling, register files
   - `cp/` - Command Processor for kernel dispatch
   - `rob/` - Reorder Buffer
   - `rdma/` - Remote DMA for multi-GPU communication
   - `pagemigrationcontroller/` - Unified memory page migration

### Key Components

- **`amd/driver/`**: GPU driver simulation - memory allocation, kernel launch, command queues
- **`amd/insts/`**: GCN3 instruction definitions, decoder, disassembler, HSACO parsing
- **`amd/kernels/`**: Kernel loading from HSACO (ELF) files
- **`amd/benchmarks/`**: Benchmark implementations with embedded HSACO binaries
- **`amd/samples/`**: Runnable simulation examples

### HSACO (Kernel Binary) Format

Kernels are compiled to HSACO format (AMD's GPU binary). Two versions:
- **V2/V3**: 256-byte header per kernel in `.text` section, used by GCN3
- **V5**: 64-byte kernel descriptor in `.rodata`, instructions in `.text`

Multi-kernel ELFs: Each kernel is a symbol pointing to its code object. Loading extracts specific kernel data using `symbol.Value` and `symbol.Size`.

### Simulation Flow

1. Benchmark loads HSACO via `kernels.LoadProgramFromMemory(data, "kernelName")`
2. Driver allocates GPU memory, copies kernel args
3. Driver creates dispatch packet with kernel address
4. Command Processor dispatches work-groups to Compute Units
5. CU executes wavefronts (64 threads) through pipeline

## Heterogeneous Accelerator Support (Approach 2)

### Goal

Extend MGPUSim to simulate **heterogeneous GPU + accelerator** systems (like AMD MI300A) where a dedicated inference accelerator sits alongside GPU compute units. The ultimate target is running an **LLM workload with high memory footprint** (e.g., GPT-2 or similar transformer model) on a heterogeneous system where compute-heavy layers are offloaded to the accelerator. Existing DNN benchmarks (Minerva, VGG16, LeNet) serve as stepping-stone test cases only.

### Current State

The core framework is built and compiles. A basic end-to-end proof of concept works (Minerva benchmark runs in timing mode with selected FC layers on the accelerator). The key pieces that are done:

**Infrastructure (complete):**
- Accelerator Akita component (`amd/timing/accelerator/`) with `Tick()` loop, request handling, and analytical cycle estimation
- Protocol messages (`AccelInferenceReq/Rsp`) with op type constants (GEMM, Conv2D, MaxPool, AvgPool, ReLU, Softmax, ElementWise)
- Driver APIs: `RegisterAccelerator()`, `SelectAccelerator()`, `AccelInference()`
- Platform builder (`accelbuilder/`) that instantiates accelerator and wires it to driver
- `tensor.Operator` interface used throughout (trainer, all benchmarks) — no concrete GPU types
- `tensor.DeviceTensor` interface for cross-operator tensor compatibility
- JSON config (`accel_config.json`) for per-layer device assignment
- Accelerator metric reporting in `runner/report.go` (busy_time, op_count, total_compute_cycles)

**Cross-operator tensor compatibility (complete):**
- Both `gputensor.GPUOperator` and `acceltensor.Operator` use `ptrOf()` helper that extracts device pointers via `tensor.DeviceTensor` interface instead of concrete type casts
- Either operator can accept tensors created by the other — critical for boundary layers during forward and backward passes

**Accelerator operator (`acceltensor/operator.go`):**
- Memory operations fully implemented: `Create`, `Free`, `Copy`, `Clone`, `Init`, `Slice`, `Repeat`, `Clear`, `Zeros`, `Reshape`
- All 19 compute ops timing-accurate (dispatch `AccelInferenceReq`): `Gemm`, `Softmax`, `CrossEntropy`, `CrossEntropyDerivative`, `SoftmaxCrossEntropyDerivative`, `ElementWiseMul`, `ScaleAdd`, `RMSProp`, `Adam`, `ReluForward`, `ReluBackward`, `Sum`, `Im2Col`, `Transpose`, `Rotate180`, `Dilate`, `MaxPoolingForward/Backward`, `AvgPoolingForward/Backward`
- Zero CPU fallbacks remaining

### Design

- **Benchmark picks the operator**: Each DNN layer is assigned either a `gputensor.GPUOperator` or `acceltensor.Operator` based on `accel_config.json`. The driver does not route transparently.
- **Shared memory space**: Both GPU and accelerator tensors are backed by the same `globalStorage`, so data can be accessed by either device without explicit transfers.
- **Multi-GPU + accelerator**: Fully supported. `DataParallelismMultiGPUTrainer` uses `tensor.Operator` interface.

### Key Files

| File | Purpose |
|------|---------|
| `amd/protocol/accelprotocol.go` | `AccelInferenceReq/Rsp` messages, op type constants |
| `amd/timing/accelerator/comp.go` | Accelerator Akita component with `Tick()`, timing model, metrics |
| `amd/timing/accelerator/builder.go` | Builder for accelerator component |
| `amd/samples/runner/timingconfig/accelbuilder/builder.go` | Platform-level builder, wires accelerator to driver + DRAM |
| `docs/approach2_design.md` | Design document with rationale for paper writing |
| `docs/phase2_5_full_traffic_plan.md` | Full-traffic interconnect plan (optical vs electrical) |
| `amd/benchmarks/dnn/acceltensor/operator.go` | `tensor.Operator` impl — accelerator dispatch + host fallbacks |
| `amd/benchmarks/dnn/acceltensor/tensor.go` | Tensor struct with device pointer |
| `amd/benchmarks/dnn/tensor/tensor.go` | `DeviceTensor` interface (extends `Tensor` with `Ptr()`) |
| `amd/driver/api.go` | `SelectAccelerator()`, `AccelInference()` driver APIs |
| `amd/driver/driver.go` | `RegisterAccelerator()`, accelerator command processing |
| `amd/samples/runner/report.go` | Accelerator metric collection and reporting |

### Phases — Path to LLM Workload

#### Phase 1: Complete acceltensor operations — DONE ✓
- All 19 ops now timing-accurate (zero CPU fallbacks) — each dispatches `AccelInferenceReq`
- Ops: Gemm, ReluForward/Backward, Softmax, ElementWiseMul, ScaleAdd, Adam, RMSProp, Sum, CrossEntropy, CrossEntropyDerivative, SoftmaxCrossEntropyDerivative, Im2Col, Transpose, Rotate180, Dilate, MaxPoolingForward/Backward, AvgPoolingForward/Backward
- Each converted op computes functionally on CPU then dispatches AccelInferenceReq for timing

#### Phase 2/2.5: Full-traffic memory with parameterizable interconnect — DONE ✓
- Full-traffic DMA: every 64 bytes of tensor data generates a real `mem.ReadReq`/`mem.WriteReq` through the interconnect (not analytical probes)
- PCIe connector between accelerator and DRAM: configurable bandwidth (`--accel-interconnect-bw`) and switch latency (`--accel-interconnect-latency`)
- Enables optical vs electrical interconnect comparison by swapping bandwidth/latency parameters
- Progressive request issuing with backpressure: `maxOutstandingReqs` limits in-flight DMA requests
- Phased execution state machine: READ → COMPUTE → WRITE (memory timing emergent, compute analytical)
- Per-tensor read traffic estimation (GEMM: input M×K, weights K×N; Adam: 4 input tensors, etc.)
- Metrics: `total_read_bytes`, `total_write_bytes`, `total_mem_reqs`, `busy_time` in SQLite
- NOT yet modeled: SRAM capacity/tiling, double-buffering, DRAM bank conflicts, GPU-accel shared memory contention

#### Full Benchmark Support — DONE ✓ (commit `0bd2af03`)
- All 8 former CPU fallback ops converted to timing-accurate dispatch
- LeNet + XOR wired for accelerator support (accel_config.json, per-layer operator selection)
- Round-robin `SelectAccelerator` fix for multi-accel runs (Minerva, VGG16, LeNet)
- Tested: Minerva (2GPU+2accel), LeNet (2GPU+2accel), XOR (1GPU+1accel) — all pass
- VGG16: OOM on 30 GB machine — needs 64+ GB RAM for full-traffic DMA testing

#### Phase 3: GPT-2 Benchmark — DONE ✓ (merged into approach_2)
- Synthetic GPT-2 benchmark with multi-GPU data parallelism and accelerator offloading
- Deprioritized per mentor feedback — existing benchmarks are the focus

#### Phase 4: Protocol and architecture extensions (future)
- Add op types to protocol: `AccelOpLayerNorm`, `AccelOpAttention`, `AccelOpGELU`
- Add fields for quantization (INT8/FP16) and data type configuration
- Consider sequence parallelism and pipeline parallelism for multi-device LLM inference
- Add memory capacity constraints (model whether tensors fit in accelerator SRAM vs need DRAM access)

## Important Patterns

- Tests use Ginkgo/Gomega with extensive mocking via `go.uber.org/mock`
- Benchmarks embed HSACO binaries using `//go:embed kernels.hsaco`
- Platform configuration in `samples/*/platform.go`, `r9nano.go`, `shaderarray.go`
- NVIDIA code (`nvidia/`) is under development and should be skipped

## Compiling HIP Kernels

For LLVM/ROCm compiler tasks (compiling HIP code to HSACO), use the Docker image:
```bash
docker run -it rocm/dev-ubuntu-24.04:7.1.1
```

## Timing Expectations

Never cancel long-running commands:
- Initial build: ~70s (downloads deps)
- Unit tests: ~17s
- Acceptance tests (1 GPU): ~3.5 min
- Multi-GPU tests: 30+ min each

## Before Completing Tasks

**IMPORTANT**: Before finishing any code modification task, always run linting to ensure code quality:

```bash
# Install golangci-lint v2.1.5 (if not already installed)
go install github.com/golangci/golangci-lint/v2/cmd/golangci-lint@v2.1.5

# Run linting
golangci-lint run ./amd/... --timeout=10m
```

Fix any linting issues before considering the task complete. Common issues include:
- Unnecessary type conversions (`unconvert`)
- Function complexity (`gocognit`, `funlen`) - add `//nolint:gocognit,funlen` if justified
- Line length (`lll`) - keep lines under 120 characters
