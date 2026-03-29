---
marp: true
theme: default
paginate: true
header: "**NC STATE UNIVERSITY**"
style: |
  header {
    background-color: #CC0000;
    color: white;
    font-size: 14px;
    padding: 8px 20px;
    width: 100%;
    position: absolute;
    top: 0;
    left: 0;
  }
  section {
    padding-top: 60px;
  }
  h1 {
    text-align: center;
    font-size: 36px;
  }
  h2 {
    font-size: 28px;
  }
  table {
    font-size: 18px;
  }
---

# Approach 2: Heterogeneous GPU + Accelerator Simulation in MGPUSim

---

## Architecture — How It Works

- **MGPUSim** extended with a dedicated **inference accelerator** alongside GPU compute units
- Inspired by heterogeneous architectures like **AMD MI300A**
- **Benchmark picks the operator** per DNN layer via `accel_config.json`
  - Each layer assigned either `GPUOperator` or `AccelOperator`
  - Trainer uses `tensor.Operator` interface — device-agnostic

**End-to-End Flow:**
1. Benchmark loads layer→device mapping from JSON config
2. Forward/backward pass dispatches each layer to assigned device
3. Accelerator ops: functional CPU compute + `AccelInferenceReq` for timing
4. Accelerator executes 3-phase pipeline: **READ → COMPUTE → WRITE**
5. Memory traffic flows through configurable PCIe interconnect to DRAM

---

## System Topology

```
                Driver                MMU
                  │                    │
          ┌───────────────────────────────────┐
          │     PCIe Root Complex              │
          │  (Gen4 x16, 140-cycle latency)     │
          └──┬──────┬──────┬──────┬───────────┘
             │      │      │      │
          Switch  Switch  Switch  Switch
           │  │    │  │    │       │
         GPU1 GPU2 GPU3 GPU4 Accel0 Accel1
```

**Each Accelerator has its own internal PCIe network:**
- Configurable bandwidth (`--accel-interconnect-bw`, default 256 GB/s)
- Configurable switch latency (`--accel-interconnect-latency`, default 10 cycles)
- Full-traffic DMA: every **64 bytes** of tensor data generates a real `mem.ReadReq`/`mem.WriteReq`

---

## Completed Work — Phase Summary

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 1** | All 19 accelerator ops timing-accurate (zero CPU fallbacks) | DONE |
| **Phase 2.5** | Full-traffic DMA with PCIe interconnect model | DONE |
| **Benchmark Wiring** | Minerva, VGG16, LeNet, XOR, GPT-2 all support accel offloading | DONE |
| **Interconnect Sweep** | 7 configs (PCIe Gen4-6 + optical) across benchmarks | DONE (Minerva, LeNet) |

**19 Timing-Accurate Operations:**
Gemm, ReluForward/Backward, Softmax, CrossEntropy, ElementWiseMul, ScaleAdd, Adam, RMSProp, Sum, Im2Col, Transpose, Rotate180, Dilate, MaxPooling, AvgPooling, and derivatives

---

## Benchmark Results

| Benchmark | Config | Accel[0] Ops | Accel[1] Ops | Mem Reqs (each) | Read (each) | Write (each) |
|-----------|--------|:---:|:---:|---:|---:|---:|
| **Minerva** | 2 GPU + 2 accel | 48 | 48 | 86,692 | 5.0 MB | 531 KB |
| **LeNet** | 2 GPU + 2 accel | 48 | 48 | 858,418 | 29.3 MB | 25.7 MB |
| **XOR** | 1 GPU + 1 accel | 450 | — | 1,400 | 32 KB | 18 KB |

**Key Observations:**
- LeNet generates **10x more memory traffic** than Minerva (Conv2D layers dominate)
- Full-traffic DMA captures real interconnect contention and backpressure
- Accelerator is **99.7% memory-bound** — bandwidth is the bottleneck, not compute

---

## Interconnect Technology Comparison

**7 configurations swept across Minerva and LeNet (2 GPU + 2 accel):**

| Config | Bandwidth | Latency | Category |
|--------|-----------|---------|----------|
| PCIe Gen4 x16 | 32 GB/s | 50 cycles | Electrical |
| PCIe Gen5 x16 | 64 GB/s | 40 cycles | Electrical |
| PCIe Gen6 x16 | 128 GB/s | 30 cycles | Electrical |
| Electrical High-BW | 256 GB/s | 20 cycles | Electrical (high-end) |
| Optical Low | 512 GB/s | 5 cycles | **Optical** (conservative) |
| Optical Mid | 1 TB/s | 3 cycles | **Optical** (mid-range) |
| Optical High | 2 TB/s | 2 cycles | **Optical** (aggressive) |

**Goal:** Quantify how optical interconnects (Co-Packaged Optics) improve accelerator performance vs. traditional electrical links — motivating package-aware design decisions.

---

# What Lies Ahead

---

## Next Phase: Chiplet Configuration Integration

**Goal:** Model a chiplet-based packaging topology where devices are grouped into chiplets with distinct intra-chiplet vs. inter-chiplet interconnects.

**Hierarchical Network Model (3 Tiers):**

```
Tier 1 — Inter-Chiplet Network
  (e.g., optical interposer, lower BW, higher latency)
  └── Connects chiplet gateways to driver/host

Tier 2 — Intra-Chiplet Network (one per chiplet)
  (e.g., silicon interposer, higher BW, lower latency)
  └── Connects GPUs + Accelerators within one chiplet

Tier 3 — Accel-to-DRAM (already exists)
  (configurable BW/latency per accelerator)
  └── Internal memory link
```

**JSON Chiplet Config** will define: which devices belong to which chiplet, intra/inter-chiplet BW and latency, and packaging technology (organic, silicon interposer, 3D stack).

---

## Analysis & Experiments Ahead

**Immediate Analysis:**
- Extract and compare Minerva + LeNet sweep results across all 7 interconnect configs
- Quantify speedup from optical vs. electrical for memory-bound accelerator workloads
- Complete VGG16 sweep (needs 64+ GB RAM machine)

**Chiplet Experiments (once integrated):**
- Compare **monolithic** (all devices on one fast network) vs. **chiplet** (hierarchical) topologies
- Measure impact of inter-chiplet latency on data-parallel DNN training throughput
- Study **optical intra-chiplet links** vs. **electrical interposer** — does CPO reduce the "communication wall"?

**Modeling Improvements (future):**
- SRAM capacity + double-buffering — overlap memory reads with compute
- Weight caching — avoid re-reading from DRAM for repeated GEMMs
- Bandwidth-limited DRAM — replace ideal memory controller
- GPU-accelerator shared DRAM contention modeling

