# MGPUSim Interconnect Design — Current State

This document describes how interconnects are modeled in MGPUSim, from the
underlying Akita simulation framework up through the system-level platform
builders. Understanding this is a prerequisite for the chiplet packaging work.

---

## 1. Akita Networking Framework (the engine under the hood)

MGPUSim does not implement its own network simulation. It uses the **Akita v4**
discrete-event simulation framework's NoC (Network-on-Chip) package. All
bandwidth limiting, switch modeling, flit serialization, and routing live in
Akita. MGPUSim's builders just configure and wire Akita components.

Source: `github.com/sarchlab/akita/v4/noc/networking/`

### 1.1 Core Abstractions

| Concept | What it is |
|---------|-----------|
| **Connector** | Factory that builds a network (switches, endpoints, links, routing tables) |
| **Switch** | Akita component with per-port pipelines, buffers, and an arbiter (crossbar) |
| **Endpoint** | Wraps device ports; converts messages ↔ flits for the network |
| **DirectConnection** | Zero-latency ideal link (the only link type currently used) |
| **Flit** | Fixed-size chunk of a message. A message of `TrafficBytes` becomes `ceil(TrafficBytes / flitByteSize)` flits |
| **Router** | Populates switch routing tables. Two built-in: Floyd-Warshall (hop-count) and BandwidthFirst |

### 1.2 How Bandwidth Is Modeled

Bandwidth emerges from three parameters working together:

```
Effective BW = flitByteSize × NumOutputChannel × frequency
```

| Parameter | Set where | Meaning |
|-----------|-----------|---------|
| `flitByteSize` | Connector (derived from bandwidth or PCIe version) | Bytes per flit |
| `NumInputChannel` / `NumOutputChannel` | Per switch port and per endpoint | Max flits that can enter/leave per cycle |
| `frequency` | Connector | Clock rate of network components |

**Example — PCIe Gen4 x16 (the system-level interconnect):**

```
Link bandwidth per lane table:
  PCIe 1: 2 × 2^30 B/s    PCIe 2: 4 × 2^30 B/s
  PCIe 3: 8 × 2^30 B/s    PCIe 4: 16 × 2^30 B/s
  PCIe 5: 32 × 2^30 B/s

Total BW = linkBandwidth × width / 8
PCIe 4 x16 = 16 × 2^30 × 16 / 8 = ~34.4 GB/s

flitByteSize = round(bandwidth / frequency)
             = round(34.4 × 10^9 / 10^9) = 34 bytes

With NumOutputChannel=1 at 1 GHz → 34 bytes/cycle → ~34 GB/s
```

**Example — Accelerator internal interconnect (default 256 GB/s):**

```
flitByteSize = round(256 × 10^9 / 10^9) = 256 bytes
With NumOutputChannel=1 at 1 GHz → 256 bytes/cycle → 256 GB/s
```

When you set `WithBandwidth(bytesPerSecond)`, the connector computes `flitByteSize`
automatically. When you set `WithVersion(gen, lanes)`, it looks up the per-lane
bandwidth, multiplies by lanes/8, then calls `WithBandwidth()`.

### 1.3 How Latency Is Modeled

Latency has two components:

**Switch latency** — Set via `WithSwitchLatency(cycles)`. Applied as a pipeline
inside each switch port. When a flit enters a switch, it takes `latency` cycles
to traverse the routing pipeline before it can be forwarded to the output port.
This is per-message-traversal, not per-flit.

**Link latency** — Currently **zero**. All links use `DirectConnection` which
delivers messages instantly. The `LinkParameter` struct has fields for pipeline
stages (`NumStage`, `CyclePerStage`) but these are only used by non-ideal links,
which are not implemented yet (`IsIdeal: false` panics).

**Queueing latency** — Emerges naturally from buffer contention. Each switch port
has incoming/outgoing buffers (default 16 flits). When buffers fill up, flits
wait, creating backpressure. This is the primary source of variable latency.

**Total latency for a message traversing N switches:**
```
latency ≈ N × switchLatency + queueing_delays + ceil(msgBytes / flitByteSize) serialization cycles
```

### 1.4 Switch Internals (per tick)

Each switch processes in this order every cycle:

1. **sendOut** — Push flits from `sendOutBuffer` → link (up to `NumOutputChannel` per port)
2. **forward** — Crossbar arbitration: `forwardBuffer` → `sendOutBuffer` (output port assignment)
3. **route** — Routing table lookup: `routeBuffer` → `forwardBuffer` (determine next hop)
4. **movePipeline** — Advance the per-port routing pipeline by one stage
5. **startProcessing** — Pull flits from link → pipeline (up to `NumInputChannel` per port)

### 1.5 Endpoint Internals (message ↔ flit conversion)

Endpoints sit between device ports and the network:

```
Device Port → Endpoint → [message split into flits] → Network Switch
Network Switch → [flits assembled into message] → Endpoint → Device Port
```

Each cycle:
1. Pull messages from device, split into `ceil(TrafficBytes / flitByteSize)` flits
2. Send up to `NumOutputChannel` flits to network
3. Receive up to `NumInputChannel` flits from network
4. Reassemble flits into messages (wait for all flits of a message to arrive)
5. Deliver complete messages to device ports

### 1.6 Routing

Two routing algorithms are available:

| Algorithm | Metric | When to use |
|-----------|--------|-------------|
| **Floyd-Warshall** (default) | Minimum hop count | Homogeneous networks |
| **BandwidthFirst** | Maximum bottleneck bandwidth on path | Heterogeneous link speeds |

Routes are computed **once** at `EstablishRoute()` and are static for the entire
simulation. There is no dynamic/adaptive routing.

### 1.7 Available Connector Types in Akita

| Connector | Location | Use case |
|-----------|----------|----------|
| **PCIe** | `noc/networking/pcie/` | PCIe tree topology (root complex → switches → devices) |
| **NVLink** | `noc/networking/nvlink/` | Multi-protocol: PCIe + NVLink + Ethernet |
| **Mesh** | `noc/networking/mesh/` | 3D mesh/torus topology |

MGPUSim currently only uses the **PCIe** connector.

---

## 2. MGPUSim Platform Topology

### 2.1 System-Level View

```
                    Driver                MMU
                      │                    │
                      ▼                    ▼
              ┌─────────────────────────────────┐
              │     PCIe Root Complex            │
              │  (Gen4 x16, 140-cycle latency)   │
              └──┬──────┬──────┬──────┬─────────┘
                 │      │      │      │
              Switch  Switch  Switch  Switch
              (pair)  (pair)  (accel) (accel)
               │  │    │  │    │       │
             GPU1 GPU2 GPU3 GPU4 Accel0 Accel1
```

**Ports on the Root Complex:**
- `Driver.GPU` — GPU kernel dispatch commands
- `Driver.Accelerator` — Accelerator inference requests
- `Driver.MMU` — Memory management
- `MMU.Migration` — Page migration service
- `MMU.Top` — Address translation responses

**GPU switch grouping:** GPUs are paired: GPU 1+2 share a switch, GPU 3+4 share
a switch, etc. (`if i%2 == 1 { newSwitch }`). This is a simplistic model — the
switch latency is the same regardless.

**Accelerator switches:** Each accelerator gets its own dedicated switch off the
root complex.

**Configuration (hardcoded in `timingconfig/builder.go:259-262`):**
```go
pcieConnector := pcie.NewConnector().
    WithEngine(b.simulation.GetEngine()).
    WithVersion(4, 16).      // PCIe Gen4 x16 → ~34 GB/s
    WithSwitchLatency(140)   // 140 cycles per switch hop
```

### 2.2 Within a GPU

**All internal connections are `DirectConnection`** — zero latency, infinite
bandwidth. The timing comes from the components themselves (cache hit/miss
latency, TLB lookup, pipeline stages in the CU), not from interconnect.

```
┌─────────────────── GPU Domain ────────────────────────┐
│                                                        │
│  ┌── Shader Array 0 ──────────────────────────────┐   │
│  │  CU[0] ─→ L1V ROB ─→ L1V AT ─→ L1V TLB       │   │
│  │          ─→ L1V Cache (16KB, writearound)      │   │
│  │  CU[1] ─→ ...                                  │   │
│  │  CU[2] ─→ ...                                  │   │
│  │  CU[3] ─→ ...                                  │   │
│  │  All CUs ─→ L1S ROB ─→ L1S AT ─→ L1S TLB     │   │
│  │           ─→ L1S Cache (16KB, writethrough)    │   │
│  │  All CUs ─→ L1I ROB ─→ L1I Cache (32KB)       │   │
│  │           ─→ L1I AT ─→ L1I TLB                │   │
│  └────────────────────────────────────────────────┘   │
│  ┌── Shader Array 1..15 ─────────────────────────┐   │
│  │  (same structure)                              │   │
│  └────────────────────────────────────────────────┘   │
│                                                        │
│  ┌── L1-to-L2 Connection (DirectConnection) ─────┐   │
│  │  16× L1V Cache.Bottom ─→ Interleaved Mapper   │   │
│  │  16× L1S Cache.Bottom ─→ (128-byte granule)   │   │
│  │  16× L1I Cache.Bottom ─→ across 16 L2 banks   │   │
│  └────────────────────────────────────────────────┘   │
│                                                        │
│  ┌── L2 Cache Layer ─────────────────────────────┐   │
│  │  L2[0] (128KB, 16-way) ──→ DRAM[0]            │   │
│  │  L2[1] ──→ DRAM[1]                            │   │
│  │  ...                                           │   │
│  │  L2[15] ──→ DRAM[15]                          │   │
│  │  (DirectConnection, SinglePortMapper per bank) │   │
│  └────────────────────────────────────────────────┘   │
│                                                        │
│  CP ←─→ DMA Engine ←─→ All caches/CUs (InternalConn) │
│  RDMA Engine (multi-GPU, local → L1 mapper)           │
│  Page Migration Controller                            │
│                                                        │
│  Exposed ports → PCIe: CommandProcessor, RDMAReq,     │
│                        RDMAData, PMC, TLB translations │
└────────────────────────────────────────────────────────┘
```

**Key internal connection instances:**
| Connection Name | Type | What it connects |
|----------------|------|-----------------|
| `InternalConn` | DirectConnection | CP ↔ DMA, caches, CUs, TLBs, ATs, RDMA, PMC |
| `L1ToL2Conn` | DirectConnection | L1 caches/TLBs ↔ L2 caches ↔ RDMA engine |
| `L2ToDRAMConn` | DirectConnection | L2 caches ↔ DRAM controllers ↔ DMA ↔ PMC |
| `L1TLBToL2TLBConn` | DirectConnection | L1 TLBs ↔ L2 TLB |
| (per-SA connections) | DirectConnection | CU ↔ ROBs ↔ ATs ↔ TLBs ↔ L1 caches |

### 2.3 Within an Accelerator

The accelerator has its **own internal PCIe network** — separate from the system
PCIe. This is where interconnect technology studies happen.

```
┌───────────── Accelerator Domain ──────────────────┐
│                                                     │
│  AccelUnit                                          │
│    ├─ ToDriver port (exposed to system PCIe)        │
│    └─ ToMem port ──┐                                │
│                     │                                │
│         ┌──── Internal PCIe Network ────┐           │
│         │  Configurable:                 │           │
│         │  • Bandwidth (default 256GB/s) │           │
│         │  • Switch latency (default 10) │           │
│         └────────────┬───────────────────┘           │
│                      │                                │
│                 DRAM Controller                       │
│              (idealmemcontroller)                     │
│              100-cycle latency                        │
│              backed by globalStorage                  │
│                                                     │
│  Memory mapper: SinglePortMapper → DRAM.Top          │
│  ToDriver is exposed as domain port                  │
│  ToMem is NOT exposed (internal only)                │
└─────────────────────────────────────────────────────┘
```

**Configuration (in `accelbuilder/builder.go:150-153`):**
```go
memPCIe := pcie.NewConnector().
    WithEngine(b.simulation.GetEngine()).
    WithBandwidth(b.interconnectBW).       // default 256 GB/s
    WithSwitchLatency(b.interconnectLatency) // default 10 cycles
```

The accelerator's `ToDriver` port is then plugged into the **system-level** PCIe
(via `pcieConnector.PlugInDevice(switchID, accel.Ports())`), so driver ↔
accelerator communication goes through the system PCIe Gen4 network.

---

## 3. All Configurable Interconnect Parameters

### 3.1 System-Level PCIe (GPU ↔ GPU, GPU ↔ Driver)

| Parameter | Current Value | Where Set | Configurable? |
|-----------|--------------|-----------|---------------|
| PCIe version | Gen4 x16 | `builder.go:261` | **No** (hardcoded) |
| Switch latency | 140 cycles | `builder.go:262` | **No** (hardcoded) |
| Bandwidth | ~34 GB/s | Derived from version | **No** |
| Buffer sizes | 16 flits | Akita PCIe default | **No** |
| Channels | 1 in / 1 out | Akita PCIe default | **No** |
| GPU switch grouping | Pairs (i%2) | `builder.go:229` | **No** (hardcoded) |

### 3.2 Accelerator Internal PCIe (AccelUnit ↔ DRAM)

| Parameter | Default | CLI Flag | Where Set |
|-----------|---------|----------|-----------|
| Bandwidth | 256 GB/s | `--accel-interconnect-bw` | `accelbuilder.go:46` |
| Switch latency | 10 cycles | `--accel-interconnect-latency` | `accelbuilder.go:47` |
| Max outstanding reqs | 64 | `--accel-max-outstanding` | `accelbuilder.go:48` |
| Buffer sizes | 16 flits | Akita PCIe default | Not configurable |
| Channels | 1 in / 1 out | Akita PCIe default | Not configurable |
| DRAM latency | 100 cycles | `accelbuilder.go:139` | Not configurable |

### 3.3 Within-GPU Connections

| Parameter | Value | Configurable? |
|-----------|-------|---------------|
| Connection type | DirectConnection | **No** — all zero-latency |
| L1V cache bank latency | 60 cycles | No (hardcoded in GPU builder) |
| L1S/L1I cache bank latency | 1 cycle | No |
| L2 cache latency | 16-way, 64 MSHR | No |
| DRAM latency | 100 cycles | No |
| DRAM frequency | 500 MHz | No |
| Memory interleaving | 128 bytes | No |

---

## 4. The Two Separate PCIe Networks

This is the most important design detail for chiplet work. There are currently
**two independent PCIe networks** that happen to share a backing memory:

| Network | Scope | Bandwidth | Latency | Purpose |
|---------|-------|-----------|---------|---------|
| **System PCIe** | Driver ↔ all GPUs ↔ all accels | ~34 GB/s (Gen4 x16) | 140 cycles/switch | Command dispatch, RDMA, page migration |
| **Accel Internal PCIe** (one per accel) | AccelUnit ↔ its DRAM | 256 GB/s (default) | 10 cycles/switch | Tensor data movement during inference |

The system PCIe carries **control traffic** (kernel dispatch, inference
requests/responses). The accelerator internal PCIe carries **bulk data traffic**
(tensor reads/writes during the READ→COMPUTE→WRITE phases).

GPU memory traffic (L1→L2→DRAM) does **not** go through any PCIe — it uses
DirectConnections within the GPU domain.

### 4.1 What this means for message latency

**Driver → GPU kernel dispatch:**
```
Driver.GPU port → Root Complex (140 cycles) → GPU Switch (140 cycles)
    → GPU.CommandProcessor
Total: 280 cycles + queueing
```

**Driver → Accelerator inference request:**
```
Driver.Accelerator port → Root Complex (140 cycles) → Accel Switch (140 cycles)
    → Accel.ToDriver
Total: 280 cycles + queueing
```

**Accelerator tensor read (64 bytes):**
```
Accel.ToMem → Internal Switch (10 cycles) → Root Complex (10 cycles)
    → DRAM.Top → 100 cycles DRAM latency → response back
Total: ~140 cycles + queueing per 64-byte read
```

**GPU L1 miss → L2 → DRAM:**
```
L1 Cache → DirectConnection (0) → L2 Cache (lookup) → DirectConnection (0)
    → DRAM (100 cycles)
Total: L2 lookup time + 100 cycles DRAM
```

---

## 5. Memory Model

### 5.1 Unified Global Storage

All GPUs and accelerators share a single `mem.Storage` byte array:

```
globalStorage size = (numGPUs + numAccelerators) × gpuMemSize + cpuMemSize
```

**Address partitioning:**
```
[0, gpuMemSize)                           → CPU
[gpuMemSize, 2×gpuMemSize)                → GPU[1]
[2×gpuMemSize, 3×gpuMemSize)              → GPU[2]
...
[(numGPUs+1)×gpuMemSize, ...]             → Accel[0]
[(numGPUs+2)×gpuMemSize, ...]             → Accel[1]
```

This means a tensor allocated on GPU[1] can be read by Accel[0] without any
explicit data transfer — they both read/write the same backing array. The
interconnect simulation only affects **timing**, not correctness.

### 5.2 DRAM Controllers

Each GPU has 16 `idealmemcontroller` instances (one per memory bank). Each
accelerator has 1 `idealmemcontroller`. All use the same `globalStorage` but
respond to different address ranges based on `memAddrOffset`.

"Ideal" means: fixed 100-cycle latency, no bandwidth limit, no bank conflicts,
no refresh cycles. The DRAM controller is not a bottleneck in the current model.

---

## 6. What Is NOT Modeled

| Missing feature | Impact |
|----------------|--------|
| Link propagation delay | All links are ideal (zero delay). Real PCIe/optical links have ns-scale propagation |
| Non-ideal DRAM | No bank conflicts, refresh, bandwidth limits. DRAM is always 100 cycles |
| GPU-to-accelerator shared DRAM contention | Each device has dedicated DRAM. No shared memory bus |
| SRAM capacity on accelerator | 32 MB SRAM is a parameter but not used to limit tiling |
| Dynamic routing | Routes are static (computed once at setup) |
| Multiple virtual channels | Only 1 input/output channel per port |
| Cache coherence traffic across devices | No snooping or directory-based coherence |
| NVLink or other high-speed GPU interconnects | Only PCIe connector is used |

---

## 7. Implications for Chiplet Design

### What we can leverage

1. **Multiple PCIe networks** are already supported — the accelerator pattern
   proves we can create independent networks with different BW/latency parameters
2. **Akita's PCIe connector** supports arbitrary tree topologies (root complex →
   switches → devices) with configurable bandwidth and switch latency per network
3. **`sim.Domain`** provides clean encapsulation — a chiplet can be a domain
   containing sub-domains (GPUs, accelerators) with its own internal network

### What we need to change

1. **System PCIe parameters are hardcoded** — need to make them configurable or
   replaced by a chiplet-aware topology
2. **GPU switch grouping is simplistic** (pairs by index) — chiplet config should
   dictate which GPUs share a switch
3. **No concept of "intra-chiplet" vs "inter-chiplet"** — need hierarchical
   networks where devices in the same chiplet communicate through a fast local
   network, and cross-chiplet traffic goes through a slower global network
4. **A single `pcie.Connector` has one bandwidth/latency setting for all its
   switches** — can't have different latencies for different switches in the same
   network. Must use separate `Connector` instances for different tiers.

### The hierarchical network approach

```
Tier 1 (inter-chiplet): pcie.NewConnector().WithBandwidth(X).WithSwitchLatency(Y)
  └─ Connects chiplet gateways to driver

Tier 2 (intra-chiplet, one per chiplet): pcie.NewConnector().WithBandwidth(A).WithSwitchLatency(B)
  └─ Connects GPUs + accelerators within one chiplet

Tier 3 (accel-to-DRAM, one per accel): pcie.NewConnector().WithBandwidth(C).WithSwitchLatency(D)
  └─ Already exists — accelerator internal memory link
```

Each tier is its own `pcie.Connector` instance with its own parameters. This is
exactly how Akita is designed to be used — the accelerator builder already
demonstrates the pattern.
