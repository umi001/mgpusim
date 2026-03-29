# Chiplet-Based Configuration Design

This document describes the design for adding chiplet-based device grouping and
hierarchical interconnects to MGPUSim's heterogeneous simulation platform.

---

## 1. Motivation

Modern GPU accelerators like AMD MI300A use chiplet packaging — multiple compute
dies (XCDs), memory dies (HBM), and I/O dies are assembled on an interposer
within a single package. Devices on the same chiplet communicate through fast,
low-latency on-package interconnects (e.g., Infinity Fabric on an organic
interposer), while cross-chiplet communication goes through slower package-level
or board-level links.

MGPUSim currently models a **flat topology** where all GPUs and accelerators hang
off a single PCIe Gen4 network with uniform 140-cycle switch latency. There is
no concept of locality — GPU[1] and GPU[2] are equidistant from GPU[3] regardless
of physical placement. This makes it impossible to study:

- How device placement within chiplets affects communication cost
- Intra-chiplet vs inter-chiplet bandwidth/latency tradeoffs
- The benefit of co-locating a GPU and accelerator on the same chiplet
- Optical vs electrical interconnects at different tiers of the hierarchy

## 2. Current Topology (Flat)

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
               │  │    │  │    │       │
             GPU1 GPU2 GPU3 GPU4 Accel0 Accel1
```

All devices share one `pcie.Connector` with one bandwidth and one switch latency.
GPU pairing into switches is by index (`i%2`), not by any physical placement model.

## 3. Proposed Topology (Chiplet-Based Hierarchical)

```
                    Driver                MMU
                      │                    │
                      ▼                    ▼
              ┌──────────────────────────────────────┐
              │        Inter-Chiplet Network          │
              │  (configurable BW + latency)          │
              └────┬──────────────────────┬──────────┘
                   │                      │
          ┌────────┴────────┐    ┌────────┴────────┐
          │   Chiplet 0     │    │   Chiplet 1     │
          │  Intra-Chiplet  │    │  Intra-Chiplet  │
          │  Network        │    │  Network        │
          │  (512 GB/s,     │    │  (512 GB/s,     │
          │   5 cycles)     │    │   5 cycles)     │
          └──┬─────┬────┬───┘    └──┬─────┬────┬───┘
             │     │    │           │     │    │
           GPU1  GPU2 Accel0      GPU3  GPU4 Accel1
             │           │           │           │
          (internal)  (internal)  (internal)  (internal)
           L1→L2→     AccelUnit→   L1→L2→     AccelUnit→
           DRAM        DRAM        DRAM        DRAM
```

Three tiers of interconnect, each its own `pcie.Connector` with independent
parameters:

| Tier | Scope | Default BW | Default Latency | Instance Count |
|------|-------|-----------|----------------|----------------|
| **Inter-chiplet** | Chiplet gateways ↔ Driver/MMU | Configurable | Configurable | 1 |
| **Intra-chiplet** | Devices within one chiplet | Configurable per chiplet | Configurable per chiplet | 1 per chiplet |
| **Accel-to-DRAM** | AccelUnit ↔ its DRAM | 256 GB/s (existing) | 10 cycles (existing) | 1 per accelerator |

### 3.1 How Communication Traverses Tiers

**GPU[1] → GPU[2] (same chiplet):**
```
GPU[1] ports → Chiplet 0 intra-chiplet switch (5 cycles)
  → GPU[2] ports
Hops: 1 switch, latency: ~5 cycles + queueing
```

**GPU[1] → GPU[3] (different chiplets):**
```
GPU[1] ports → Chiplet 0 intra-chiplet switch (5 cycles)
  → Chiplet 0 gateway → inter-chiplet switch (50 cycles)
  → Chiplet 1 gateway → Chiplet 1 intra-chiplet switch (5 cycles)
  → GPU[3] ports
Hops: 3 switches, latency: ~60 cycles + queueing
```

**Driver → Accel[0] (control path):**
```
Driver.Accelerator → inter-chiplet root complex (50 cycles)
  → Chiplet 0 gateway → intra-chiplet switch (5 cycles)
  → Accel[0].ToDriver
Hops: 2 switches, latency: ~55 cycles + queueing
```

**Accel[0] tensor read (data path, unchanged):**
```
AccelUnit.ToMem → internal PCIe switch (10 cycles) → DRAM (100 cycles)
This path is entirely within the accelerator domain, unaffected by chiplet topology.
```

## 4. Configuration File Format

Each sample directory gets a `chiplet_config.json` file (alongside the existing
`accel_config.json`):

```json
{
  "chiplets": [
    {
      "name": "chiplet_0",
      "gpus": [1, 2],
      "accelerators": [0],
      "intra_bw_gbps": 512,
      "intra_latency_cycles": 5
    },
    {
      "name": "chiplet_1",
      "gpus": [3, 4],
      "accelerators": [1],
      "intra_bw_gbps": 512,
      "intra_latency_cycles": 5
    }
  ],
  "inter_chiplet_bw_gbps": 128,
  "inter_chiplet_latency_cycles": 50
}
```

### 4.1 Field Definitions

| Field | Type | Description |
|-------|------|-------------|
| `chiplets` | array | List of chiplet definitions |
| `chiplets[].name` | string | Human-readable chiplet identifier |
| `chiplets[].gpus` | int array | GPU IDs assigned to this chiplet (1-indexed, matching `-gpus` flag) |
| `chiplets[].accelerators` | int array | Accelerator indices assigned to this chiplet (0-indexed) |
| `chiplets[].intra_bw_gbps` | int | Intra-chiplet interconnect bandwidth in GB/s |
| `chiplets[].intra_latency_cycles` | int | Intra-chiplet switch latency in cycles |
| `inter_chiplet_bw_gbps` | int | Inter-chiplet interconnect bandwidth in GB/s |
| `inter_chiplet_latency_cycles` | int | Inter-chiplet switch latency in cycles |

### 4.2 Design Decisions

**Per-chiplet intra parameters** — Different chiplets can have different internal
interconnect characteristics. This supports heterogeneous packaging (e.g., one
chiplet with optical on-package links, another with electrical).

**Backward compatibility** — If no `chiplet_config.json` exists, the platform
builder falls back to the current flat PCIe topology. Existing benchmarks and
experiments continue to work unchanged.

**GPU IDs are 1-indexed** — Matching the existing `-gpus=1,2,3,4` CLI convention.

**Accelerator indices are 0-indexed** — Matching the existing `Accel[0]`, `Accel[1]`
naming convention.

**Validation rules:**
- Every GPU in `-gpus` must appear in exactly one chiplet
- Every accelerator (0 to `num-accel - 1`) must appear in exactly one chiplet
- No device can appear in multiple chiplets
- GPU IDs in config must match the IDs passed via `-gpus` flag

### 4.3 Example Configurations

**Single chiplet (all devices co-located, like MI300A):**
```json
{
  "chiplets": [
    {
      "name": "package_0",
      "gpus": [1, 2],
      "accelerators": [0, 1],
      "intra_bw_gbps": 512,
      "intra_latency_cycles": 5
    }
  ],
  "inter_chiplet_bw_gbps": 128,
  "inter_chiplet_latency_cycles": 50
}
```
(Inter-chiplet parameters are unused but required for schema consistency.)

**Four chiplets (one GPU + one accelerator each):**
```json
{
  "chiplets": [
    {"name": "c0", "gpus": [1], "accelerators": [0], "intra_bw_gbps": 256, "intra_latency_cycles": 10},
    {"name": "c1", "gpus": [2], "accelerators": [1], "intra_bw_gbps": 256, "intra_latency_cycles": 10},
    {"name": "c2", "gpus": [3], "accelerators": [2], "intra_bw_gbps": 256, "intra_latency_cycles": 10},
    {"name": "c3", "gpus": [4], "accelerators": [3], "intra_bw_gbps": 256, "intra_latency_cycles": 10}
  ],
  "inter_chiplet_bw_gbps": 64,
  "inter_chiplet_latency_cycles": 100
}
```

**Optical intra-chiplet, electrical inter-chiplet:**
```json
{
  "chiplets": [
    {"name": "c0", "gpus": [1, 2], "accelerators": [0], "intra_bw_gbps": 2000, "intra_latency_cycles": 2},
    {"name": "c1", "gpus": [3, 4], "accelerators": [1], "intra_bw_gbps": 2000, "intra_latency_cycles": 2}
  ],
  "inter_chiplet_bw_gbps": 128,
  "inter_chiplet_latency_cycles": 50
}
```

## 5. Implementation Plan

### 5.1 New Package: `chipletbuilder/`

Location: `amd/samples/runner/timingconfig/chipletbuilder/`

Responsible for:
1. Parsing `chiplet_config.json`
2. Creating one intra-chiplet `pcie.Connector` per chiplet
3. Creating the inter-chiplet `pcie.Connector`
4. Wiring GPU and accelerator domain ports into the correct chiplet network
5. Connecting chiplet gateway ports to the inter-chiplet network
6. Connecting Driver and MMU ports to the inter-chiplet root complex

```go
// chipletbuilder/config.go — JSON parsing
type ChipletConfig struct {
    Chiplets              []ChipletDef `json:"chiplets"`
    InterChipletBWGbps    int          `json:"inter_chiplet_bw_gbps"`
    InterChipletLatency   int          `json:"inter_chiplet_latency_cycles"`
}

type ChipletDef struct {
    Name            string `json:"name"`
    GPUs            []int  `json:"gpus"`
    Accelerators    []int  `json:"accelerators"`
    IntraBWGbps     int    `json:"intra_bw_gbps"`
    IntraLatency    int    `json:"intra_latency_cycles"`
}
```

```go
// chipletbuilder/builder.go — network construction
type Builder struct {
    simulation          *simulation.Simulation
    config              *ChipletConfig
    gpuDomains          map[int]*sim.Domain    // gpuID → built GPU domain
    accelDomains        map[int]*sim.Domain    // accelIdx → built accel domain
    driverPorts         []sim.Port             // Driver + MMU ports for root complex
}

func (b *Builder) Build() {
    // 1. Create inter-chiplet PCIe connector
    // 2. Add root complex with driver/MMU ports
    // 3. For each chiplet:
    //    a. Create intra-chiplet PCIe connector
    //    b. Add root complex (becomes the "gateway")
    //    c. Plug in GPU domains' ports
    //    d. Plug in accelerator domains' ToDriver ports
    //    e. Connect gateway to inter-chiplet switch
    // 4. EstablishRoute() on all connectors
}
```

### 5.2 Modifications to `timingconfig/builder.go`

The main `Build()` method gains a conditional path:

```go
func (b Builder) Build() *sim.Domain {
    // ... existing setup (globalStorage, MMU, driver, GPU builder) ...

    if b.chipletConfig != nil {
        // Chiplet path: build GPUs and accels first (without plugging into PCIe),
        // then hand them to chipletbuilder to wire the hierarchical topology.
        b.buildChipletTopology(gpuDriver, mmuComp, gpuBuilder, pmcAddressTable)
    } else {
        // Legacy flat path (unchanged)
        pcieConnector, rootComplexID := b.createConnection(gpuDriver, mmuComp)
        b.createGPUs(rootComplexID, pcieConnector, ...)
        b.createAccelerators(rootComplexID, pcieConnector, gpuDriver)
        pcieConnector.EstablishRoute()
    }

    return b.platform
}
```

### 5.3 New CLI Flag

```go
var chipletConfigFlag = flag.String("chiplet-config", "",
    "Path to chiplet configuration JSON file. "+
        "If not specified, uses flat PCIe topology.")
```

### 5.4 Metrics and Reporting

Add chiplet-aware metrics to `runner/report.go`:
- Traffic per chiplet (intra-chiplet bytes transferred)
- Cross-chiplet traffic (inter-chiplet bytes transferred)
- Per-tier utilization (how saturated each network is)

This enables analysis like: "Moving GPU[3] from chiplet_1 to chiplet_0 reduced
cross-chiplet traffic by 40%."

## 6. What Changes vs What Stays the Same

### Unchanged
- GPU internal architecture (DirectConnection within GPU domain)
- Accelerator internal architecture (AccelUnit ↔ DRAM via internal PCIe)
- `accel_config.json` layer assignment
- Driver APIs (`SelectGPU`, `SelectAccelerator`, `AccelInference`)
- Benchmark code (Minerva, LeNet, VGG16, XOR)
- Memory model (unified `globalStorage`)
- All existing CLI flags

### Changed
- System-level PCIe topology (flat → hierarchical when chiplet config present)
- How GPU/accel domain ports are wired to the network
- Platform builder gains chiplet-aware path
- New CLI flag: `--chiplet-config`
- New package: `chipletbuilder/`
- Report metrics gain chiplet awareness

## 7. Experiments Enabled

### 7.1 Placement Studies

Run the same benchmark with different chiplet configs to measure how device
placement affects performance:

```
Config A: GPU[1]+Accel[0] on chiplet_0, GPU[2]+Accel[1] on chiplet_1
Config B: GPU[1]+GPU[2] on chiplet_0, Accel[0]+Accel[1] on chiplet_1
```

Compare: cross-chiplet traffic, total simulation time, accelerator busy time.

### 7.2 Interconnect Technology Sweep (Extended)

The existing optical vs electrical sweep (7 configs for accel-to-DRAM) can be
extended to study interconnect technology at each tier:

| Experiment | Intra-chiplet | Inter-chiplet | Accel-to-DRAM |
|-----------|--------------|--------------|---------------|
| All electrical | 256 GB/s, 20 cy | 128 GB/s, 50 cy | 256 GB/s, 10 cy |
| Optical intra | 2 TB/s, 2 cy | 128 GB/s, 50 cy | 256 GB/s, 10 cy |
| Optical inter | 256 GB/s, 20 cy | 1 TB/s, 3 cy | 256 GB/s, 10 cy |
| All optical | 2 TB/s, 2 cy | 1 TB/s, 3 cy | 2 TB/s, 2 cy |

### 7.3 Scaling Studies

How does adding more chiplets affect performance? Run with 1, 2, 4 chiplets
with proportionally more GPUs/accels.

## 8. Relationship to Existing Interconnect Experiments

The current interconnect comparison experiments (`experiments/` directory) sweep
the **accel-to-DRAM** interconnect parameters (Tier 3). The chiplet work adds
Tier 1 (inter-chiplet) and Tier 2 (intra-chiplet) as new dimensions to sweep.
The existing experiments remain valid and serve as a baseline — they show
accelerator sensitivity to its memory link, while chiplet experiments will show
sensitivity to device placement and system-level topology.
