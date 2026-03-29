# Phase 2.5: Full-Traffic Memory Simulation for Interconnect Studies

## Context

The user's research goal is to compare **optical vs electrical interconnects** between the accelerator and its DRAM. The current Phase 2 implementation uses an analytical bandwidth model (`mem_cycles = total_bytes / memBandwidthBW`) with 1 probe request per tensor. This means:

- No real traffic flows through the interconnect — changing interconnect parameters has zero effect
- The `directconnection` (zero latency) between accelerator and DRAM doesn't model any interconnect behavior
- The `idealmemcontroller` processes all requests in parallel with no bandwidth limiting
- **Result**: Impossible to study interconnect effects because there's nothing to study

To enable optical vs electrical interconnect comparison, we need:
1. **Full traffic volume**: Every 64 bytes of data generates a real `mem.ReadReq`/`mem.WriteReq`
2. **Parameterizable interconnect**: Replace `directconnection` with a bandwidth/latency-limited connection
3. **Emergent timing**: Memory access time comes from the simulation, not a formula

## Design

### Architecture Change

```
BEFORE (analytical):
  AccelComp.ToMem → directconnection (0 latency) → idealmemcontroller (100 cycles)
  Bandwidth: hardcoded formula inside comp.go
  Traffic: 1 probe per tensor (3 reads + 1 write per op)

AFTER (full traffic):
  AccelComp.ToMem → PCIe network (configurable BW + latency) → idealmemcontroller (100 cycles)
  Bandwidth: emerges from PCIe flit serialization
  Traffic: ceil(tensorBytes/64) requests per tensor (thousands per op)
```

### Why PCIe Connector

The Akita PCIe connector (`github.com/sarchlab/akita/v4/noc/networking/pcie`) provides:
- **Flit-based bandwidth**: Messages serialized into flits, naturally limiting throughput
- **Configurable bandwidth**: `WithBandwidth(bytesPerSecond)` — directly parameterizable
- **Configurable latency**: `WithSwitchLatency(cycles)` — per-switch hop delay
- **Already used in MGPUSim**: Proven pattern for GPU-to-GPU communication
- **No code to write**: Reuse existing framework component

For optical vs electrical, change two parameters:
- Electrical: `WithBandwidth(32e9)`, `WithSwitchLatency(140)` (PCIe 4.0 x16)
- Optical: `WithBandwidth(200e9)`, `WithSwitchLatency(10)` (photonic link)

### Sequential Phases (READ → COMPUTE → WRITE)

Keep the existing phase model. Memory time now emerges from simulation:
- **READ**: Issue all read requests progressively (backpressure-limited), wait for all responses
- **COMPUTE**: Count down analytical compute cycles (PE array model unchanged)
- **WRITE**: Issue all write requests progressively, wait for all responses

Total time = read_time (emergent) + compute_time (analytical) + write_time (emergent)

### Alignment with Approach 2

This plan is strictly internal to the accelerator's memory path. Nothing changes in:
- Driver APIs (`SelectAccelerator`, `AccelInference`)
- Protocol messages (`AccelInferenceReq/Rsp`)
- Benchmark operator interface (`tensor.Operator`)
- Per-layer device assignment (`accel_config.json`)
- Shared memory model (`globalStorage`)

It's an evolution of Phase 2's DMA model: from analytical bandwidth to simulation-based, enabling interconnect technology comparison studies.

## Implementation Plan

### Step 1: `amd/timing/accelerator/comp.go` — Full traffic generation

**Core change**: Replace 1-probe-per-tensor with full-volume 64-byte requests.

#### 1a. New `tensorTransfer` struct and updated `transaction`
```go
type tensorTransfer struct {
    baseAddr          uint64
    totalBytes        uint64
    bytesSent         uint64  // bytes for which requests have been issued
    responsesExpected int
    responsesReceived int
}

type transaction struct {
    req           *protocol.AccelInferenceReq
    phase         int
    computeCycles int  // analytical compute time (PE array)

    readTensors    []tensorTransfer
    currentReadIdx int
    writeTransfer  tensorTransfer

    outstandingReqs int  // currently in-flight (for backpressure)
    maxOutstanding  int

    totalReadBytes  uint64
    totalWriteBytes uint64
}
```

#### 1b. Remove `memBandwidthBW` field from `Comp`
- Remove `memBandwidthBW float64` from struct
- Remove all roofline calculation code (`memCycles`, `rooflineCycles`)
- `totalCycles` metric now tracks compute cycles only

#### 1c. Add `maxOutstandingReqs` field to `Comp`
Default: 64 (controls DMA queue depth — how many requests can be in-flight simultaneously)

#### 1d. New `estimatePerTensorReadBytes()` function
Returns per-tensor `{addr, bytes}` breakdown instead of a single total. For GEMM:
- InputAddr: M*K*4 bytes
- WeightsAddr: K*N*4 bytes
- BiasAddr: M*N*4 bytes (if present)

Uses existing `estimateMemoryTraffic()` logic split by tensor. The existing function can be kept for metrics; the new one drives the DMA requests.

#### 1e. Rewrite `startOperation()`
1. Call `estimateComputeCycles(req)` → store in `transaction.computeCycles`
2. Call `estimatePerTensorReadBytes(req)` → build `readTensors` list
3. Calculate `writeTransfer` from OutputAddr + writeBytes
4. Set `maxOutstanding` from `c.maxOutstandingReqs`
5. Set `phase = phaseReading`, start issuing reads

#### 1f. Rewrite `processReadPhase()`
Progressive request issuing with backpressure:
```go
for txn.outstandingReqs < txn.maxOutstanding {
    // find next byte range to issue
    // call issueReadReq(addr)
    // increment bytesSent, responsesExpected, outstandingReqs
    // break if port send fails (backpressure)
}
// check if all tensors fully responded → transition to phaseCompute
```

#### 1g. Rewrite `processWritePhase()`
Same progressive pattern for write requests.

#### 1h. Update `processMemRsp()`
- Loop to drain all pending responses per tick (not just one)
- Decrement `outstandingReqs` per response
- Increment appropriate tensor's `responsesReceived`

#### 1i. Handle partial request sizes
If `totalBytes % 64 != 0`, last request uses `min(64, remaining)`.

#### 1j. New metric: `totalMemReqs` counting actual memory transactions generated

### Step 2: `amd/timing/accelerator/builder.go` — Builder updates

- Remove `memBandwidthBW` field and `WithMemBandwidth()` method
- Add `maxOutstandingReqs int` field with default 64
- Add `WithMaxOutstandingReqs(n int)` method
- Wire `maxOutstandingReqs` to `Comp` in `Build()`
- Increase default `bufferSize` from 128 to 512 (handle more in-flight messages)

### Step 3: `amd/samples/runner/timingconfig/accelbuilder/builder.go` — PCIe interconnect

**Replace `directconnection` with PCIe connector.**

#### 3a. Add interconnect parameters to Builder
```go
type Builder struct {
    // ... existing ...
    interconnectBW      uint64  // bytes/sec (default 256 GB/s)
    interconnectLatency int     // switch cycles (default 10)
    maxOutstandingReqs  int     // DMA queue depth (default 64)
}
```

#### 3b. Add builder methods
- `WithInterconnectBW(bw uint64)`
- `WithInterconnectLatency(lat int)`
- `WithMaxOutstandingReqs(n int)`

#### 3c. Replace connection in `Build()`

Remove:
```go
memConn := directconnection.MakeBuilder()...Build(name + ".MemConn")
memConn.PlugIn(accelComp.ToMem)
memConn.PlugIn(accelDRAM.GetPortByName("Top"))
```

Replace with:
```go
memPCIe := pcie.NewConnector().
    WithEngine(b.simulation.GetEngine()).
    WithBandwidth(b.interconnectBW).
    WithSwitchLatency(b.interconnectLatency)
memPCIe.CreateNetwork(name + ".MemNet")
rootID := memPCIe.AddRootComplex([]sim.Port{accelDRAM.GetPortByName("Top")})
switchID := memPCIe.AddSwitch(rootID)
memPCIe.PlugInDevice(switchID, []sim.Port{accelComp.ToMem})
memPCIe.EstablishRoute()
```

The `SinglePortMapper` still points to `accelDRAM.Top` — PCIe routing handles intermediate hops transparently.

**Note**: This internal PCIe network is separate from the system-level PCIe. `ToMem` is not a domain port, so no conflict with the system `pcieConnector.PlugInDevice(switchID, accel.Ports())` at line 340 of `timingconfig/builder.go`.

#### 3d. Update imports
Add `pcie`, remove `directconnection`.

#### 3e. Pass `maxOutstandingReqs` to accelerator builder

### Step 4: `amd/samples/runner/timingconfig/builder.go` — Thread parameters

- Add fields: `accelInterconnectBW`, `accelInterconnectLatency`, `accelMaxOutstandingReqs`
- Add builder methods: `WithAccelInterconnectBW()`, `WithAccelInterconnectLatency()`, `WithAccelMaxOutstandingReqs()`
- Pass to `accelbuilder.MakeBuilder()` in `createAccelerator()`

### Step 5: `amd/samples/runner/flag.go` + `runner.go` — CLI flags

#### flag.go:
```go
var accelInterconnectBWFlag = flag.Uint64("accel-interconnect-bw", 256000000000,
    "Accelerator-to-DRAM interconnect bandwidth in bytes/sec (default 256 GB/s)")
var accelInterconnectLatencyFlag = flag.Int("accel-interconnect-latency", 10,
    "Accelerator-to-DRAM interconnect switch latency in cycles (default 10)")
var accelMaxOutstandingFlag = flag.Int("accel-max-outstanding", 64,
    "Maximum outstanding DMA memory requests (default 64)")
```

#### runner.go:
- Add fields to `Runner` struct
- Wire in `parseSimulationFlags()`
- Pass to timing builder

### Step 6: `amd/samples/runner/report.go` — New metric

Add `total_mem_reqs` metric to verify full traffic is flowing:
```go
r.dataRecorder.InsertData(tableName, metric{
    Location: t.comp.Name(),
    What:     "total_mem_reqs",
    Value:    float64(accelComp.TotalMemReqs()),
    Unit:     "count",
})
```

### Step 7: `docs/approach2_design.md` — Document the design

Add section covering:
- Full-traffic rationale (why analytical model is insufficient for interconnect studies)
- PCIe connector as parameterizable interconnect
- How to configure for optical vs electrical comparison
- Sequential phases (READ → COMPUTE → WRITE)
- What emerges from simulation vs what's analytical

## Files to Modify

| File | Changes |
|------|---------|
| `amd/timing/accelerator/comp.go` | Full traffic DMA, remove analytical BW, progressive request issuing |
| `amd/timing/accelerator/builder.go` | Remove memBandwidthBW, add maxOutstandingReqs |
| `amd/samples/runner/timingconfig/accelbuilder/builder.go` | Replace directconnection with PCIe, add interconnect params |
| `amd/samples/runner/timingconfig/builder.go` | Thread interconnect params to accelbuilder |
| `amd/samples/runner/flag.go` | New CLI flags for interconnect BW/latency/outstanding |
| `amd/samples/runner/runner.go` | Wire flags to builder |
| `amd/samples/runner/report.go` | Add total_mem_reqs metric |
| `docs/approach2_design.md` | Document full-traffic design + interconnect study setup |

## Verification

1. `go build ./...` — must compile
2. `~/go/bin/golangci-lint run ./amd/... --timeout=10m` — no new lint issues
3. Run Minerva with accelerator:
   ```
   cd amd/samples/minerva && go build && \
     ./minerva -timing -gpus=1 -num-accel=1 -report-all -epoch=1
   ```
   - Must complete without deadlock
   - `total_mem_reqs` should be >> 18 (was 18 ops * 1-3 probes; now should be thousands)
   - `total_mem_reqs` ~ `ceil(total_read_bytes/64) + ceil(total_write_bytes/64)`
4. Bandwidth sensitivity test:
   ```
   ./minerva -timing -gpus=1 -num-accel=1 -accel-interconnect-bw=32000000000 -report-all
   ./minerva -timing -gpus=1 -num-accel=1 -accel-interconnect-bw=256000000000 -report-all
   ```
   Higher bandwidth should result in less `busy_time` for memory-bound ops.

## Potential Issues

1. **Simulation speed**: A GEMM reading 100KB generates ~1600 requests. Each goes through PCIe flit serialization. This will slow simulation vs the analytical model. Acceptable for research accuracy.
2. **Port buffer sizing**: PCIe endpoints have default buffer size ~16. The `maxOutstandingReqs` (64) limits the accelerator's send rate. Backpressure from the PCIe endpoint naturally rate-limits traffic — this is desirable.
3. **idealmemcontroller has no bandwidth limit**: All requests complete in 100 cycles regardless of load. The bandwidth limitation comes entirely from the PCIe interconnect. For the interconnect study this is correct — we're studying the interconnect, not the DRAM controller.
