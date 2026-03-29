// Package accelbuilder constructs an inference accelerator domain for
// integration into the MGPUSim timing simulation platform.
package accelbuilder

import (
	"github.com/sarchlab/akita/v4/mem/cache/writearound"
	"github.com/sarchlab/akita/v4/mem/idealmemcontroller"
	"github.com/sarchlab/akita/v4/mem/mem"
	"github.com/sarchlab/akita/v4/noc/networking/pcie"
	"github.com/sarchlab/akita/v4/sim"
	"github.com/sarchlab/akita/v4/sim/directconnection"
	"github.com/sarchlab/akita/v4/simulation"
	"github.com/sarchlab/mgpusim/v4/amd/timing/accelerator"
)

// Builder creates an accelerator sim.Domain that can be wired into the
// platform alongside GPUs.
type Builder struct {
	simulation    *simulation.Simulation
	globalStorage *mem.Storage
	freq          sim.Freq

	peArrayRows    int
	peArrayCols    int
	sramSizeBytes  uint64
	dramSize       uint64

	// Interconnect parameters (between accelerator and its DRAM).
	// These can be changed to model different interconnect technologies
	// (e.g., electrical vs optical).
	interconnectBW      uint64 // bytes per second
	interconnectLatency int    // switch latency in cycles

	// DMA queue depth: how many memory requests can be in-flight.
	maxOutstandingReqs int

	memAddrOffset uint64
}

// MakeBuilder creates a new Builder with default parameters.
func MakeBuilder() Builder {
	return Builder{
		freq:                1 * sim.GHz,
		peArrayRows:         256,
		peArrayCols:         256,
		sramSizeBytes:       32 * 1024 * 1024,  // 32 MB
		dramSize:            4 * mem.GB,
		interconnectBW:      256 * 1000 * 1000 * 1000, // 256 GB/s
		interconnectLatency: 10,
		maxOutstandingReqs:  64,
	}
}

// WithSimulation sets the simulation instance.
func (b Builder) WithSimulation(s *simulation.Simulation) Builder {
	b.simulation = s
	return b
}

// WithGlobalStorage sets the global storage.
func (b Builder) WithGlobalStorage(s *mem.Storage) Builder {
	b.globalStorage = s
	return b
}

// WithFreq sets the accelerator clock frequency.
func (b Builder) WithFreq(freq sim.Freq) Builder {
	b.freq = freq
	return b
}

// WithPEArraySize sets the systolic array dimensions.
func (b Builder) WithPEArraySize(rows, cols int) Builder {
	b.peArrayRows = rows
	b.peArrayCols = cols
	return b
}

// WithSRAMSize sets the on-chip SRAM size in bytes.
func (b Builder) WithSRAMSize(sizeBytes uint64) Builder {
	b.sramSizeBytes = sizeBytes
	return b
}

// WithDRAMSize sets the device memory size.
func (b Builder) WithDRAMSize(size uint64) Builder {
	b.dramSize = size
	return b
}

// WithMemAddrOffset sets the physical address offset for this device.
func (b Builder) WithMemAddrOffset(offset uint64) Builder {
	b.memAddrOffset = offset
	return b
}

// WithInterconnectBW sets the interconnect bandwidth in bytes/second.
// Controls the flit-based bandwidth of the PCIe network between the
// accelerator and its DRAM controller.
func (b Builder) WithInterconnectBW(bw uint64) Builder {
	b.interconnectBW = bw
	return b
}

// WithInterconnectLatency sets the per-switch latency in cycles for the
// interconnect between accelerator and DRAM.
func (b Builder) WithInterconnectLatency(lat int) Builder {
	b.interconnectLatency = lat
	return b
}

// WithMaxOutstandingReqs sets the DMA queue depth.
func (b Builder) WithMaxOutstandingReqs(n int) Builder {
	b.maxOutstandingReqs = n
	return b
}

// Build creates the accelerator domain containing the accelerator
// component, its DRAM controller, and the interconnect between them.
//
//nolint:funlen
func (b Builder) Build(name string) *sim.Domain {
	domain := sim.NewDomain(name)

	accelComp := accelerator.MakeBuilder().
		WithEngine(b.simulation.GetEngine()).
		WithFreq(b.freq).
		WithPEArraySize(b.peArrayRows, b.peArrayCols).
		WithSRAMSize(b.sramSizeBytes).
		WithMaxOutstandingReqs(b.maxOutstandingReqs).
		Build(name + ".AccelUnit")

	b.simulation.RegisterComponent(accelComp)

	domain.AddPort("ToDriver", accelComp.ToDriver)

	// Create DRAM controller (shared globalStorage for unified memory).
	accelDRAM := idealmemcontroller.MakeBuilder().
		WithEngine(b.simulation.GetEngine()).
		WithFreq(b.freq).
		WithLatency(100).
		WithStorage(b.globalStorage).
		Build(name + ".DRAM")
	b.simulation.RegisterComponent(accelDRAM)

	// Build the on-chip SRAM cache. This sits between the AccelUnit and
	// the DRAM interconnect, modeling the accelerator's on-chip buffer.
	// Reads that hit in the cache avoid DRAM traffic entirely (e.g.,
	// weight reuse across epochs). Writes use writearound policy and
	// bypass the cache, going directly to DRAM — correct for output
	// tensors that are never reread from the same op.
	sramCache := writearound.MakeBuilder().
		WithEngine(b.simulation.GetEngine()).
		WithFreq(b.freq).
		WithTotalByteSize(b.sramSizeBytes).
		WithLog2BlockSize(6). // 64-byte blocks, matches DMA granularity
		WithWayAssociativity(4).
		WithNumMSHREntry(64). // match maxOutstandingReqs
		WithNumReqsPerCycle(16).
		WithAddressToPortMapper(&mem.SinglePortMapper{
			Port: accelDRAM.GetPortByName("Top").AsRemote(),
		}).
		Build(name + ".SRAM")
	b.simulation.RegisterComponent(sramCache)

	// Wire AccelUnit → SRAM cache via DirectConnection (on-chip, zero
	// latency). The AccelUnit issues 64-byte ReadReqs; the cache checks
	// its tags and either returns a hit or forwards a miss to DRAM.
	internalConn := directconnection.MakeBuilder().
		WithEngine(b.simulation.GetEngine()).
		WithFreq(b.freq).
		Build(name + ".InternalConn")
	b.simulation.RegisterComponent(internalConn)
	internalConn.PlugIn(accelComp.ToMem)
	internalConn.PlugIn(sramCache.GetPortByName("Top"))

	// Create a parameterizable interconnect between the SRAM cache and
	// DRAM. This is the packaging link being studied — sweep
	// interconnectBW and interconnectLatency to compare technologies.
	memPCIe := pcie.NewConnector().
		WithEngine(b.simulation.GetEngine()).
		WithBandwidth(b.interconnectBW).
		WithSwitchLatency(b.interconnectLatency)

	memPCIe.CreateNetwork(name + ".MemNet")

	rootID := memPCIe.AddRootComplex(
		[]sim.Port{accelDRAM.GetPortByName("Top")},
	)

	switchID := memPCIe.AddSwitch(rootID)

	// Cache Bottom port now drives the PCIe interconnect (was AccelUnit.ToMem).
	memPCIe.PlugInDevice(switchID,
		[]sim.Port{sramCache.GetPortByName("Bottom")},
	)

	memPCIe.EstablishRoute()

	// Tell the accelerator where to route memory requests — now the
	// SRAM cache Top port, not DRAM directly.
	localModules := &mem.SinglePortMapper{
		Port: sramCache.GetPortByName("Top").AsRemote(),
	}
	accelComp.SetLocalModuleFinder(localModules)

	// ToMem is wired internally to the DRAM controller via PCIe;
	// do NOT expose it as a domain port, otherwise the system-level
	// PCIe connector will try to connect it a second time.

	return domain
}
