// Package accelbuilder constructs an inference accelerator domain for
// integration into the MGPUSim timing simulation platform.
package accelbuilder

import (
	"github.com/sarchlab/akita/v4/mem/idealmemcontroller"
	"github.com/sarchlab/akita/v4/mem/mem"
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
	memBandwidthBW float64
	dramSize       uint64

	memAddrOffset uint64
}

// MakeBuilder creates a new Builder with default parameters.
func MakeBuilder() Builder {
	return Builder{
		freq:           1 * sim.GHz,
		peArrayRows:    256,
		peArrayCols:    256,
		sramSizeBytes:  32 * 1024 * 1024, // 32 MB
		memBandwidthBW: 256,              // 256 bytes/cycle
		dramSize:       4 * mem.GB,
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

// WithMemBandwidth sets the memory bandwidth in bytes per cycle.
func (b Builder) WithMemBandwidth(bw float64) Builder {
	b.memBandwidthBW = bw
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

// Build creates the accelerator domain containing the accelerator component.
func (b Builder) Build(name string) *sim.Domain {
	domain := sim.NewDomain(name)

	accelComp := accelerator.MakeBuilder().
		WithEngine(b.simulation.GetEngine()).
		WithFreq(b.freq).
		WithPEArraySize(b.peArrayRows, b.peArrayCols).
		WithSRAMSize(b.sramSizeBytes).
		WithMemBandwidth(b.memBandwidthBW).
		Build(name + ".AccelUnit")

	b.simulation.RegisterComponent(accelComp)

	domain.AddPort("ToDriver", accelComp.ToDriver)

	// Wire the accelerator's ToMem port to an ideal DRAM controller.
	// This uses the same idealmemcontroller (100-cycle latency) that GPUs
	// use, backed by the shared globalStorage for unified memory access.
	accelDRAM := idealmemcontroller.MakeBuilder().
		WithEngine(b.simulation.GetEngine()).
		WithFreq(b.freq).
		WithLatency(100).
		WithStorage(b.globalStorage).
		Build(name + ".DRAM")
	b.simulation.RegisterComponent(accelDRAM)

	memConn := directconnection.MakeBuilder().
		WithEngine(b.simulation.GetEngine()).
		WithFreq(b.freq).
		Build(name + ".MemConn")
	b.simulation.RegisterComponent(memConn)

	memConn.PlugIn(accelComp.ToMem)
	memConn.PlugIn(accelDRAM.GetPortByName("Top"))

	// Tell the accelerator where to route memory requests.
	localModules := &mem.SinglePortMapper{
		Port: accelDRAM.GetPortByName("Top").AsRemote(),
	}
	accelComp.SetLocalModuleFinder(localModules)

	// ToMem is now wired internally to the DRAM controller; do NOT expose
	// it as a domain port, otherwise the PCIe connector will try to
	// connect it a second time.

	return domain
}
