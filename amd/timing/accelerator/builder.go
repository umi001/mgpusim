package accelerator

import (
	"github.com/sarchlab/akita/v4/mem/mem"
	"github.com/sarchlab/akita/v4/sim"
)

// Builder constructs an accelerator Comp with configurable parameters.
type Builder struct {
	engine       sim.Engine
	freq         sim.Freq
	localModules mem.AddressToPortMapper
	bufferSize   int

	peArrayRows        int
	peArrayCols        int
	sramSizeBytes      uint64
	maxOutstandingReqs int
}

// MakeBuilder creates a new Builder with default configuration.
// Defaults model a 256x256 systolic array at 1 GHz with 32MB SRAM.
func MakeBuilder() Builder {
	return Builder{
		freq:               1 * sim.GHz,
		bufferSize:         512,
		peArrayRows:        256,
		peArrayCols:        256,
		sramSizeBytes:      32 * 1024 * 1024, // 32 MB
		maxOutstandingReqs: 64,
	}
}

// WithEngine sets the simulation engine.
func (b Builder) WithEngine(engine sim.Engine) Builder {
	b.engine = engine
	return b
}

// WithFreq sets the accelerator clock frequency.
func (b Builder) WithFreq(freq sim.Freq) Builder {
	b.freq = freq
	return b
}

// WithBufferSize sets the port buffer capacity.
func (b Builder) WithBufferSize(n int) Builder {
	b.bufferSize = n
	return b
}

// WithLocalModules sets the address-to-port mapper for memory access.
func (b Builder) WithLocalModules(m mem.AddressToPortMapper) Builder {
	b.localModules = m
	return b
}

// WithPEArraySize sets the systolic array dimensions.
func (b Builder) WithPEArraySize(rows, cols int) Builder {
	b.peArrayRows = rows
	b.peArrayCols = cols
	return b
}

// WithSRAMSize sets the on-chip SRAM buffer size in bytes.
func (b Builder) WithSRAMSize(sizeBytes uint64) Builder {
	b.sramSizeBytes = sizeBytes
	return b
}

// WithMaxOutstandingReqs sets the DMA queue depth — how many memory
// requests can be in-flight simultaneously.
func (b Builder) WithMaxOutstandingReqs(n int) Builder {
	b.maxOutstandingReqs = n
	return b
}

// Build creates an accelerator Comp with the configured parameters.
func (b Builder) Build(name string) *Comp {
	accel := &Comp{}

	accel.TickingComponent = sim.NewTickingComponent(
		name, b.engine, b.freq, accel)

	accel.peArrayRows = b.peArrayRows
	accel.peArrayCols = b.peArrayCols
	accel.sramSizeBytes = b.sramSizeBytes
	accel.maxOutstandingReqs = b.maxOutstandingReqs
	accel.localModules = b.localModules

	accel.ToDriver = sim.NewPort(
		accel, b.bufferSize, b.bufferSize, name+".ToDriver")
	accel.ToMem = sim.NewPort(
		accel, b.bufferSize, b.bufferSize, name+".ToMem")

	accel.AddPort("ToDriver", accel.ToDriver)
	accel.AddPort("ToMem", accel.ToMem)

	return accel
}
