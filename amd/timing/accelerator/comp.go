// Package accelerator provides a cycle-level timing model for a
// fixed-function inference accelerator (e.g., systolic-array / TPU-like).
// It plugs into the Akita simulation engine alongside GPU compute units.
//
// The timing model uses a roofline approach:
//
//	latency = max(compute_cycles, memory_cycles) + DRAM_latency
//
// where compute_cycles comes from analytical per-op estimates and
// memory_cycles = total_bytes / memBandwidthBW. Memory access is modeled
// via DMA-style bulk transfers: one mem.ReadReq per input tensor for DRAM
// latency, with bandwidth modeled analytically.
package accelerator

import (
	"log"
	"math"
	"reflect"

	"github.com/sarchlab/akita/v4/mem/mem"
	"github.com/sarchlab/akita/v4/sim"
	"github.com/sarchlab/akita/v4/tracing"
	"github.com/sarchlab/mgpusim/v4/amd/protocol"
)

// Execution phases for the accelerator state machine.
// Each operation goes through: READ → COMPUTE → WRITE → done.
const (
	phaseReading = iota // Waiting for DRAM read responses
	phaseCompute        // Counting down compute/roofline cycles
	phaseWriting        // Waiting for DRAM write responses
)

// transaction tracks one in-flight accelerator operation.
type transaction struct {
	req            *protocol.AccelInferenceReq
	phase          int // phaseReading → phaseCompute → phaseWriting
	remainCycles   int // roofline cycles: max(compute, memory bandwidth)
	memReqsPending int // outstanding DRAM read/write requests

	totalReadBytes  uint64
	totalWriteBytes uint64
}

// Comp is the top-level accelerator component.
// It receives AccelInferenceReq from the driver, models execution latency
// using a roofline model (max of compute and memory bandwidth), issues
// actual memory transactions through ToMem for DRAM latency modeling,
// and sends AccelInferenceRsp when done.
type Comp struct {
	*sim.TickingComponent

	// ToDriver receives AccelInferenceReq and sends AccelInferenceRsp.
	ToDriver sim.Port

	// ToMem issues memory read/write requests to the DRAM controller.
	ToMem sim.Port

	// Hardware configuration — set via Builder.
	peArrayRows    int     // systolic array rows (e.g., 256)
	peArrayCols    int     // systolic array cols (e.g., 256)
	sramSizeBytes  uint64  // on-chip SRAM buffer size
	memBandwidthBW float64 // bytes per cycle to off-chip memory

	// Internal state
	pendingReqs  []*protocol.AccelInferenceReq
	currentTxn   *transaction
	localModules mem.AddressToPortMapper

	// Metrics
	totalOps        int
	totalCycles     int
	totalReadBytes  uint64
	totalWriteBytes uint64
}

// SetLocalModuleFinder sets the address-to-port mapper for memory access.
func (c *Comp) SetLocalModuleFinder(lmf mem.AddressToPortMapper) {
	c.localModules = lmf
}

// Tick is called every cycle by the Akita engine.
func (c *Comp) Tick() bool {
	madeProgress := false

	madeProgress = c.processMemRsp() || madeProgress
	madeProgress = c.processCurrentOp() || madeProgress
	madeProgress = c.acceptNewReq() || madeProgress

	return madeProgress
}

// acceptNewReq checks if a new AccelInferenceReq has arrived from the driver.
func (c *Comp) acceptNewReq() bool {
	if c.currentTxn != nil {
		return false
	}

	msg := c.ToDriver.PeekIncoming()
	if msg == nil {
		return false
	}

	switch req := msg.(type) {
	case *protocol.AccelInferenceReq:
		c.ToDriver.RetrieveIncoming()
		c.startOperation(req)
		return true
	default:
		log.Panicf("accelerator: unexpected message type %s",
			reflect.TypeOf(msg))
		return false
	}
}

// startOperation initializes a new transaction with roofline timing and
// issues DMA read requests for input tensors.
//
//nolint:funlen
func (c *Comp) startOperation(req *protocol.AccelInferenceReq) {
	computeCycles := c.estimateComputeCycles(req)
	readBytes, writeBytes := c.estimateMemoryTraffic(req)

	totalBytes := readBytes + writeBytes
	memCycles := int(math.Ceil(
		float64(totalBytes) / c.memBandwidthBW))

	// Roofline: total time is dominated by the slower of compute or memory.
	rooflineCycles := computeCycles
	if memCycles > rooflineCycles {
		rooflineCycles = memCycles
	}

	if rooflineCycles < 1 {
		rooflineCycles = 1
	}

	c.totalOps++
	c.totalCycles += rooflineCycles

	c.currentTxn = &transaction{
		req:             req,
		remainCycles:    rooflineCycles,
		totalReadBytes:  readBytes,
		totalWriteBytes: writeBytes,
	}

	tracing.TraceReqReceive(req, c)

	// Accumulate memory traffic metrics.
	c.totalReadBytes += readBytes
	c.totalWriteBytes += writeBytes

	// Issue DMA read requests for input tensors through the DRAM hierarchy.
	// We issue one 64-byte "probe" ReadReq per non-zero input address.
	// The actual bandwidth is modeled analytically in rooflineCycles;
	// these requests model the DRAM round-trip latency (100 cycles).
	readsIssued := 0
	if req.InputAddr != 0 && c.localModules != nil {
		if c.issueReadReq(req.InputAddr) {
			readsIssued++
		}
	}

	if req.WeightsAddr != 0 && c.localModules != nil {
		if c.issueReadReq(req.WeightsAddr) {
			readsIssued++
		}
	}

	if req.BiasAddr != 0 && c.localModules != nil {
		if c.issueReadReq(req.BiasAddr) {
			readsIssued++
		}
	}

	if readsIssued > 0 {
		c.currentTxn.phase = phaseReading
	} else {
		c.currentTxn.phase = phaseCompute
	}
}

// issueReadReq sends a 64-byte read request to the DRAM controller
// for DMA latency modeling. Returns true if the request was sent.
func (c *Comp) issueReadReq(addr uint64) bool {
	readReq := mem.ReadReqBuilder{}.
		WithSrc(c.ToMem.AsRemote()).
		WithDst(c.localModules.Find(addr)).
		WithAddress(addr).
		WithByteSize(64).
		Build()

	err := c.ToMem.Send(readReq)
	if err != nil {
		return false
	}

	c.currentTxn.memReqsPending++

	return true
}

// issueWriteReq sends a 64-byte write request to the DRAM controller
// for DMA latency modeling. Returns true if the request was sent.
func (c *Comp) issueWriteReq(addr uint64) bool {
	writeReq := mem.WriteReqBuilder{}.
		WithSrc(c.ToMem.AsRemote()).
		WithDst(c.localModules.Find(addr)).
		WithAddress(addr).
		WithData(make([]byte, 64)).
		Build()

	err := c.ToMem.Send(writeReq)
	if err != nil {
		return false
	}

	c.currentTxn.memReqsPending++

	return true
}

// processCurrentOp advances the operation through its execution phases:
// READ → COMPUTE → WRITE → complete.
func (c *Comp) processCurrentOp() bool {
	if c.currentTxn == nil {
		return false
	}

	switch c.currentTxn.phase {
	case phaseReading:
		return c.processReadPhase()
	case phaseCompute:
		return c.processComputePhase()
	case phaseWriting:
		return c.processWritePhase()
	default:
		return false
	}
}

// processReadPhase waits for all DRAM read responses before starting compute.
func (c *Comp) processReadPhase() bool {
	if c.currentTxn.memReqsPending > 0 {
		return false
	}

	// All reads complete — transition to compute phase.
	c.currentTxn.phase = phaseCompute

	return true
}

// processComputePhase counts down the roofline cycles.
func (c *Comp) processComputePhase() bool {
	c.currentTxn.remainCycles--

	if c.currentTxn.remainCycles <= 0 {
		// Compute done — issue write for output tensor.
		return c.transitionToWritePhase()
	}

	return true
}

// transitionToWritePhase issues a DMA write for the output tensor
// and transitions to the write phase.
func (c *Comp) transitionToWritePhase() bool {
	req := c.currentTxn.req

	if req.OutputAddr != 0 && c.localModules != nil {
		if c.issueWriteReq(req.OutputAddr) {
			c.currentTxn.phase = phaseWriting
			return true
		}

		// Port full — retry next cycle. Keep remainCycles at 0 so
		// we'll retry the transition on next tick.
		c.currentTxn.remainCycles = 0

		return false
	}

	// No output to write — complete immediately.
	return c.completeOperation()
}

// processWritePhase waits for DRAM write response before completing.
func (c *Comp) processWritePhase() bool {
	if c.currentTxn.memReqsPending > 0 {
		return false
	}

	return c.completeOperation()
}

// completeOperation sends the response back to the driver and clears
// the current transaction.
func (c *Comp) completeOperation() bool {
	req := c.currentTxn.req

	rsp := protocol.NewAccelInferenceRsp(
		c.ToDriver.AsRemote(),
		req.Src,
		req.Meta().ID,
	)

	err := c.ToDriver.Send(rsp)
	if err != nil {
		return false
	}

	tracing.TraceReqComplete(req, c)

	c.currentTxn = nil

	return true
}

// processMemRsp handles memory read/write responses from the DRAM controller.
func (c *Comp) processMemRsp() bool {
	msg := c.ToMem.PeekIncoming()
	if msg == nil {
		return false
	}

	c.ToMem.RetrieveIncoming()

	if c.currentTxn != nil {
		c.currentTxn.memReqsPending--
	}

	return true
}

// --- Roofline timing model: compute cycle estimates ---

// estimateComputeCycles returns the compute-bound cycle count for an
// operation. The roofline model takes max(compute, memory) — this function
// provides the compute component.
func (c *Comp) estimateComputeCycles(
	req *protocol.AccelInferenceReq,
) int {
	switch req.OpType {
	case protocol.AccelOpGEMM, protocol.AccelOpMatMul:
		return c.estimateGEMMComputeCycles(req)
	case protocol.AccelOpConv2D:
		return c.estimateConv2DComputeCycles(req)
	case protocol.AccelOpMaxPool, protocol.AccelOpAvgPool:
		return c.estimatePoolingComputeCycles(req)
	case protocol.AccelOpReLU, protocol.AccelOpElementWise:
		return c.estimateElementWiseComputeCycles(req)
	case protocol.AccelOpScaleAdd:
		return c.estimateScaleAddComputeCycles(req)
	case protocol.AccelOpSoftmax:
		return c.estimateSoftmaxComputeCycles(req)
	case protocol.AccelOpReduction:
		return c.estimateReductionComputeCycles(req)
	case protocol.AccelOpAdam:
		return c.estimateAdamComputeCycles(req)
	case protocol.AccelOpRMSProp:
		return c.estimateRMSPropComputeCycles(req)
	case protocol.AccelOpCrossEntropy,
		protocol.AccelOpCrossEntropyDeriv,
		protocol.AccelOpSoftmaxCrossEntropyDeriv:
		return c.estimateCrossEntropyComputeCycles(req)
	default:
		return 1000 // fallback
	}
}

// inputElements computes the total elements from InputSize dimensions.
func inputElements(req *protocol.AccelInferenceReq) int {
	return int(req.InputSize[0]) *
		int(req.InputSize[1]) *
		int(req.InputSize[2]) *
		int(req.InputSize[3])
}

// outputElements computes the total elements from OutputSize dimensions.
func outputElements(req *protocol.AccelInferenceReq) int {
	return int(req.OutputSize[0]) *
		int(req.OutputSize[1]) *
		int(req.OutputSize[2]) *
		int(req.OutputSize[3])
}

func (c *Comp) estimateGEMMComputeCycles(
	req *protocol.AccelInferenceReq,
) int {
	m := int(req.Params.M)
	n := int(req.Params.N)
	k := int(req.Params.K)

	// total_MACs / (array_rows * array_cols)
	totalMACs := m * n * k
	cycles := totalMACs / (c.peArrayRows * c.peArrayCols)

	if cycles < 1 {
		cycles = 1
	}

	return cycles
}

func (c *Comp) estimateConv2DComputeCycles(
	req *protocol.AccelInferenceReq,
) int {
	// Conv2D lowered to GEMM: M=OutChannels, N=batch*outH*outW,
	// K=InChannels*kH*kW.
	outChannels := int(req.Params.OutChannels)
	inChannels := int(req.Params.InChannels)
	kH := int(req.Params.KernelSize[0])
	kW := int(req.Params.KernelSize[1])
	batch := int(req.InputSize[0])
	outH := int(req.OutputSize[2])
	outW := int(req.OutputSize[3])

	m := outChannels
	n := batch * outH * outW
	k := inChannels * kH * kW

	totalMACs := m * n * k
	cycles := totalMACs / (c.peArrayRows * c.peArrayCols)

	if cycles < 1 {
		cycles = 1
	}

	return cycles
}

func (c *Comp) estimatePoolingComputeCycles(
	req *protocol.AccelInferenceReq,
) int {
	// Pooling is memory-bound with trivial compute.
	total := inputElements(req)
	cycles := total / c.peArrayCols

	if cycles < 1 {
		cycles = 1
	}

	return cycles
}

func (c *Comp) estimateElementWiseComputeCycles(
	req *protocol.AccelInferenceReq,
) int {
	total := inputElements(req)
	cycles := total / (c.peArrayRows * c.peArrayCols)

	if cycles < 1 {
		cycles = 1
	}

	return cycles
}

// Softmax: 3 passes (exp, sum, div). Compute per element is trivial;
// each pass pipelines through vector lanes.
func (c *Comp) estimateSoftmaxComputeCycles(
	req *protocol.AccelInferenceReq,
) int {
	total := inputElements(req)
	cycles := 3 * total / c.peArrayCols

	if cycles < 1 {
		cycles = 1
	}

	return cycles
}

// ScaleAdd: alpha*A + beta*B. One multiply-add per element, trivial.
func (c *Comp) estimateScaleAddComputeCycles(
	req *protocol.AccelInferenceReq,
) int {
	total := inputElements(req)
	cycles := total / c.peArrayCols

	if cycles < 1 {
		cycles = 1
	}

	return cycles
}

// Reduction: single pass read + log-tree reduction (dominated by read).
func (c *Comp) estimateReductionComputeCycles(
	req *protocol.AccelInferenceReq,
) int {
	total := inputElements(req)
	cycles := total / c.peArrayCols

	if cycles < 1 {
		cycles = 1
	}

	return cycles
}

// Adam: trivial per-element arithmetic (a few multiply-adds).
func (c *Comp) estimateAdamComputeCycles(
	req *protocol.AccelInferenceReq,
) int {
	total := inputElements(req)
	cycles := total / c.peArrayCols

	if cycles < 1 {
		cycles = 1
	}

	return cycles
}

// RMSProp: trivial per-element arithmetic.
func (c *Comp) estimateRMSPropComputeCycles(
	req *protocol.AccelInferenceReq,
) int {
	total := inputElements(req)
	cycles := total / c.peArrayCols

	if cycles < 1 {
		cycles = 1
	}

	return cycles
}

// CrossEntropy: trivial per-element arithmetic.
func (c *Comp) estimateCrossEntropyComputeCycles(
	req *protocol.AccelInferenceReq,
) int {
	total := inputElements(req)
	cycles := total / c.peArrayCols

	if cycles < 1 {
		cycles = 1
	}

	return cycles
}

// --- Roofline timing model: memory traffic estimates ---

// estimateMemoryTraffic returns (readBytes, writeBytes) for the operation.
// These are used for the bandwidth component of the roofline model:
//
//	mem_cycles = (readBytes + writeBytes) / memBandwidthBW
//
//nolint:gocognit,funlen
func (c *Comp) estimateMemoryTraffic(
	req *protocol.AccelInferenceReq,
) (readBytes, writeBytes uint64) {
	const bytesPerFloat32 = 4

	nIn := uint64(inputElements(req))
	nOut := uint64(outputElements(req))

	switch req.OpType {
	case protocol.AccelOpGEMM, protocol.AccelOpMatMul:
		m := uint64(req.Params.M)
		n := uint64(req.Params.N)
		k := uint64(req.Params.K)
		// Read A[M×K] + B[K×N], write C[M×N]
		readBytes = (m*k + k*n) * bytesPerFloat32
		writeBytes = m * n * bytesPerFloat32

	case protocol.AccelOpConv2D:
		inC := uint64(req.Params.InChannels)
		outC := uint64(req.Params.OutChannels)
		kH := uint64(req.Params.KernelSize[0])
		kW := uint64(req.Params.KernelSize[1])
		// Read input + filter, write output
		readBytes = (nIn + outC*inC*kH*kW) * bytesPerFloat32
		writeBytes = nOut * bytesPerFloat32

	case protocol.AccelOpReLU, protocol.AccelOpElementWise:
		// Read 1 input, write 1 output (same size)
		readBytes = nIn * bytesPerFloat32
		writeBytes = nIn * bytesPerFloat32

	case protocol.AccelOpScaleAdd:
		// Read 2 inputs, write 1 output
		readBytes = 2 * nIn * bytesPerFloat32
		writeBytes = nIn * bytesPerFloat32

	case protocol.AccelOpSoftmax:
		// Read input, write output (same size)
		readBytes = nIn * bytesPerFloat32
		writeBytes = nIn * bytesPerFloat32

	case protocol.AccelOpReduction:
		// Read all input, write scalar output
		readBytes = nIn * bytesPerFloat32
		writeBytes = bytesPerFloat32

	case protocol.AccelOpAdam:
		// Read params + gradients + vHistory + sHistory (4 tensors)
		// Write params + vHistory + sHistory (3 tensors)
		readBytes = 4 * nIn * bytesPerFloat32
		writeBytes = 3 * nIn * bytesPerFloat32

	case protocol.AccelOpRMSProp:
		// Read params + gradients + sHistory (3 tensors)
		// Write params + sHistory (2 tensors)
		readBytes = 3 * nIn * bytesPerFloat32
		writeBytes = 2 * nIn * bytesPerFloat32

	case protocol.AccelOpCrossEntropy:
		// Read predictions + labels, write scalar loss
		readBytes = 2 * nIn * bytesPerFloat32
		writeBytes = bytesPerFloat32

	case protocol.AccelOpCrossEntropyDeriv,
		protocol.AccelOpSoftmaxCrossEntropyDeriv:
		// Read predictions + labels, write gradient tensor
		readBytes = 2 * nIn * bytesPerFloat32
		writeBytes = nIn * bytesPerFloat32

	case protocol.AccelOpMaxPool, protocol.AccelOpAvgPool:
		// Read input, write (smaller) output
		readBytes = nIn * bytesPerFloat32
		writeBytes = nOut * bytesPerFloat32

	default:
		// Conservative fallback: assume 1 read + 1 write
		readBytes = nIn * bytesPerFloat32
		writeBytes = nIn * bytesPerFloat32
	}

	return readBytes, writeBytes
}

// --- Metric accessors ---

// TotalOps returns the number of operations processed.
func (c *Comp) TotalOps() int {
	return c.totalOps
}

// TotalCycles returns the total roofline cycles (max of compute, memory).
func (c *Comp) TotalCycles() int {
	return c.totalCycles
}

// TotalReadBytes returns the total bytes read from DRAM across all ops.
func (c *Comp) TotalReadBytes() uint64 {
	return c.totalReadBytes
}

// TotalWriteBytes returns the total bytes written to DRAM across all ops.
func (c *Comp) TotalWriteBytes() uint64 {
	return c.totalWriteBytes
}

// SetFreq sets the operating frequency of the accelerator.
func (c *Comp) SetFreq(freq sim.Freq) {
	c.TickingComponent.Freq = freq
}
