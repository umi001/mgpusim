// Package accelerator provides a cycle-level timing model for a
// fixed-function inference accelerator (e.g., systolic-array / TPU-like).
// It plugs into the Akita simulation engine alongside GPU compute units.
package accelerator

import (
	"log"
	"reflect"

	"github.com/sarchlab/akita/v4/mem/mem"
	"github.com/sarchlab/akita/v4/sim"
	"github.com/sarchlab/akita/v4/tracing"
	"github.com/sarchlab/mgpusim/v4/amd/protocol"
)

// transaction tracks one in-flight accelerator operation.
type transaction struct {
	req            *protocol.AccelInferenceReq
	remainCycles   int
	memReqsPending int // outstanding memory read/write requests
}

// Comp is the top-level accelerator component.
// It receives AccelInferenceReq from the driver, models execution latency,
// and sends AccelInferenceRsp when done.
type Comp struct {
	*sim.TickingComponent

	// ToDriver receives AccelInferenceReq and sends AccelInferenceRsp.
	ToDriver sim.Port

	// ToMem issues memory read/write requests to the memory hierarchy.
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
	totalOps    int
	totalCycles int

	// TODO: Add more internal state as the model is refined:
	//   - pendingMemReads / pendingMemWrites for tracking memory transactions
	//   - sramOccupancy for modeling on-chip buffer contention
	//   - pipeline stage registers if modeling a multi-stage pipeline
	//   - weight-stationary / output-stationary dataflow configuration
}

// SetLocalModuleFinder sets the address-to-port mapper for memory access.
func (c *Comp) SetLocalModuleFinder(lmf mem.AddressToPortMapper) {
	c.localModules = lmf
}

// Tick is called every cycle by the Akita engine.
func (c *Comp) Tick() bool {
	madeProgress := false

	madeProgress = c.acceptNewReq() || madeProgress
	madeProgress = c.processCurrentOp() || madeProgress
	madeProgress = c.processMemRsp() || madeProgress

	return madeProgress
}

// acceptNewReq checks if a new AccelInferenceReq has arrived from the driver.
func (c *Comp) acceptNewReq() bool {
	if c.currentTxn != nil {
		// Accelerator is busy — cannot accept new work.
		// TODO: Implement a request queue if you want to support
		// pipelining / multiple concurrent operations.
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

// startOperation initializes a new transaction for the given request.
func (c *Comp) startOperation(req *protocol.AccelInferenceReq) {
	cycles := c.estimateCycles(req)

	c.totalOps++
	c.totalCycles += cycles

	c.currentTxn = &transaction{
		req:          req,
		remainCycles: cycles,
	}

	tracing.TraceReqReceive(req, c)

	// TODO: Issue memory read requests for input + weight tensors here.
	// For now we model the operation as a fixed latency without
	// explicit memory transactions.
	//
	// Example of issuing a memory read:
	//   readReq := mem.ReadReqBuilder{}.
	//       WithSrc(c.ToMem.AsRemote()).
	//       WithDst(c.localModules.Find(addr)).
	//       WithAddress(addr).
	//       WithByteSize(size).
	//       Build()
	//   c.ToMem.Send(readReq)
	//   c.currentTxn.memReqsPending++
}

// estimateCycles computes how many cycles the operation will take.
// This is the core timing model — replace / refine this for accuracy.
func (c *Comp) estimateCycles(
	req *protocol.AccelInferenceReq,
) int {
	// TODO: Implement a proper roofline / analytical timing model.
	// The model should consider:
	//   1. Compute bound: FLOPs / (peArrayRows * peArrayCols * freq)
	//   2. Memory bound:  bytes_to_transfer / memBandwidthBW
	//   3. Latency = max(compute_cycles, memory_cycles) + pipeline_overhead
	//
	// For now, return a rough estimate based on operation type and size.

	switch req.OpType {
	case protocol.AccelOpGEMM, protocol.AccelOpMatMul:
		return c.estimateGEMMCycles(req)
	case protocol.AccelOpConv2D:
		return c.estimateConv2DCycles(req)
	case protocol.AccelOpMaxPool, protocol.AccelOpAvgPool:
		return c.estimatePoolingCycles(req)
	case protocol.AccelOpReLU, protocol.AccelOpElementWise:
		return c.estimateElementWiseCycles(req)
	case protocol.AccelOpSoftmax:
		return c.estimateSoftmaxCycles(req)
	default:
		// TODO: Add timing models for LayerNorm, attention, etc.
		return 1000 // fallback
	}
}

func (c *Comp) estimateGEMMCycles(
	req *protocol.AccelInferenceReq,
) int {
	m := int(req.Params.M)
	n := int(req.Params.N)
	k := int(req.Params.K)

	// TODO: Model tiling over the systolic array.
	// Real model: tiles_M * tiles_N * (K / array_dim + pipeline_depth)
	//
	// Simplified: total_MACs / (array_rows * array_cols)
	totalMACs := m * n * k
	computeCycles := totalMACs / (c.peArrayRows * c.peArrayCols)
	if computeCycles < 1 {
		computeCycles = 1
	}

	// TODO: Add memory transfer cycles for weights + activations.
	// memCycles := (m*k + k*n + m*n) * 4 / memBandwidthBW

	return computeCycles
}

func (c *Comp) estimateConv2DCycles(
	req *protocol.AccelInferenceReq,
) int {
	// TODO: Implement im2col-based or direct convolution timing.
	// Conv2D is typically lowered to GEMM:
	//   M = OutChannels
	//   N = batch * outH * outW
	//   K = InChannels * kernelH * kernelW
	//
	// For now, compute a rough estimate.
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
	computeCycles := totalMACs / (c.peArrayRows * c.peArrayCols)
	if computeCycles < 1 {
		computeCycles = 1
	}

	return computeCycles
}

func (c *Comp) estimatePoolingCycles(
	req *protocol.AccelInferenceReq,
) int {
	// TODO: Pooling is memory-bound. Model as:
	//   cycles = total_elements * comparisons_per_element / throughput
	batch := int(req.InputSize[0])
	channels := int(req.InputSize[1])
	height := int(req.InputSize[2])
	width := int(req.InputSize[3])

	totalElements := batch * channels * height * width
	// Rough: 1 cycle per element (pooling is simple)
	return totalElements / c.peArrayCols
}

func (c *Comp) estimateElementWiseCycles(
	req *protocol.AccelInferenceReq,
) int {
	// TODO: Element-wise ops (ReLU, add, mul) are memory-bound.
	// Model as bytes_transferred / memory_bandwidth.
	totalElements := int(req.InputSize[0]) *
		int(req.InputSize[1]) *
		int(req.InputSize[2]) *
		int(req.InputSize[3])
	cycles := totalElements / (c.peArrayRows * c.peArrayCols)
	if cycles < 1 {
		cycles = 1
	}
	return cycles
}

func (c *Comp) estimateSoftmaxCycles(
	req *protocol.AccelInferenceReq,
) int {
	// TODO: Softmax requires exp + sum + div — model multi-pass.
	totalElements := int(req.InputSize[0]) *
		int(req.InputSize[1]) *
		int(req.InputSize[2]) *
		int(req.InputSize[3])
	// 3 passes over data (exp, sum, div)
	return 3 * totalElements / c.peArrayCols
}

// processCurrentOp decrements the remaining cycles and completes the
// operation when done.
func (c *Comp) processCurrentOp() bool {
	if c.currentTxn == nil {
		return false
	}

	if c.currentTxn.memReqsPending > 0 {
		// Still waiting for memory — don't count down.
		// TODO: Model compute-memory overlap (pipelining tiles with
		// prefetch of next tile's data).
		return false
	}

	c.currentTxn.remainCycles--

	if c.currentTxn.remainCycles <= 0 {
		return c.completeOperation()
	}

	return true
}

// completeOperation sends the response back to the driver and clears
// the current transaction.
func (c *Comp) completeOperation() bool {
	req := c.currentTxn.req

	// TODO: Before completing, issue memory writes for the output tensor.
	// For now, we assume the functional result is handled by the driver /
	// global storage (similar to magic memory copy mode).
	//
	// In a full model you would:
	//   1. Compute the result functionally (or read from golden data)
	//   2. Write output to memory via c.ToMem
	//   3. Wait for write acknowledgments
	//   4. Then send the response

	rsp := protocol.NewAccelInferenceRsp(
		c.ToDriver.AsRemote(),
		req.Src,
		req.Meta().ID,
	)

	err := c.ToDriver.Send(rsp)
	if err != nil {
		// Port is full — retry next cycle.
		c.currentTxn.remainCycles = 0
		return false
	}

	tracing.TraceReqComplete(req, c)

	c.currentTxn = nil

	return true
}

// processMemRsp handles memory read/write responses.
func (c *Comp) processMemRsp() bool {
	msg := c.ToMem.PeekIncoming()
	if msg == nil {
		return false
	}

	c.ToMem.RetrieveIncoming()

	if c.currentTxn != nil {
		c.currentTxn.memReqsPending--
	}

	// TODO: Process the memory response:
	//   - For reads: store data in on-chip SRAM buffer
	//   - For writes: mark output tile as committed
	//   - Track which tiles are ready for compute

	return true
}

// TotalOps returns the number of operations processed.
func (c *Comp) TotalOps() int {
	return c.totalOps
}

// TotalCycles returns the total estimated compute cycles.
func (c *Comp) TotalCycles() int {
	return c.totalCycles
}

// SetFreq sets the operating frequency of the accelerator.
func (c *Comp) SetFreq(freq sim.Freq) {
	c.TickingComponent.Freq = freq
}
