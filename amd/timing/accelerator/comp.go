// Package accelerator provides a cycle-level timing model for a
// fixed-function inference accelerator (e.g., systolic-array / TPU-like).
// It plugs into the Akita simulation engine alongside GPU compute units.
//
// The timing model uses full-traffic memory simulation:
//   - Every 64 bytes of tensor data generates a real mem.ReadReq/WriteReq
//   - Memory bandwidth emerges from the interconnect (PCIe flit serialization)
//   - Compute cycles are estimated analytically (PE array model)
//   - Total time = read_time + compute_time + write_time (sequential phases)
package accelerator

import (
	"log"
	"reflect"

	"github.com/sarchlab/akita/v4/mem/mem"
	"github.com/sarchlab/akita/v4/sim"
	"github.com/sarchlab/akita/v4/tracing"
	"github.com/sarchlab/mgpusim/v4/amd/protocol"
)

// memReqSize is the granularity of DMA memory requests (cache line size).
const memReqSize = 64

// Execution phases for the accelerator state machine.
// Each operation goes through: READ → COMPUTE → WRITE → done.
const (
	phaseReading = iota // Issuing reads and waiting for responses
	phaseCompute        // Counting down analytical compute cycles
	phaseWriting        // Issuing writes and waiting for responses
)

// tensorTransfer tracks progressive DMA transfer of a single tensor.
type tensorTransfer struct {
	baseAddr          uint64
	totalBytes        uint64
	bytesSent         uint64 // bytes for which requests have been issued
	responsesExpected int
	responsesReceived int
}

// allIssued returns true if all requests for this tensor have been sent.
func (t *tensorTransfer) allIssued() bool {
	return t.bytesSent >= t.totalBytes
}

// allDone returns true if all responses for this tensor have been received.
func (t *tensorTransfer) allDone() bool {
	return t.allIssued() && t.responsesReceived >= t.responsesExpected
}

// transaction tracks one in-flight accelerator operation.
type transaction struct {
	req           *protocol.AccelInferenceReq
	phase         int // phaseReading → phaseCompute → phaseWriting
	computeCycles int // analytical compute time (PE array)

	// Read tracking: one entry per input tensor.
	readTensors    []tensorTransfer
	currentReadIdx int

	// Write tracking: single output tensor.
	writeTransfer tensorTransfer

	// Flow control: limits in-flight memory requests.
	outstandingReqs int
	maxOutstanding  int

	// Metrics for this operation.
	totalReadBytes  uint64
	totalWriteBytes uint64
}

// Comp is the top-level accelerator component.
// It receives AccelInferenceReq from the driver, issues full-volume
// memory traffic through ToMem, models compute cycles analytically,
// and sends AccelInferenceRsp when done.
type Comp struct {
	*sim.TickingComponent

	// ToDriver receives AccelInferenceReq and sends AccelInferenceRsp.
	ToDriver sim.Port

	// ToMem issues memory read/write requests through the interconnect
	// to the DRAM controller. Full traffic volume flows through this port.
	ToMem sim.Port

	// Hardware configuration — set via Builder.
	peArrayRows       int    // systolic array rows (e.g., 256)
	peArrayCols       int    // systolic array cols (e.g., 256)
	sramSizeBytes     uint64 // on-chip SRAM buffer size
	maxOutstandingReqs int   // DMA queue depth

	// Internal state
	pendingReqs  []*protocol.AccelInferenceReq
	currentTxn   *transaction
	localModules mem.AddressToPortMapper

	// Metrics
	totalOps        int
	totalCycles     int // compute cycles only (memory is emergent)
	totalReadBytes  uint64
	totalWriteBytes uint64
	totalMemReqs    int // actual memory transactions generated
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

// startOperation initializes a new transaction with full-traffic DMA
// and begins issuing read requests for input tensors.
//
//nolint:funlen
func (c *Comp) startOperation(req *protocol.AccelInferenceReq) {
	computeCycles := c.estimateComputeCycles(req)
	readTensors := c.buildReadTensors(req)
	_, writeBytes := c.estimateMemoryTraffic(req)

	var totalReadBytes uint64
	for i := range readTensors {
		totalReadBytes += readTensors[i].totalBytes
	}

	c.totalOps++
	c.totalCycles += computeCycles

	c.currentTxn = &transaction{
		req:           req,
		computeCycles: computeCycles,
		readTensors:   readTensors,
		writeTransfer: tensorTransfer{
			baseAddr:   req.OutputAddr,
			totalBytes: writeBytes,
		},
		maxOutstanding:  c.maxOutstandingReqs,
		totalReadBytes:  totalReadBytes,
		totalWriteBytes: writeBytes,
	}

	tracing.TraceReqReceive(req, c)

	c.totalReadBytes += totalReadBytes
	c.totalWriteBytes += writeBytes

	if len(readTensors) > 0 && totalReadBytes > 0 {
		c.currentTxn.phase = phaseReading
	} else {
		c.currentTxn.phase = phaseCompute
		c.currentTxn.computeCycles = computeCycles
	}
}

// buildReadTensors creates per-tensor transfer descriptors for all
// input tensors that need to be read from DRAM.
//
//nolint:gocognit,funlen
func (c *Comp) buildReadTensors(
	req *protocol.AccelInferenceReq,
) []tensorTransfer {
	const bpf = 4 // bytes per float32
	var transfers []tensorTransfer

	switch req.OpType {
	case protocol.AccelOpGEMM, protocol.AccelOpMatMul:
		m := uint64(req.Params.M)
		k := uint64(req.Params.K)
		n := uint64(req.Params.N)
		if req.InputAddr != 0 {
			transfers = append(transfers, tensorTransfer{
				baseAddr: req.InputAddr, totalBytes: m * k * bpf,
			})
		}
		if req.WeightsAddr != 0 {
			transfers = append(transfers, tensorTransfer{
				baseAddr: req.WeightsAddr, totalBytes: k * n * bpf,
			})
		}
		if req.BiasAddr != 0 {
			transfers = append(transfers, tensorTransfer{
				baseAddr: req.BiasAddr, totalBytes: n * bpf,
			})
		}

	case protocol.AccelOpConv2D:
		nIn := uint64(inputElements(req))
		inC := uint64(req.Params.InChannels)
		outC := uint64(req.Params.OutChannels)
		kH := uint64(req.Params.KernelSize[0])
		kW := uint64(req.Params.KernelSize[1])
		if req.InputAddr != 0 {
			transfers = append(transfers, tensorTransfer{
				baseAddr: req.InputAddr, totalBytes: nIn * bpf,
			})
		}
		if req.WeightsAddr != 0 {
			transfers = append(transfers, tensorTransfer{
				baseAddr:   req.WeightsAddr,
				totalBytes: outC * inC * kH * kW * bpf,
			})
		}

	case protocol.AccelOpScaleAdd:
		nIn := uint64(inputElements(req))
		if req.InputAddr != 0 {
			transfers = append(transfers, tensorTransfer{
				baseAddr: req.InputAddr, totalBytes: nIn * bpf,
			})
		}
		if req.WeightsAddr != 0 {
			transfers = append(transfers, tensorTransfer{
				baseAddr: req.WeightsAddr, totalBytes: nIn * bpf,
			})
		}

	case protocol.AccelOpAdam:
		nIn := uint64(inputElements(req))
		// params + gradients + vHistory + sHistory
		if req.InputAddr != 0 {
			transfers = append(transfers, tensorTransfer{
				baseAddr: req.InputAddr, totalBytes: nIn * bpf,
			})
		}
		if req.WeightsAddr != 0 {
			transfers = append(transfers, tensorTransfer{
				baseAddr: req.WeightsAddr, totalBytes: nIn * bpf,
			})
		}
		if req.BiasAddr != 0 {
			transfers = append(transfers, tensorTransfer{
				baseAddr: req.BiasAddr, totalBytes: nIn * bpf,
			})
		}
		if req.OutputAddr != 0 {
			// sHistory read as 4th input (reuse OutputAddr)
			transfers = append(transfers, tensorTransfer{
				baseAddr: req.OutputAddr, totalBytes: nIn * bpf,
			})
		}

	case protocol.AccelOpRMSProp:
		nIn := uint64(inputElements(req))
		// params + gradients + sHistory
		if req.InputAddr != 0 {
			transfers = append(transfers, tensorTransfer{
				baseAddr: req.InputAddr, totalBytes: nIn * bpf,
			})
		}
		if req.WeightsAddr != 0 {
			transfers = append(transfers, tensorTransfer{
				baseAddr: req.WeightsAddr, totalBytes: nIn * bpf,
			})
		}
		if req.BiasAddr != 0 {
			transfers = append(transfers, tensorTransfer{
				baseAddr: req.BiasAddr, totalBytes: nIn * bpf,
			})
		}

	case protocol.AccelOpCrossEntropy,
		protocol.AccelOpCrossEntropyDeriv,
		protocol.AccelOpSoftmaxCrossEntropyDeriv:
		nIn := uint64(inputElements(req))
		// predictions + labels
		if req.InputAddr != 0 {
			transfers = append(transfers, tensorTransfer{
				baseAddr: req.InputAddr, totalBytes: nIn * bpf,
			})
		}
		if req.WeightsAddr != 0 {
			transfers = append(transfers, tensorTransfer{
				baseAddr: req.WeightsAddr, totalBytes: nIn * bpf,
			})
		}

	default:
		// ReLU, ElementWise, Softmax, Reduction, Pooling:
		// single input tensor
		nIn := uint64(inputElements(req))
		if req.InputAddr != 0 {
			transfers = append(transfers, tensorTransfer{
				baseAddr: req.InputAddr, totalBytes: nIn * bpf,
			})
		}
	}

	return transfers
}

// issueReadReq sends a 64-byte read request through the interconnect
// to the DRAM controller. Returns true if the request was sent.
func (c *Comp) issueReadReq(addr uint64, size uint64) bool {
	if c.localModules == nil {
		return false
	}

	readReq := mem.ReadReqBuilder{}.
		WithSrc(c.ToMem.AsRemote()).
		WithDst(c.localModules.Find(addr)).
		WithAddress(addr).
		WithByteSize(size).
		Build()

	err := c.ToMem.Send(readReq)
	if err != nil {
		return false
	}

	c.totalMemReqs++

	return true
}

// issueWriteReq sends a 64-byte write request through the interconnect
// to the DRAM controller. Returns true if the request was sent.
func (c *Comp) issueWriteReq(addr uint64, size uint64) bool {
	if c.localModules == nil {
		return false
	}

	writeReq := mem.WriteReqBuilder{}.
		WithSrc(c.ToMem.AsRemote()).
		WithDst(c.localModules.Find(addr)).
		WithAddress(addr).
		WithData(make([]byte, size)).
		Build()

	err := c.ToMem.Send(writeReq)
	if err != nil {
		return false
	}

	c.totalMemReqs++

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

// processReadPhase progressively issues 64-byte read requests for all
// input tensors, respecting the maxOutstanding limit for backpressure.
func (c *Comp) processReadPhase() bool {
	txn := c.currentTxn
	madeProgress := false

	// Issue read requests up to maxOutstanding limit.
	for txn.outstandingReqs < txn.maxOutstanding {
		if txn.currentReadIdx >= len(txn.readTensors) {
			break // all tensors fully issued
		}

		rt := &txn.readTensors[txn.currentReadIdx]
		if rt.allIssued() {
			txn.currentReadIdx++
			continue
		}

		addr := rt.baseAddr + rt.bytesSent
		remaining := rt.totalBytes - rt.bytesSent
		reqSize := uint64(memReqSize)
		if remaining < reqSize {
			reqSize = remaining
		}

		if c.issueReadReq(addr, reqSize) {
			rt.bytesSent += reqSize
			rt.responsesExpected++
			txn.outstandingReqs++
			madeProgress = true
		} else {
			break // port backpressure
		}
	}

	// Check if all reads are complete.
	if c.allReadsComplete() {
		txn.phase = phaseCompute
		madeProgress = true
	}

	return madeProgress
}

// allReadsComplete returns true when all tensor reads have been issued
// and all responses received.
func (c *Comp) allReadsComplete() bool {
	txn := c.currentTxn
	if txn.currentReadIdx < len(txn.readTensors) {
		return false
	}

	for i := range txn.readTensors {
		if !txn.readTensors[i].allDone() {
			return false
		}
	}

	return true
}

// processComputePhase counts down the analytical compute cycles.
func (c *Comp) processComputePhase() bool {
	c.currentTxn.computeCycles--

	if c.currentTxn.computeCycles <= 0 {
		return c.transitionToWritePhase()
	}

	return true
}

// transitionToWritePhase moves to the write phase. If there's no output
// to write, completes the operation immediately.
func (c *Comp) transitionToWritePhase() bool {
	wt := &c.currentTxn.writeTransfer
	if wt.baseAddr == 0 || wt.totalBytes == 0 {
		return c.completeOperation()
	}

	c.currentTxn.phase = phaseWriting

	return true
}

// processWritePhase progressively issues 64-byte write requests for the
// output tensor, respecting the maxOutstanding limit.
func (c *Comp) processWritePhase() bool {
	txn := c.currentTxn
	wt := &txn.writeTransfer
	madeProgress := false

	// Issue write requests up to maxOutstanding limit.
	for txn.outstandingReqs < txn.maxOutstanding {
		if wt.allIssued() {
			break
		}

		addr := wt.baseAddr + wt.bytesSent
		remaining := wt.totalBytes - wt.bytesSent
		reqSize := uint64(memReqSize)
		if remaining < reqSize {
			reqSize = remaining
		}

		if c.issueWriteReq(addr, reqSize) {
			wt.bytesSent += reqSize
			wt.responsesExpected++
			txn.outstandingReqs++
			madeProgress = true
		} else {
			break // port backpressure
		}
	}

	// Check if all writes are complete.
	if wt.allDone() {
		return c.completeOperation()
	}

	return madeProgress
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

// processMemRsp handles memory read/write responses from the DRAM
// controller. Drains all available responses per tick.
func (c *Comp) processMemRsp() bool {
	madeProgress := false

	for {
		msg := c.ToMem.PeekIncoming()
		if msg == nil {
			break
		}

		c.ToMem.RetrieveIncoming()
		madeProgress = true

		if c.currentTxn == nil {
			continue
		}

		c.currentTxn.outstandingReqs--

		switch c.currentTxn.phase {
		case phaseReading:
			c.creditReadResponse()
		case phaseWriting:
			c.currentTxn.writeTransfer.responsesReceived++
		}
	}

	return madeProgress
}

// creditReadResponse assigns a response to the first read tensor that
// still has outstanding responses.
func (c *Comp) creditReadResponse() {
	for i := range c.currentTxn.readTensors {
		rt := &c.currentTxn.readTensors[i]
		if rt.responsesReceived < rt.responsesExpected {
			rt.responsesReceived++
			return
		}
	}
}

// --- Compute cycle estimates (analytical PE array model) ---

// estimateComputeCycles returns the compute-bound cycle count for an
// operation, based on the PE array dimensions.
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

// --- Memory traffic estimates (used for metrics only) ---

// estimateMemoryTraffic returns (readBytes, writeBytes) for the operation.
// Used for reporting metrics. The actual traffic is driven by
// buildReadTensors() and the write transfer.
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
		readBytes = (m*k + k*n) * bytesPerFloat32
		writeBytes = m * n * bytesPerFloat32

	case protocol.AccelOpConv2D:
		inC := uint64(req.Params.InChannels)
		outC := uint64(req.Params.OutChannels)
		kH := uint64(req.Params.KernelSize[0])
		kW := uint64(req.Params.KernelSize[1])
		readBytes = (nIn + outC*inC*kH*kW) * bytesPerFloat32
		writeBytes = nOut * bytesPerFloat32

	case protocol.AccelOpReLU, protocol.AccelOpElementWise:
		readBytes = nIn * bytesPerFloat32
		writeBytes = nIn * bytesPerFloat32

	case protocol.AccelOpScaleAdd:
		readBytes = 2 * nIn * bytesPerFloat32
		writeBytes = nIn * bytesPerFloat32

	case protocol.AccelOpSoftmax:
		readBytes = nIn * bytesPerFloat32
		writeBytes = nIn * bytesPerFloat32

	case protocol.AccelOpReduction:
		readBytes = nIn * bytesPerFloat32
		writeBytes = bytesPerFloat32

	case protocol.AccelOpAdam:
		readBytes = 4 * nIn * bytesPerFloat32
		writeBytes = 3 * nIn * bytesPerFloat32

	case protocol.AccelOpRMSProp:
		readBytes = 3 * nIn * bytesPerFloat32
		writeBytes = 2 * nIn * bytesPerFloat32

	case protocol.AccelOpCrossEntropy:
		readBytes = 2 * nIn * bytesPerFloat32
		writeBytes = bytesPerFloat32

	case protocol.AccelOpCrossEntropyDeriv,
		protocol.AccelOpSoftmaxCrossEntropyDeriv:
		readBytes = 2 * nIn * bytesPerFloat32
		writeBytes = nIn * bytesPerFloat32

	case protocol.AccelOpMaxPool, protocol.AccelOpAvgPool:
		readBytes = nIn * bytesPerFloat32
		writeBytes = nOut * bytesPerFloat32

	default:
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

// TotalCycles returns the total compute cycles (PE array model).
// Memory latency is emergent from the interconnect simulation.
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

// TotalMemReqs returns the total number of memory transactions generated.
func (c *Comp) TotalMemReqs() int {
	return c.totalMemReqs
}

// SetFreq sets the operating frequency of the accelerator.
func (c *Comp) SetFreq(freq sim.Freq) {
	c.TickingComponent.Freq = freq
}
