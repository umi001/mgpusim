// Package protocol - accelerator protocol messages for heterogeneous simulation.
package protocol

import (
	"github.com/sarchlab/akita/v4/mem/vm"
	"github.com/sarchlab/akita/v4/sim"
)

// AccelOpType defines the type of accelerator operation.
type AccelOpType int

// Supported accelerator operations.
const (
	AccelOpMatMul AccelOpType = iota
	AccelOpConv2D
	AccelOpMaxPool
	AccelOpAvgPool
	AccelOpSoftmax
	AccelOpReLU
	AccelOpGEMM
	AccelOpLayerNorm
	AccelOpElementWise
	AccelOpScaleAdd                  // alpha*A + beta*B (2 input tensors)
	AccelOpReduction                 // sum/mean reduction over axes
	AccelOpAdam                      // Adam optimizer step (4 tensors in, 3 written)
	AccelOpRMSProp                   // RMSProp optimizer step (3 tensors in, 2 written)
	AccelOpCrossEntropy              // cross-entropy loss (read-only, scalar output)
	AccelOpCrossEntropyDeriv         // cross-entropy derivative
	AccelOpSoftmaxCrossEntropyDeriv  // fused softmax + cross-entropy derivative
)

// AccelOpParams holds the parameters for an accelerator operation.
type AccelOpParams struct {
	// For Conv2D / Pooling
	KernelSize  [2]uint32
	Stride      [2]uint32
	Padding     [2]uint32
	InChannels  uint32
	OutChannels uint32

	// For GEMM
	M, N, K   uint32
	TransposeA bool
	TransposeB bool
	Alpha      float64
	Beta       float64

	// TODO: Add fields for other operation-specific parameters
	// (e.g., epsilon for LayerNorm, axis for Softmax, etc.)
}

// AccelInferenceReq is a request sent from the driver to the accelerator
// to execute a tensor operation.
type AccelInferenceReq struct {
	sim.MsgMeta

	PID vm.PID

	OpType AccelOpType
	Params AccelOpParams

	// Input/output tensor addresses in the shared memory space
	InputAddr  uint64
	OutputAddr uint64

	// Optional second input (e.g., weights for GEMM, filter for Conv2D)
	WeightsAddr uint64

	// Optional third input (e.g., bias)
	BiasAddr uint64

	// Tensor shapes (number of elements)
	InputSize   [4]uint32 // [N, C, H, W] — unused dims set to 1
	OutputSize  [4]uint32
	WeightsSize [4]uint32

	// TODO: Add fields for quantization info, data type (fp16/fp32/int8),
	// and any accelerator-specific scheduling hints
}

// Meta returns the meta data associated with the message.
func (m *AccelInferenceReq) Meta() *sim.MsgMeta {
	return &m.MsgMeta
}

// Clone returns a clone of the AccelInferenceReq with different ID.
func (m *AccelInferenceReq) Clone() sim.Msg {
	cloneMsg := *m
	cloneMsg.ID = sim.GetIDGenerator().Generate()
	return &cloneMsg
}

// NewAccelInferenceReq creates a new AccelInferenceReq.
func NewAccelInferenceReq(src, dst sim.Port) *AccelInferenceReq {
	req := new(AccelInferenceReq)
	req.ID = sim.GetIDGenerator().Generate()
	req.Src = src.AsRemote()
	req.Dst = dst.AsRemote()
	return req
}

// AccelInferenceRsp is the response sent from the accelerator back to the
// driver when an inference operation completes.
type AccelInferenceRsp struct {
	sim.MsgMeta

	RspTo string
}

// Meta returns the meta data associated with the message.
func (m *AccelInferenceRsp) Meta() *sim.MsgMeta {
	return &m.MsgMeta
}

// Clone returns a clone of the AccelInferenceRsp with different ID.
func (m *AccelInferenceRsp) Clone() sim.Msg {
	cloneMsg := *m
	cloneMsg.ID = sim.GetIDGenerator().Generate()
	return &cloneMsg
}

// NewAccelInferenceRsp creates a new AccelInferenceRsp.
func NewAccelInferenceRsp(
	src, dst sim.RemotePort,
	rspTo string,
) *AccelInferenceRsp {
	rsp := new(AccelInferenceRsp)
	rsp.ID = sim.GetIDGenerator().Generate()
	rsp.Src = src
	rsp.Dst = dst
	rsp.RspTo = rspTo
	return rsp
}
