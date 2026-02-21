package acceltensor

import (
	"fmt"

	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/tensor"
	"github.com/sarchlab/mgpusim/v4/amd/driver"
	"github.com/sarchlab/mgpusim/v4/amd/protocol"
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/mat"
)

var sizeOfFloat32 = 4

// Operator implements tensor.Operator by dispatching operations to a
// hardware inference accelerator via the driver's accelerator API.
//
// It shares the same memory space as GPUOperator — tensors allocated here
// can be passed to GPU operators and vice versa.
type Operator struct {
	driver *driver.Driver
	ctx    *driver.Context

	// TODO: Add a CPU operator for verification, similar to
	// gputensor.GPUOperator.EnableVerification().
}

// NewOperator creates a new accelerator tensor operator.
func NewOperator(
	d *driver.Driver,
	ctx *driver.Context,
) *Operator {
	return &Operator{
		driver: d,
		ctx:    ctx,
	}
}

// Create allocates a new tensor on the device.
func (o *Operator) Create(size []int) tensor.Tensor {
	t := &Tensor{
		driver: o.driver,
		ctx:    o.ctx,
		size:   make([]int, len(size)),
	}
	copy(t.size, size)

	numElem := t.NumElement()
	t.ptr = o.driver.AllocateMemory(o.ctx, uint64(numElem*sizeOfFloat32))

	return t
}

// CreateWithData creates a tensor and initializes it with data.
func (o *Operator) CreateWithData(
	data []float64, size []int, descriptor string,
) tensor.Tensor {
	t := o.Create(size)
	t.SetDescriptor(descriptor)
	o.Init(t, data)
	return t
}

// Free releases the device memory for a tensor.
func (o *Operator) Free(t tensor.Tensor) {
	at := t.(*Tensor)
	if at.ptr != 0 {
		_ = o.driver.FreeMemory(o.ctx, at.ptr)
	}
}

// Copy copies data from src to dst tensor.
func (o *Operator) Copy(dst, src tensor.Tensor) {
	d := dst.(*Tensor)
	s := src.(*Tensor)
	o.driver.MemCopyD2D(o.ctx, d.ptr, s.ptr, s.NumElement()*sizeOfFloat32)
}

// Clone duplicates a tensor.
func (o *Operator) Clone(t tensor.Tensor) tensor.Tensor {
	out := o.Create(t.Size())
	out.SetDescriptor(t.Descriptor())
	o.Copy(out, t)
	return out
}

// Dump converts tensor to string for debugging.
func (o *Operator) Dump(t tensor.Tensor) string {
	v := t.Vector()
	return fmt.Sprintf("%v", v)
}

// Init initializes tensor with float64 data from host.
func (o *Operator) Init(t tensor.Tensor, data []float64) {
	at := t.(*Tensor)
	f32 := make([]float32, len(data))
	for i, v := range data {
		f32[i] = float32(v)
	}
	o.driver.MemCopyH2D(o.ctx, at.ptr, f32)
}

// Slice creates a tensor that shares part of the underlying buffer.
func (o *Operator) Slice(t tensor.Tensor, start, end int) tensor.Tensor {
	at := t.(*Tensor)
	return &Tensor{
		driver: o.driver,
		ctx:    o.ctx,
		size:   []int{end - start},
		ptr:    at.ptr + driver.Ptr(start*sizeOfFloat32),
	}
}

// Repeat creates a new tensor that duplicates input n times.
func (o *Operator) Repeat(t tensor.Tensor, times int) tensor.Tensor {
	// TODO: Dispatch to accelerator if beneficial, otherwise use
	// host-side copy. For now, copy to host, repeat, copy back.
	inData := t.Vector()
	numElem := t.NumElement()
	outData := make([]float64, numElem*times)
	for i := 0; i < times; i++ {
		copy(outData[i*numElem:(i+1)*numElem], inData)
	}
	out := o.Create([]int{numElem * times})
	o.Init(out, outData)
	return out
}

// Clear sets all elements to 0.
func (o *Operator) Clear(t tensor.Tensor) {
	zeros := make([]float32, t.NumElement())
	o.driver.MemCopyH2D(o.ctx, t.(*Tensor).ptr, zeros)
}

// Zeros creates a zero-filled tensor.
func (o *Operator) Zeros(size []int) tensor.Tensor {
	t := o.Create(size)
	o.Clear(t)
	return t
}

// Reshape creates a tensor with the same data but different shape.
func (o *Operator) Reshape(t tensor.Tensor, newSize []int) tensor.Tensor {
	out := o.Clone(t)
	out.SetSize(newSize)
	return out
}

// ---- Operations dispatched to accelerator ----

// Gemm performs alpha * A * B + beta * C via the accelerator.
func (o *Operator) Gemm(
	transA, transB bool,
	alpha, beta float64,
	a, b, c tensor.Tensor,
) tensor.Tensor {
	aT, bT, cT := a.(*Tensor), b.(*Tensor), c.(*Tensor)

	m := uint32(a.Size()[0])
	k := uint32(a.Size()[1])
	n := uint32(b.Size()[1])

	out := o.Clone(c)
	outT := out.(*Tensor)

	params := protocol.AccelOpParams{
		M: m, N: n, K: k,
		TransposeA: transA,
		TransposeB: transB,
		Alpha:      alpha,
		Beta:       beta,
	}

	o.driver.AccelInference(
		o.ctx,
		protocol.AccelOpGEMM,
		params,
		uint64(aT.ptr),
		uint64(outT.ptr),
		uint64(bT.ptr),
		uint64(cT.ptr),
		[4]uint32{m, k, 1, 1},
		[4]uint32{m, n, 1, 1},
		[4]uint32{k, n, 1, 1},
	)

	return out
}

func (to CPUOperator) Gemm(
	transA, transB bool,
	alpha, beta float64,
	a, b, c Tensor,
) Tensor {
	to.mustBeTwoDimension(a)
	to.mustBeTwoDimension(b)
	to.mustBeTwoDimension(c)

	out := to.Clone(c)

	ma := mat.NewDense(a.Size()[0], a.Size()[1], a.Vector())
	mb := mat.NewDense(b.Size()[0], b.Size()[1], b.Vector())
	mc := mat.NewDense(c.Size()[0], c.Size()[1], out.Vector())

	gemmTransA := blas.NoTrans
	if transA {
		gemmTransA = blas.Trans
	}

	gemmTransB := blas.NoTrans
	if transB {
		gemmTransB = blas.Trans
	}

	blas64.Gemm(gemmTransA, gemmTransB,
		1, ma.RawMatrix(), mb.RawMatrix(), 1, mc.RawMatrix())

	return out
}

// Im2Col performs the im2col transformation via the accelerator.
func (o *Operator) Im2Col(
	t tensor.Tensor,
	kernelSize, padding, stride, dilation []int,
) tensor.Tensor {
	// TODO: Dispatch to accelerator with AccelOpConv2D or a dedicated
	// im2col op type. For now, fall back to CPU implementation.
	//
	// A full implementation would:
	//   1. Compute output dimensions
	//   2. Allocate output tensor
	//   3. Send AccelInferenceReq with im2col parameters
	//   4. Return the output tensor
	panic("acceltensor: Im2Col not yet implemented — " +
		"TODO: implement as accelerator op or CPU fallback")
}

// Transpose reorders tensor axes.
func (o *Operator) Transpose(t tensor.Tensor, order []int) tensor.Tensor {
	// TODO: Dispatch to accelerator or implement via memory reshuffling.
	// Transpose is typically memory-bound and may be better on the
	// accelerator's DMA engine.
	panic("acceltensor: Transpose not yet implemented — " +
		"TODO: implement as accelerator op or CPU fallback")
}

// Rotate180 rotates the lowest-level matrices by 180 degrees.
func (o *Operator) Rotate180(t tensor.Tensor) tensor.Tensor {
	// TODO: Implement — used in conv2d backward pass.
	panic("acceltensor: Rotate180 not yet implemented")
}

// Dilate adds zeros between elements.
func (o *Operator) Dilate(t tensor.Tensor, dilate []int) tensor.Tensor {
	// TODO: Implement — used in conv2d backward pass.
	panic("acceltensor: Dilate not yet implemented")
}

// Sum calculates sums over given axes.
func (o *Operator) Sum(t tensor.Tensor, axis []int) tensor.Tensor {
	// TODO: Dispatch as AccelOpElementWise reduction.
	panic("acceltensor: Sum not yet implemented")
}

// MaxPoolingForward performs max pooling forward pass.
func (o *Operator) MaxPoolingForward(
	t tensor.Tensor,
	kernelSize, padding, stride []int,
) (tensor.Tensor, tensor.Tensor) {
	// TODO: Dispatch as AccelOpMaxPool.
	panic("acceltensor: MaxPoolingForward not yet implemented")
}

// MaxPoolingBackward performs max pooling backward pass.
func (o *Operator) MaxPoolingBackward(
	forwardIn, backwardIn, mask tensor.Tensor,
	kernelSize, padding, stride []int,
) tensor.Tensor {
	// TODO: Implement backward pass for max pooling.
	panic("acceltensor: MaxPoolingBackward not yet implemented")
}

// AvgPoolingForward performs average pooling forward pass.
func (o *Operator) AvgPoolingForward(
	t tensor.Tensor,
	kernelSize, padding, stride []int,
) tensor.Tensor {
	// TODO: Dispatch as AccelOpAvgPool.
	panic("acceltensor: AvgPoolingForward not yet implemented")
}

// AvgPoolingBackward performs average pooling backward pass.
func (o *Operator) AvgPoolingBackward(
	forwardIn, backwardIn tensor.Tensor,
	kernelSize, padding, stride []int,
) tensor.Tensor {
	// TODO: Implement backward pass for avg pooling.
	panic("acceltensor: AvgPoolingBackward not yet implemented")
}

// Softmax computes softmax.
func (o *Operator) Softmax(t tensor.Tensor) tensor.Tensor {
	// TODO: Dispatch as AccelOpSoftmax.
	panic("acceltensor: Softmax not yet implemented")
}

// CrossEntropy computes cross entropy loss.
func (o *Operator) CrossEntropy(t tensor.Tensor, label []int) float64 {
	// TODO: Implement — may run on host or accelerator.
	panic("acceltensor: CrossEntropy not yet implemented")
}

// CrossEntropyDerivative computes cross entropy derivative.
func (o *Operator) CrossEntropyDerivative(
	t tensor.Tensor, label []int,
) tensor.Tensor {
	// TODO: Implement.
	panic("acceltensor: CrossEntropyDerivative not yet implemented")
}

// SoftmaxCrossEntropyDerivative computes fused softmax + cross entropy
// derivative.
func (o *Operator) SoftmaxCrossEntropyDerivative(
	t tensor.Tensor, label []int,
) tensor.Tensor {
	// TODO: Implement.
	panic("acceltensor: SoftmaxCrossEntropyDerivative not yet implemented")
}

// ElementWiseMul performs element-wise multiplication.
func (o *Operator) ElementWiseMul(t1, t2 tensor.Tensor) tensor.Tensor {
	// TODO: Dispatch as AccelOpElementWise.
	panic("acceltensor: ElementWiseMul not yet implemented")
}

// ScaleAdd performs alpha*A + beta*B.
func (o *Operator) ScaleAdd(
	alpha, beta float64, a, b tensor.Tensor,
) tensor.Tensor {
	// TODO: Dispatch as AccelOpElementWise.
	panic("acceltensor: ScaleAdd not yet implemented")
}

// RMSProp runs the RMSProp optimization step.
func (o *Operator) RMSProp(
	params, gradient, sHistory tensor.Tensor,
	smoothFactor, learningRate float64,
) {
	// TODO: Implement optimizer step on accelerator or host.
	panic("acceltensor: RMSProp not yet implemented")
}

// Adam runs the Adam optimization step.
func (o *Operator) Adam(
	params, gradients, vHistory, sHistory tensor.Tensor,
	smoothFactor1, smoothFactor2, learningRate float64,
) {
	// TODO: Implement optimizer step on accelerator or host.
	panic("acceltensor: Adam not yet implemented")
}

// ReluForward performs ReLU forward pass.
func (o *Operator) ReluForward(in tensor.Tensor) tensor.Tensor {
	// TODO: Dispatch as AccelOpReLU.
	panic("acceltensor: ReluForward not yet implemented")
}

// ReluBackward performs ReLU backward pass.
func (o *Operator) ReluBackward(
	forwardIn, backwardIn tensor.Tensor,
) tensor.Tensor {
	// TODO: Implement ReLU backward on accelerator.
	panic("acceltensor: ReluBackward not yet implemented")
}
