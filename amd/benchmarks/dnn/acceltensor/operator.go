package acceltensor

import (
	"fmt"
	"math"

	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/tensor"
	"github.com/sarchlab/mgpusim/v4/amd/driver"
	"github.com/sarchlab/mgpusim/v4/amd/protocol"
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
	p := ptrOf(t)
	if p != 0 {
		_ = o.driver.FreeMemory(o.ctx, p)
	}
}

// ptrOf extracts the device pointer from any tensor that implements
// DeviceTensor (both acceltensor.Tensor and gputensor.Tensor).
func ptrOf(t tensor.Tensor) driver.Ptr {
	if dt, ok := t.(tensor.DeviceTensor); ok {
		return dt.Ptr()
	}

	panic("acceltensor: tensor does not implement DeviceTensor")
}

// Copy copies data from src to dst tensor.
func (o *Operator) Copy(dst, src tensor.Tensor) {
	o.driver.MemCopyD2D(o.ctx, ptrOf(dst), ptrOf(src),
		src.NumElement()*sizeOfFloat32)
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
	f32 := make([]float32, len(data))
	for i, v := range data {
		f32[i] = float32(v)
	}
	o.driver.MemCopyH2D(o.ctx, ptrOf(t), f32)
}

// Slice creates a tensor that shares part of the underlying buffer.
func (o *Operator) Slice(t tensor.Tensor, start, end int) tensor.Tensor {
	return &Tensor{
		driver: o.driver,
		ctx:    o.ctx,
		size:   []int{end - start},
		ptr:    ptrOf(t) + driver.Ptr(start*sizeOfFloat32),
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
	o.driver.MemCopyH2D(o.ctx, ptrOf(t), zeros)
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

// sizeToU32x4 encodes a tensor shape into the [4]uint32 format used by
// AccelInferenceReq. Unused trailing dimensions are set to 1.
func sizeToU32x4(size []int) [4]uint32 {
	var s [4]uint32
	for i := range s {
		s[i] = 1
	}

	for i := 0; i < len(size) && i < 4; i++ {
		s[i] = uint32(size[i])
	}

	return s
}

// toU32x2 converts an int slice to [2]uint32 for AccelOpParams fields.
func toU32x2(v []int) [2]uint32 {
	var out [2]uint32
	for i := 0; i < len(v) && i < 2; i++ {
		out[i] = uint32(v[i])
	}

	return out
}

// ---- Operations dispatched to accelerator ----

// Gemm performs alpha * A * B + beta * C via the accelerator.
func (o *Operator) Gemm(
	transA, transB bool,
	alpha, beta float64,
	a, b, c tensor.Tensor,
) tensor.Tensor {
	m := uint32(a.Size()[0])
	k := uint32(a.Size()[1])
	n := uint32(b.Size()[1])

	out := o.Clone(c)

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
		uint64(ptrOf(a)),
		uint64(ptrOf(out)),
		uint64(ptrOf(b)),
		uint64(ptrOf(c)),
		[4]uint32{m, k, 1, 1},
		[4]uint32{m, n, 1, 1},
		[4]uint32{k, n, 1, 1},
	)

	return out
}

// Im2Col performs the im2col transformation via the accelerator.
func (o *Operator) Im2Col(
	t tensor.Tensor,
	kernelSize, padding, stride, dilation []int,
) tensor.Tensor {
	cpuOp := tensor.CPUOperator{}
	cpuIn := cpuOp.CreateWithData(t.Vector(), t.Size(), t.Descriptor())
	cpuOut := cpuOp.Im2Col(cpuIn, kernelSize, padding, stride, dilation)

	out := o.Create(cpuOut.Size())
	o.Init(out, cpuOut.Vector())

	// Dispatch to accelerator for timing.
	inSize := sizeToU32x4(t.Size())
	outSize := sizeToU32x4(cpuOut.Size())

	o.driver.AccelInference(
		o.ctx,
		protocol.AccelOpIm2Col,
		protocol.AccelOpParams{
			KernelSize: toU32x2(kernelSize),
			Stride:     toU32x2(stride),
			Padding:    toU32x2(padding),
		},
		uint64(ptrOf(t)),
		uint64(ptrOf(out)),
		0, 0,
		inSize, outSize, [4]uint32{},
	)

	return out
}

// Transpose reorders tensor axes via the accelerator.
func (o *Operator) Transpose(t tensor.Tensor, order []int) tensor.Tensor {
	cpuOp := tensor.CPUOperator{}
	cpuIn := cpuOp.CreateWithData(t.Vector(), t.Size(), t.Descriptor())
	cpuOut := cpuOp.Transpose(cpuIn, order)

	out := o.Create(cpuOut.Size())
	out.SetDescriptor(cpuOut.Descriptor())
	o.Init(out, cpuOut.Vector())

	// Dispatch to accelerator for timing.
	inSize := sizeToU32x4(t.Size())
	outSize := sizeToU32x4(cpuOut.Size())

	o.driver.AccelInference(
		o.ctx,
		protocol.AccelOpTranspose,
		protocol.AccelOpParams{},
		uint64(ptrOf(t)),
		uint64(ptrOf(out)),
		0, 0,
		inSize, outSize, [4]uint32{},
	)

	return out
}

// Rotate180 rotates the lowest-level matrices by 180 degrees via the
// accelerator.
func (o *Operator) Rotate180(t tensor.Tensor) tensor.Tensor {
	cpuOp := tensor.CPUOperator{}
	cpuIn := cpuOp.CreateWithData(t.Vector(), t.Size(), t.Descriptor())
	cpuOut := cpuOp.Rotate180(cpuIn)

	out := o.Create(cpuOut.Size())
	o.Init(out, cpuOut.Vector())

	// Dispatch to accelerator for timing.
	inSize := sizeToU32x4(t.Size())

	o.driver.AccelInference(
		o.ctx,
		protocol.AccelOpRotate180,
		protocol.AccelOpParams{},
		uint64(ptrOf(t)),
		uint64(ptrOf(out)),
		0, 0,
		inSize, inSize, [4]uint32{},
	)

	return out
}

// Dilate adds zeros between elements via the accelerator.
func (o *Operator) Dilate(t tensor.Tensor, dilate []int) tensor.Tensor {
	cpuOp := tensor.CPUOperator{}
	cpuIn := cpuOp.CreateWithData(t.Vector(), t.Size(), t.Descriptor())
	cpuOut := cpuOp.Dilate(cpuIn, dilate)

	out := o.Create(cpuOut.Size())
	o.Init(out, cpuOut.Vector())

	// Dispatch to accelerator for timing.
	inSize := sizeToU32x4(t.Size())
	outSize := sizeToU32x4(cpuOut.Size())

	o.driver.AccelInference(
		o.ctx,
		protocol.AccelOpDilate,
		protocol.AccelOpParams{},
		uint64(ptrOf(t)),
		uint64(ptrOf(out)),
		0, 0,
		inSize, outSize, [4]uint32{},
	)

	return out
}

// Sum calculates sums over given axes via the accelerator.
func (o *Operator) Sum(t tensor.Tensor, axis []int) tensor.Tensor {
	// Functional computation on CPU.
	cpuOp := tensor.CPUOperator{}
	cpuIn := cpuOp.CreateWithData(t.Vector(), t.Size(), t.Descriptor())
	cpuOut := cpuOp.Sum(cpuIn, axis)

	out := o.Create(cpuOut.Size())
	o.Init(out, cpuOut.Vector())

	// Dispatch to accelerator for timing.
	// The reduction reads all input elements and writes a smaller output.
	inSize := sizeToU32x4(t.Size())
	outSize := sizeToU32x4(cpuOut.Size())

	o.driver.AccelInference(
		o.ctx,
		protocol.AccelOpReduction,
		protocol.AccelOpParams{},
		uint64(ptrOf(t)),
		uint64(ptrOf(out)),
		0, 0,
		inSize, outSize, [4]uint32{},
	)

	return out
}

// MaxPoolingForward performs max pooling forward pass via the accelerator.
func (o *Operator) MaxPoolingForward(
	t tensor.Tensor,
	kernelSize, padding, stride []int,
) (tensor.Tensor, tensor.Tensor) {
	cpuOp := tensor.CPUOperator{}
	cpuIn := cpuOp.CreateWithData(t.Vector(), t.Size(), t.Descriptor())
	cpuOut, cpuMask := cpuOp.MaxPoolingForward(
		cpuIn, kernelSize, padding, stride)

	out := o.Create(cpuOut.Size())
	o.Init(out, cpuOut.Vector())

	mask := o.Create(cpuMask.Size())
	o.Init(mask, cpuMask.Vector())

	// Dispatch to accelerator for timing.
	inSize := sizeToU32x4(t.Size())
	outSize := sizeToU32x4(cpuOut.Size())

	o.driver.AccelInference(
		o.ctx,
		protocol.AccelOpMaxPool,
		protocol.AccelOpParams{
			KernelSize: toU32x2(kernelSize),
			Stride:     toU32x2(stride),
			Padding:    toU32x2(padding),
		},
		uint64(ptrOf(t)),
		uint64(ptrOf(out)),
		0, 0,
		inSize, outSize, [4]uint32{},
	)

	return out, mask
}

// MaxPoolingBackward performs max pooling backward pass via the accelerator.
func (o *Operator) MaxPoolingBackward(
	forwardIn, backwardIn, mask tensor.Tensor,
	kernelSize, padding, stride []int,
) tensor.Tensor {
	cpuOp := tensor.CPUOperator{}
	cpuFwd := cpuOp.CreateWithData(
		forwardIn.Vector(), forwardIn.Size(), "")
	cpuBwd := cpuOp.CreateWithData(
		backwardIn.Vector(), backwardIn.Size(), "")
	cpuMask := cpuOp.CreateWithData(
		mask.Vector(), mask.Size(), "")
	cpuOut := cpuOp.MaxPoolingBackward(
		cpuFwd, cpuBwd, cpuMask, kernelSize, padding, stride)

	out := o.Create(cpuOut.Size())
	o.Init(out, cpuOut.Vector())

	// Dispatch to accelerator for timing.
	// Backward uses 3 inputs: forwardIn, backwardIn, mask.
	inSize := sizeToU32x4(forwardIn.Size())
	outSize := sizeToU32x4(cpuOut.Size())
	bwdSize := sizeToU32x4(backwardIn.Size())

	o.driver.AccelInference(
		o.ctx,
		protocol.AccelOpMaxPool,
		protocol.AccelOpParams{
			KernelSize: toU32x2(kernelSize),
			Stride:     toU32x2(stride),
			Padding:    toU32x2(padding),
		},
		uint64(ptrOf(forwardIn)),
		uint64(ptrOf(out)),
		uint64(ptrOf(backwardIn)),
		uint64(ptrOf(mask)),
		inSize, outSize, bwdSize,
	)

	return out
}

// AvgPoolingForward performs average pooling forward pass via the accelerator.
func (o *Operator) AvgPoolingForward(
	t tensor.Tensor,
	kernelSize, padding, stride []int,
) tensor.Tensor {
	cpuOp := tensor.CPUOperator{}
	cpuIn := cpuOp.CreateWithData(t.Vector(), t.Size(), t.Descriptor())
	cpuOut := cpuOp.AvgPoolingForward(
		cpuIn, kernelSize, padding, stride)

	out := o.Create(cpuOut.Size())
	o.Init(out, cpuOut.Vector())

	// Dispatch to accelerator for timing.
	inSize := sizeToU32x4(t.Size())
	outSize := sizeToU32x4(cpuOut.Size())

	o.driver.AccelInference(
		o.ctx,
		protocol.AccelOpAvgPool,
		protocol.AccelOpParams{
			KernelSize: toU32x2(kernelSize),
			Stride:     toU32x2(stride),
			Padding:    toU32x2(padding),
		},
		uint64(ptrOf(t)),
		uint64(ptrOf(out)),
		0, 0,
		inSize, outSize, [4]uint32{},
	)

	return out
}

// AvgPoolingBackward performs average pooling backward pass via the
// accelerator.
func (o *Operator) AvgPoolingBackward(
	forwardIn, backwardIn tensor.Tensor,
	kernelSize, padding, stride []int,
) tensor.Tensor {
	cpuOp := tensor.CPUOperator{}
	cpuFwd := cpuOp.CreateWithData(
		forwardIn.Vector(), forwardIn.Size(), "")
	cpuBwd := cpuOp.CreateWithData(
		backwardIn.Vector(), backwardIn.Size(), "")
	cpuOut := cpuOp.AvgPoolingBackward(
		cpuFwd, cpuBwd, kernelSize, padding, stride)

	out := o.Create(cpuOut.Size())
	o.Init(out, cpuOut.Vector())

	// Dispatch to accelerator for timing.
	// Backward uses 2 inputs: forwardIn, backwardIn.
	inSize := sizeToU32x4(forwardIn.Size())
	outSize := sizeToU32x4(cpuOut.Size())
	bwdSize := sizeToU32x4(backwardIn.Size())

	o.driver.AccelInference(
		o.ctx,
		protocol.AccelOpAvgPool,
		protocol.AccelOpParams{
			KernelSize: toU32x2(kernelSize),
			Stride:     toU32x2(stride),
			Padding:    toU32x2(padding),
		},
		uint64(ptrOf(forwardIn)),
		uint64(ptrOf(out)),
		uint64(ptrOf(backwardIn)),
		0,
		inSize, outSize, bwdSize,
	)

	return out
}

// Softmax computes softmax via the accelerator.
func (o *Operator) Softmax(t tensor.Tensor) tensor.Tensor {
	size := t.Size()
	inData := t.Vector()
	outData := make([]float64, len(inData))

	// Functional computation: per-row softmax with numerical stability.
	for i := 0; i < size[0]; i++ {
		start := i * size[1]
		end := start + size[1]

		// Find max for numerical stability.
		maxVal := inData[start]
		for j := start + 1; j < end; j++ {
			if inData[j] > maxVal {
				maxVal = inData[j]
			}
		}

		sum := 0.0
		for j := start; j < end; j++ {
			sum += math.Exp(inData[j] - maxVal)
		}

		for j := start; j < end; j++ {
			outData[j] = math.Exp(inData[j]-maxVal) / sum
		}
	}

	out := o.Create(size)
	o.Init(out, outData)

	// Dispatch to accelerator for timing.
	inSize := sizeToU32x4(size)

	o.driver.AccelInference(
		o.ctx,
		protocol.AccelOpSoftmax,
		protocol.AccelOpParams{},
		uint64(ptrOf(t)),
		uint64(ptrOf(out)),
		0, 0,
		inSize, inSize, [4]uint32{},
	)

	return out
}

// CrossEntropy computes cross entropy loss via the accelerator.
// Returns a scalar (float64). The accelerator dispatch models the time to
// read the prediction tensor and compute the per-sample log-loss.
func (o *Operator) CrossEntropy(t tensor.Tensor, label []int) float64 {
	size := t.Size()
	data := t.Vector()

	// Functional computation on CPU.
	loss := 0.0
	for i := 0; i < size[0]; i++ {
		idx := i*size[1] + label[i]
		loss += -math.Log(data[idx])
	}

	loss /= float64(size[0])

	// Dispatch to accelerator for timing.
	// Cross-entropy reads the full prediction tensor (batch × classes).
	inSize := sizeToU32x4(size)

	o.driver.AccelInference(
		o.ctx,
		protocol.AccelOpCrossEntropy,
		protocol.AccelOpParams{},
		uint64(ptrOf(t)),
		0, 0, 0,
		inSize, [4]uint32{1, 1, 1, 1}, [4]uint32{},
	)

	return loss
}

// CrossEntropyDerivative computes cross entropy derivative via the
// accelerator.
func (o *Operator) CrossEntropyDerivative(
	t tensor.Tensor, label []int,
) tensor.Tensor {
	size := t.Size()
	inData := t.Vector()
	outData := make([]float64, len(inData))

	// Functional computation on CPU.
	for i := 0; i < size[0]; i++ {
		idx := i*size[1] + label[i]
		outData[idx] = -1 / inData[idx]
	}

	out := o.Create(size)
	o.Init(out, outData)

	// Dispatch to accelerator for timing.
	inSize := sizeToU32x4(size)

	o.driver.AccelInference(
		o.ctx,
		protocol.AccelOpCrossEntropyDeriv,
		protocol.AccelOpParams{},
		uint64(ptrOf(t)),
		uint64(ptrOf(out)),
		0, 0,
		inSize, inSize, [4]uint32{},
	)

	return out
}

// SoftmaxCrossEntropyDerivative computes fused softmax + cross entropy
// derivative via the accelerator.
func (o *Operator) SoftmaxCrossEntropyDerivative(
	t tensor.Tensor, label []int,
) tensor.Tensor {
	inData := t.Vector()
	size := t.Size()
	outData := make([]float64, len(inData))

	// Functional computation on CPU.
	for i := 0; i < size[0]; i++ {
		for j := 0; j < size[1]; j++ {
			idx := i*size[1] + j
			if label[i] == j {
				outData[idx] = inData[idx] - 1
			} else {
				outData[idx] = inData[idx]
			}
		}
	}

	out := o.Create(size)
	o.Init(out, outData)

	// Dispatch to accelerator for timing.
	inSize := sizeToU32x4(size)

	o.driver.AccelInference(
		o.ctx,
		protocol.AccelOpSoftmaxCrossEntropyDeriv,
		protocol.AccelOpParams{},
		uint64(ptrOf(t)),
		uint64(ptrOf(out)),
		0, 0,
		inSize, inSize, [4]uint32{},
	)

	return out
}

// ElementWiseMul performs element-wise multiplication via the accelerator.
func (o *Operator) ElementWiseMul(t1, t2 tensor.Tensor) tensor.Tensor {
	// Functional computation on CPU.
	d1 := t1.Vector()
	d2 := t2.Vector()
	outData := make([]float64, len(d1))

	for i := range d1 {
		outData[i] = d1[i] * d2[i]
	}

	out := o.Create(t1.Size())
	o.Init(out, outData)

	// Dispatch to accelerator for timing.
	inSize := sizeToU32x4(t1.Size())

	o.driver.AccelInference(
		o.ctx,
		protocol.AccelOpElementWise,
		protocol.AccelOpParams{},
		uint64(ptrOf(t1)),
		uint64(ptrOf(out)),
		uint64(ptrOf(t2)),
		0,
		inSize, inSize, inSize,
	)

	return out
}

// ScaleAdd performs alpha*A + beta*B via the accelerator.
func (o *Operator) ScaleAdd(
	alpha, beta float64, a, b tensor.Tensor,
) tensor.Tensor {
	// Functional computation on CPU for correctness.
	da := a.Vector()
	db := b.Vector()
	outData := make([]float64, len(da))

	for i := range da {
		outData[i] = alpha*da[i] + beta*db[i]
	}

	out := o.Create(a.Size())
	o.Init(out, outData)

	// Dispatch to accelerator for timing.
	inSize := sizeToU32x4(a.Size())

	o.driver.AccelInference(
		o.ctx,
		protocol.AccelOpScaleAdd,
		protocol.AccelOpParams{Alpha: alpha, Beta: beta},
		uint64(ptrOf(a)),
		uint64(ptrOf(out)),
		uint64(ptrOf(b)),
		0,
		inSize, inSize, inSize,
	)

	return out
}

// RMSProp runs the RMSProp optimization step via the accelerator.
// Updates params and sHistory in-place.
func (o *Operator) RMSProp(
	params, gradient, sHistory tensor.Tensor,
	smoothFactor, learningRate float64,
) {
	// Functional computation on CPU.
	p := params.Vector()
	g := gradient.Vector()
	s := sHistory.Vector()

	for i := range p {
		s[i] = smoothFactor*s[i] + (1-smoothFactor)*g[i]*g[i]
		p[i] -= learningRate * (1.0 / (math.Sqrt(s[i]*1e-8)) * g[i])
	}

	o.Init(params, p)
	o.Init(sHistory, s)

	// Dispatch to accelerator for timing.
	// RMSProp reads 3 tensors (params, gradient, sHistory) and writes 2
	// (params, sHistory). Memory-bound.
	inSize := sizeToU32x4(params.Size())

	o.driver.AccelInference(
		o.ctx,
		protocol.AccelOpRMSProp,
		protocol.AccelOpParams{
			Alpha: smoothFactor,
			Beta:  learningRate,
		},
		uint64(ptrOf(params)),
		uint64(ptrOf(sHistory)),
		uint64(ptrOf(gradient)),
		0,
		inSize, inSize, inSize,
	)
}

// Adam runs the Adam optimization step via the accelerator.
// Updates params, vHistory, and sHistory in-place.
func (o *Operator) Adam(
	params, gradients, vHistory, sHistory tensor.Tensor,
	smoothFactor1, smoothFactor2, learningRate float64,
) {
	// Functional computation on CPU.
	p := params.Vector()
	g := gradients.Vector()
	v := vHistory.Vector()
	s := sHistory.Vector()

	for i := range p {
		v[i] = smoothFactor1*v[i] + (1-smoothFactor1)*g[i]
		s[i] = smoothFactor2*s[i] + (1-smoothFactor2)*g[i]*g[i]
		p[i] -= learningRate * (1.0 / (math.Sqrt(s[i]) + 1e-8)) * v[i]
	}

	o.Init(params, p)
	o.Init(vHistory, v)
	o.Init(sHistory, s)

	// Dispatch to accelerator for timing.
	// Adam reads 4 tensors (params, gradients, vHistory, sHistory) and
	// writes 3 (params, vHistory, sHistory). Memory-bound.
	inSize := sizeToU32x4(params.Size())

	o.driver.AccelInference(
		o.ctx,
		protocol.AccelOpAdam,
		protocol.AccelOpParams{
			Alpha: smoothFactor1,
			Beta:  learningRate,
		},
		uint64(ptrOf(params)),
		uint64(ptrOf(vHistory)),
		uint64(ptrOf(gradients)),
		uint64(ptrOf(sHistory)),
		inSize, inSize, inSize,
	)
}

// ReluForward performs ReLU forward pass via the accelerator.
func (o *Operator) ReluForward(in tensor.Tensor) tensor.Tensor {
	// Functional computation on CPU.
	inData := in.Vector()
	outData := make([]float64, len(inData))

	for i, v := range inData {
		if v > 0 {
			outData[i] = v
		}
	}

	out := o.Create(in.Size())
	out.SetDescriptor(in.Descriptor())
	o.Init(out, outData)

	// Dispatch to accelerator for timing.
	inSize := sizeToU32x4(in.Size())

	o.driver.AccelInference(
		o.ctx,
		protocol.AccelOpReLU,
		protocol.AccelOpParams{},
		uint64(ptrOf(in)),
		uint64(ptrOf(out)),
		0, 0,
		inSize, inSize, [4]uint32{},
	)

	return out
}

// ReluBackward performs ReLU backward pass via the accelerator.
// The backward pass applies the same mask as forward (zero where input < 0).
func (o *Operator) ReluBackward(
	forwardIn, backwardIn tensor.Tensor,
) tensor.Tensor {
	// Functional computation on CPU.
	fIn := forwardIn.Vector()
	bIn := backwardIn.Vector()
	outData := make([]float64, len(fIn))

	for i := range outData {
		if fIn[i] >= 0 {
			outData[i] = bIn[i]
		}
	}

	out := o.Create(forwardIn.Size())
	o.Init(out, outData)

	// Dispatch to accelerator for timing — same cost as forward (element-wise).
	inSize := sizeToU32x4(forwardIn.Size())

	o.driver.AccelInference(
		o.ctx,
		protocol.AccelOpReLU,
		protocol.AccelOpParams{},
		uint64(ptrOf(forwardIn)),
		uint64(ptrOf(out)),
		uint64(ptrOf(backwardIn)),
		0,
		inSize, inSize, inSize,
	)

	return out
}
