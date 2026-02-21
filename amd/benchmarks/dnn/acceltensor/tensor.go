// Package acceltensor provides a tensor.Tensor implementation backed by
// device memory, and a tensor.Operator that dispatches operations to a
// hardware inference accelerator via the driver's accelerator API.
//
// The benchmark code (e.g., VGG16) does NOT change — only the operator
// instantiation is swapped:
//
//	// GPU:   to := gputensor.NewGPUOperator(driver, ctx)
//	// Accel: to := acceltensor.NewOperator(driver, ctx)
package acceltensor

import (
	"github.com/sarchlab/mgpusim/v4/amd/driver"
)

// Tensor is a tensor stored in device memory, identical in structure to
// gputensor.Tensor.
type Tensor struct {
	driver     *driver.Driver
	ctx        *driver.Context
	size       []int
	ptr        driver.Ptr
	descriptor string
}

// Dim returns the number of dimensions.
func (t *Tensor) Dim() int {
	return len(t.size)
}

// Size returns the tensor dimensions.
func (t *Tensor) Size() []int {
	return t.size
}

// SetSize updates the tensor dimensions without reallocating.
func (t *Tensor) SetSize(size []int) {
	t.size = size
}

// NumElement returns the total number of elements.
func (t *Tensor) NumElement() int {
	n := 1
	for _, s := range t.size {
		n *= s
	}
	return n
}

// Descriptor returns the tensor descriptor string (e.g., "NCHW").
func (t *Tensor) Descriptor() string {
	return t.descriptor
}

// SetDescriptor sets the tensor descriptor.
func (t *Tensor) SetDescriptor(d string) {
	t.descriptor = d
}

// Ptr returns the device memory pointer.
func (t *Tensor) Ptr() driver.Ptr {
	return t.ptr
}

// Vector copies the tensor data back to host as float64 slice.
func (t *Tensor) Vector() []float64 {
	raw := make([]float32, t.NumElement())
	t.driver.MemCopyD2H(t.ctx, raw, t.ptr)

	out := make([]float64, t.NumElement())
	for i := 0; i < t.NumElement(); i++ {
		out[i] = float64(raw[i])
	}
	return out
}
