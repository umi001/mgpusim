// Package xor implements a extremely simple network that can perform the xor
// operation.
package xor

import (
	"encoding/json"
	"fmt"
	"log"
	"os"

	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/acceltensor"
	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/gputensor"
	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/layers"
	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/tensor"
	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/training"
	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/training/optimization"
	"github.com/sarchlab/mgpusim/v4/amd/driver"
)

// accelConfig defines the JSON structure for accelerator configuration.
type accelConfig struct {
	AccelLayers []int `json:"accel_layers"`
}

// Benchmark defines the XOR network training benchmark.
type Benchmark struct {
	driver  *driver.Driver
	context *driver.Context
	to      tensor.Operator

	gpus          []int
	accelLayerSet map[int]bool

	network training.Network
	trainer training.Trainer

	AccelConfigPath string
}

// NewBenchmark creates a new benchmark.
func NewBenchmark(driver *driver.Driver) *Benchmark {
	b := new(Benchmark)
	b.driver = driver
	b.context = b.driver.Init()

	return b
}

func (b *Benchmark) loadAccelConfig() map[int]bool {
	set := make(map[int]bool)

	path := b.AccelConfigPath
	if path == "" {
		path = "accel_config.json"
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return set
	}

	var cfg accelConfig

	err = json.Unmarshal(data, &cfg)
	if err != nil {
		log.Printf("Warning: invalid accel_config.json: %v", err)
		return set
	}

	for _, id := range cfg.AccelLayers {
		set[id] = true
	}

	return set
}

func (b *Benchmark) opForLayer(
	layerID int,
	gpuOp, accelOp tensor.Operator,
	current tensor.Operator,
) tensor.Operator {
	if layerID < 0 {
		return current
	}

	if b.accelLayerSet[layerID] {
		return accelOp
	}

	return gpuOp
}

func (b *Benchmark) init() {
	b.accelLayerSet = b.loadAccelConfig()

	if len(b.gpus) > 0 {
		b.driver.SelectGPU(b.context, b.gpus[0])
	}

	gpuOp := gputensor.NewGPUOperator(b.driver, b.context)
	gpuOp.EnableVerification()

	var accelOp tensor.Operator
	if b.driver.GetNumAccelerators() > 0 && len(b.accelLayerSet) > 0 {
		b.driver.SelectAccelerator(b.context, 0)
		accelOp = acceltensor.NewOperator(b.driver, b.context)
	} else {
		accelOp = gpuOp
	}

	cur := b.opForLayer(0, gpuOp, accelOp, gpuOp)
	b.to = cur

	b.network = training.Network{
		Layers: []layers.Layer{
			layers.NewFullyConnectedLayer(0, cur, 2, 4),
			layers.NewReluLayer(cur),
			func() layers.Layer {
				cur = b.opForLayer(2, gpuOp, accelOp, cur)
				return layers.NewFullyConnectedLayer(2, cur, 4, 2)
			}(),
		},
	}

	b.trainer = training.Trainer{
		TO:              gpuOp,
		DataSource:      NewDataSource(gpuOp),
		Network:         b.network,
		LossFunc:        training.NewSoftmaxCrossEntropy(gpuOp),
		OptimizationAlg: optimization.NewAdam(gpuOp, 0.03),
		Epoch:           50,
		BatchSize:       4,
		ShowBatchInfo:   true,
	}

	b.enableLayerVerification(&b.network)
}

func (b *Benchmark) enableLayerVerification(network *training.Network) {

}

// SelectGPU selects the GPU to use.
func (b *Benchmark) SelectGPU(gpuIDs []int) {
	if len(gpuIDs) > 1 {
		panic("multi-GPU is not supported by DNN workloads")
	}

	b.gpus = gpuIDs
}

// Run executes the benchmark.
func (b *Benchmark) Run() {
	b.init()

	for _, l := range b.network.Layers {
		l.Randomize()
	}

	b.trainer.Train()
}

func (b *Benchmark) printLayerParams() {
	for i, l := range b.network.Layers {
		params := l.Parameters()
		if params != nil {
			fmt.Println("Layer ", i, params.Vector())
		}
	}
}

// Verify runs the benchmark on the CPU and checks the result.
func (b *Benchmark) Verify() {
	panic("not implemented")
}

// SetUnifiedMemory asks the benchmark to use unified memory.
func (b *Benchmark) SetUnifiedMemory() {
	panic("unified memory is not supported by dnn workloads")
}
