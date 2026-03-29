// Package lenet implements lenet network training.
package lenet

import (
	"encoding/json"
	"log"
	"math"
	"os"

	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/acceltensor"
	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/gputensor"
	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/tensor"
	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/mccl"

	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/dataset/mnist"
	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/gputraining"
	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/layers"
	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/training"
	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/training/optimization"
	"github.com/sarchlab/mgpusim/v4/amd/driver"
)

// accelConfig defines the JSON structure for accelerator configuration.
type accelConfig struct {
	AccelLayers []int `json:"accel_layers"`
}

// Benchmark defines the LeNet network training benchmark.
type Benchmark struct {
	driver   *driver.Driver
	ctx      *driver.Context
	to       []tensor.Operator
	gpus     []int
	contexts []*driver.Context

	networks      []training.Network
	trainer       gputraining.DataParallelismMultiGPUTrainer
	accelLayerSet map[int]bool

	BatchSize          int
	Epoch              int
	MaxBatchPerEpoch   int
	EnableVerification bool
	EnableTesting      bool
	AccelConfigPath    string
}

// NewBenchmark creates a new benchmark.
func NewBenchmark(driver *driver.Driver) *Benchmark {
	b := new(Benchmark)

	b.driver = driver
	b.ctx = driver.Init()

	return b
}

// SelectGPU selects the GPU to use.
func (b *Benchmark) SelectGPU(gpuIDs []int) {
	b.gpus = gpuIDs
}

func (b *Benchmark) init() {
	b.accelLayerSet = b.loadAccelConfig()

	for _, gpu := range b.gpus {
		b.defineNetwork(gpu)
	}

	b.createTrainer()
	b.randomizeParams()
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

// opForLayer returns the accelerator operator if the layer ID is in the
// accel set, otherwise the GPU operator.
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

func (b *Benchmark) defineNetwork(gpuID int) {
	context := b.driver.InitWithExistingPID(b.ctx)
	b.driver.SelectGPU(context, gpuID)

	gpuOp := gputensor.NewGPUOperator(b.driver, context)
	if b.EnableVerification {
		gpuOp.EnableVerification()
	}

	var accelOp tensor.Operator
	if b.driver.GetNumAccelerators() > 0 && len(b.accelLayerSet) > 0 {
		accelIdx := len(b.networks) % b.driver.GetNumAccelerators()
		b.driver.SelectAccelerator(context, accelIdx)
		accelOp = acceltensor.NewOperator(b.driver, context)
	} else {
		accelOp = gpuOp
	}

	// cur tracks the active operator; layers without an explicit ID
	// (ReLU, AvgPooling) inherit the operator of the preceding layer.
	cur := b.opForLayer(0, gpuOp, accelOp, gpuOp)

	network := training.Network{
		Layers: []layers.Layer{
			layers.NewConv2D(0, cur,
				[]int{1, 28, 28}, []int{6, 1, 5, 5},
				[]int{1, 1}, []int{2, 2}),
			layers.NewReluLayer(cur),
			layers.NewAvgPoolingLayer(cur,
				[]int{2, 2}, []int{0, 0}, []int{2, 2}),

			func() layers.Layer {
				cur = b.opForLayer(3, gpuOp, accelOp, cur)
				return layers.NewConv2D(3, cur,
					[]int{6, 14, 14}, []int{16, 6, 5, 5},
					[]int{1, 1}, []int{0, 0})
			}(),
			layers.NewReluLayer(cur),
			layers.NewAvgPoolingLayer(cur,
				[]int{2, 2}, []int{0, 0}, []int{2, 2}),

			func() layers.Layer {
				cur = b.opForLayer(6, gpuOp, accelOp, cur)
				return layers.NewFullyConnectedLayer(6, cur, 400, 120)
			}(),
			layers.NewReluLayer(cur),

			func() layers.Layer {
				cur = b.opForLayer(8, gpuOp, accelOp, cur)
				return layers.NewFullyConnectedLayer(8, cur, 120, 84)
			}(),
			layers.NewReluLayer(cur),

			func() layers.Layer {
				cur = b.opForLayer(10, gpuOp, accelOp, cur)
				return layers.NewFullyConnectedLayer(10, cur, 84, 10)
			}(),
		},
	}

	b.networks = append(b.networks, network)
	b.contexts = append(b.contexts, context)
	b.to = append(b.to, gpuOp)
}

func (b *Benchmark) createTrainer() {
	sources := make([]training.DataSource, len(b.networks))
	alg := make([]optimization.Alg, len(b.networks))
	testers := make([]*training.Tester, len(b.networks))
	lossFuncs := make([]training.LossFunction, len(b.networks))

	for i := 0; i < len(b.networks); i++ {
		sources[i] = mnist.NewTrainingDataSource(b.to[i])
		alg[i] = optimization.NewAdam(b.to[i], 0.001)
		lossFuncs[i] = training.NewSoftmaxCrossEntropy(b.to[i])

		if b.EnableTesting {
			testers[i] = &training.Tester{
				DataSource: mnist.NewTestDataSource(b.to[i]),
				Network:    b.networks[i],
				BatchSize:  math.MaxInt32,
			}
		}
	}

	b.trainer = gputraining.DataParallelismMultiGPUTrainer{
		TensorOperators:  b.to,
		DataSource:       sources,
		Networks:         b.networks,
		LossFunc:         lossFuncs,
		OptimizationAlg:  alg,
		Tester:           testers,
		Epoch:            b.Epoch,
		MaxBatchPerEpoch: b.MaxBatchPerEpoch,
		BatchSize:        b.BatchSize,
		ShowBatchInfo:    true,
		GPUs:             b.gpus,
		Contexts:         b.contexts,
		Driver:           b.driver,
	}
}

func (b *Benchmark) randomizeParams() {
	initNet := b.networks[0]
	for _, l := range initNet.Layers {
		l.Randomize()
	}

	gpuNum := len(b.networks)

	for i := range b.networks[0].Layers {
		if b.networks[0].Layers[i].Parameters() == nil {
			continue
		}

		params := make([]tensor.DeviceTensor, gpuNum)
		datas := make([]driver.Ptr, gpuNum)

		for j := 0; j < gpuNum; j++ {
			params[j] = b.networks[j].Layers[i].
				Parameters().(tensor.DeviceTensor)
		}

		dataSizeArr := params[0].Size()
		dataSize := 1
		for i := 0; i < len(dataSizeArr); i++ {
			dataSize *= dataSizeArr[i]
		}

		for i := 0; i < len(params); i++ {
			datas[i] = params[i].Ptr()
		}
		comms := mccl.CommInitAllMultipleContexts(
			gpuNum, b.driver, b.contexts, b.gpus)
		mccl.BroadcastRing(b.driver, comms, 1, datas, dataSize)
	}
}

// Run executes the benchmark.
func (b *Benchmark) Run() {
	b.init()
	b.trainer.Train()
}

// Verify runs the benchmark on the CPU and checks the result.
func (b *Benchmark) Verify() {
	panic("not implemented")
}

// SetUnifiedMemory asks the benchmark to use unified memory.
func (b *Benchmark) SetUnifiedMemory() {
	panic("unified memory is not supported by dnn workloads")
}
