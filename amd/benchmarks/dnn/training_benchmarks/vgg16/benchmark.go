// Package vgg16 implements VGG16 network training.
package vgg16

import (
	"encoding/json"
	"log"
	"math"
	"os"

	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/acceltensor"
	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/dataset/imagenet"
	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/gputensor"
	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/gputraining"
	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/layers"
	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/tensor"
	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/training"
	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/training/optimization"
	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/mccl"
	"github.com/sarchlab/mgpusim/v4/amd/driver"
)

// accelConfig defines the JSON structure for accelerator configuration.
type accelConfig struct {
	AccelLayers []int `json:"accel_layers"`
}

// Benchmark defines the VGG16 network training benchmark.
type Benchmark struct {
	driver   *driver.Driver
	ctx      *driver.Context
	to       []tensor.Operator
	gpus     []int
	contexts []*driver.Context

	networks []training.Network
	trainer  gputraining.DataParallelismMultiGPUTrainer

	accelLayerSet map[int]bool

	BatchSize          int
	Epoch              int
	MaxBatchPerEpoch   int
	EnableTesting      bool
	EnableVerification bool
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
		// No config file — all layers on GPU (backward compatible).
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
// accel set, otherwise the GPU operator. For layers without an explicit ID
// (ReLU, MaxPooling), pass -1 and the current operator is kept.
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

	// Create accelerator operator if accelerator is available and
	// configured. Falls back to GPU operator if no accelerator exists.
	var accelOp tensor.Operator
	if b.driver.GetNumAccelerators() > 0 && len(b.accelLayerSet) > 0 {
		b.driver.SelectAccelerator(context, 0)
		accelOp = acceltensor.NewOperator(b.driver, context)
	} else {
		accelOp = gpuOp
	}

	// cur tracks the active operator; layers without an explicit ID
	// (ReLU, MaxPooling) inherit the operator of the preceding layer.
	cur := b.opForLayer(0, gpuOp, accelOp, gpuOp)

	network := training.Network{
		Layers: []layers.Layer{
			layers.NewConv2D(0, cur, []int{3, 224, 224}, []int{64, 3, 3, 3}, []int{1, 1}, []int{1, 1}),
			layers.NewReluLayer(cur),

			func() layers.Layer {
				cur = b.opForLayer(2, gpuOp, accelOp, cur)
				return layers.NewConv2D(2, cur, []int{64, 224, 224}, []int{64, 64, 3, 3}, []int{1, 1}, []int{1, 1})
			}(),
			layers.NewReluLayer(cur),
			layers.NewMaxPoolingLayer(cur, []int{2, 2}, []int{0, 0}, []int{2, 2}),

			func() layers.Layer {
				cur = b.opForLayer(5, gpuOp, accelOp, cur)
				return layers.NewConv2D(5, cur, []int{64, 112, 112}, []int{128, 64, 3, 3}, []int{1, 1}, []int{1, 1})
			}(),
			layers.NewReluLayer(cur),
			func() layers.Layer {
				cur = b.opForLayer(7, gpuOp, accelOp, cur)
				return layers.NewConv2D(7, cur, []int{128, 112, 112}, []int{128, 128, 3, 3}, []int{1, 1}, []int{1, 1})
			}(),
			layers.NewReluLayer(cur),
			func() layers.Layer {
				cur = b.opForLayer(9, gpuOp, accelOp, cur)
				return layers.NewConv2D(9, cur, []int{128, 112, 112}, []int{128, 128, 3, 3}, []int{1, 1}, []int{1, 1})
			}(),
			layers.NewReluLayer(cur),
			layers.NewMaxPoolingLayer(cur, []int{2, 2}, []int{0, 0}, []int{2, 2}),

			func() layers.Layer {
				cur = b.opForLayer(12, gpuOp, accelOp, cur)
				return layers.NewConv2D(12, cur, []int{128, 56, 56}, []int{256, 128, 3, 3}, []int{1, 1}, []int{1, 1})
			}(),
			layers.NewReluLayer(cur),
			func() layers.Layer {
				cur = b.opForLayer(14, gpuOp, accelOp, cur)
				return layers.NewConv2D(14, cur, []int{256, 56, 56}, []int{256, 256, 3, 3}, []int{1, 1}, []int{1, 1})
			}(),
			layers.NewReluLayer(cur),
			func() layers.Layer {
				cur = b.opForLayer(16, gpuOp, accelOp, cur)
				return layers.NewConv2D(16, cur, []int{256, 56, 56}, []int{256, 256, 3, 3}, []int{1, 1}, []int{1, 1})
			}(),
			layers.NewReluLayer(cur),
			layers.NewMaxPoolingLayer(cur, []int{2, 2}, []int{0, 0}, []int{2, 2}),

			func() layers.Layer {
				cur = b.opForLayer(19, gpuOp, accelOp, cur)
				return layers.NewConv2D(19, cur, []int{256, 28, 28}, []int{512, 256, 3, 3}, []int{1, 1}, []int{1, 1})
			}(),
			layers.NewReluLayer(cur),
			func() layers.Layer {
				cur = b.opForLayer(21, gpuOp, accelOp, cur)
				return layers.NewConv2D(21, cur, []int{512, 28, 28}, []int{512, 512, 3, 3}, []int{1, 1}, []int{1, 1})
			}(),
			layers.NewReluLayer(cur),
			func() layers.Layer {
				cur = b.opForLayer(23, gpuOp, accelOp, cur)
				return layers.NewConv2D(23, cur, []int{512, 28, 28}, []int{512, 512, 3, 3}, []int{1, 1}, []int{1, 1})
			}(),
			layers.NewReluLayer(cur),
			layers.NewMaxPoolingLayer(cur, []int{2, 2}, []int{0, 0}, []int{2, 2}),

			func() layers.Layer {
				cur = b.opForLayer(26, gpuOp, accelOp, cur)
				return layers.NewConv2D(26, cur, []int{512, 14, 14}, []int{512, 512, 3, 3}, []int{1, 1}, []int{1, 1})
			}(),
			layers.NewReluLayer(cur),
			func() layers.Layer {
				cur = b.opForLayer(27, gpuOp, accelOp, cur)
				return layers.NewConv2D(27, cur, []int{512, 14, 14}, []int{512, 512, 3, 3}, []int{1, 1}, []int{1, 1})
			}(),
			layers.NewReluLayer(cur),
			func() layers.Layer {
				cur = b.opForLayer(29, gpuOp, accelOp, cur)
				return layers.NewConv2D(29, cur, []int{512, 14, 14}, []int{512, 512, 3, 3}, []int{1, 1}, []int{1, 1})
			}(),
			layers.NewReluLayer(cur),
			layers.NewMaxPoolingLayer(cur, []int{2, 2}, []int{0, 0}, []int{2, 2}),

			func() layers.Layer {
				cur = b.opForLayer(32, gpuOp, accelOp, cur)
				return layers.NewFullyConnectedLayer(32, cur, 7*7*512, 2*2*512)
			}(),
			layers.NewReluLayer(cur),
			func() layers.Layer {
				cur = b.opForLayer(34, gpuOp, accelOp, cur)
				return layers.NewFullyConnectedLayer(34, cur, 2*2*512, 200)
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
		sources[i] = imagenet.NewTrainingDataSource(b.to[i])
		alg[i] = optimization.NewAdam(b.to[i], 0.001)
		lossFuncs[i] = training.NewSoftmaxCrossEntropy(b.to[i])

		if b.EnableTesting {
			testers[i] = &training.Tester{
				DataSource: imagenet.NewTestDataSource(b.to[i]),
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
