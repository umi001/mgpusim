// Package minerva implements minerva network training.
package minerva

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

// Benchmark defines the Mineva network training benchmark.
type Benchmark struct {
	driver   *driver.Driver
	ctx      *driver.Context
	to       []tensor.Operator
	lastOp   []tensor.Operator // operator of the last layer (for loss func)
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
) tensor.Operator {
	if b.accelLayerSet[layerID] {
		return accelOp
	}

	return gpuOp
}

// pickReluOp chooses the operator for a ReLU layer sitting between two FC
// layers. During the backward pass the ReLU receives a tensor produced by the
// layer *after* it (nextOp). If there is an operator mismatch at the boundary
// (one is GPU, the other is accel), we must use the accel operator because it
// can handle both tensor types via ptrOf, whereas the GPU operator only
// accepts *gputensor.Tensor.
func (b *Benchmark) pickReluOp(
	currentOp, nextOp, accelOp tensor.Operator,
) tensor.Operator {
	if currentOp == accelOp || nextOp == accelOp {
		return accelOp
	}

	return currentOp
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
		b.driver.SelectAccelerator(context, 0)
		accelOp = acceltensor.NewOperator(b.driver, context)
	} else {
		accelOp = gpuOp
	}

	op0 := b.opForLayer(0, gpuOp, accelOp)
	op2 := b.opForLayer(2, gpuOp, accelOp)
	op4 := b.opForLayer(4, gpuOp, accelOp)
	op6 := b.opForLayer(6, gpuOp, accelOp)

	// ReLU layers use the operator of the *next* FC layer when there is a
	// boundary crossing, because during backward pass the ReLU receives a
	// tensor from the layer behind it. The accel operator can handle both
	// tensor types via ptrOf/DeviceTensor, but the GPU operator cannot.
	relu0Op := b.pickReluOp(op0, op2, accelOp)
	relu2Op := b.pickReluOp(op2, op4, accelOp)
	relu4Op := b.pickReluOp(op4, op6, accelOp)

	network := training.Network{
		Layers: []layers.Layer{
			layers.NewFullyConnectedLayer(0, op0, 784, 256),
			layers.NewReluLayer(relu0Op),
			layers.NewFullyConnectedLayer(2, op2, 256, 100),
			layers.NewReluLayer(relu2Op),
			layers.NewFullyConnectedLayer(4, op4, 100, 100),
			layers.NewReluLayer(relu4Op),
			layers.NewFullyConnectedLayer(6, op6, 100, 10),
		},
	}

	b.networks = append(b.networks, network)
	b.contexts = append(b.contexts, context)
	b.to = append(b.to, gpuOp)
	b.lastOp = append(b.lastOp, op6)
}

func (b *Benchmark) createTrainer() {
	sources := make([]training.DataSource, len(b.networks))
	alg := make([]optimization.Alg, len(b.networks))
	testers := make([]*training.Tester, len(b.networks))
	lossFuncs := make([]training.LossFunction, len(b.networks))

	for i := 0; i < len(b.networks); i++ {
		sources[i] = mnist.NewTrainingDataSource(b.to[i])
		alg[i] = optimization.NewAdam(b.to[i], 0.001)
		lossFuncs[i] = training.NewSoftmaxCrossEntropy(b.lastOp[i])

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
